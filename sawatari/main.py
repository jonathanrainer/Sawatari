import csv

import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

from pathlib import Path

from pyexcel_xlsx import get_data


class Sawatari(object):

    raw_gap_data_filename = "/home/jonathanrainer/Documents/PhD/Ichijou/data/gap_results.xlsx"
    raw_results_data_filename = "/home/jonathanrainer/Documents/PhD/Ichijou/data/results.xlsx"
    variants = {
        "sc_dm": "Standard Cache - DM",
        "cc_dm": "Complex Cache - DM",
        "sc_nway": "Standard Cache - Nway",
        "cc_nway": "Complex Cache - Nway"
    }
    display_names = {
        "sc_dm": "Direct-Mapped Cache",
        "cc_dm": "Direct-Mapped TAC",
        "sc_nway": "Set-Associative Cache",
        "cc_nway": "Set-Associative TAC"
    }
    colours_and_labels = [
        ("black", "Program Has Completed"),
        ("white", "black", "Memory Idle"),
        ("blue", "Memory Active")
    ]

    def plot_heatmaps(self, benchmarks, results_folder):
        for benchmark in benchmarks:
            benchmark_data = {}
            for variant in self.variants.items():
                data = s.amass_data(benchmark, variant, results_folder)
                benchmark_data[variant[0]] = data
            benchmark_data = self.make_data_same_length(benchmark_data)
            self.plot_data(benchmark_data, benchmark)

    def amass_data(self, benchmark, variant, results_folder):
        data = [[], [], []]
        start_time = 0

        # Get Runtime
        runtime_data = get_data(self.raw_results_data_filename)
        runtime = [x for x in runtime_data[variant[1]] if x and x[1] == benchmark][0][2]

        with open(str(Path(results_folder, "{}_{}_mg_results.csv".format(benchmark, variant[0])).absolute()), "r") as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if i > 0:
                    data[0].append(int(row[0]))
                    data[1].append(int(row[1]))
                    data[2].append(int(row[3]))
                    start_time = int(row[-1])

        # Correct the initial gap so that it only starts counting when the runtime does, otherwise we could miss stuff
        if data[0][0] > 0:
            data[0][0] = data[1][0] - 1 - start_time

        plotting_results = []
        for i in range(0, len(data[0])):
            plotting_results += ([0.0] * int(data[0][i]))
            if int(data[0][i]) == 0 and int(data[1][i]) == 0:
                return [0.0] * runtime
            try:
                transaction_length = (int(data[1][i + 1]) - 1 - int(data[0][i + 1])) - (int(data[1][i]) - 1)
            except IndexError:
                transaction_length = (int(data[2][i]) + 1) - (int(data[1][i]) - 1)
            plotting_results += ([1.0] * transaction_length)
        return plotting_results

    def make_data_same_length(self, benchmark_data):
        desired_data_length = max([len(x) for x in benchmark_data.values()])
        return {key: self.expand_data(value, desired_data_length) for key, value in benchmark_data.items()}

    @staticmethod
    def expand_data(list_of_values, desired_length):
        if len(list_of_values) < desired_length:
            return list_of_values + [-1.0] * (desired_length - len(list_of_values))
        else:
            return list_of_values

    def plot_data(self, data, benchmark):
        fig, axs = plt.subplots(len(data), 1, sharex=True)
        fig.set_size_inches(16, 11)
        for i, (variant_key, variant_data) in enumerate(data.items()):
            cmap = col.ListedColormap([x[0] for x in self.colours_and_labels])
            sns.heatmap(np.asarray(variant_data).reshape(1, len(variant_data)), cmap=cmap, ax=axs[i], cbar=False,
                        vmin=-1.0, vmax=1)
            axs[i].set_ylabel(self.display_names[variant_key], rotation=0, labelpad=60, wrap=True)
            axs[i].set_yticks([])
        # Sort Graph Formatting
        max_x = max([len(vals) for _, vals in data.items()])
        plt.xlim(right=max_x)
        plt.xticks(np.arange(0, max_x, step=max_x/30),
                   ["{:4.0f}".format(x) for x in np.arange(0, max_x, step=max_x/30)])

        patches = [mpatches.Patch(color=x[0], label=x[1]) if len(x) == 2 else
                   mpatches.Patch(facecolor=x[0], edgecolor=x[1], label=x[2])
                   for x in self.colours_and_labels]
        fig.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, 0.95),
                   ncol=len(self.colours_and_labels), fontsize="large", edgecolor="black")
        plt.subplots_adjust(top=0.9)

        plt.xlabel("Clock Cycles Elapsed")
        fig.suptitle("Heatmap of Memory Activity for benchmark {}".format(benchmark))
        fig.savefig("output/{}_memory_results.png".format(benchmark))


if __name__ == "__main__":
    s = Sawatari()
    s.plot_heatmaps(["fac", "ns", "fft1", "bsort100", "adpcm", "janne_complex", "fibcall", "prime", "insertsort",
                     "duff", "matmult"],
                    Path("/home/jonathanrainer/Documents/PhD/Ichijou/data/Gap Results"))
