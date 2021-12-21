#!/usr/bin/env python3

import argparse
import sys
from json import loads
from subprocess import Popen, PIPE

# Holds code regions statistics.
class Summary:
    def __init__(
        self,
        name,
        block_rthroughput,
        dispatch_width,
        ipc,
        instructions,
        iterations,
        total_cycles,
        total_uops,
        uops_per_cycle,
        iteration_resource_pressure,
        name_target_info_resources,
    ):
        self.name = name
        self.block_rthroughput = block_rthroughput
        self.dispatch_width = dispatch_width
        self.ipc = ipc
        self.instructions = instructions
        self.iterations = iterations
        self.total_cycles = total_cycles
        self.total_uops = total_uops
        self.uops_per_cycle = uops_per_cycle
        self.iteration_resource_pressure = iteration_resource_pressure
        self.name_target_info_resources = name_target_info_resources


# Parse the program arguments.
def parse_program_args(parser):
    parser.add_argument(
        "file_names",
        nargs="+",
        type=str,
        help="Names of files which llvm-mca tool process.",
    )
    parser.add_argument(
        "--llvm-mca-binary",
        nargs=1,
        required=True,
        type=str,
        action="store",
        metavar="[=<path to llvm-mca>]",
        help="Specified relative path to binary of llvm-mca.",
    )
    parser.add_argument(
        "--args",
        nargs=1,
        type=str,
        action="store",
        metavar="[='-option1=<arg> -option2=<arg> ...']",
        default=["-"],
        help="Forward options to lvm-mca tool.",
    )
    parser.add_argument(
        "-v",
        action="store_true",
        default=False,
        help="More details about the running lvm-mca tool.",
    )
    return parser.parse_args()


# Returns the name of the file to be analyzed from the path it is on.
def get_filename_from_path(path):
    index_of_slash = path.rfind("/")
    return path[(index_of_slash + 1) : len(path)]


# Returns the results of the running llvm-mca tool for the input file.
def run_llvm_mca_tool(opts, file_name):
    # Get the path of the llvm-mca binary file.
    llvm_mca_cmd = opts.llvm_mca_binary[0]

    # The statistics llvm-mca options.
    if opts.args[0] != "-":
        llvm_mca_cmd += " " + opts.args[0]
    llvm_mca_cmd += " -json"

    # Set file which llvm-mca tool will process.
    llvm_mca_cmd += " " + file_name

    if opts.v:
        print("run: $ " + llvm_mca_cmd + "\n")

    # Generate the stats with the llvm-mca.
    subproc = Popen(
        llvm_mca_cmd.split(" "),
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
    )

    cmd_stdout, cmd_stderr = subproc.communicate()

    try:
        json_parsed = loads(cmd_stdout)
    except:
        print("error: No valid llvm-mca statistics found.")
        print(cmd_stderr)
        sys.exit(1)

    if opts.v:
        print("Simulation Parameters: ")
        simulation_parameters = json_parsed["SimulationParameters"]
        for key in simulation_parameters:
            print(key, ":", simulation_parameters[key])
        print("\n")

    code_regions_len = len(json_parsed["CodeRegions"])
    array_of_code_regions = [None] * code_regions_len

    for i in range(code_regions_len):
        code_region_instructions_len = len(
            json_parsed["CodeRegions"][i]["Instructions"]
        )
        target_info_resources_len = len(json_parsed["TargetInfo"]["Resources"])
        iteration_resource_pressure = ["-" for k in range(target_info_resources_len)]
        resource_pressure_info = json_parsed["CodeRegions"][i]["ResourcePressureView"][
            "ResourcePressureInfo"
        ]

        name_target_info_resources = [" "] + json_parsed["TargetInfo"]["Resources"]

        for s in range(len(resource_pressure_info)):
            obj_of_resource_pressure_info = resource_pressure_info[s]
            if (
                obj_of_resource_pressure_info["InstructionIndex"]
                == code_region_instructions_len
            ):
                iteration_resource_pressure[
                    obj_of_resource_pressure_info["ResourceIndex"]
                ] = str(round(obj_of_resource_pressure_info["ResourceUsage"], 2))

        array_of_code_regions[i] = Summary(
            file_name,
            json_parsed["CodeRegions"][i]["SummaryView"]["BlockRThroughput"],
            json_parsed["CodeRegions"][i]["SummaryView"]["DispatchWidth"],
            json_parsed["CodeRegions"][i]["SummaryView"]["IPC"],
            json_parsed["CodeRegions"][i]["SummaryView"]["Instructions"],
            json_parsed["CodeRegions"][i]["SummaryView"]["Iterations"],
            json_parsed["CodeRegions"][i]["SummaryView"]["TotalCycles"],
            json_parsed["CodeRegions"][i]["SummaryView"]["TotaluOps"],
            json_parsed["CodeRegions"][i]["SummaryView"]["uOpsPerCycle"],
            iteration_resource_pressure,
            name_target_info_resources,
        )

    return array_of_code_regions


# Print statistics in console for single file or for multiple files.
def console_print_results(matrix_of_code_regions, opts):
    try:
        import termtables as tt
    except ImportError:
        print("error: termtables not found.")
        sys.exit(1)

    headers_names = [None] * (len(opts.file_names) + 1)
    headers_names[0] = " "

    max_code_regions = 0

    print("Input files:")
    for i in range(len(matrix_of_code_regions)):
        if max_code_regions < len(matrix_of_code_regions[i]):
            max_code_regions = len(matrix_of_code_regions[i])
        print("[f" + str(i + 1) + "]: " + get_filename_from_path(opts.file_names[i]))
        headers_names[i + 1] = "[f" + str(i + 1) + "]: "

    print("\nITERATIONS: " + str(matrix_of_code_regions[0][0].iterations) + "\n")

    for i in range(max_code_regions):

        print(
            "\n-----------------------------------------\nCode region: "
            + str(i + 1)
            + "\n"
        )

        table_values = [
            [[None] for i in range(len(matrix_of_code_regions) + 1)] for j in range(7)
        ]

        table_values[0][0] = "Instructions: "
        table_values[1][0] = "Total Cycles: "
        table_values[2][0] = "Total uOps: "
        table_values[3][0] = "Dispatch Width: "
        table_values[4][0] = "uOps Per Cycle: "
        table_values[5][0] = "IPC: "
        table_values[6][0] = "Block RThroughput: "

        for j in range(len(matrix_of_code_regions)):
            if len(matrix_of_code_regions[j]) > i:
                table_values[0][j + 1] = str(matrix_of_code_regions[j][i].instructions)
                table_values[1][j + 1] = str(matrix_of_code_regions[j][i].total_cycles)
                table_values[2][j + 1] = str(matrix_of_code_regions[j][i].total_uops)
                table_values[3][j + 1] = str(
                    matrix_of_code_regions[j][i].dispatch_width
                )
                table_values[4][j + 1] = str(
                    round(matrix_of_code_regions[j][i].uops_per_cycle, 2)
                )
                table_values[5][j + 1] = str(round(matrix_of_code_regions[j][i].ipc, 2))
                table_values[6][j + 1] = str(
                    round(matrix_of_code_regions[j][i].block_rthroughput, 2)
                )
            else:
                table_values[0][j + 1] = "-"
                table_values[1][j + 1] = "-"
                table_values[2][j + 1] = "-"
                table_values[3][j + 1] = "-"
                table_values[4][j + 1] = "-"
                table_values[5][j + 1] = "-"
                table_values[6][j + 1] = "-"

        tt.print(
            table_values,
            header=headers_names,
            style=tt.styles.ascii_thin_double,
            padding=(0, 1),
        )

        print("\nResource pressure per iteration: \n")

        table_values = [
            [
                [None]
                for i in range(
                    len(matrix_of_code_regions[0][0].iteration_resource_pressure) + 1
                )
            ]
            for j in range(len(matrix_of_code_regions) + 1)
        ]

        table_values[0] = matrix_of_code_regions[0][0].name_target_info_resources

        for j in range(len(matrix_of_code_regions)):
            if len(matrix_of_code_regions[j]) > i:
                table_values[j + 1] = [
                    "[f" + str(j + 1) + "]: "
                ] + matrix_of_code_regions[j][i].iteration_resource_pressure
            else:
                table_values[j + 1] = ["[f" + str(j + 1) + "]: "] + len(
                    matrix_of_code_regions[0][0].iteration_resource_pressure
                ) * ["-"]

        tt.print(
            table_values,
            style=tt.styles.ascii_thin_double,
            padding=(0, 1),
        )
        print("\n")


def Main():
    parser = argparse.ArgumentParser()
    opts = parse_program_args(parser)

    matrix_of_code_regions = [None] * len(opts.file_names)

    for i in range(len(opts.file_names)):
        matrix_of_code_regions[i] = run_llvm_mca_tool(opts, opts.file_names[i])
    console_print_results(matrix_of_code_regions, opts)


if __name__ == "__main__":
    Main()
    sys.exit(0)
