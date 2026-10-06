# LLVM opt and llc parameter search

## Overview

`opt` and `llc` have many hidden flags, shown through the `--help-hidden` option, which can affect performance by adjusting optimization pass parameters. The goal of this repository is to provide scripts that automatically collect data about how different values of a user-defined list of `opt` and `llc` flags affects performance of OpenMP Target GPU kernels. The scripts automatically replay an isolated kernel with a predefined set of `opt`/`llc` flags and their values through AMD's [rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) and collect runtime data into a CSV file.

`autorun.py` is the primary automatic search script, which uses the [hyperopt](https://hyperopt.github.io/) library to automatically choose values for the user-defined flags within the user-defined range. The chosen values are then passed to `llc`, which then modifies the kernel image. The script then runs the modified kernel via rocprofv3 and records runtime data.

`run.py` can be used to run a search through a predefined set of `llc` or `opt` flags and their values. Whereas `autorun.py` automatically selects values, `run.py` requires the user to specify all possible flag values themselves.

## Requirements

- RoCM (rocprofv3)
- Python 3.10+
- LLVM

## Setup

### Install Python Dependencies

Make sure that the python version on your system is **at least** 3.10.

Then, install dependencies either globally, using pip:

```bash
pip install -r requirements.txt
```

Or into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Record your kernel

Firstly, when compiling your application, ensure that the `-fopenmp-target-jit` flag is included in the compilation command. Without a binary compiled with this flag, the scripts won't work. See more information about the flag here: https://openmp.llvm.org/CommandLineArgumentReference.html#fopenmp-target-jit

Assuming that your OpenMP target application is compiled with `-fopenmp-target-jit` and can be run on the GPU, record the kernel by launching your application with the following environment variables:

```bash
LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_REPORT=1 LIBOMPTARGET_RECORD_DIR=records ./openmp-app
```

If all goes well, you should see the output of the kernel recording tool and a new folder called `records`.

> [!IMPORTANT]
> Make sure that the kernel records directory (which would be `records` if you followed the instruction above) contains a file that has the `.bc` extension. This is the device IR bitcode file, without which the scripts won't work. If you don't see the file, make sure that your binary is compiled with the `-fopenmp-target-jit` option and you're using the latest version of LLVM.

See more about the kernel record and replay process here: https://openmp.llvm.org/design/Runtimes.html#kernel-record-replay

### Create your JSON configuration file

Feel free to copy and modify the example JSON files in this repository. If you're running `autorun.py`, you will need to copy `example-autorun.json`. Else, copy `example-opt-passes.json`. For more details about what each field means, see the [JSON configuration files section](#json-configuration-files).

## Usage

### The autorun script

Assuming that you're on a system with the AMD MI250A GPU, you can run automatic search like this:

```bash
autorun.py --arch gfx90a records/example.bc example.json
```

If your system is using a different architecture than `gfx90a`, you need to specify your architecture through the `--arch` argument, otherwise `llc` will return an error. You can find out the correct value for this flag by looking at the output of `rocm-smi --showhw`.

The script will read the JSON configuration file `example.json` and construct a search space, which it will explore for 100 iterations. Feel free to get started by copying and modifying `example-autorun.json`. If a longer run is desired, you can pass the desired number of maximum iteration to the `--max-trials` argument like this:

```bash
autorun.py --arch gfx90a --max-trials 500 records/example.bc example.json
```

At the end of the run you should see two new files in your current directory: an `autosearch-results.csv` file that will have the current date prepended to it before the file extension and a `loss_history.png` file with a similar date affix.

You can run `autorun.py --help` to get information about all arguments.

### The run script

The `run.py` script can be invoked like this:

```
run.py --pipelines-file example-opt-passes.json records/example.bc
```

By default, the script only transform the device IR bitcode using `opt`. If you also desire to use `llc` to influence the backend, you can use the `--llc` and `--llc-also-use-opt` flags to switch the search to using llc and to pass the bitcode to `opt` first before passing it through `llc`, respectively.

## JSON configuration files

### Autorun script JSON configuration file

The JSON configuration file accepted by `autorun.py` looks like this:

```jsonc
{
  "persistent_llc_flags": ["-flag_1", "-flag_2"],
  "llc_flags": [
    {
      "flag": "example",
      "type": "boolean | value | uint | int | number | string"
      "range":
        // For type "boolean"
        null |
        {
          // For types "value" and "string"
          "choice": ["A", "B", "C"] |
          // For types "uint", "int", "number"
          "low": 1,
          "high": 1e6,
          "step": 1e2,
          "distribution": "logarithmic" | "uniform"
      }
    }
  ]
}
```

`persistent_llc_flags` is a list of flags that must persist across all trials, such as `-O3`. `llc_flags` is a list of flags with a type and range. This list defines a search space that is automatically explored as the script runs.

There are a few things to take into account when assembling your configuration file:

1. The flag names in the `flag` field are given without hyphens, i.e. `--unroll-threshold` should be represented as `"unroll-threshold"`.
2. For flags of type `uint`, `int` and `number`, the fields `low`, `high`, `step`, and `distribution` in the `range` object are required.
3. For flags of type `value` and `string` the field `choice` in the `range` object is required.
4. If you're using the logarithmic distribution, make sure to not include the value of 0 in the `low` or `high` fields as log(0) = ∞.

### Run script JSON configuration file

The JSON configuration file accepted by `run.py` looks like this:

```
{
  "pipeline_1": {
    "opt_passes": "",
    "opt_args": ["-O1"],
    "llc_args": ["-O1"],
    "replay_args": ["--repetitions", "20"]
  }
}
```

Each pipeline is defined as a `"name": object` pair. In the object:

- `opt_passes` is a string that gets passed to `opt -passes=""`
- `opt_args` is a list of arguments that gets directly added to the `opt` invocation.
- `llc_args` is the same but for `llc`
- `replay_args` is the same but for the `llvm-omp-kernel-replay` tool that is used to launch kernels.

Keep in mind that if you want to specify any flag that takes a value, you should separate the flag name and value like this: `["flag_name", "value"]`. If you don't want to have any arguments for opt, llc, or kernel replay, leave the list empty; do not delete the field as it will result in an error.
