# MBR - MLIR Benchmark Runner
MBR is a tool to run benchmarks. It measures compilation and running times of
benchmark programs. It uses MLIR's python bindings for MLIR benchmarks.

## Installation
To build and enable MLIR benchmarks, pass `-DMLIR_ENABLE_PYTHON_BENCHMARKS=ON`
while building MLIR. If you make some changes to the `mbr` files itself, build
again with `-DMLIR_ENABLE_PYTHON_BENCHMARKS=ON`.

## Writing benchmarks
As mentioned in the intro, this tool measures compilation and running times.
An MBR benchmark is a python function that returns two callables, a compiler
and a runner. Here's an outline of a benchmark; we explain its working after
the example code.

```python
def benchmark_something():
    # Preliminary setup
    def compiler():
        # Compiles a program and creates an "executable object" that can be
        # called to invoke the compiled program.
        ...

    def runner(executable_object):
        # Sets up arguments for executable_object and calls it. The
        # executable_object is returned by the compiler.
        # Returns an integer representing running time in nanoseconds.
        ...

    return compiler, runner
```

The benchmark function's name must be prefixed by `"benchmark_"` and benchmarks
must be in the  python files prefixed by `"benchmark_` for them to be
discoverable. The file and function prefixes are configurable using the
configuration file `mbr/config.ini` relative to this  README's directory.

A benchmark returns two functions, a `compiler` and a `runner`. The `compiler`
returns a callable which is accepted as an argument by the runner function.
So the two functions work like this
1. `compiler`: configures and returns a callable.
2. `runner`: takes that callable in as input, sets up its arguments, and calls
    it. Returns an int representing running time in nanoseconds.

The `compiler` callable is optional if there is no compilation step, for
example, for benchmarks involving numpy. In that case, the benchmarks look
like this.

```python
def benchmark_something():
    # Preliminary setup
    def runner():
        # Run the program and return the running time in nanoseconds.
        ...

    return None, runner
```
In this case, the runner does not take any input as there is no compiled object
to invoke.

## Running benchmarks
MLIR benchmarks can be run like this

```bash
PYTHONPATH=<path_to_python_mlir_core> <other_env_vars> python <llvm-build-path>/bin/mlir-mbr --machine <machine_identifier> --revision <revision_string> --result-stdout <path_to_start_search_for_benchmarks>
```
For a description of command line arguments, run

```bash
python mlir/utils/mbr/mbr/main.py -h
```
And to learn more about the other arguments, check out the LNT's
documentation page [here](https://llvm.org/docs/lnt/concepts.html).

If you want to run only specific benchmarks, you can use the positional argument
`top_level_path` appropriately.

1. If you want to run benchmarks in a specific directory or a file, set
   `top_level_path` to that.
2. If you want to run a specific benchmark function, set the `top_level_path` to 
   the file containing that benchmark function, followed by a `::`, and then the
   benchmark function name. For example, `mlir/benchmark/python/benchmark_sparse.py::benchmark_sparse_mlir_multiplication`.

## Configuration
Various aspects about the framework can be configured using the configuration
file in the `mbr/config.ini` relative to the directory of this README.
