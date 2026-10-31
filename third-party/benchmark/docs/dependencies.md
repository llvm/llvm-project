# Build tool dependency policy

We follow the [Foundational C++ support policy](https://opensource.google/documentation/policies/cplusplus-support) for our build tools. In
particular the ["Build Systems" section](https://opensource.google/documentation/policies/cplusplus-support#build-systems).

## CMake

The current supported version is CMake 3.10 as of 2023-08-10. Most modern
distributions include newer versions, for example:

* Ubuntu 20.04 provides CMake 3.16.3
* Debian 11.4 provides CMake 3.18.4
* Ubuntu 22.04 provides CMake 3.22.1

## Python

The Python bindings require Python 3.10+ as of v1.9.0 (2024-08-16) for installation from PyPI.
Building from source for older versions probably still works, though. See the [user guide](python_bindings.md) for details on how to build from source.
The minimum theoretically supported version is Python 3.8, since the used bindings generator (nanobind) only supports Python 3.8+.
