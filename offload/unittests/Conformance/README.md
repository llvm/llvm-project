# GPU Math Conformance Tests

## Overview

This test suite provides a framework to systematically measure the accuracy of math functions on GPUs and verify their conformance with standards like OpenCL.

While the primary focus is validating the implementations in the C standard math library (LLVM-libm), these tests can also be executed against other math library providers, such as CUDA Math and HIP Math, for comparison.

The goals of this project are to empower LLVM-libm contributors with a robust tool for validating their implementations and to build trust with end-users by providing transparent accuracy data.

### Table of Contents

- [Getting Started](#getting-started)
- [Running the Tests](#running-the-tests)
- [Adding New Tests](#adding-new-tests)

## Getting Started

This guide covers how to build the necessary dependencies, which include the new Offload API and the C standard library for both host and GPU targets.

### System Requirements

Before you begin, ensure your system meets the following requirements:

- A system with an AMD or NVIDIA GPU.
- The latest proprietary GPU drivers installed.
- The corresponding development SDK for your hardware:
  - **AMD:** [ROCm SDK](https://rocm.docs.amd.com)
  - **NVIDIA:** [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

### Building the Dependencies

The official documentation for building LLVM-libc for GPUs provides a detailed guide and should be considered the primary reference. Please follow the instructions in the **"Standard runtimes build"** section of that guide:

- [Building the GPU C library (Official Documentation)](https://libc.llvm.org/gpu/building.html)

> [!IMPORTANT]
> For the conformance tests, the standard `cmake` command from the official documentation must be adapted slightly. You must also add `libc` to the main `-DLLVM_ENABLE_RUNTIMES` list. This is a crucial step because the tests need a host-side build of `libc` to use as the reference oracle for validating GPU results.

## Running the Tests

### Default Test

To build and run the conformance test for a given function (e.g., `logf`) against the default C standard math library `llvm-libm` provider, use the following command. This will execute the test on all available and supported platforms.

```bash
ninja -C build/runtimes/runtimes-bins offload.conformance.logf
```

### Testing Other Providers

Once the test binary has been built, you can run it against other math library providers using the `--test-configs` flag.

- **For `cuda-math` on an NVIDIA GPU:**

  ```bash
  ./build/runtimes/runtimes-bins/offload/logf.conformance --test-configs=cuda-math:cuda
  ```

- **For `hip-math` on an AMD GPU:**

  ```bash
  ./build/runtimes/runtimes-bins/offload/logf.conformance --test-configs=hip-math:amdgpu
  ```

You can also run all available configurations for a test with:

```bash
./build/runtimes/runtimes-bins/offload/logf.conformance --test-configs=all
```

## Adding New Tests

To add a conformance test for a new math function, follow these steps:

1. **Implement the Device Kernels**: Create a kernel wrapper for the new function in each provider's source file. For CUDA Math and HIP Math, you must also add a forward declaration for the vendor function in `/device_code/DeviceAPIs.hpp`.

2. **Implement the Host Test**: Create a new `.cpp` file in `/tests`. This file defines the `FunctionConfig` (function and kernel names, as well as ULP tolerance) and the input generation strategy.

    - Use **exhaustive testing** (`ExhaustiveGenerator`) for functions with small input spaces (e.g., half-precision functions and single-precision univariate functions). This strategy iterates over every representable point in the input space, ensuring complete coverage.
    - Use **randomized testing** (`RandomGenerator`) for functions with large input spaces (e.g., single-precision bivariate and double-precision functions), where exhaustive testing is computationally infeasible. Although not exhaustive, this strategy is deterministic, using a fixed seed to sample a large, reproducible subset of points from the input space.

3. **Add the Build Target**: Add a new `add_conformance_test(...)` entry to `/tests/CMakeLists.txt` to make the test buildable.
