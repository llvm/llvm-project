<head>
  <meta charset="UTF-8">
  <meta name="description" content="OpenMP support in ROCm">
  <meta name="keywords" content="OpenMP, LLVM, OpenMP toolchain">
</head>

# OpenMP support in ROCm

## Introduction

The ROCm™ installation includes an LLVM-based implementation that fully supports
the OpenMP 4.5 standard and a subset of OpenMP 5.0, 5.1, and 5.2 standards.
Fortran, C/C++ compilers, and corresponding runtime libraries are included.
Along with host APIs, the OpenMP compilers support offloading code and data onto
GPU devices. This document briefly describes the installation location of the
OpenMP toolchain, example usage of device offloading, and usage of `rocprof`
with OpenMP applications. The GPUs supported are the same as those supported by
this ROCm release. See the list of supported GPUs for {doc}`Linux<rocm-install-on-linux:reference/system-requirements>` and
{doc}`Windows<rocm-install-on-windows:reference/system-requirements>`.

The ROCm OpenMP compiler is implemented using LLVM compiler technology.
The following image illustrates the internal steps taken to translate a user’s application into an executable that can offload computation to the AMDGPU. The compilation is a two-pass process. Pass 1 compiles the application to generate the CPU code and Pass 2 links the CPU code to the AMDGPU device code.

![OpenMP toolchain](../../data/reference/openmp/openmp-toolchain.svg "OpenMP toolchain")

### Installation

The OpenMP toolchain is automatically installed as part of the standard ROCm
installation and is available under `/opt/rocm-{version}/llvm`. The
sub-directories are:

* bin: Compilers (`flang` and `clang`) and other binaries.
* examples: The usage section below shows how to compile and run these programs.
* include: Header files.
* lib: Libraries including those required for target offload.
* lib-debug: Debug versions of the above libraries.

## OpenMP: usage

The example programs can be compiled and run by pointing the environment
variable `ROCM_PATH` to the ROCm install directory.

**Example:**

```bash
export ROCM_PATH=/opt/rocm-{version}
cd $ROCM_PATH/share/openmp-extras/examples/openmp/veccopy
sudo make run
```

:::{note}
`sudo` is required since we are building inside the `/opt` directory.
Alternatively, copy the files to your home directory first.
:::

The above invocation of Make compiles and runs the program. Note the options
that are required for target offload from an OpenMP program:

```bash
-fopenmp --offload-arch=<gpu-arch>
```

:::{note}
The compiler also accepts the alternative offloading notation:

```bash
-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=<gpu-arch>
```

:::

Obtain the value of `gpu-arch` by running the following command:

```bash
% /opt/rocm-{version}/bin/rocminfo | grep gfx
```

[//]: # (dated link below, needs updating)

See the complete list of compiler command-line references
[here](https://github.com/ROCm/llvm-project/blob/amd-staging/openmp/docs/CommandLineArgumentReference.rst).

### Using `rocprof` with OpenMP

The following steps describe a typical workflow for using `rocprof` with OpenMP
code compiled with AOMP:

1. Run `rocprof` with the program command line:

    ```bash
    % rocprof <application> <args>
    ```

    This produces a `results.csv` file in the user’s current directory that
    shows basic stats such as kernel names, grid size, number of registers used,
    etc. The user can choose to specify the preferred output file name using the
    o option.

2. Add options for a detailed result:

   ```bash
   --stats: % rocprof --stats <application> <args>
   ```

   The stats option produces timestamps for the kernels. Look into the output
   CSV file for the field, `DurationNs`, which is useful in getting an
   understanding of the critical kernels in the code.

   Apart from `--stats`, the option `--timestamp` on produces a timestamp for
   the kernels.

3. After learning about the required kernels, the user can take a detailed look
   at each one of them. `rocprof` has support for hardware counters: a set of
   basic and a set of derived ones. See the complete list of counters using
   options --list-basic and --list-derived. `rocprof` accepts either a text or
   an XML file as an input.

For more details on `rocprof`, refer to the {doc}`ROCProfilerV1 User Manual <rocprofiler:rocprofv1>`.

### Using tracing options

**Prerequisite:** When using the `--sys-trace` option, compile the OpenMP
program with:

```bash
    -Wl,-rpath,/opt/rocm-{version}/lib -lamdhip64
```

The following tracing options are widely used to generate useful information:

* **`--hsa-trace`**: This option is used to get a JSON output file with the HSA
  API execution traces and a flat profile in a CSV file.

* **`--sys-trace`**: This allows programmers to trace both HIP and HSA calls.
  Since this option results in loading ``libamdhip64.so``, follow the
  prerequisite as mentioned above.

A CSV and a JSON file are produced by the above trace options. The CSV file
presents the data in a tabular format, and the JSON file can be visualized using
Google Chrome at chrome://tracing/ or [Perfetto](https://perfetto.dev/).
Navigate to Chrome or Perfetto and load the JSON file to see the timeline of the
HSA calls.

For more details on tracing, refer to the {doc}`ROCProfilerV1 User Manual <rocprofiler:rocprofv1>`.

### Environment variables

:::{table}
:widths: auto
| Environment Variable        | Purpose                  |
| --------------------------- | ---------------------------- |
| `OMP_NUM_TEAMS`             | To set the number of teams for kernel launch, which is otherwise chosen by the implementation by default. You can set this number (subject to implementation limits) for performance tuning. |
| `LIBOMPTARGET_KERNEL_TRACE` | To print useful statistics for device operations. Setting it to 1 and running the program emits the name of every kernel launched, the number of teams and threads used, and the corresponding register usage. Setting it to 2 additionally emits timing information for kernel launches and data transfer operations between the host and the device. |
| `LIBOMPTARGET_INFO`         | To print informational messages from the device runtime as the program executes. Setting it to a value of 1 or higher, prints fine-grain information and setting it to -1 prints complete information. |
| `LIBOMPTARGET_DEBUG`        | To get detailed debugging information about data transfer operations and kernel launch when using a debug version of the device library. Set this environment variable to 1 to get the detailed information from the library. |
| `GPU_MAX_HW_QUEUES`         | To set the number of HSA queues in the OpenMP runtime. The HSA queues are created on demand up to the maximum value as supplied here. The queue creation starts with a single initialized queue to avoid unnecessary allocation of resources. The provided value is capped if it exceeds the recommended, device-specific value. |
| `LIBOMPTARGET_AMDGPU_MAX_ASYNC_COPY_BYTES` | To set the threshold size up to which data transfers are initiated asynchronously. The default threshold size is 1*1024*1024 bytes (1MB). |
| `OMPX_FORCE_SYNC_REGIONS` | To force the runtime to execute all operations synchronously, i.e., wait for an operation to complete immediately. This affects data transfers and kernel execution. While it is mainly designed for debugging, it may have a minor positive effect on performance in certain situations. |
:::

## OpenMP: features

The OpenMP programming model is greatly enhanced with the following new features
implemented in the past releases.

(openmp_usm)=

### Asynchronous behavior in OpenMP target regions

* Controlling Asynchronous Behavior

The OpenMP offloading runtime executes in an asynchronous fashion by default, allowing multiple data transfers to start concurrently. However, if the data to be transferred becomes larger than the default threshold of 1MB, the runtime falls back to a synchronous data transfer. The buffers that have been locked already are always executed asynchronously.
You can overrule this default behavior by setting `LIBOMPTARGET_AMDGPU_MAX_ASYNC_COPY_BYTES` and `OMPX_FORCE_SYNC_REGIONS`. See the [Environment Variables](#environment-variables) table for details.

* Multithreaded Offloading on the Same Device

The `libomptarget` plugin for GPU offloading allows creation of separate configurable HSA queues per chiplet, which enables two or more threads to concurrently offload to the same device.

* Parallel Memory Copy Invocations

Implicit asynchronous execution of single target region enables parallel memory copy invocations.

### Unified shared memory

Unified Shared Memory (USM) provides a pointer-based approach to memory
management. To implement USM, fulfill the following system requirements along
with Xnack capability.

#### Prerequisites

* Linux Kernel versions above 5.14
* Latest KFD driver packaged in ROCm stack
* Xnack, as USM support can only be tested with applications compiled with Xnack
  capability

#### Xnack capability

When enabled, Xnack capability allows GPU threads to access CPU (system) memory,
allocated with OS-allocators, such as `malloc`, `new`, and `mmap`. Xnack must be
enabled both at compile- and run-time. To enable Xnack support at compile-time,
use:

```bash
--offload-arch=gfx908:xnack+
```

Or use another functionally equivalent option Xnack-any:

```bash
--offload-arch=gfx908
```

To enable Xnack functionality at runtime on a per-application basis,
use environment variable:

```bash
HSA_XNACK=1
```

When Xnack support is not needed:

* Build the applications to maximize resource utilization using:

```bash
--offload-arch=gfx908:xnack-
```

* At runtime, set the `HSA_XNACK` environment variable to 0.

#### Unified shared memory pragma

This OpenMP pragma is available on MI200 through `xnack+` support.

```bash
omp requires unified_shared_memory
```

As stated in the OpenMP specifications, this pragma makes the map clause on
target constructs optional. By default, on MI200, all memory allocated on the
host is fine grain. Using the map clause on a target clause is allowed, which
transforms the access semantics of the associated memory to coarse grain.

```bash
A simple program demonstrating the use of this feature is:
$ cat parallel_for.cpp
#include <stdlib.h>
#include <stdio.h>

#define N 64
#pragma omp requires unified_shared_memory
int main() {
  int n = N;
  int *a = new int[n];
  int *b = new int[n];

  for(int i = 0; i < n; i++)
    b[i] = i;

  #pragma omp target parallel for map(to:b[:n])
  for(int i = 0; i < n; i++)
    a[i] = b[i];

  for(int i = 0; i < n; i++)
    if(a[i] != i)
      printf("error at %d: expected %d, got %d\n", i, i+1, a[i]);

  return 0;
}
$ clang++ -O2 -target x86_64-pc-linux-gnu -fopenmp --offload-arch=gfx90a:xnack+ parallel_for.cpp
$ HSA_XNACK=1 ./a.out
```

In the above code example, pointer “a” is not mapped in the target region, while
pointer “b” is. Both are valid pointers on the GPU device and passed by-value to
the kernel implementing the target region. This means the pointer values on the
host and the device are the same.

The difference between the memory pages pointed to by these two variables is
that the pages pointed by “a” are in fine-grain memory, while the pages pointed
to by “b” are in coarse-grain memory during and after the execution of the
target region. This is accomplished in the OpenMP runtime library with calls to
the ROCr runtime to set the pages pointed by “b” as coarse grain.

### OMPT target support

The OpenMP runtime in ROCm implements a subset of the OMPT device APIs, as
described in the OpenMP specification document. These APIs allow first-party
tools to examine the profile and kernel traces that execute on a device. A tool
can register callbacks for data transfer and kernel dispatch entry points or use
APIs to start and stop tracing for device-related activities such as data
transfer and kernel dispatch timings and associated metadata. If device tracing
is enabled, trace records for device activities are collected during program
execution and returned to the tool using the APIs described in the
specification.

The following example demonstrates how a tool uses the supported OMPT target
APIs. The `README` in `/opt/rocm/llvm/examples/tools/ompt` outlines the steps to
be followed, and the provided example can be run as shown below:

```bash
cd $ROCM_PATH/share/openmp-extras/examples/tools/ompt/veccopy-ompt-target-tracing
sudo make run
```

The file `veccopy-ompt-target-tracing.c` simulates how a tool initiates device
activity tracing. The file `callbacks.h` shows the callbacks registered and
implemented by the tool.

### Floating point atomic operations

The MI200-series GPUs support the generation of hardware floating-point atomics
using the OpenMP atomic pragma. The support includes single- and
double-precision floating-point atomic operations. The programmer must ensure
that the memory subjected to the atomic operation is in coarse-grain memory by
mapping it explicitly with the help of map clauses when not implicitly mapped by
the compiler as per the [OpenMP
specifications](https://www.openmp.org/specifications/). This makes these
hardware floating-point atomic instructions “fast,” as they are faster than
using a default compare-and-swap loop scheme, but at the same time “unsafe,” as
they are not supported on fine-grain memory. The operation in
`unified_shared_memory` mode also requires programmers to map the memory
explicitly when not implicitly mapped by the compiler.

To request fast floating-point atomic instructions at the file level, use
compiler flag `-munsafe-fp-atomics` or a hint clause on a specific pragma:

```bash
double a = 0.0;
#pragma omp atomic hint(AMD_fast_fp_atomics)
a = a + 1.0;
```

:::{note}
`AMD_unsafe_fp_atomics` is an alias for `AMD_fast_fp_atomics`, and
`AMD_safe_fp_atomics` is implemented with a compare-and-swap loop.
:::

To disable the generation of fast floating-point atomic instructions at the file
level, build using the option `-msafe-fp-atomics` or use a hint clause on a
specific pragma:

```bash
double a = 0.0;
#pragma omp atomic hint(AMD_safe_fp_atomics)
a = a + 1.0;
```

The hint clause value always has a precedence over the compiler flag, which
allows programmers to create atomic constructs with a different behavior than
the rest of the file.

See the example below, where the user builds the program using
`-msafe-fp-atomics` to select a file-wide “safe atomic” compilation. However,
the fast atomics hint clause over variable “a” takes precedence and operates on
“a” using a fast/unsafe floating-point atomic, while the variable “b” in the
absence of a hint clause is operated upon using safe floating-point atomics as
per the compiler flag.

```bash
double a = 0.0;.
#pragma omp atomic hint(AMD_fast_fp_atomics)
a = a + 1.0;

double b = 0.0;
#pragma omp atomic
b = b + 1.0;
```

### AddressSanitizer tool

AddressSanitizer (ASan) is a memory error detector tool utilized by applications to
detect various errors ranging from spatial issues such as out-of-bound access to
temporal issues such as use-after-free. The AOMP compiler supports ASan for AMD
GPUs with applications written in both HIP and OpenMP.

**Features supported on host platform (Target x86_64):**

* Use-after-free
* Buffer overflows
* Heap buffer overflow
* Stack buffer overflow
* Global buffer overflow
* Use-after-return
* Use-after-scope
* Initialization order bugs

**Features supported on AMDGPU platform (`amdgcn-amd-amdhsa`):**

* Heap buffer overflow
* Global buffer overflow

**Software (kernel/OS) requirements:** Unified Shared Memory support with Xnack
capability. See the section on [Unified Shared Memory](#unified-shared-memory)
for prerequisites and details on Xnack.

**Example:**

* Heap buffer overflow

```bash
void  main() {
.......  // Some program statements
.......  // Some program statements
#pragma omp target map(to : A[0:N], B[0:N]) map(from: C[0:N])
{
#pragma omp parallel for
    for(int i =0 ; i < N; i++){
    C[i+10] = A[i] + B[i];
  }   // end of for loop
}
.......   // Some program statements
}// end of main
```

See the complete sample code for heap buffer overflow
[here](https://github.com/ROCm/aomp/blob/aomp-dev/examples/tools/asan/heap_buffer_overflow/openmp/vecadd-HBO.cpp).

* Global buffer overflow

```bash
#pragma omp declare target
   int A[N],B[N],C[N];
#pragma omp end declare target
void main(){
......  // some program statements
......  // some program statements
#pragma omp target data map(to:A[0:N],B[0:N]) map(from: C[0:N])
{
#pragma omp target update to(A,B)
#pragma omp target parallel for
for(int i=0; i<N; i++){
    C[i]=A[i*100]+B[i+22];
} // end of for loop
#pragma omp target update from(C)
}
........  // some program statements
} // end of main
```

See the complete sample code for global buffer overflow
[here](https://github.com/ROCm/aomp/blob/aomp-dev/examples/tools/asan/global_buffer_overflow/openmp/vecadd-GBO.cpp).

### Clang compiler option for kernel optimization

You can use the clang compiler option `-fopenmp-target-fast` for kernel optimization if certain constraints implied by its component options are satisfied. `-fopenmp-target-fast` enables the following options:

* `-fopenmp-target-ignore-env-vars`: It enables code generation of specialized kernels including no-loop and Cross-team reductions.

* `-fopenmp-assume-no-thread-state`: It enables the compiler to assume that no thread in a parallel region modifies an Internal Control Variable (`ICV`), thus potentially reducing the device runtime code execution.

* `-fopenmp-assume-no-nested-parallelism`: It enables the compiler to assume that no thread in a parallel region encounters a parallel region, thus potentially reducing the device runtime code execution.

* `-O3` if no `-O*` is specified by the user.

### Specialized kernels

Clang will attempt to generate specialized kernels based on compiler options and OpenMP constructs. The following specialized kernels are supported:

* No-loop
* Big-jump-loop
* Cross-team reductions

To enable the generation of specialized kernels, follow these guidelines:

* Do not specify teams, threads, and schedule-related environment variables. The `num_teams` clause in an OpenMP target construct acts as an override and prevents the generation of the no-loop kernel. If the specification of `num_teams` clause is a user requirement then clang tries to generate the big-jump-loop kernel instead of the no-loop kernel.

* Assert the absence of the teams, threads, and schedule-related environment variables by adding the command-line option `-fopenmp-target-ignore-env-vars`.

* To automatically enable the specialized kernel generation, use `-Ofast` or `-fopenmp-target-fast` for compilation.

* To disable specialized kernel generation, use `-fno-openmp-target-ignore-env-vars`.

#### No-loop kernel generation

The no-loop kernel generation feature optimizes the compiler performance by generating a specialized kernel for certain OpenMP target constructs such as target teams distribute parallel for. The specialized kernel generation feature assumes every thread executes a single iteration of the user loop, which leads the runtime to launch a total number of GPU threads equal to or greater than the iteration space size of the target region loop. This allows the compiler to generate code for the loop body without an enclosing loop, resulting in reduced control-flow complexity and potentially better performance.

#### Big-jump-loop kernel generation

A no-loop kernel is not generated if the OpenMP teams construct uses a `num_teams` clause. Instead, the compiler attempts to generate a different specialized kernel called the big-jump-loop kernel. The compiler launches the kernel with a grid size determined by the number of teams specified by the OpenMP `num_teams` clause and the `blocksize` chosen either by the compiler or specified by the corresponding OpenMP clause.

#### Cross-team optimized reduction kernel generation

If the OpenMP construct has a reduction clause, the compiler attempts to generate optimized code by utilizing efficient cross-team communication. New APIs for cross-team reduction are implemented in the device runtime and are automatically generated by clang.
