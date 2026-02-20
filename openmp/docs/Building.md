# Building the OpenMP Libraries

LLVM OpenMP uses a CMake-based build system. For generic information on the LLVM
build system see
[LLVM's Getting Started](https://llvm.org/docs/GettingStarted.html) and
[Advanced Build](https://llvm.org//docs/AdvancedBuilds.html) pages.

```{contents}
:depth: 3
```


## Requirements

LLVM OpenMP shares the same requirements as LLVM itself. See
[LLVM's Requirements](https://llvm.org/docs/GettingStarted.html#requirements)
for those requirements.


### Requirements for Building with Nvidia GPU support

The CUDA SDK is required on the machine that will build and execute the
offloading application. Normally this is only required at runtime by dynamically
opening the CUDA driver API. This can be disabled in the build by omitting
`cuda` from the [`LIBOMPTARGET_DLOPEN_PLUGINS`](LIBOMPTARGET_DLOPEN_PLUGINS)
list which is present by default. With this setting we will instead find the
CUDA library at LLVM build time and link against it directly.


### Requirements for Building with AMD GPU support

The OpenMP AMDGPU offloading support depends on the ROCm math libraries and the
HSA ROCr / ROCt runtimes. These are normally provided by a standard ROCm
installation, but can be built and used independently if desired. Building the
libraries does not depend on these libraries by default by dynamically loading
the HSA runtime at program execution. As in the CUDA case, this can be change by
omitting `amdgpu` from the
[`LIBOMPTARGET_DLOPEN_PLUGINS`](LIBOMPTARGET_DLOPEN_PLUGINS) list.


## Building on Linux

### Bootstrapping Build (Build together with LLVM)

An LLVM *bootstrapping build* compiles LLVM and Clang first, then uses this
just-built Clang to build the runtimes such as OpenMP.

```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
cmake ../llvm -G Ninja            \
    -DCMAKE_BUILD_TYPE=Release    \
    -DCMAKE_INSTALL_PREFIX=<PATH> \
    -DLLVM_ENABLE_PROJECTS=clang  \
    -DLLVM_ENABLE_RUNTIMES=openmp
ninja              # Build
ninja check-openmp # Run regression and unit tests
ninja install      # Installs files to <PATH>/bin, <PATH>/lib, etc
```

Without any further options, this builds the OpenMP libraries for the host
triple (e.g. when the host is `x86_64-linux-gnu`, this builds `libomp.so`
also for `x86_64-linux-gnu`). For building the libraries for additional,
cross-compilation target, they can be passed using `LLVM_RUNTIME_TARGETS`.
Internally, a new CMake build directory for each target triple will be created.
Configuration parameters with `OPENMP_` and `LIBOMP_` prefix are automatically
forwarded to all runtime build directories (but not others such as `LIBOMPT_` or
`LIBOMPTARGET_` prefixes). Other configuration parameters that should apply to
the runtimes can be passed via `RUNTIMES_CMAKE_ARGS`. For a parameter to be
passed to the build of only one target triple, set the parameter
`RUNTIMES_<triple>_<runtimes-parameter>`. For example:

```sh
cmake ../llvm -G Ninja                                           \
    -DCMAKE_BUILD_TYPE=Release                                   \
    -DCMAKE_INSTALL_PREFIX=<PATH>                                \
    -DLLVM_ENABLE_PROJECTS=clang                                 \
    -DLLVM_ENABLE_RUNTIMES=openmp                                \
    -DLLVM_RUNTIME_TARGETS="default;aarch64-linux-gnu"           \
    -DOPENMP_ENABLE_OMPT_TOOLS=ON                                \
    -DRUNTIMES_CMAKE_ARGS="-DLIBOMPTEST_INSTALL_COMPONENTS=ON"   \
    -DRUNTIMES_arch64-linux-gnu_CMAKE_CXX_FLAGS="-march=armv8-a"
```

Note that this requires having an `aarch64-linux-gnu` cross-compilation
toolchain to be available on the host system. While Clang is able to
cross-compile this triple when `LLVM_TARGETS_TO_BUILD` includes `AArch64` (which
it does by default), a linker and certain libraries such as pthread are required
as well.

If [`CMAKE_INSTALL_PREFIX`][CMAKE_INSTALL_PREFIX] is omitted, CMake defaults to
`/usr/local` to install the libraries globally. This is not recommended since it
might interfere with the system's OpenMP installation, such as `omp.h` from
gcc.


(default_runtimes_build)=
### Runtimes Default/Standalone Build (Using a pre-built LLVM)

An LLVM *default runtimes build* (sometimes also *standalone runtimes build*)
uses an pre-existing LLVM and Clang builds to directly compile the OpenMP
libraries.

```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Building LLVM
mkdir build
cd build
cmake ../llvm -G Ninja            \
    -DCMAKE_BUILD_TYPE=Release    \
    -DCMAKE_INSTALL_PREFIX=<PATH> \
    -DLLVM_ENABLE_PROJECTS=clang
ninja
ninja install
cd ..

# Building the OpenMP libraries
mkdir build-runtimes
cd build-runtimes
cmake ../runtimes -G Ninja        \
    -DCMAKE_BUILD_TYPE=Release    \
    -DCMAKE_INSTALL_PREFIX=<PATH> \
    -DLLVM_BINARY_DIR=../build    \
    -DLLVM_ENABLE_RUNTIMES=openmp
ninja              # Build
ninja check-openmp # Run regression and unit tests
ninja install      # Installs files to <PATH>/bin, <PATH>/lib, etc
```

Here, `../build` is the path the build of LLVM completed in the first step. It
is expected to have been built from the same Git commit as OpenMP. It will,
however, use the compiler detected by CMake, usually gcc.
To also make it use Clang, add
`-DCMAKE_C_COMPILER=../build/bin/clang -DCMAKE_C_COMPILER=../build/bin/clang++`.
It will use Clang from `LLVM_BINARY_DIR` for running the regression tests, if Clang is included in that build.
`LLVM_BINARY_DIR` can also be omitted in which case testing
(`ninja check-openmp`) is disabled.

The `CMAKE_INSTALL_PREFIX` can be the same, but does not need to. Using the same
path will allow Clang to automatically find the OpenMP files.


(build_offload_capable_compiler)=
### Building with Offload Support

Enabling support for offloading (i.e. `#pragma omp target`) additionally
requires the offload runtime. Host offloading (i.e. using the CPU itself as
an offloading target) should work out of the box, but each GPU architecture
requires its own runtime. Currently supported GPU architectures are
`amdgcn-amd-amdhsa` and `nvptx-nvidia-cuda`. Use the aforementioned
`RUNTIMES_<triple>_<runtimes-parameter>` form to restrict an option
`<runtimes-parameter>` to only only one of these architectures. A minimal build
configuration supporting both architectures would be the following.

```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
cmake ../llvm -G Ninja                                                     \
    -DCMAKE_BUILD_TYPE=Release                                             \
    -DCMAKE_INSTALL_PREFIX=<PATH>                                          \
    -DLLVM_ENABLE_PROJECTS="clang;lld"                                     \
    -DLLVM_ENABLE_RUNTIMES="openmp;offload"                                \
    -DLLVM_RUNTIME_TARGETS="default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda"
ninja                            # Build
ninja check-openmp check-offload # Run regression and unit tests
ninja install                    # Installs files to <PATH>/bin, <PATH>/lib, etc
```

The additional `LLVM_ENABLE_PROJECTS=lld` is needed to compile LLVM
bitcode of the GPU-side device runtime which uses LTO.

If using a [default/standalone runtimes build](default_runtimes_build), ensure
that in addition to `LLVM_BINARY_DIR`, `CMAKE_C_COMPILER` and
`CMAKE_CXX_COMPILER` is Clang built from the same git commit as OpenMP, as well
as lld, and that `AMDGPU` and `NVPTX` is enabled in its
``LLVM_TARGETS_TO_BUILD`` configuration (which it is by default).

In practice the setup above will probably missing requirements for actually
running programs on GPUs such as device-side toolchain libraries. A more
complete build on the device requires more options. Using CMake's
[`-C`](https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-C)
option allows to conveniently use pre-defined set from a file.

```sh
cmake ../llvm -G Ninja                       \
    -C ../offload/cmake/caches/Offload.cmake \
    -DCMAKE_BUILD_TYPE=Release               \
    -DCMAKE_INSTALL_PREFIX=<PATH>
```

Additionally, the `FlangOffload.cmake` file is provided for users that wish to
build a complete Fortran offloading toolchain.

## Building on Windows

Building the OpenMP libraries in Windows is not much different than on Linux,
only accounting for some differences of the shell (`cmd.exe`; for PowerShell
replace the end-of-line escape character `^` with a backtick `` ` ``).

```bat
  git clone https://github.com/llvm/llvm-project.git
  cd llvm-project
  mkdir build
  cd build
  cmake ..\llvm -G Ninja          ^
    -DCMAKE_BUILD_TYPE=Release    ^
    -DCMAKE_INSTALL_PREFIX=<PATH> ^
    -DLLVM_ENABLE_PROJECTS=clang  ^
    -DLLVM_ENABLE_RUNTIMES=openmp
  ninja
  ninja check-openmp
  ninja install
```

Compiling OpenMP with the MSVC compiler for a
[runtimes default build](default_runtimes_build) is possible as well:

```bat
  cmake ..\runtimes -G Ninja      ^
    -DCMAKE_BUILD_TYPE=Release    ^
    -DCMAKE_INSTALL_PREFIX=<PATH> ^
    -DLLVM_BINARY_DIR=../build    ^
    -DLLVM_ENABLE_RUNTIMES=openmp
```

However, offloading is not supported on the Windows platform.


## Building on macOS

On macOS machines, it is possible to build universal (or fat) libraries which
include both i386 and x86_64 architecture objects in a single archive.

```console
$ cmake ../llvm -G Ninja                    \
    -DCMAKE_C_COMPILER=clang                \
    -DCMAKE_CXX_COMPILER=clang++            \
    -DCMAKE_OSX_ARCHITECTURES='i386;x86_64' \
    ..
$ ninja
```


## CMake Configuration Parameter Reference

CMake configuration parameters specific to OpenMP are prefixed with `OPENMP_`,
`LIBOMP_`, ``LIBOMPTEST_`, `LIBOMPD_`, or `LIBARCHER_`. Additional configuration
parameters for the offloading are prefixed with `OFFLOAD_` or `LIBOMPTARGET_`.

The following is a selection of CMake build options recognized by the LLVM
OpenMP libraries.

[CMAKE_INSTALL_PREFIX]: https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html


### Options for All Libraries

**OPENMP_TEST_FLAGS**:STRING (default: *empty*), **OPENMP_TEST_OPENMP_FLAGS**:STRING (default: `-fopenmp`)
: Additional command line flags passed to Clang when compiling the regression
tests.

**OPENMP_INSTALL_LIBDIR**:STRING (default: `lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE}`)
: Location, relative to [`CMAKE_INSTALL_PREFIX`][CMAKE_INSTALL_PREFIX], where to
install the OpenMP libraries (`.a` and `.so`)

**OPENMP_TEST_C_COMPILER**:STRING (default: Clang built in the same build configuration, or **CMAKE_C_COMPILER**)
: C compiler to use for testing OpenMP runtime libraries.

**OPENMP_TEST_CXX_COMPILER**:STRING (default: Clang built in the same build configuration, or **CMAKE_CXX_COMPILER**)
: C++ compiler to use for testing OpenMP runtime libraries.

**OPENMP_TEST_Fortran_COMPILER**:STRING (default: Flang built in the same build configuration, or **CMAKE_Fortran_COMPILER**)
: Fortran compiler to use for testing OpenMP runtime libraries.

### Options for `libomp`

**LIBOMP_MIC_ARCH** = `knc|knf`
: Intel(R) Many Integrated Core Architecture (Intel(R) MIC Architecture) to
build for.  This value is ignored if `LIBOMP_ARCH` does not equal `mic`.

**LIBOMP_LIB_TYPE** = `normal|profile|stubs`
: Library type can be `normal`, `profile`, or `stubs`.

**LIBOMP_USE_VERSION_SYMBOLS**:BOOL
: Use versioned symbols for building the library.  This option only makes sense
for ELF based libraries where version symbols are supported (Linux*, some BSD*
variants).  It is `OFF` by default for Windows and macOS, but `ON` for
other Unix based operating systems.

**LIBOMP_ENABLE_SHARED**:BOOL (default: `ON`)
: Build a shared library.  If this option is `OFF`, static OpenMP libraries
will be built instead of dynamic ones.

:::{note}
Static libraries are not supported on Windows.
:::

(LIBOMP_OSX_ARCHITECTURES)=
**LIBOMP_OSX_ARCHITECTURES**
: For Mac builds, semicolon separated list of architectures to build for
universal fat binary.

**LIBOMP_USE_ADAPTIVE_LOCKS**:BOOL
: Include adaptive locks, based on Intel(R) Transactional Synchronization
Extensions (Intel(R) TSX).  This feature is x86 specific and turned `ON`
by default for IA-32 architecture and Intel(R) 64 architecture.

**LIBOMP_USE_INTERNODE_ALIGNMENT**:BOOL
: Align certain data structures on 4096-byte.  This option is useful on
multi-node systems where a small `CACHE_LINE` setting leads to false sharing.

**LIBOMP_STATS**:BOOL
: Include stats-gathering code.

**LIBOMP_USE_DEBUGGER**:BOOL
: Include the friendly debugger interface.

(LIBOMP_USE_HWLOC)=
**LIBOMP_USE_HWLOC**:BOOL
: Use [OpenMPI's hwloc library](https://www.open-mpi.org/projects/hwloc/) for
topology detection and affinity.

**LIBOMP_HWLOC_INSTALL_DIR**:PATH
: Specify install location of hwloc. The configuration system will look for
`hwloc.h` in `${LIBOMP_HWLOC_INSTALL_DIR}/include` and the library in
`${LIBOMP_HWLOC_INSTALL_DIR}/lib`.  The default is `/usr/local`.
This option is only used if [`LIBOMP_USE_HWLOC`](LIBOMP_USE_HWLOC) is `ON`.

**LIBOMP_CPPFLAGS** = <space-separated flags>
: Additional C preprocessor flags.

**LIBOMP_CXXFLAGS** = <space-separated flags>
: Additional C++ compiler flags.

**LIBOMP_ASMFLAGS** = <space-separated flags>
: Additional assembler flags.

**LIBOMP_LDFLAGS** = <space-separated flags>
: Additional linker flags.

**LIBOMP_LIBFLAGS** = <space-separated flags>
: Additional libraries to link.

**LIBOMP_FFLAGS** = <space-separated flags>
: Additional Fortran compiler flags.


### Options for OMPT

(LIBOMP_OMPT_SUPPORT)=
**LIBOMP_OMPT_SUPPORT**:BOOL
: Include support for the OpenMP Tools Interface (OMPT).
This option is supported and `ON` by default for x86, x86_64, AArch64,
PPC64, RISCV64, LoongArch64, and s390x on Linux and macOS.
This option is `OFF` if this feature is not supported for the platform.

**OPENMP_ENABLE_OMPT_TOOLS**:BOOL
: Enable building ompt based tools for OpenMP.

**LIBOMP_ARCHER_SUPPORT**:BOOL
: Build libomp with archer support.

**LIBOMP_OMPT_OPTIONAL**:BOOL
: Include support for optional OMPT functionality.  This option is ignored if
[`LIBOMP_OMPT_SUPPORT`](LIBOMP_OMPT_SUPPORT) is `OFF`.

**LIBOMPTEST_INSTALL_COMPONENTS**: BOOL (default: `OFF`)
: Whether to also copy `libomptest.so` into
[`CMAKE_INSTALL_PREFIX`][CMAKE_INSTALL_PREFIX] during `ninja install`.


### Options for `libompd`

**LIBOMP_OMPD_SUPPORT**:BOOL
: Enable building the libompd library.

**LIBOMPD_LD_STD_FLAGS**:STRING
: Use `-stdlibc++` instead of `-libc++` library for C++.


### Options for `libomptarget`/offload

**LIBOMPTARGET_OPENMP_HEADER_FOLDER**:PATH
: Path of the folder that contains `omp.h`.  This is required for testing
out-of-tree builds.

**LIBOMPTARGET_OPENMP_HOST_RTL_FOLDER**:PATH
: Path of the folder that contains `libomp.so`, and `libLLVMSupport.so`
when profiling is enabled. This is required for testing.

**LIBOMPTARGET_LIT_ARGS**:STRING
: Arguments given to lit. `make check-libomptarget` and
`make check-libomptarget-*` are affected. For example, use
`LIBOMPTARGET_LIT_ARGS="-j4"` to force `lit` to start only four parallel
jobs instead of by default the number of threads in the system.

**LIBOMPTARGET_ENABLE_DEBUG**:BOOL
: Enable printing of debug messages with the `LIBOMPTARGET_DEBUG=1` environment
variable.

**LIBOMPTARGET_PLUGINS_TO_BUILD** = semicolon-separated list of `cuda|amdgpu|host` or `all` (default: `all`)
: List of offload plugins to build.

(LIBOMPTARGET_DLOPEN_PLUGINS)=
**LIBOMPTARGET_DLOPEN_PLUGINS** = semicolon-separated list of `cuda|amdgpu` or `all` (default: `${LIBOMPTARGET_PLUGINS_TO_BUILD}`)
: List of plugins to use `dlopen` instead of the `ld.so` dynamic linker for
runtime linking. `dlopen` does not require the vendor runtime libraries to be
present at build-time of OpenMP, but imposes higher runtime overhead.
