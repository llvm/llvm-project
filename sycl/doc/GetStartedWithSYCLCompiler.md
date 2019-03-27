# Overview

The SYCL* Compiler compiles C++\-based SYCL source files with code for both CPU
and a wide range of compute accelerators. The compiler uses Khronos*
OpenCL&trade; API to offload computations to accelerators.

# Before You Begin

Software requirements:

Installing OpenCL 2.1 compatible software stack:
1. OpenCL headers:

   a. Download the OpenCL headers from
[github.com/KhronosGroup/OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers)
to your local machine. e.g. `/usr/local/include/CL` with environment var
`$OPENCL_HEADERS`.
2. OpenCL runtime for CPU and GPU:

   a. OpenCL runtime for GPU: follow instructions on
[github.com/intel/compute-runtime/releases](https://github.com/intel/compute-runtime/releases)
to install.

   b. OpenCL runtime for CPU: follow instructions under section "Intel&reg; CPU
Runtime for OpenCL. Applications 18.1 for Linux* OS (64bit only)" on
[https://software.intel.com/en-us/articles/opencl-drivers#cpu-section](https://software.intel.com/en-us/articles/opencl-drivers#cpu-section)
and click on orange "Download" button to download & install.

# Build the SYCL compiler

Download the LLVM* repository with SYCL support to your local machine folder
e.g. `$HOME/sycl` (assuming environment var `$SYCL_HOME`) folder using
following command:

```bash
git clone https://github.com/intel/llvm -b sycl $HOME/sycl
```

Follow regular LLVM build instructions under:
[llvm.org/docs/CMake.html](https://llvm.org/docs/CMake.html). To build SYCL
runtime use modified CMake command below:

```bash
mkdir $SYCL_HOME/build
cd $SYCL_HOME/build
cmake -DCMAKE_BUILD_TYPE=Release -DOpenCL_INCLUDE_DIR=$OPENCL_HEADERS \
-DLLVM_ENABLE_PROJECTS="clang" -DLLVM_EXTERNAL_PROJECTS="sycl;llvm-spirv" \
-DLLVM_EXTERNAL_SYCL_SOURCE_DIR=$SYCL_HOME/sycl \
-DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR=$SYCL_HOME/llvm-spirv \
-DLLVM_TOOL_SYCL_BUILD=ON -DLLVM_TOOL_LLVM_SPIRV_BUILD=ON $SYCL_HOME/llvm
make -j`nproc` check-all
```

After the build completed, the SYCL compiler/include/libraries can be found
under `$SYCL_HOME/build` directory.

# Creating a simple SYCL program

A simple SYCL program consists of following parts:
1. Header section
2. Allocating buffer for data
3. Creating SYCL queue
4. Submitting command group to SYCL queue which includes the kernel
5. Wait for the queue to complete the work
6. Use buffer accessor to retrieve the result on the device and verify the data
7. The end

Creating a file `simple-sycl-app.cpp` with the following C++ SYCL code in it:

```c++

#include <CL/sycl.hpp>

int main() {
  // Creating buffer of 4 ints to be used inside the kernel code
  cl::sycl::buffer<cl::sycl::cl_int, 1> Buffer(4);

  // Creating SYCL queue
  cl::sycl::queue Queue;

  // Size of index space for kernel
  cl::sycl::range<1> NumOfWorkItems{Buffer.get_count()};

  // Submitting command group(work) to queue
  Queue.submit([&](cl::sycl::handler &cgh) {
    // Getting write only access to the buffer on a device
    auto Accessor = Buffer.get_access<cl::sycl::access::mode::write>(cgh);
    // Executing kernel
    cgh.parallel_for<class FillBuffer>(
        NumOfWorkItems, [=](cl::sycl::id<1> WIid) {
          // Fill buffer with indexes
          Accessor[WIid] = (cl::sycl::cl_int)WIid.get(0);
        });
  });

  // Getting read only access to the buffer on the host.
  // Implicit barrier waiting for queue to complete the work.
  const auto HostAccessor = Buffer.get_access<cl::sycl::access::mode::read>();

  // Check the results
  bool MismatchFound = false;
  for (size_t I = 0; I < Buffer.get_count(); ++I) {
    if (HostAccessor[I] != I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  if (!MismatchFound) {
    std::cout << "The results are correct!" << std::endl;
  }

  return MismatchFound;
}

```

# Build and Test a simple SYCL program

To build simple-sycl-app run following command:

   ```console
   clang++ -std=c++11 -fsycl simple-sycl-app.cpp -o simple-sycl-app -lOpenCL
   ```

This `simple-sycl-app` application doesn't specify SYCL device for execution,
so SYCL runtime will first try to execute on OpenCL GPU device first, if OpenCL
GPU device is not found, it will try to run OpenCL CPU device; and if OpenCL
CPU device is also not available, SYCL runtime will run on SYCL host device.

To run the `simple-sycl-app`:

    LD_LIBRARY_PATH=$SYCL_HOME/build/lib ./simple-sycl-app
    The results are correct!

NOTE: SYCL developer can specify SYCL device for execution using device
selectors (e.g. `cl::sycl::cpu_selector`, `cl::sycl::gpu_selector`) as
explained in following section [Code the program for a specific
GPU](#code-the-program-for-a-specific-gpu).

# Code the program for a specific GPU

To specify OpenCL device SYCL provides the abstract `cl::sycl::device_selector`
class which the can be used to define how the runtime should select the best
device.

The method `cl::sycl::device_selector::operator()` of the SYCL
`cl::sycl::device_selector` is an abstract member function which takes a
reference to a SYCL device and returns an integer score. This abstract member
function can be implemented in a derived class to provide a logic for selecting
a SYCL device. SYCL runtime uses the device for with the highest score is
returned. Such object can be passed to `cl::sycl::queue` and `cl::sycl::device`
constructors.

The example below illustrates how to use `cl::sycl::device_selector` to create
device and queue objects bound to Intel GPU device:

```c++
#include <CL/sycl.hpp>

int main() {
  class NEOGPUDeviceSelector : public cl::sycl::device_selector {
  public:
    int operator()(const cl::sycl::device &Device) const override {
      using namespace cl::sycl::info;

      const std::string DeviceName = Device.get_info<device::name>();
      const std::string DeviceVendor = Device.get_info<device::vendor>();

      return Device.is_gpu() && DeviceName.find("HD Graphics NEO") ? 1 : -1;
    }
  };

  NEOGPUDeviceSelector Selector;
  try {
    cl::sycl::queue Queue(Selector);
    cl::sycl::device Device(Selector);
  } catch (cl::sycl::invalid_parameter_error &E) {
    std::cout << E.what() << std::endl;
  }
}

```


# Known Issues or Limitations

- SYCL device compiler fails if the same kernel was used in different
  translation units.
- SYCL host device is not fully supported.
- SYCL works only with OpenCL implementations supporting out-of-order queues.
- `math.h` header is conflicting with SYCL headers. Please use `cmath` as a
  workaround for now like below:

```c++
//#include <math.h>  // conflicting
#include <cmath>
```

# Find More

SYCL 1.2.1 specification: [www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf](https://www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf)

