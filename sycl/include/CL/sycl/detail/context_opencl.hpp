//==------------ context_opencl.hpp - SYCL OpenCL context ------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>

// 4.6.2 Context class

namespace cl {
namespace sycl {
// Forward declaration
class platform;
namespace detail {
class context_opencl : public context_impl {
public:
  context_opencl(const vector_class<cl::sycl::device> devices,
                 async_handler asyncHandler)
      : context_impl(asyncHandler) {
    dev_list = devices;
    plt = dev_list[0].get_platform();
    vector_class<cl_device_id> dev_ids;
    for (const auto &d : dev_list)
      dev_ids.push_back(d.get());
    cl_int error;
    id = clCreateContext(0, dev_ids.size(), dev_ids.data(), 0, 0, &error);
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(error);
  }

  context_opencl(cl_context clContext, async_handler asyncHandler)
      : context_impl(asyncHandler) {
    id = clContext;
    vector_class<cl_device_id> dev_ids;
    size_t devicesBuffer = 0;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(
        clGetContextInfo(id, CL_CONTEXT_DEVICES, 0, nullptr, &devicesBuffer));
    dev_ids.resize(devicesBuffer / sizeof(cl_device_id));
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetContextInfo(id, CL_CONTEXT_DEVICES, devicesBuffer,
                                    &dev_ids[0], nullptr));

    for (auto dev : dev_ids) {
      dev_list.emplace_back(dev);
    }
    // TODO What if dev_list if empty? dev_list[0].get_platform()
    plt = platform(dev_list[0].get_platform());
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clRetainContext(id));
  }

  cl_context get() const override {
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clRetainContext(id));
    return id;
  }

  bool is_host() const override { return false; }

  platform get_platform() const override { return plt; }

  vector_class<device> get_devices() const override { return dev_list; }

  ~context_opencl() {
    // TODO replace CHECK_OCL_CODE_NO_EXC to CHECK_OCL_CODE and
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE_NO_EXC(clReleaseContext(id));
  }
  // TODO: implement param traits
  // template <info::context param>
  // typename param_traits<info::context, param>::type get_info() const;
private:
  vector_class<device> dev_list;
  cl_context id;
  platform plt;
};
} // namespace detail
} // namespace sycl
} // namespace cl
