//==---------------- context_impl.cpp - SYCL context -----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/detail/context_info.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>

namespace cl {
namespace sycl {
namespace detail {

context_impl::context_impl(const device &Device, async_handler AsyncHandler)
    : m_AsyncHandler(AsyncHandler), m_Devices(1, Device), m_ClContext(nullptr),
      m_Platform(), m_OpenCLInterop(false), m_HostContext(true) {}

context_impl::context_impl(const vector_class<cl::sycl::device> Devices,
                           async_handler AsyncHandler)
    : m_AsyncHandler(AsyncHandler), m_Devices(Devices), m_ClContext(nullptr),
      m_Platform(), m_OpenCLInterop(true), m_HostContext(false) {
  m_Platform = m_Devices[0].get_platform();
  vector_class<cl_device_id> DeviceIds;
  for (const auto &D : m_Devices) {
    DeviceIds.push_back(D.get());
  }
  cl_int Err;
  m_ClContext =
      clCreateContext(0, DeviceIds.size(), DeviceIds.data(), 0, 0, &Err);
  // TODO catch an exception and put it to list of asynchronous exceptions
  CHECK_OCL_CODE(Err);
}

context_impl::context_impl(cl_context ClContext, async_handler AsyncHandler)
    : m_AsyncHandler(AsyncHandler), m_Devices(), m_ClContext(ClContext),
      m_Platform(), m_OpenCLInterop(true), m_HostContext(false) {
  vector_class<cl_device_id> DeviceIds;
  size_t DevicesBuffer = 0;
  // TODO catch an exception and put it to list of asynchronous exceptions
  CHECK_OCL_CODE(clGetContextInfo(m_ClContext, CL_CONTEXT_DEVICES, 0, nullptr,
                                  &DevicesBuffer));
  DeviceIds.resize(DevicesBuffer / sizeof(cl_device_id));
  // TODO catch an exception and put it to list of asynchronous exceptions
  CHECK_OCL_CODE(clGetContextInfo(m_ClContext, CL_CONTEXT_DEVICES,
                                  DevicesBuffer, &DeviceIds[0], nullptr));

  for (auto Dev : DeviceIds) {
    m_Devices.emplace_back(Dev);
  }
  // TODO What if m_Devices if empty? m_Devices[0].get_platform()
  m_Platform = platform(m_Devices[0].get_platform());
  // TODO catch an exception and put it to list of asynchronous exceptions
  CHECK_OCL_CODE(clRetainContext(m_ClContext));
}

cl_context context_impl::get() const {
  if (m_OpenCLInterop) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clRetainContext(m_ClContext));
    return m_ClContext;
  }
  throw invalid_object_error(
      "This instance of event doesn't support OpenCL interoperability.");
}

bool context_impl::is_host() const { return m_HostContext || !m_OpenCLInterop; }
platform context_impl::get_platform() const { return m_Platform; }
vector_class<device> context_impl::get_devices() const { return m_Devices; }

context_impl::~context_impl() {
  if (m_OpenCLInterop) {
    // TODO replace CHECK_OCL_CODE_NO_EXC to CHECK_OCL_CODE and
    // catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE_NO_EXC(clReleaseContext(m_ClContext));
  }
}

const async_handler &context_impl::get_async_handler() const {
  return m_AsyncHandler;
}

template <>
cl_uint context_impl::get_info<info::context::reference_count>() const {
  if (is_host()) {
    return 0;
  }
  return get_context_info_cl<info::context::reference_count>::_(this->get());
}
template <> platform context_impl::get_info<info::context::platform>() const {
  return get_platform();
}
template <>
vector_class<cl::sycl::device>
context_impl::get_info<info::context::devices>() const {
  return get_devices();
}

cl_context &context_impl::getHandleRef() { return m_ClContext; }

} // namespace detail
} // namespace sycl
} // namespace cl
