//==---------------- context.cpp - SYCL context ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>

#include <memory>
#include <utility>

// 4.6.2 Context class

namespace cl {
namespace sycl {
context::context(const async_handler &asyncHandler)
    : context(default_selector().select_device(), asyncHandler) {}

context::context(const device &dev, async_handler asyncHandler)
    : context(vector_class<device>(1, dev), asyncHandler) {}

context::context(const platform &plt, async_handler asyncHandler)
    : context(plt.get_devices(), asyncHandler) {}

context::context(const vector_class<device> &deviceList,
                 async_handler asyncHandler) {
  if (deviceList.empty()) {
    throw invalid_parameter_error("First argument deviceList is empty.");
  }
  if (deviceList[0].is_host()) {
    impl = std::make_shared<detail::context_impl>(deviceList[0], asyncHandler);
  } else {
    // TODO also check that devices belongs to the same platform
    impl = std::make_shared<detail::context_impl>(deviceList, asyncHandler);
  }
}

context::context(cl_context clContext, async_handler asyncHandler) {
  impl = std::make_shared<detail::context_impl>(clContext, asyncHandler);
}

template <> cl_uint context::get_info<info::context::reference_count>() const {
  return impl->get_info<info::context::reference_count>();
}

template <>
cl::sycl::platform context::get_info<info::context::platform>() const {
  return impl->get_info<info::context::platform>();
}

template <>
vector_class<cl::sycl::device>
context::get_info<info::context::devices>() const {
  return impl->get_info<info::context::devices>();
}

bool context::operator==(const context &rhs) const { return impl == rhs.impl; }

bool context::operator!=(const context &rhs) const { return !(*this == rhs); }

cl_context context::get() const { return impl->get(); }

bool context::is_host() const { return impl->is_host(); }

platform context::get_platform() const { return impl->get_platform(); }

vector_class<device> context::get_devices() const {
  return impl->get_devices();
}

} // namespace sycl
} // namespace cl
