//==---------------- context.hpp - SYCL context ----------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/context_host.hpp>
#include <CL/sycl/detail/context_opencl.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>
#include <memory>
#include <utility>
// 4.6.2 Context class

namespace cl {
namespace sycl {
class context {
public:
  explicit context(const async_handler &asyncHandler = {})
      : context(default_selector().select_device(), asyncHandler) {}

  context(const device &dev, async_handler asyncHandler = {})
      : context(vector_class<device>(1, dev), asyncHandler) {}

  context(const platform &plt, async_handler asyncHandler = {})
      : context(plt.get_devices(), asyncHandler) {}

  context(const vector_class<device> &deviceList,
          async_handler asyncHandler = {});

  context(cl_context clContext, async_handler asyncHandler = {});

  template <info::context param>
  typename info::param_traits<info::context, param>::return_type
  get_info() const {
    return impl->get_info<param>();
  }

  context(const context &rhs) = default;

  context(context &&rhs) = default;

  context &operator=(const context &rhs) = default;

  context &operator=(context &&rhs) = default;

  bool operator==(const context &rhs) const { return impl == rhs.impl; }

  bool operator!=(const context &rhs) const { return !(*this == rhs); }

  cl_context get() const { return impl->get(); }

  bool is_host() const { return impl->is_host(); }

  platform get_platform() const { return impl->get_platform(); }

  vector_class<device> get_devices() const { return impl->get_devices(); }

private:
  std::shared_ptr<detail::context_impl> impl;
  template <class T>
  friend decltype(T::impl) detail::getSyclObjImpl(const T &SyclObject);
};

} // namespace sycl
} // namespace cl

namespace std {
template <> struct hash<cl::sycl::context> {
  size_t operator()(const cl::sycl::context &c) const {
    return hash<std::shared_ptr<cl::sycl::detail::context_impl>>()(
        cl::sycl::detail::getSyclObjImpl(c));
  }
};
} // namespace std
