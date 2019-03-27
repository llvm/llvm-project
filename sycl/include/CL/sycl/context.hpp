//==---------------- context.hpp - SYCL context ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stl.hpp>
#include <memory>
#include <type_traits>
// 4.6.2 Context class

namespace cl {
namespace sycl {
// Forward declarations
class device;
class platform;
class context {
public:
  explicit context(const async_handler &asyncHandler = {});

  context(const device &dev, async_handler asyncHandler = {});

  context(const platform &plt, async_handler asyncHandler = {});

  context(const vector_class<device> &deviceList,
          async_handler asyncHandler = {});

  context(cl_context clContext, async_handler asyncHandler = {});

  template <info::context param>
  typename info::param_traits<info::context, param>::return_type
  get_info() const;

  context(const context &rhs) = default;

  context(context &&rhs) = default;

  context &operator=(const context &rhs) = default;

  context &operator=(context &&rhs) = default;

  bool operator==(const context &rhs) const;

  bool operator!=(const context &rhs) const;

  cl_context get() const;

  bool is_host() const;

  platform get_platform() const;

  vector_class<device> get_devices() const;

private:
  std::shared_ptr<detail::context_impl> impl;
  template <class T>
  friend decltype(T::impl) detail::getSyclObjImpl(const T &SyclObject);

  template <class T>
  friend
      typename std::add_pointer<typename decltype(T::impl)::element_type>::type
      detail::getRawSyclObjImpl(const T &SyclObject);
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
