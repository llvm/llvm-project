//==-------------- platform_impl.hpp - SYCL platform -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/platform_info.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stl.hpp>

// 4.6.2 Platform class
namespace cl {
namespace sycl {

// Forward declaration
class device_selector;
class device;

namespace detail {

class platform_impl {
public:
  platform_impl() = default;

  explicit platform_impl(const device_selector &);

  virtual bool has_extension(const string_class &extension_name) const = 0;

  virtual vector_class<device>
      get_devices(info::device_type = info::device_type::all) const = 0;

  template <info::platform param>
  typename info::param_traits<info::platform, param>::return_type
  get_info() const {
    if (is_host()) {
      return get_platform_info_host<param>();
    }
    return get_platform_info_cl<
        typename info::param_traits<info::platform, param>::return_type,
        param>::_(this->get());
  }

  virtual bool is_host() const = 0;

  virtual cl_platform_id get() const = 0;

  virtual ~platform_impl() = default;
}; // class platform_impl

} // namespace detail
} // namespace sycl
} // namespace cl
