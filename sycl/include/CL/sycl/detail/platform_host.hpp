//==------------ platform_host.hpp - SYCL host platform --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/platform_impl.hpp>

// 4.6.2 Platform class for host platform
namespace cl {
namespace sycl {

// Forward declaration
class device;

namespace detail {
// TODO: implement extension management
// TODO: implement parameters treatment

class platform_host : public platform_impl {
public:
  vector_class<device> get_devices(
      info::device_type dev_type = info::device_type::all) const override;

  bool has_extension(const string_class &extension_name) const override {
    return false;
  }

  cl_platform_id get() const override {
    throw invalid_object_error("This instance of platform is a host instance");
  }

  bool is_host() const override { return true; }
}; // class platform_host
} // namespace detail
} // namespace sycl
} // namespace cl
