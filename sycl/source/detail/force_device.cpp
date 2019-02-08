//==---------- force_device.cpp - Forcing SYCL device ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stl.hpp>
#include <cstdlib>
#include "force_device.hpp"

namespace cl {
namespace sycl {
namespace detail {

bool match_types(const info::device_type &l, const info::device_type &r) {
  return l == info::device_type::all || l == r || r == info::device_type::all;
}

info::device_type get_forced_type() {
  if (const char *val = std::getenv("SYCL_DEVICE_TYPE")) {
    if (string_class(val) == "CPU") {
      return info::device_type::cpu;
    }
    if (string_class(val) == "GPU") {
      return info::device_type::gpu;
    }
    if (string_class(val) == "ACC") {
      return info::device_type::accelerator;
    }
    if (string_class(val) == "HOST") {
      return info::device_type::host;
    }
  }
  return info::device_type::all;
}

} // namespace detail
} // namespace sycl
} // namespace cl
