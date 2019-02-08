//==----------- platform_host.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/platform_host.hpp>
#include <CL/sycl/device.hpp>

namespace cl {
namespace sycl {
namespace detail {

vector_class<device>
platform_host::get_devices(info::device_type dev_type) const {
  vector_class<device> res;
  if (dev_type == info::device_type::host || dev_type == info::device_type::all)
    res.resize(1); // default device construct creates host device
  return res;
}

} // namespace detail
} // namespace sycl
} // namespace cl
