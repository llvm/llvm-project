//==----------- platform_host.cpp -----------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
