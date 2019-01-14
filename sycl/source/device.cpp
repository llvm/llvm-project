//==------------------- device.cpp -----------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/device_host.hpp>
#include <CL/sycl/detail/device_opencl.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include "detail/force_device.hpp"

namespace cl {
namespace sycl {
namespace detail {
void force_type(info::device_type &t, const info::device_type &ft) {
  if (t == info::device_type::all) {
    t = ft;
  } else if (ft != info::device_type::all && t != ft) {
    throw cl::sycl::invalid_parameter_error("No device of forced type.");
  }
}
} // namespace detail

device::device() : impl(std::make_shared<detail::device_host>()) {}

device::device(cl_device_id deviceId)
    : impl(std::make_shared<detail::device_opencl>(deviceId)) {}

device::device(const device_selector &deviceSelector) {
  *this = deviceSelector.select_device();
}

vector_class<device> device::get_devices(info::device_type deviceType) {
  vector_class<device> devices;
  info::device_type forced_type = detail::get_forced_type();
  // Exclude devices which do not match requested device type
  if (detail::match_types(deviceType, forced_type)) {
    detail::force_type(deviceType, forced_type);
    for (const auto &plt : platform::get_platforms()) {
      vector_class<device> found_devices(plt.get_devices(deviceType));
      if (!found_devices.empty())
        devices.insert(devices.end(), found_devices.begin(),
                       found_devices.end());
    }
  }
  return devices;
}

} // namespace sycl
} // namespace cl
