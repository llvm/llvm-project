//==----------- platform.cpp -----------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/platform_host.hpp>
#include <CL/sycl/detail/platform_opencl.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/platform.hpp>
#include "detail/force_device.hpp"

namespace cl {
namespace sycl {

platform::platform() : impl(std::make_shared<detail::platform_host>()) {}

platform::platform(cl_platform_id platform_id)
    : impl(std::make_shared<detail::platform_opencl>(platform_id)) {}

platform::platform(const device_selector &dev_selector) {
  *this = dev_selector.select_device().get_platform();
}

vector_class<device> platform::get_devices(info::device_type dev_type) const {
  return impl->get_devices(dev_type);
}

vector_class<platform> platform::get_platforms() {
  static vector_class<platform> platforms;

  if (!platforms.empty()) {
    return platforms;
  }

  cl_uint num_platforms = 0;
  info::device_type forced_type = detail::get_forced_type();

  auto error = clGetPlatformIDs(0, 0, &num_platforms);
  if (error != CL_PLATFORM_NOT_FOUND_KHR)
    CHECK_OCL_CODE(error); // Skip check if no OpenCL available
  if (num_platforms) {
    vector_class<cl_platform_id> platform_ids(num_platforms);
    error = clGetPlatformIDs(num_platforms, platform_ids.data(), 0);
    CHECK_OCL_CODE(error);

    for (cl_uint i = 0; i < num_platforms; i++) {
      platform plt(platform_ids[i]);

      // Skip platforms which do not contain requested device types
      if (!plt.get_devices(forced_type).empty())
        platforms.push_back(plt);
    }
  }

  // Add host device platform if required
  if (detail::match_types(forced_type, info::device_type::host))
    platforms.push_back(platform());

  return platforms;
}

} // namespace sycl
} // namespace cl
