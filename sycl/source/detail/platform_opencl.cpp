//==----------- platform_opencl.cpp -----------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/platform_opencl.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/stl.hpp>

namespace cl {
namespace sycl {
namespace detail {

vector_class<device>
platform_opencl::get_devices(info::device_type deviceType) const {
  vector_class<device> res;
  if (deviceType == info::device_type::host)
    return res;
  cl_uint num_devices;
  auto err = clGetDeviceIDs(id, (cl_device_type)deviceType, 0, 0, &num_devices);
  if (err == CL_DEVICE_NOT_FOUND) {
    return res;
  }
  // TODO catch an exception and put it to list of asynchronous exceptions
  CHECK_OCL_CODE(err);
  vector_class<cl_device_id> device_ids(num_devices);
  // TODO catch an exception and put it to list of asynchronous exceptions
  CHECK_OCL_CODE(clGetDeviceIDs(id, (cl_device_type)deviceType, num_devices,
                                device_ids.data(), 0));
  vector_class<device> devices =
      vector_class<device>(device_ids.data(), device_ids.data() + num_devices);
  res.insert(res.end(), devices.begin(), devices.end());
  return res;
}

} // namespace detail
} // namespace sycl
} // namespace cl
