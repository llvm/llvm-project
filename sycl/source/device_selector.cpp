//==------ device_selector.cpp - SYCL device selector ----------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>
// 4.6.1 Device selection class

namespace cl {
namespace sycl {
device device_selector::select_device() const {
  vector_class<device> devices = device::get_devices();
  int score = -1;
  const device *res = nullptr;
  for (const auto &dev : devices)
    if (score < operator()(dev)) {
      res = &dev;
      score = operator()(dev);
    }

  if (res != nullptr)
    return *res;

  throw cl::sycl::invalid_parameter_error(
      "No device of requested type available.");
}

int default_selector::operator()(const device &dev) const {
  if (dev.is_gpu())
    return 500;

  if (dev.is_accelerator())
    return 400;

  if (dev.is_cpu())
    return 300;

  if (dev.is_host())
    return 100;

  return -1;
}

int gpu_selector::operator()(const device &dev) const {
  return dev.is_gpu() ? 1000 : -1;
}

int cpu_selector::operator()(const device &dev) const {
  return dev.is_cpu() ? 1000 : -1;
}

int accelerator_selector::operator()(const device &dev) const {
  return dev.is_accelerator() ? 1000 : -1;
}

int host_selector::operator()(const device &dev) const {
  return dev.is_host() ? 1000 : -1;
}

} // namespace sycl
} // namespace cl
