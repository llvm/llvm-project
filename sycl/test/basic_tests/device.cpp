// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out

//==--------------- device.cpp - SYCL device test --------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>
#include <utility>

using namespace cl::sycl;

string_class get_type(const device &dev) {
  if (dev.is_host()) {
    return "host";
  } else if (dev.is_gpu()) {
    return "OpenCL.GPU";
  } else if (dev.is_accelerator()) {
    return "OpenCL.ACC";
  } else {
    return "OpenCL.CPU";
  }
}

int main() {
  device d;
  std::cout << "Default device type: " << get_type(d) << std::endl;

  int i = 1;
  std::cout << "Get all devices in the system" << std::endl;
  for (const auto &dev : device::get_devices()) {
    std::cout << "Device " << i++ << " is available: " << get_type(dev)
              << std::endl;
  }
  i = 1;
  std::cout << "Get host devices in the system" << std::endl;
  for (const auto &dev : device::get_devices(info::device_type::host)) {
    std::cout << "Device " << i++ << " is available: " << get_type(dev)
              << std::endl;
  }
  i = 1;
  std::cout << "Get OpenCL.CPU devices in the system" << std::endl;
  for (const auto &dev : device::get_devices(info::device_type::cpu)) {
    std::cout << "Device " << i++ << " is available: " << get_type(dev)
              << std::endl;
  }
  i = 1;
  std::cout << "Get OpenCL.GPU devices in the system" << std::endl;
  for (const auto &dev : device::get_devices(info::device_type::gpu)) {
    std::cout << "Device " << i++ << " is available: " << get_type(dev)
              << std::endl;
  }
  i = 1;
  std::cout << "Get OpenCL.ACC devices in the system" << std::endl;
  for (const auto &dev : device::get_devices(info::device_type::accelerator)) {
    std::cout << "Device " << i++ << " is available: " << get_type(dev)
              << std::endl;
  }

  auto devices = device::get_devices();
  device &deviceA = devices[0];
  device &deviceB = (devices.size() > 1 ? devices[1] : devices[0]);
  {
    std::cout << "move constructor" << std::endl;
    device Device(deviceA);
    size_t hash = hash_class<device>()(Device);
    device MovedDevice(std::move(Device));
    assert(hash == hash_class<device>()(MovedDevice));
    assert(deviceA.is_host() == MovedDevice.is_host());
    if (!deviceA.is_host()) {
      assert(MovedDevice.get() != nullptr);
    }
  }
  {
    std::cout << "move assignment operator" << std::endl;
    device Device(deviceA);
    size_t hash = hash_class<device>()(Device);
    device WillMovedDevice(deviceB);
    WillMovedDevice = std::move(Device);
    assert(hash == hash_class<device>()(WillMovedDevice));
    assert(deviceA.is_host() == WillMovedDevice.is_host());
    if (!deviceA.is_host()) {
      assert(WillMovedDevice.get() != nullptr);
    }
  }
  {
    std::cout << "copy constructor" << std::endl;
    device Device(deviceA);
    size_t hash = hash_class<device>()(Device);
    device DeviceCopy(Device);
    assert(hash == hash_class<device>()(Device));
    assert(hash == hash_class<device>()(DeviceCopy));
    assert(Device == DeviceCopy);
    assert(Device.is_host() == DeviceCopy.is_host());
  }
  {
    std::cout << "copy assignment operator" << std::endl;
    device Device(deviceA);
    size_t hash = hash_class<device>()(Device);
    device WillDeviceCopy(deviceB);
    WillDeviceCopy = Device;
    assert(hash == hash_class<device>()(Device));
    assert(hash == hash_class<device>()(WillDeviceCopy));
    assert(Device == WillDeviceCopy);
    assert(Device.is_host() == WillDeviceCopy.is_host());
  }
}

