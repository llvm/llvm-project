// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------ subdevice.cpp - SYCL subdevice basic test -----------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <utility>

using namespace cl::sycl;

int main() {
  try {
    auto devices = device::get_devices();
    for (const auto &dev : devices) {
      // TODO: implement subdevices creation for host device
      if (dev.is_host())
        continue;

      assert(dev.get_info<info::device::partition_type_property>() ==
             info::partition_property::no_partition);

      size_t MaxSubDevices =
          dev.get_info<info::device::partition_max_sub_devices>();
      if (MaxSubDevices == 0)
        continue;

      try {
        auto SubDevicesEq =
            dev.create_sub_devices<info::partition_property::partition_equally>(
                1);
        assert(SubDevicesEq.size() == MaxSubDevices &&
               "Requested 1 compute unit in each subdevice, expected maximum "
               "number of subdevices in output");
        std::cout << "Created " << SubDevicesEq.size()
                  << " subdevices using equal partition scheme" << std::endl;

        assert(
            SubDevicesEq[0].get_info<info::device::partition_type_property>() ==
            info::partition_property::partition_equally);

        assert(SubDevicesEq[0].get_info<info::device::parent_device>().get() ==
               dev.get());
      } catch (feature_not_supported) {
        // okay skip it
      }

      try {
        vector_class<size_t> Counts(MaxSubDevices, 1);
        auto SubDevicesByCount = dev.create_sub_devices<
            info::partition_property::partition_by_counts>(Counts);
        assert(SubDevicesByCount.size() == MaxSubDevices &&
               "Maximum number of subdevices was requested with 1 compute unit "
               "on each");
        std::cout << "Created " << SubDevicesByCount.size()
                  << " subdevices using partition by counts scheme."
                  << std::endl;
        assert(SubDevicesByCount[0]
                   .get_info<info::device::partition_type_property>() ==
               info::partition_property::partition_by_counts);
      } catch (feature_not_supported) {
        // okay skip it
      }

      try {
        auto SubDevicesDomainNuma = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::numa);
        std::cout
            << "Created " << SubDevicesDomainNuma.size()
            << " subdevices using partition by numa affinity domain scheme."
            << std::endl;
      } catch (feature_not_supported) {
        // okay skip it
      }

      try {
        auto SubDevicesDomainL4 = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::L4_cache);
        std::cout << "Created " << SubDevicesDomainL4.size()
                  << " subdevices using partition by L4 cache domain scheme."
                  << std::endl;
      } catch (feature_not_supported) {
        // okay skip it
      }

      try {
        auto SubDevicesDomainL3 = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::L3_cache);
        std::cout << "Created " << SubDevicesDomainL3.size()
                  << " subdevices using partition by L3 cache domain scheme."
                  << std::endl;
      } catch (feature_not_supported) {
        // okay skip it
      }

      try {
        auto SubDevicesDomainL2 = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::L2_cache);
        std::cout << "Created " << SubDevicesDomainL2.size()
                  << " subdevices using partition by L2 cache domain scheme."
                  << std::endl;
      } catch (feature_not_supported) {
        // okay skip it
      }

      try {
        auto SubDevicesDomainL1 = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::L1_cache);
        std::cout << "Created " << SubDevicesDomainL1.size()
                  << " subdevices using partition by L1 cache domain scheme."
                  << std::endl;
      } catch (feature_not_supported) {
        // okay skip it
      }

      try {
        auto SubDevicesDomainNextPart = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::next_partitionable);
        std::cout << "Created " << SubDevicesDomainNextPart.size()
                  << " subdevices using partition by next partitionable "
                     "domain scheme."
                  << std::endl;
      } catch (feature_not_supported) {
        // okay skip it
      }
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
