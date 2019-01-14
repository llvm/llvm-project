//==--------------- device_host.hpp - SYCL host device --------------------== //
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>

namespace cl {
namespace sycl {
namespace detail {
// TODO: 4.6.4 Partitioning into multiple SYCL devices
// TODO: 4.6.4.2 Device information descriptors
// TODO: Make code thread-safe
class device_host : public device_impl {
public:
  device_host() = default;
  cl_device_id get() const override {
    throw invalid_object_error("This instance of device is a host instance");
  }

  bool is_host() const override { return true; }

  bool is_cpu() const override { return false; }

  bool is_gpu() const override { return false; }

  bool is_accelerator() const override { return false; }

  platform get_platform() const override { return platform(); }

  bool has_extension(const string_class &extension_name) const override {
    // TODO: implement extension management;
    return false;
  }

  vector_class<device> create_sub_devices(size_t nbSubDev) const {
    // TODO: implement host device partitioning
    throw runtime_error(
        "Partitioning to subdevices of the host device is not implemented yet");
  }

  vector_class<device>
  create_sub_devices(const vector_class<size_t> &counts) const {
    // TODO: implement host device partitioning
    throw runtime_error(
        "Partitioning to subdevices of the host device is not implemented yet");
  }

  vector_class<device>
  create_sub_devices(info::partition_affinity_domain affinityDomain) const {
    // TODO: implement host device partitioning
    throw runtime_error(
        "Partitioning to subdevices of the host device is not implemented yet");
  }
};
} // namespace detail
} // namespace sycl
} // namespace cl
