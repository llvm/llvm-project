//==----------------- device_impl.hpp - SYCL device ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/device_info.hpp>
#include <CL/sycl/stl.hpp>
#include <algorithm>
#include <memory>

namespace cl {
namespace sycl {

// Forward declaration
class platform;

namespace detail {
// TODO: 4.6.4 Partitioning into multiple SYCL devices
// TODO: 4.6.4.2 Device information descriptors
// TODO: Make code thread-safe
class device_impl {
public:
  virtual ~device_impl() = default;

  virtual cl_device_id get() const = 0;

  // Returns underlying native device object (if any) w/o reference count
  // modification. Caller must ensure the returned object lives on stack only.
  // It can also be safely passed to the underlying native runtime API.
  // Warning. Returned reference will be invalid if device_impl was destroyed.
  virtual cl_device_id &getHandleRef() = 0;

  virtual bool is_host() const = 0;

  virtual bool is_cpu() const = 0;

  virtual bool is_gpu() const = 0;

  virtual bool is_accelerator() const = 0;

  virtual platform get_platform() const = 0;

  virtual vector_class<device> create_sub_devices(size_t nbSubDev) const = 0;

  virtual vector_class<device>
  create_sub_devices(const vector_class<size_t> &counts) const = 0;

  virtual vector_class<device>
  create_sub_devices(info::partition_affinity_domain affinityDomain) const = 0;

  static vector_class<device>
  get_devices(info::device_type deviceType = info::device_type::all);

  template <info::device param>
  typename info::param_traits<info::device, param>::return_type
  get_info() const {
    if (is_host()) {
      return get_device_info_host<param>();
    }
    return get_device_info_cl<
        typename info::param_traits<info::device, param>::return_type,
        param>::_(this->get());
  }

  bool is_partition_supported(info::partition_property Prop) const {
    auto SupportedProperties = get_info<info::device::partition_properties>();
    return std::find(SupportedProperties.begin(), SupportedProperties.end(),
                     Prop) != SupportedProperties.end();
  }

  bool
  is_affinity_supported(info::partition_affinity_domain AffinityDomain) const {
    auto SupportedDomains =
        get_info<info::device::partition_affinity_domains>();
    return std::find(SupportedDomains.begin(), SupportedDomains.end(),
                     AffinityDomain) != SupportedDomains.end();
  }

  virtual bool has_extension(const string_class &extension_name) const = 0;
};
} // namespace detail
} // namespace sycl
} // namespace cl
