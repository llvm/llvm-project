//==------------------- device.hpp - SYCL device ---------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>
#include <memory>
#include <utility>

namespace cl {
namespace sycl {
// Forward declarations
class device_selector;

// TODO: 4.6.4 Partitioning into multiple SYCL devices
// TODO: 4.6.4.2 Device information descriptors
// TODO: Make code thread-safe
class device {
public:
  device();

  explicit device(cl_device_id deviceId);

  explicit device(const device_selector &deviceSelector);

  bool operator==(const device &rhs) const { return impl == rhs.impl; }

  bool operator!=(const device &rhs) const { return !(*this == rhs); }

  device(const device &rhs) = default;

  device(device &&rhs) = default;

  device &operator=(const device &rhs) = default;

  device &operator=(device &&rhs) = default;

  cl_device_id get() const { return impl->get(); }

  bool is_host() const { return impl->is_host(); }

  bool is_cpu() const { return impl->is_cpu(); }

  bool is_gpu() const { return impl->is_gpu(); }

  bool is_accelerator() const { return impl->is_accelerator(); }

  platform get_platform() const { return impl->get_platform(); }

  // Available only when prop == info::partition_property::partition_equally
  template <info::partition_property prop>
  typename std::enable_if<(prop == info::partition_property::partition_equally),
                          vector_class<device>>::type
  create_sub_devices(size_t ComputeUnits) const {
    return impl->create_sub_devices(ComputeUnits);
  }

  // Available only when prop == info::partition_property::partition_by_counts
  template <info::partition_property prop>
  typename std::enable_if<(prop ==
                           info::partition_property::partition_by_counts),
                          vector_class<device>>::type
  create_sub_devices(const vector_class<size_t> &Counts) const {
    return impl->create_sub_devices(Counts);
  }

  // Available only when prop ==
  // info::partition_property::partition_by_affinity_domain
  template <info::partition_property prop>
  typename std::enable_if<
      (prop == info::partition_property::partition_by_affinity_domain),
      vector_class<device>>::type
  create_sub_devices(info::partition_affinity_domain AffinityDomain) const {
    return impl->create_sub_devices(AffinityDomain);
  }

  template <info::device param>
  typename info::param_traits<info::device, param>::return_type
  get_info() const {
    return impl->get_info<param>();
  }

  bool has_extension(const string_class &extension_name) const {
    return impl->has_extension(extension_name);
  }

  static vector_class<device>
  get_devices(info::device_type deviceType = info::device_type::all);

private:
  std::shared_ptr<detail::device_impl> impl;
  template <class T>
  friend decltype(T::impl) detail::getSyclObjImpl(const T &SyclObject);
};

} // namespace sycl
} // namespace cl

namespace std {
template <> struct hash<cl::sycl::device> {
  size_t operator()(const cl::sycl::device &d) const {
    return hash<std::shared_ptr<cl::sycl::detail::device_impl>>()(
        cl::sycl::detail::getSyclObjImpl(d));
  }
};
} // namespace std
