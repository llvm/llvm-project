//==------------ device_opencl.hpp - SYCL OpenCL device --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>

class device_selector;

namespace cl {
namespace sycl {
namespace detail {
// TODO: 4.6.4 Partitioning into multiple SYCL devices
// TODO: 4.6.4.2 Device information descriptors
// TODO: Make code thread-safe
class device_opencl : public device_impl {
public:
  /** Constructs a device class instance using cl device_id of the OpenCL
   * device. */
  explicit device_opencl(cl_device_id deviceId) {
    id = deviceId;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(
        clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0));
    cl_device_id parent;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetDeviceInfo(id, CL_DEVICE_PARENT_DEVICE,
                                   sizeof(cl_device_id), &parent, nullptr));
    isRootDevice = (nullptr == parent);
    if (!isRootDevice) {
      // TODO catch an exception and put it to list of asynchronous exceptions
      CHECK_OCL_CODE(clRetainDevice(id));
    }
  }

  ~device_opencl() {
    if (!isRootDevice) {
      // TODO replace CHECK_OCL_CODE_NO_EXC to CHECK_OCL_CODE and
      // TODO catch an exception and put it to list of asynchronous exceptions
      CHECK_OCL_CODE_NO_EXC(clReleaseDevice(id));
    }
  }

  cl_device_id get() const override {
    if (!isRootDevice) {
      // TODO catch an exception and put it to list of asynchronous exceptions
      CHECK_OCL_CODE(clRetainDevice(id));
    }
    return id;
  }

  cl_device_id &getHandleRef() override{
    return id;
  }

  bool is_host() const override { return false; }

  bool is_cpu() const override { return (type == CL_DEVICE_TYPE_CPU); }

  bool is_gpu() const override { return (type == CL_DEVICE_TYPE_GPU); }

  bool is_accelerator() const override {
    return (type == CL_DEVICE_TYPE_ACCELERATOR);
  }

  platform get_platform() const override {
    cl_platform_id plt_id;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(
        clGetDeviceInfo(id, CL_DEVICE_PLATFORM, sizeof(plt_id), &plt_id, 0));
    return platform(plt_id);
  }

  bool has_extension(const string_class &extension_name) const override {
    string_class all_extension_names =
        get_device_info_cl<string_class, info::device::extensions>::_(id);
    return (all_extension_names.find(extension_name) != std::string::npos);
  }

  vector_class<device>
  create_sub_devices(const cl_device_partition_property *Properties,
                     size_t SubDevicesCount) const {
    vector_class<cl_device_id> SubDevices(SubDevicesCount);
    cl_uint ReturnedSubDevices;
    CHECK_OCL_CODE(clCreateSubDevices(id, Properties, SubDevicesCount,
                                      SubDevices.data(), &ReturnedSubDevices));
    return vector_class<device>(SubDevices.begin(), SubDevices.end());
  }

  vector_class<device> create_sub_devices(size_t ComputeUnits) const {
    if (!is_partition_supported(info::partition_property::partition_equally)) {
      throw cl::sycl::feature_not_supported();
    }
    size_t SubDevicesCount =
        get_info<info::device::max_compute_units>() / ComputeUnits;
    const cl_device_partition_property Properties[3] = {
        CL_DEVICE_PARTITION_EQUALLY, (cl_device_partition_property)ComputeUnits,
        0};
    return create_sub_devices(Properties, SubDevicesCount);
  }

  vector_class<device>
  create_sub_devices(const vector_class<size_t> &Counts) const {
    if (!is_partition_supported(
            info::partition_property::partition_by_counts)) {
      throw cl::sycl::feature_not_supported();
    }
    static const cl_device_partition_property P[] = {
        CL_DEVICE_PARTITION_BY_COUNTS, CL_DEVICE_PARTITION_BY_COUNTS_LIST_END,
        0};
    vector_class<cl_device_partition_property> Properties(P, P + 3);
    Properties.insert(Properties.begin() + 1, Counts.begin(), Counts.end());
    return create_sub_devices(Properties.data(), Counts.size());
  }

  vector_class<device>
  create_sub_devices(info::partition_affinity_domain AffinityDomain) const {
    if (!is_partition_supported(
            info::partition_property::partition_by_affinity_domain) ||
        !is_affinity_supported(AffinityDomain)) {
      throw cl::sycl::feature_not_supported();
    }
    const cl_device_partition_property Properties[3] = {
        CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
        (cl_device_partition_property)AffinityDomain, 0};
    size_t SubDevicesCount =
        get_info<info::device::partition_max_sub_devices>();
    return create_sub_devices(Properties, SubDevicesCount);
  }

private:
  cl_device_id id = 0;
  cl_device_type type = 0;
  bool isRootDevice = false;
};
} // namespace detail
} // namespace sycl
} // namespace cl
