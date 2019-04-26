//==----------- device_info.cpp --------------------------------*- C ++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/device_info.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/platform_util.hpp>
#include <CL/sycl/device.hpp>
#include <chrono>
#include <thread>

#ifdef __GNUG__
#define GCC_VERSION                                                            \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif

namespace cl {
namespace sycl {
namespace detail {

vector_class<info::fp_config> read_fp_bitfield(cl_device_fp_config bits) {
  vector_class<info::fp_config> result;
  if (bits & CL_FP_DENORM)
    result.push_back(info::fp_config::denorm);
  if (bits & CL_FP_INF_NAN)
    result.push_back(info::fp_config::inf_nan);
  if (bits & CL_FP_ROUND_TO_NEAREST)
    result.push_back(info::fp_config::round_to_nearest);
  if (bits & CL_FP_ROUND_TO_ZERO)
    result.push_back(info::fp_config::round_to_zero);
  if (bits & CL_FP_ROUND_TO_INF)
    result.push_back(info::fp_config::round_to_inf);
  if (bits & CL_FP_FMA)
    result.push_back(info::fp_config::fma);
  if (bits & CL_FP_SOFT_FLOAT)
    result.push_back(info::fp_config::soft_float);
  if (bits & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)
    result.push_back(info::fp_config::correctly_rounded_divide_sqrt);
  return result;
}

vector_class<info::partition_affinity_domain>
read_domain_bitfield(cl_device_affinity_domain bits) {
  vector_class<info::partition_affinity_domain> result;
  if (bits & CL_DEVICE_AFFINITY_DOMAIN_NUMA)
    result.push_back(info::partition_affinity_domain::numa);
  if (bits & CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE)
    result.push_back(info::partition_affinity_domain::L4_cache);
  if (bits & CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE)
    result.push_back(info::partition_affinity_domain::L3_cache);
  if (bits & CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE)
    result.push_back(info::partition_affinity_domain::L2_cache);
  if (bits & CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE)
    result.push_back(info::partition_affinity_domain::L1_cache);
  if (bits & CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE)
    result.push_back(info::partition_affinity_domain::next_partitionable);
  return result;
}

vector_class<info::execution_capability>
read_execution_bitfield(cl_device_exec_capabilities bits) {
  vector_class<info::execution_capability> result;
  if (bits & CL_EXEC_KERNEL)
    result.push_back(info::execution_capability::exec_kernel);
  if (bits & CL_EXEC_NATIVE_KERNEL)
    result.push_back(info::execution_capability::exec_native_kernel);
  return result;
}

template <>
info::device_type get_device_info_host<info::device::device_type>() {
  return info::device_type::host;
}

template <> cl_uint get_device_info_host<info::device::vendor_id>() {
  return 0x8086;
}

template <> cl_uint get_device_info_host<info::device::max_compute_units>() {
  return std::thread::hardware_concurrency();
}

template <>
cl_uint get_device_info_host<info::device::max_work_item_dimensions>() {
  return 3;
}

template <> id<3> get_device_info_host<info::device::max_work_item_sizes>() {
  // current value is the required minimum
  return {1, 1, 1};
}

template <> size_t get_device_info_host<info::device::max_work_group_size>() {
  // current value is the required minimum
  return 1;
}

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_char>() {
  // TODO update when appropriate
  return 1;
}

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_short>() {
  // TODO update when appropriate
  return 1;
}

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_int>() {
  // TODO update when appropriate
  return 1;
}

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_long>() {
  // TODO update when appropriate
  return 1;
}

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_float>() {
  // TODO update when appropriate
  return 1;
}

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_double>() {
  // TODO update when appropriate
  return 1;
}

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_half>() {
  // TODO update when appropriate
  return 0;
}

template <>
cl_uint get_device_info_host<info::device::native_vector_width_char>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Char);
}

template <>
cl_uint get_device_info_host<info::device::native_vector_width_short>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Short);
}

template <>
cl_uint get_device_info_host<info::device::native_vector_width_int>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Int);
}

template <>
cl_uint get_device_info_host<info::device::native_vector_width_long>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Long);
}

template <>
cl_uint get_device_info_host<info::device::native_vector_width_float>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Float);
}

template <>
cl_uint get_device_info_host<info::device::native_vector_width_double>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Double);
}

template <>
cl_uint get_device_info_host<info::device::native_vector_width_half>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Half);
}

template <> cl_uint get_device_info_host<info::device::max_clock_frequency>() {
  return PlatformUtil::getMaxClockFrequency();
}

template <> cl_uint get_device_info_host<info::device::address_bits>() {
  return sizeof(void *) * 8;
}

template <> cl_ulong get_device_info_host<info::device::global_mem_size>() {
  return static_cast<cl_ulong>(OSUtil::getOSMemSize());
}

template <> cl_ulong get_device_info_host<info::device::max_mem_alloc_size>() {
  // current value is the required minimum
  const cl_ulong a = get_device_info_host<info::device::global_mem_size>() / 4;
  const cl_ulong b = 128ul * 1024 * 1024;
  return (a > b) ? a : b;
}

template <> bool get_device_info_host<info::device::image_support>() {
  return true;
}

template <> cl_uint get_device_info_host<info::device::max_read_image_args>() {
  // current value is the required minimum
  return 128;
}

template <> cl_uint get_device_info_host<info::device::max_write_image_args>() {
  // current value is the required minimum
  return 8;
}

template <> size_t get_device_info_host<info::device::image2d_max_width>() {
  // current value is the required minimum
  return 8192;
}

template <> size_t get_device_info_host<info::device::image2d_max_height>() {
  // current value is the required minimum
  return 8192;
}

template <> size_t get_device_info_host<info::device::image3d_max_width>() {
  // current value is the required minimum
  return 2048;
}

template <> size_t get_device_info_host<info::device::image3d_max_height>() {
  // current value is the required minimum
  return 2048;
}

template <> size_t get_device_info_host<info::device::image3d_max_depth>() {
  // current value is the required minimum
  return 2048;
}

template <> size_t get_device_info_host<info::device::image_max_buffer_size>() {
  // Not supported in SYCL
  return 0;
}

template <> size_t get_device_info_host<info::device::image_max_array_size>() {
  // current value is the required minimum
  return 2048;
}

template <> cl_uint get_device_info_host<info::device::max_samplers>() {
  // current value is the required minimum
  return 16;
}

template <> size_t get_device_info_host<info::device::max_parameter_size>() {
  // current value is the required minimum
  return 1024;
}

template <> cl_uint get_device_info_host<info::device::mem_base_addr_align>() {
  return 1024;
}

template <>
vector_class<info::fp_config>
get_device_info_host<info::device::half_fp_config>() {
  // current value is the required minimum
  return {};
}

template <>
vector_class<info::fp_config>
get_device_info_host<info::device::single_fp_config>() {
  // current value is the required minimum
  return {info::fp_config::round_to_nearest, info::fp_config::inf_nan};
}

template <>
vector_class<info::fp_config>
get_device_info_host<info::device::double_fp_config>() {
  // current value is the required minimum
  return {info::fp_config::fma,           info::fp_config::round_to_nearest,
          info::fp_config::round_to_zero, info::fp_config::round_to_inf,
          info::fp_config::inf_nan,       info::fp_config::denorm};
}

template <>
info::global_mem_cache_type
get_device_info_host<info::device::global_mem_cache_type>() {
  return info::global_mem_cache_type::write_only;
}

template <>
cl_uint get_device_info_host<info::device::global_mem_cache_line_size>() {
  return PlatformUtil::getMemCacheLineSize();
}

template <>
cl_ulong get_device_info_host<info::device::global_mem_cache_size>() {
  return PlatformUtil::getMemCacheSize();
}

template <>
cl_ulong get_device_info_host<info::device::max_constant_buffer_size>() {
  // current value is the required minimum
  return 64 * 1024;
}

template <> cl_uint get_device_info_host<info::device::max_constant_args>() {
  // current value is the required minimum
  return 8;
}

template <>
info::local_mem_type get_device_info_host<info::device::local_mem_type>() {
  return info::local_mem_type::global;
}

template <> cl_ulong get_device_info_host<info::device::local_mem_size>() {
  // current value is the required minimum
  return 32 * 1024;
}

template <>
bool get_device_info_host<info::device::error_correction_support>() {
  return false;
}

template <> bool get_device_info_host<info::device::host_unified_memory>() {
  return true;
}

template <>
size_t get_device_info_host<info::device::profiling_timer_resolution>() {
  typedef std::ratio_divide<std::chrono::high_resolution_clock::period,
                            std::nano>
      ns_period;
  return ns_period::num / ns_period::den;
}

template <> bool get_device_info_host<info::device::is_endian_little>() {
  union {
    uint16_t a;
    uint8_t b[2];
  } u = {0x0100};

  return u.b[1];
}

template <> bool get_device_info_host<info::device::is_available>() {
  return true;
}

template <> bool get_device_info_host<info::device::is_compiler_available>() {
  return true;
}

template <> bool get_device_info_host<info::device::is_linker_available>() {
  return true;
}

template <>
vector_class<info::execution_capability>
get_device_info_host<info::device::execution_capabilities>() {
  return {info::execution_capability::exec_kernel};
}

template <> bool get_device_info_host<info::device::queue_profiling>() {
  return true;
}

template <>
vector_class<string_class>
get_device_info_host<info::device::built_in_kernels>() {
  return {};
}

template <> platform get_device_info_host<info::device::platform>() {
  return platform();
}

template <> string_class get_device_info_host<info::device::name>() {
  return "SYCL host device";
}

template <> string_class get_device_info_host<info::device::vendor>() {
  return "";
}

template <> string_class get_device_info_host<info::device::driver_version>() {
  return "1.2";
}

template <> string_class get_device_info_host<info::device::profile>() {
  return "FULL PROFILE";
}

template <> string_class get_device_info_host<info::device::version>() {
  return "1.2";
}

template <>
string_class get_device_info_host<info::device::opencl_c_version>() {
  return "not applicable";
}

template <>
vector_class<string_class> get_device_info_host<info::device::extensions>() {
  // TODO update when appropriate
  return {};
}

template <> size_t get_device_info_host<info::device::printf_buffer_size>() {
  // current value is the required minimum
  return 1024 * 1024;
}

template <>
bool get_device_info_host<info::device::preferred_interop_user_sync>() {
  return false;
}

template <> device get_device_info_host<info::device::parent_device>() {
  // TODO: implement host device partitioning
  throw runtime_error(
      "Partitioning to subdevices of the host device is not implemented yet");
}

template <>
cl_uint get_device_info_host<info::device::partition_max_sub_devices>() {
  // TODO update once subdevice creation is enabled
  return 1;
}

template <>
vector_class<info::partition_property>
get_device_info_host<info::device::partition_properties>() {
  // TODO update once subdevice creation is enabled
  return {};
}

template <>
vector_class<info::partition_affinity_domain>
get_device_info_host<info::device::partition_affinity_domains>() {
  // TODO update once subdevice creation is enabled
  return {};
}

template <>
info::partition_property
get_device_info_host<info::device::partition_type_property>() {
  return info::partition_property::no_partition;
}

template <>
info::partition_affinity_domain
get_device_info_host<info::device::partition_type_affinity_domain>() {
  // TODO update once subdevice creation is enabled
  return info::partition_affinity_domain::not_applicable;
}

template <> cl_uint get_device_info_host<info::device::reference_count>() {
  // TODO update once subdevice creation is enabled
  return 1;
}

template <> cl_uint get_device_info_host<info::device::max_num_sub_groups>() {
  // TODO update once subgroups are enabled
  throw runtime_error("Sub-group feature is not supported on HOST device.");
}

template <>
bool get_device_info_host<
    info::device::sub_group_independent_forward_progress>() {
  // TODO update once subgroups are enabled
  throw runtime_error("Sub-group feature is not supported on HOST device.");
}

} // namespace detail
} // namespace sycl
} // namespace cl
