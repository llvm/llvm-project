//==-------- device_info.hpp - SYCL device info methods --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common_info.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>

namespace cl {
namespace sycl {
namespace detail {

vector_class<info::fp_config> read_fp_bitfield(cl_device_fp_config bits);

vector_class<info::partition_affinity_domain>
read_domain_bitfield(cl_device_affinity_domain bits);

vector_class<info::execution_capability>
read_execution_bitfield(cl_device_exec_capabilities bits);

// Mapping expected SYCL return types to those returned by OpenCL calls
template <typename T> struct sycl_to_ocl { using type = T; };

template <> struct sycl_to_ocl<bool> { using type = cl_bool; };

template <> struct sycl_to_ocl<device> { using type = cl_device_id; };

template <> struct sycl_to_ocl<platform> { using type = cl_platform_id; };

// Mapping fp_config device info types to the values used to check fp support
template <info::device param> struct check_fp_support {};

template <> struct check_fp_support<info::device::half_fp_config> {
  static const info::device value = info::device::native_vector_width_half;
};

template <> struct check_fp_support<info::device::double_fp_config> {
  static const info::device value = info::device::native_vector_width_double;
};

// Structs for emulating function template partial specialization
// Default template for the general case
template <typename T, info::device param> struct get_device_info_cl {
  static T _(cl_device_id dev) {
    typename sycl_to_ocl<T>::type result;
    CHECK_OCL_CODE(clGetDeviceInfo(dev, (cl_device_info)param, sizeof(result),
                                   &result, NULL));
    return T(result);
  }
};

// Specialization for string return type, variable OpenCL return size
template <info::device param> struct get_device_info_cl<string_class, param> {
  static string_class _(cl_device_id dev) {
    size_t resultSize;
    CHECK_OCL_CODE(
        clGetDeviceInfo(dev, (cl_device_info)param, 0, NULL, &resultSize));
    if (resultSize == 0) {
      return string_class();
    }
    unique_ptr_class<char[]> result(new char[resultSize]);
    CHECK_OCL_CODE(clGetDeviceInfo(dev, (cl_device_info)param, resultSize,
                                   result.get(), NULL));
    return string_class(result.get());
  }
};

// Specialization for id return type
template <info::device param> struct get_device_info_cl<id<3>, param> {
  static id<3> _(cl_device_id dev) {
    size_t result[3];
    CHECK_OCL_CODE(clGetDeviceInfo(dev, (cl_device_info)param, sizeof(result),
                                   &result, NULL));
    return id<3>(result[0], result[1], result[2]);
  }
};

// Specialization for fp_config types, checks the corresponding fp type support
template <info::device param>
struct get_device_info_cl<vector_class<info::fp_config>, param> {
  static vector_class<info::fp_config> _(cl_device_id dev) {
    // Check if fp type is supported
    if (!get_device_info_cl<
            typename info::param_traits<
                info::device, check_fp_support<param>::value>::return_type,
            check_fp_support<param>::value>::_(dev)) {
      return {};
    }
    cl_device_fp_config result;
    CHECK_OCL_CODE(clGetDeviceInfo(dev, (cl_device_info)param, sizeof(result),
                                   &result, NULL));
    return read_fp_bitfield(result);
  }
};

// Specialization for single_fp_config, no type support check required
template <>
struct get_device_info_cl<vector_class<info::fp_config>,
                          info::device::single_fp_config> {
  static vector_class<info::fp_config> _(cl_device_id dev) {
    cl_device_fp_config result;
    CHECK_OCL_CODE(
        clGetDeviceInfo(dev, (cl_device_info)info::device::single_fp_config,
                        sizeof(result), &result, NULL));
    return read_fp_bitfield(result);
  }
};

// Specialization for queue_profiling, OpenCL returns a bitfield
template <> struct get_device_info_cl<bool, info::device::queue_profiling> {
  static bool _(cl_device_id dev) {
    cl_command_queue_properties result;
    CHECK_OCL_CODE(
        clGetDeviceInfo(dev, (cl_device_info)info::device::queue_profiling,
                        sizeof(result), &result, NULL));
    return (result & CL_QUEUE_PROFILING_ENABLE);
  }
};

// Specialization for exec_capabilities, OpenCL returns a bitfield
template <>
struct get_device_info_cl<vector_class<info::execution_capability>,
                          info::device::execution_capabilities> {
  static vector_class<info::execution_capability> _(cl_device_id dev) {
    cl_device_exec_capabilities result;
    CHECK_OCL_CODE(clGetDeviceInfo(
        dev, (cl_device_info)info::device::execution_capabilities,
        sizeof(result), &result, NULL));
    return read_execution_bitfield(result);
  }
};

// Specialization for built in kernels, splits the string returned by OpenCL
template <>
struct get_device_info_cl<vector_class<string_class>,
                          info::device::built_in_kernels> {
  static vector_class<string_class> _(cl_device_id dev) {
    string_class result =
        get_device_info_cl<string_class, info::device::built_in_kernels>::_(
            dev);
    return split_string(result, ';');
  }
};

// Specialization for extensions, splits the string returned by OpenCL
template <>
struct get_device_info_cl<vector_class<string_class>,
                          info::device::extensions> {
  static vector_class<string_class> _(cl_device_id dev) {
    string_class result =
        get_device_info_cl<string_class, info::device::extensions>::_(dev);
    return split_string(result, ' ');
  }
};

// Specialization for partition properties, variable OpenCL return size
template <>
struct get_device_info_cl<vector_class<info::partition_property>,
                          info::device::partition_properties> {
  static vector_class<info::partition_property> _(cl_device_id dev) {
    size_t resultSize;
    CHECK_OCL_CODE(
        clGetDeviceInfo(dev, (cl_device_info)info::device::partition_properties,
                        0, NULL, &resultSize));
    size_t arrayLength = resultSize / sizeof(cl_device_partition_property);
    if (arrayLength == 0) {
      return {};
    }
    unique_ptr_class<cl_device_partition_property[]> arrayResult(
        new cl_device_partition_property[arrayLength]);
    CHECK_OCL_CODE(
        clGetDeviceInfo(dev, (cl_device_info)info::device::partition_properties,
                        resultSize, arrayResult.get(), NULL));

    vector_class<info::partition_property> result;
    for (size_t i = 0; i < arrayLength - 1; ++i) {
      result.push_back(info::partition_property(arrayResult[i]));
    }
    return result;
  }
};

// Specialization for partition affinity domains, OpenCL returns a bitfield
template <>
struct get_device_info_cl<vector_class<info::partition_affinity_domain>,
                          info::device::partition_affinity_domains> {
  static vector_class<info::partition_affinity_domain> _(cl_device_id dev) {
    cl_device_affinity_domain result;
    CHECK_OCL_CODE(clGetDeviceInfo(
        dev, (cl_device_info)info::device::partition_affinity_domains,
        sizeof(result), &result, NULL));
    return read_domain_bitfield(result);
  }
};

// Specialization for partition type affinity domain, OpenCL can return other
// partition properties instead
template <>
struct get_device_info_cl<info::partition_affinity_domain,
                          info::device::partition_type_affinity_domain> {
  static info::partition_affinity_domain _(cl_device_id dev) {
    size_t resultSize;
    CHECK_OCL_CODE(clGetDeviceInfo(
        dev, (cl_device_info)info::device::partition_type_affinity_domain, 0,
        NULL, &resultSize));
    if (resultSize != 1) {
      return info::partition_affinity_domain::not_applicable;
    }
    cl_device_partition_property result;
    CHECK_OCL_CODE(clGetDeviceInfo(
        dev, (cl_device_info)info::device::partition_type_affinity_domain,
        sizeof(result), &result, NULL));
    if (result == CL_DEVICE_AFFINITY_DOMAIN_NUMA ||
        result == CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE ||
        result == CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE ||
        result == CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE ||
        result == CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE) {
      return info::partition_affinity_domain(result);
    }

    return info::partition_affinity_domain::not_applicable;
  }
};

// Specialization for partition type
template <>
struct get_device_info_cl<info::partition_property,
                          info::device::partition_type_property> {
  static info::partition_property _(cl_device_id dev) {
    size_t resultSize;
    CHECK_OCL_CODE(
        clGetDeviceInfo(dev, CL_DEVICE_PARTITION_TYPE, 0, NULL, &resultSize));
    if (!resultSize)
      return info::partition_property::no_partition;

    size_t arrayLength = resultSize / sizeof(cl_device_partition_property);

    unique_ptr_class<cl_device_partition_property[]> arrayResult(
        new cl_device_partition_property[arrayLength]);
    CHECK_OCL_CODE(clGetDeviceInfo(dev, CL_DEVICE_PARTITION_TYPE, resultSize,
                                   arrayResult.get(), NULL));
    if (!arrayResult[0])
      return info::partition_property::no_partition;
    return info::partition_property(arrayResult[0]);
  }
};

// Specialization for parent device
template <typename T>
struct get_device_info_cl<T, info::device::parent_device> {
  static T _(cl_device_id dev) {
    typename sycl_to_ocl<T>::type result;
    CHECK_OCL_CODE(
        clGetDeviceInfo(dev, (cl_device_info)info::device::parent_device,
                        sizeof(result), &result, NULL));
    if (result == nullptr)
      throw invalid_object_error(
          "No parent for device because it is not a subdevice");
    return T(result);
  }
};

// Specialization for supported subgroup sizes
template <>
struct get_device_info_cl<vector_class<size_t>,
                          info::device::sub_group_sizes> {
  static vector_class<size_t> _(cl_device_id dev) {
    size_t resultSize = 0;
    CHECK_OCL_CODE(
        clGetDeviceInfo(dev, (cl_device_info)info::device::sub_group_sizes,
                        0, nullptr, &resultSize));
    vector_class<size_t> result(resultSize);
    CHECK_OCL_CODE(
        clGetDeviceInfo(dev, (cl_device_info)info::device::sub_group_sizes,
                        resultSize, result.data(), nullptr));
    return result;
  }
};


// SYCL host device information

// Default template is disabled, all possible instantiations are
// specified explicitly.
template <info::device param>
typename info::param_traits<info::device, param>::return_type
get_device_info_host() = delete;

template <> info::device_type get_device_info_host<info::device::device_type>();

template <> cl_uint get_device_info_host<info::device::vendor_id>();

template <> cl_uint get_device_info_host<info::device::max_compute_units>();

template <>
cl_uint get_device_info_host<info::device::max_work_item_dimensions>();

template <> id<3> get_device_info_host<info::device::max_work_item_sizes>();

template <> size_t get_device_info_host<info::device::max_work_group_size>();

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_char>();

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_short>();

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_int>();

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_long>();

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_float>();

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_double>();

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_half>();

cl_uint get_native_vector_width(size_t idx);

template <>
cl_uint get_device_info_host<info::device::preferred_vector_width_half>();

template <>
cl_uint get_device_info_host<info::device::native_vector_width_char>();

template <>
cl_uint get_device_info_host<info::device::native_vector_width_short>();

template <>
cl_uint get_device_info_host<info::device::native_vector_width_int>();

template <>
cl_uint get_device_info_host<info::device::native_vector_width_long>();

template <>
cl_uint get_device_info_host<info::device::native_vector_width_float>();

template <>
cl_uint get_device_info_host<info::device::native_vector_width_double>();

template <>
cl_uint get_device_info_host<info::device::native_vector_width_half>();

template <> cl_uint get_device_info_host<info::device::max_clock_frequency>();

template <> cl_uint get_device_info_host<info::device::address_bits>();

template <> cl_ulong get_device_info_host<info::device::global_mem_size>();

template <> cl_ulong get_device_info_host<info::device::max_mem_alloc_size>();

template <> bool get_device_info_host<info::device::image_support>();

template <> cl_uint get_device_info_host<info::device::max_read_image_args>();

template <> cl_uint get_device_info_host<info::device::max_write_image_args>();

template <> size_t get_device_info_host<info::device::image2d_max_width>();

template <> size_t get_device_info_host<info::device::image2d_max_height>();

template <> size_t get_device_info_host<info::device::image3d_max_width>();

template <> size_t get_device_info_host<info::device::image3d_max_height>();

template <> size_t get_device_info_host<info::device::image3d_max_depth>();

template <> size_t get_device_info_host<info::device::image_max_buffer_size>();

template <> size_t get_device_info_host<info::device::image_max_array_size>();

template <> cl_uint get_device_info_host<info::device::max_samplers>();

template <> size_t get_device_info_host<info::device::max_parameter_size>();

template <> cl_uint get_device_info_host<info::device::mem_base_addr_align>();

template <>
vector_class<info::fp_config>
get_device_info_host<info::device::half_fp_config>();

template <>
vector_class<info::fp_config>
get_device_info_host<info::device::single_fp_config>();

template <>
vector_class<info::fp_config>
get_device_info_host<info::device::double_fp_config>();

template <>
info::global_mem_cache_type
get_device_info_host<info::device::global_mem_cache_type>();

template <>
cl_uint get_device_info_host<info::device::global_mem_cache_line_size>();

template <>
cl_ulong get_device_info_host<info::device::global_mem_cache_size>();

template <>
cl_ulong get_device_info_host<info::device::max_constant_buffer_size>();

template <> cl_uint get_device_info_host<info::device::max_constant_args>();

template <>
info::local_mem_type get_device_info_host<info::device::local_mem_type>();

template <> cl_ulong get_device_info_host<info::device::local_mem_size>();

template <> bool get_device_info_host<info::device::error_correction_support>();

template <> bool get_device_info_host<info::device::host_unified_memory>();

template <>
size_t get_device_info_host<info::device::profiling_timer_resolution>();

template <> bool get_device_info_host<info::device::is_endian_little>();

template <> bool get_device_info_host<info::device::is_available>();

template <> bool get_device_info_host<info::device::is_compiler_available>();

template <> bool get_device_info_host<info::device::is_linker_available>();

template <>
vector_class<info::execution_capability>
get_device_info_host<info::device::execution_capabilities>();

template <> bool get_device_info_host<info::device::queue_profiling>();

template <>
vector_class<string_class>
get_device_info_host<info::device::built_in_kernels>();

template <> platform get_device_info_host<info::device::platform>();

template <> string_class get_device_info_host<info::device::name>();

template <> string_class get_device_info_host<info::device::vendor>();

template <> string_class get_device_info_host<info::device::driver_version>();

template <> string_class get_device_info_host<info::device::profile>();

template <> string_class get_device_info_host<info::device::version>();

template <> string_class get_device_info_host<info::device::opencl_c_version>();

template <>
vector_class<string_class> get_device_info_host<info::device::extensions>();

template <> size_t get_device_info_host<info::device::printf_buffer_size>();

template <>
bool get_device_info_host<info::device::preferred_interop_user_sync>();

template <> device get_device_info_host<info::device::parent_device>();

template <>
cl_uint get_device_info_host<info::device::partition_max_sub_devices>();

template <>
vector_class<info::partition_property>
get_device_info_host<info::device::partition_properties>();

template <>
vector_class<info::partition_affinity_domain>
get_device_info_host<info::device::partition_affinity_domains>();

template <>
info::partition_property
get_device_info_host<info::device::partition_type_property>();

template <>
info::partition_affinity_domain
get_device_info_host<info::device::partition_type_affinity_domain>();

template <> cl_uint get_device_info_host<info::device::reference_count>();

template <> cl_uint get_device_info_host<info::device::max_num_sub_groups>();

template <>
vector_class<size_t> get_device_info_host<info::device::sub_group_sizes>();

template <>
bool get_device_info_host<
    info::device::sub_group_independent_forward_progress>();

} // namespace detail
} // namespace sycl
} // namespace cl
