//==------- info_desc.hpp - SYCL information descriptors -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/id.hpp>

namespace cl {
namespace sycl {

class program;
class device;
class platform;

namespace info {

// Information descriptors
// A.1 Platform information descriptors
enum class platform : cl_platform_info {
  profile = CL_PLATFORM_PROFILE,
  version = CL_PLATFORM_VERSION,
  name = CL_PLATFORM_NAME,
  vendor = CL_PLATFORM_VENDOR,
  extensions = CL_PLATFORM_EXTENSIONS
};

// A.2 Context information desctiptors
enum class context : cl_context_info {
  reference_count = CL_CONTEXT_REFERENCE_COUNT,
  platform = CL_CONTEXT_PLATFORM,
  devices = CL_CONTEXT_DEVICES,
};

// A.3 Device information descriptors
enum class device : cl_device_info {
  device_type = CL_DEVICE_TYPE,
  vendor_id = CL_DEVICE_VENDOR_ID,
  max_compute_units = CL_DEVICE_MAX_COMPUTE_UNITS,
  max_work_item_dimensions = CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
  max_work_item_sizes = CL_DEVICE_MAX_WORK_ITEM_SIZES,
  max_work_group_size = CL_DEVICE_MAX_WORK_GROUP_SIZE,

  preferred_vector_width_char = CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
  preferred_vector_width_short = CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
  preferred_vector_width_int = CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
  preferred_vector_width_long = CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
  preferred_vector_width_float = CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
  preferred_vector_width_double = CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
  preferred_vector_width_half = CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,

  native_vector_width_char = CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
  native_vector_width_short = CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
  native_vector_width_int = CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
  native_vector_width_long = CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
  native_vector_width_float = CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
  native_vector_width_double = CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
  native_vector_width_half = CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,

  max_clock_frequency = CL_DEVICE_MAX_CLOCK_FREQUENCY,
  address_bits = CL_DEVICE_ADDRESS_BITS,
  max_mem_alloc_size = CL_DEVICE_MAX_MEM_ALLOC_SIZE,
  image_support = CL_DEVICE_IMAGE_SUPPORT,
  max_read_image_args = CL_DEVICE_MAX_READ_IMAGE_ARGS,
  max_write_image_args = CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
  image2d_max_width = CL_DEVICE_IMAGE2D_MAX_WIDTH,
  image2d_max_height = CL_DEVICE_IMAGE2D_MAX_HEIGHT,
  image3d_max_width = CL_DEVICE_IMAGE3D_MAX_WIDTH,
  image3d_max_height = CL_DEVICE_IMAGE3D_MAX_HEIGHT,
  image3d_max_depth = CL_DEVICE_IMAGE3D_MAX_DEPTH,
  image_max_buffer_size = CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
  image_max_array_size = CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,
  max_samplers = CL_DEVICE_MAX_SAMPLERS,
  max_parameter_size = CL_DEVICE_MAX_PARAMETER_SIZE,
  mem_base_addr_align = CL_DEVICE_MEM_BASE_ADDR_ALIGN,
  half_fp_config = CL_DEVICE_HALF_FP_CONFIG,
  single_fp_config = CL_DEVICE_SINGLE_FP_CONFIG,
  double_fp_config = CL_DEVICE_DOUBLE_FP_CONFIG,
  global_mem_cache_type = CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
  global_mem_cache_line_size = CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
  global_mem_cache_size = CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
  global_mem_size = CL_DEVICE_GLOBAL_MEM_SIZE,
  max_constant_buffer_size = CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
  max_constant_args = CL_DEVICE_MAX_CONSTANT_ARGS,
  local_mem_type = CL_DEVICE_LOCAL_MEM_TYPE,
  local_mem_size = CL_DEVICE_LOCAL_MEM_SIZE,
  error_correction_support = CL_DEVICE_ERROR_CORRECTION_SUPPORT,
  host_unified_memory = CL_DEVICE_HOST_UNIFIED_MEMORY,
  profiling_timer_resolution = CL_DEVICE_PROFILING_TIMER_RESOLUTION,
  is_endian_little = CL_DEVICE_ENDIAN_LITTLE,
  is_available = CL_DEVICE_AVAILABLE,
  is_compiler_available = CL_DEVICE_COMPILER_AVAILABLE,
  is_linker_available = CL_DEVICE_LINKER_AVAILABLE,
  execution_capabilities = CL_DEVICE_EXECUTION_CAPABILITIES,
  queue_profiling = CL_DEVICE_QUEUE_PROPERTIES,
  built_in_kernels = CL_DEVICE_BUILT_IN_KERNELS,
  platform = CL_DEVICE_PLATFORM,
  name = CL_DEVICE_NAME,
  vendor = CL_DEVICE_VENDOR,
  driver_version = CL_DRIVER_VERSION,
  profile = CL_DEVICE_PROFILE,
  version = CL_DEVICE_VERSION,
  opencl_c_version = CL_DEVICE_OPENCL_C_VERSION,
  extensions = CL_DEVICE_EXTENSIONS,
  printf_buffer_size = CL_DEVICE_PRINTF_BUFFER_SIZE,
  preferred_interop_user_sync = CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,
  parent_device = CL_DEVICE_PARENT_DEVICE,
  partition_max_sub_devices = CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
  partition_properties = CL_DEVICE_PARTITION_PROPERTIES,
  partition_affinity_domains = CL_DEVICE_PARTITION_AFFINITY_DOMAIN,
  partition_type_affinity_domain = CL_DEVICE_PARTITION_TYPE,
  reference_count = CL_DEVICE_REFERENCE_COUNT,
  max_num_sub_groups = CL_DEVICE_MAX_NUM_SUB_GROUPS,
  sub_group_independent_forward_progress =
      CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
  sub_group_sizes = CL_DEVICE_SUB_GROUP_SIZES_INTEL,
  partition_type_property
};

enum class device_type : cl_device_type {
  cpu = CL_DEVICE_TYPE_CPU,
  gpu = CL_DEVICE_TYPE_GPU,
  accelerator = CL_DEVICE_TYPE_ACCELERATOR,
  custom = CL_DEVICE_TYPE_CUSTOM,
  automatic,
  host,
  all = CL_DEVICE_TYPE_ALL
};

enum class partition_property : cl_device_partition_property {
  partition_equally = CL_DEVICE_PARTITION_EQUALLY,
  partition_by_counts = CL_DEVICE_PARTITION_BY_COUNTS,
  partition_by_affinity_domain = CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
  no_partition
};

enum class partition_affinity_domain : cl_device_affinity_domain {
  not_applicable = 0,
  numa = CL_DEVICE_AFFINITY_DOMAIN_NUMA,
  L4_cache = CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE,
  L3_cache = CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE,
  L2_cache = CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE,
  L1_cache = CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE,
  next_partitionable = CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE
};

enum class local_mem_type : int { none, local, global };

enum class fp_config : cl_device_fp_config {
  denorm = CL_FP_DENORM,
  inf_nan = CL_FP_INF_NAN,
  round_to_nearest = CL_FP_ROUND_TO_NEAREST,
  round_to_zero = CL_FP_ROUND_TO_ZERO,
  round_to_inf = CL_FP_ROUND_TO_INF,
  fma = CL_FP_FMA,
  correctly_rounded_divide_sqrt,
  soft_float
};

enum class global_mem_cache_type : int { none, read_only, write_only };

enum class execution_capability : unsigned int {
  exec_kernel,
  exec_native_kernel
};

// A.4 Queue information desctiptors
enum class queue : cl_command_queue_info {
  context = CL_QUEUE_CONTEXT,
  device = CL_QUEUE_DEVICE,
  reference_count = CL_QUEUE_REFERENCE_COUNT
};

// A.5 Kernel information desctiptors
enum class kernel : cl_kernel_info {
  function_name = CL_KERNEL_FUNCTION_NAME,
  num_args = CL_KERNEL_NUM_ARGS,
  context = CL_KERNEL_CONTEXT,
  program = CL_KERNEL_PROGRAM,
  reference_count = CL_KERNEL_REFERENCE_COUNT,
  attributes = CL_KERNEL_ATTRIBUTES
};

enum class kernel_work_group : cl_kernel_work_group_info {
  global_work_size = CL_KERNEL_GLOBAL_WORK_SIZE,
  work_group_size = CL_KERNEL_WORK_GROUP_SIZE,
  compile_work_group_size = CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
  preferred_work_group_size_multiple =
      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
  private_mem_size = CL_KERNEL_PRIVATE_MEM_SIZE
};

enum class kernel_sub_group : cl_kernel_sub_group_info {
  max_sub_group_size_for_ndrange = CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
  sub_group_count_for_ndrange = CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
  local_size_for_sub_group_count = CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT,
  max_num_sub_groups = CL_KERNEL_MAX_NUM_SUB_GROUPS,
  compile_num_sub_groups = CL_KERNEL_COMPILE_NUM_SUB_GROUPS
};

// A.6 Program information desctiptors
enum class program : cl_program_info {
  context = CL_PROGRAM_CONTEXT,
  devices = CL_PROGRAM_DEVICES,
  reference_count = CL_PROGRAM_REFERENCE_COUNT
};

// A.7 Event information desctiptors
enum class event : cl_event_info {
  reference_count = CL_EVENT_REFERENCE_COUNT,
  command_execution_status = CL_EVENT_COMMAND_EXECUTION_STATUS
};

enum class event_command_status : cl_int {
  submitted = CL_SUBMITTED,
  running = CL_RUNNING,
  complete = CL_COMPLETE
};

enum class event_profiling : cl_profiling_info {
  command_submit = CL_PROFILING_COMMAND_SUBMIT,
  command_start = CL_PROFILING_COMMAND_START,
  command_end = CL_PROFILING_COMMAND_END
};

// Provide an alias to the return type for each of the info parameters
template <typename T, T param> class param_traits {};

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template <> class param_traits<param_type, param_type::param> {              \
  public:                                                                      \
    using return_type = ret_type;                                              \
  };

#define PARAM_TRAITS_SPEC_WITH_INPUT(param_type, param, ret_type, in_type)     \
  template <> class param_traits<param_type, param_type::param> {              \
  public:                                                                      \
    using return_type = ret_type;                                              \
    using input_type = in_type;                                                \
  };

PARAM_TRAITS_SPEC(device, device_type, device_type)
PARAM_TRAITS_SPEC(device, vendor_id, cl_uint)
PARAM_TRAITS_SPEC(device, max_compute_units, cl_uint)
PARAM_TRAITS_SPEC(device, max_work_item_dimensions, cl_uint)
PARAM_TRAITS_SPEC(device, max_work_item_sizes, id<3>)
PARAM_TRAITS_SPEC(device, max_work_group_size, size_t)
PARAM_TRAITS_SPEC(device, preferred_vector_width_char, cl_uint)
PARAM_TRAITS_SPEC(device, preferred_vector_width_short, cl_uint)
PARAM_TRAITS_SPEC(device, preferred_vector_width_int, cl_uint)
PARAM_TRAITS_SPEC(device, preferred_vector_width_long, cl_uint)
PARAM_TRAITS_SPEC(device, preferred_vector_width_float, cl_uint)
PARAM_TRAITS_SPEC(device, preferred_vector_width_double, cl_uint)
PARAM_TRAITS_SPEC(device, preferred_vector_width_half, cl_uint)
PARAM_TRAITS_SPEC(device, native_vector_width_char, cl_uint)
PARAM_TRAITS_SPEC(device, native_vector_width_short, cl_uint)
PARAM_TRAITS_SPEC(device, native_vector_width_int, cl_uint)
PARAM_TRAITS_SPEC(device, native_vector_width_long, cl_uint)
PARAM_TRAITS_SPEC(device, native_vector_width_float, cl_uint)
PARAM_TRAITS_SPEC(device, native_vector_width_double, cl_uint)
PARAM_TRAITS_SPEC(device, native_vector_width_half, cl_uint)
PARAM_TRAITS_SPEC(device, max_clock_frequency, cl_uint)
PARAM_TRAITS_SPEC(device, address_bits, cl_uint)
PARAM_TRAITS_SPEC(device, max_mem_alloc_size, cl_ulong)
PARAM_TRAITS_SPEC(device, image_support, bool)
PARAM_TRAITS_SPEC(device, max_read_image_args, cl_uint)
PARAM_TRAITS_SPEC(device, max_write_image_args, cl_uint)
PARAM_TRAITS_SPEC(device, image2d_max_width, size_t)
PARAM_TRAITS_SPEC(device, image2d_max_height, size_t)
PARAM_TRAITS_SPEC(device, image3d_max_width, size_t)
PARAM_TRAITS_SPEC(device, image3d_max_height, size_t)
PARAM_TRAITS_SPEC(device, image3d_max_depth, size_t)
PARAM_TRAITS_SPEC(device, image_max_buffer_size, size_t)
PARAM_TRAITS_SPEC(device, image_max_array_size, size_t)
PARAM_TRAITS_SPEC(device, max_samplers, cl_uint)
PARAM_TRAITS_SPEC(device, max_parameter_size, size_t)
PARAM_TRAITS_SPEC(device, mem_base_addr_align, cl_uint)
PARAM_TRAITS_SPEC(device, half_fp_config, vector_class<info::fp_config>)
PARAM_TRAITS_SPEC(device, single_fp_config, vector_class<info::fp_config>)
PARAM_TRAITS_SPEC(device, double_fp_config, vector_class<info::fp_config>)
PARAM_TRAITS_SPEC(device, global_mem_cache_type, info::global_mem_cache_type)
PARAM_TRAITS_SPEC(device, global_mem_cache_line_size, cl_uint)
PARAM_TRAITS_SPEC(device, global_mem_cache_size, cl_ulong)
PARAM_TRAITS_SPEC(device, global_mem_size, cl_ulong)
PARAM_TRAITS_SPEC(device, max_constant_buffer_size, cl_ulong)
PARAM_TRAITS_SPEC(device, max_constant_args, cl_uint)
PARAM_TRAITS_SPEC(device, local_mem_type, info::local_mem_type)
PARAM_TRAITS_SPEC(device, local_mem_size, cl_ulong)
PARAM_TRAITS_SPEC(device, error_correction_support, bool)
PARAM_TRAITS_SPEC(device, host_unified_memory, bool)
PARAM_TRAITS_SPEC(device, profiling_timer_resolution, size_t)
PARAM_TRAITS_SPEC(device, is_endian_little, bool)
PARAM_TRAITS_SPEC(device, is_available, bool)
PARAM_TRAITS_SPEC(device, is_compiler_available, bool)
PARAM_TRAITS_SPEC(device, is_linker_available, bool)
PARAM_TRAITS_SPEC(device, execution_capabilities,
                  vector_class<info::execution_capability>)
PARAM_TRAITS_SPEC(device, queue_profiling, bool)
PARAM_TRAITS_SPEC(device, built_in_kernels, vector_class<string_class>)
PARAM_TRAITS_SPEC(device, platform, cl::sycl::platform)
PARAM_TRAITS_SPEC(device, name, string_class)
PARAM_TRAITS_SPEC(device, vendor, string_class)
PARAM_TRAITS_SPEC(device, driver_version, string_class)
PARAM_TRAITS_SPEC(device, profile, string_class)
PARAM_TRAITS_SPEC(device, version, string_class)
PARAM_TRAITS_SPEC(device, opencl_c_version, string_class)
PARAM_TRAITS_SPEC(device, extensions, vector_class<string_class>)
PARAM_TRAITS_SPEC(device, printf_buffer_size, size_t)
PARAM_TRAITS_SPEC(device, preferred_interop_user_sync, bool)
PARAM_TRAITS_SPEC(device, parent_device, cl::sycl::device)
PARAM_TRAITS_SPEC(device, partition_max_sub_devices, cl_uint)
PARAM_TRAITS_SPEC(device, partition_properties,
                  vector_class<info::partition_property>)
PARAM_TRAITS_SPEC(device, partition_affinity_domains,
                  vector_class<info::partition_affinity_domain>)
PARAM_TRAITS_SPEC(device, partition_type_property, info::partition_property)
PARAM_TRAITS_SPEC(device, partition_type_affinity_domain,
                  info::partition_affinity_domain)
PARAM_TRAITS_SPEC(device, reference_count, cl_uint)
PARAM_TRAITS_SPEC(device, max_num_sub_groups, cl_uint)
PARAM_TRAITS_SPEC(device, sub_group_independent_forward_progress, bool)
PARAM_TRAITS_SPEC(device, sub_group_sizes, vector_class<size_t>)

PARAM_TRAITS_SPEC(context, reference_count, cl_uint)
PARAM_TRAITS_SPEC(context, platform, cl::sycl::platform)
PARAM_TRAITS_SPEC(context, devices, vector_class<cl::sycl::device>)

PARAM_TRAITS_SPEC(event, command_execution_status, event_command_status)
PARAM_TRAITS_SPEC(event, reference_count, cl_uint)

PARAM_TRAITS_SPEC(event_profiling, command_submit, cl_ulong)
PARAM_TRAITS_SPEC(event_profiling, command_start, cl_ulong)
PARAM_TRAITS_SPEC(event_profiling, command_end, cl_ulong)

PARAM_TRAITS_SPEC(kernel, function_name, string_class)
PARAM_TRAITS_SPEC(kernel, num_args, cl_uint)
PARAM_TRAITS_SPEC(kernel, reference_count, cl_uint)
PARAM_TRAITS_SPEC(kernel, attributes, string_class)
// Shilei: The following two traits are not covered in the current version of
// CTS (SYCL-1.2.1/master)
PARAM_TRAITS_SPEC(kernel, context, cl::sycl::context)
PARAM_TRAITS_SPEC(kernel, program, cl::sycl::program)

PARAM_TRAITS_SPEC(kernel_work_group, compile_work_group_size,
                  cl::sycl::range<3>)
PARAM_TRAITS_SPEC(kernel_work_group, global_work_size, cl::sycl::range<3>)
PARAM_TRAITS_SPEC(kernel_work_group, preferred_work_group_size_multiple, size_t)
PARAM_TRAITS_SPEC(kernel_work_group, private_mem_size, cl_ulong)
PARAM_TRAITS_SPEC(kernel_work_group, work_group_size, size_t)

PARAM_TRAITS_SPEC_WITH_INPUT(kernel_sub_group, max_sub_group_size_for_ndrange,
                             size_t, cl::sycl::range<3>)
PARAM_TRAITS_SPEC_WITH_INPUT(kernel_sub_group, sub_group_count_for_ndrange,
                             size_t, cl::sycl::range<3>)
PARAM_TRAITS_SPEC_WITH_INPUT(kernel_sub_group, local_size_for_sub_group_count,
                             cl::sycl::range<3>, size_t)
PARAM_TRAITS_SPEC(kernel_sub_group, max_num_sub_groups, size_t)
PARAM_TRAITS_SPEC(kernel_sub_group, compile_num_sub_groups, size_t)

PARAM_TRAITS_SPEC(platform, profile, string_class)
PARAM_TRAITS_SPEC(platform, version, string_class)
PARAM_TRAITS_SPEC(platform, name, string_class)
PARAM_TRAITS_SPEC(platform, vendor, string_class)
PARAM_TRAITS_SPEC(platform, extensions, vector_class<string_class>)

PARAM_TRAITS_SPEC(program, context, cl::sycl::context)
PARAM_TRAITS_SPEC(program, devices, vector_class<cl::sycl::device>)
PARAM_TRAITS_SPEC(program, reference_count, cl_uint)

PARAM_TRAITS_SPEC(queue, reference_count, cl_uint)
PARAM_TRAITS_SPEC(queue, context, cl::sycl::context)
PARAM_TRAITS_SPEC(queue, device, cl::sycl::device)

#undef PARAM_TRAITS_SPEC

} // namespace info
} // namespace sycl
} // namespace cl
