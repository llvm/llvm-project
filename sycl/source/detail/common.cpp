//==----------- common.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/common_info.hpp>

const char *stringifyErrorCode(cl_int error) {
  switch (error) {
    case CL_INVALID_ACCELERATOR_INTEL:
      return "CL_INVALID_ACCELERATOR_INTEL";
    case CL_INVALID_ACCELERATOR_TYPE_INTEL:
      return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
    case CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL:
      return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
    case CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL:
      return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
    case CL_PLATFORM_NOT_FOUND_KHR:
      return "CL_PLATFORM_NOT_FOUND_KHR";
    case CL_DEVICE_PARTITION_FAILED_EXT:
      return "CL_DEVICE_PARTITION_FAILED_EXT";
    case CL_INVALID_PARTITION_COUNT_EXT:
      return "CL_INVALID_PARTITION_COUNT_EXT";
    case CL_INVALID_PARTITION_NAME_EXT:
      return "CL_INVALID_PARTITION_NAME_EXT";
      /*    case CL_INVALID_DX9_DEVICE_INTEL:
            return "CL_INVALID_DX9_DEVICE_INTEL";
          case CL_INVALID_DX9_RESOURCE_INTEL:
            return "CL_INVALID_DX9_RESOURCE_INTEL";
          case CL_DX9_RESOURCE_ALREADY_ACQUIRED_INTEL:
            return "CL_DX9_RESOURCE_ALREADY_ACQUIRED_INTEL";
          case CL_DX9_RESOURCE_NOT_ACQUIRED_INTEL:
            return "CL_DX9_RESOURCE_NOT_ACQUIRED_INTEL";
          case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
            return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
      */
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_COMPILE_PROGRAM_FAILURE:
      return "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE:
      return "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE:
      return "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED:
      return "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
      return "CL_INVALID_PROPERTY";
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS:
      return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS:
      return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return "CL_INVALID_DEVICE_PARTITION_COUNT";
    case CL_INVALID_PIPE_SIZE:
      return "CL_INVALID_PIPE_SIZE";
    case CL_INVALID_DEVICE_QUEUE:
      return "CL_INVALID_DEVICE_QUEUE";
    case CL_INVALID_SPEC_ID:
      return "CL_INVALID_SPEC_ID";
    case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
      return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
    /*
        case CL_BUILD_NONE:
          return "CL_BUILD_NONE";
        case CL_BUILD_ERROR:
          return "CL_BUILD_ERROR";
        case CL_BUILD_IN_PROGRESS:
          return "CL_BUILD_IN_PROGRESS";
        case CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL:
          return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
        case CL_INVALID_VA_API_MEDIA_SURFACE_INTEL:
          return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
        case CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL:
          return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
        case CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL:
          return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
        case CL_INVALID_EGL_OBJECT_KHR:
          return "CL_INVALID_EGL_OBJECT_KHR";
        case CL_EGL_RESOURCE_NOT_ACQUIRED_KHR:
          return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
        case CL_INVALID_D3D11_DEVICE_KHR:
          return "CL_INVALID_D3D11_DEVICE_KHR";
        case CL_INVALID_D3D11_RESOURCE_KHR:
          return "CL_INVALID_D3D11_RESOURCE_KHR";
        case CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR:
          return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
        case CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR:
          return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
        case CL_INVALID_D3D10_DEVICE_KHR:
          return "CL_INVALID_D3D10_DEVICE_KHR";
        case CL_INVALID_D3D10_RESOURCE_KHR:
          return "CL_INVALID_D3D10_RESOURCE_KHR";
        case CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR:
          return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR:
          return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        case CL_INVALID_DX9_MEDIA_ADAPTER_KHR:
          return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
        case CL_INVALID_DX9_MEDIA_SURFACE_KHR:
          return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
        case CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR:
          return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
        case CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR:
          return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
          */
    default:
      return "Unknown OpenCL error code";
  }
}

namespace cl {
namespace sycl {
namespace detail {

vector_class<string_class> split_string(const string_class &str,
                                        char delimeter) {
  vector_class<string_class> result;
  size_t beg = 0;
  size_t length = 0;
  for (const auto &x : str) {
    if (x == delimeter) {
      result.push_back(str.substr(beg, length));
      beg += length + 1;
      length = 0;
      continue;
    }
    length++;
  }
  if (length != 0) {
    result.push_back(str.substr(beg, length));
  }
  return result;
}

} // namespace detail
} // namespace sycl
} // namespace cl
