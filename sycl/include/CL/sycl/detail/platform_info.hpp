//==------ platform_info.hpp - SYCL platform info methods ------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/common_info.hpp>
#include <CL/sycl/info/info_desc.hpp>

namespace cl {
namespace sycl {
namespace detail {

// OpenCL platform information methods
template <typename T, info::platform param> struct get_platform_info_cl {};

template <info::platform param>
struct get_platform_info_cl<string_class, param> {
  static string_class _(cl_platform_id plt) {
    size_t resultSize;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(
        clGetPlatformInfo(plt, cl_platform_info(param), 0, NULL, &resultSize));
    if (resultSize == 0) {
      return "";
    }
    unique_ptr_class<char[]> result(new char[resultSize]);
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetPlatformInfo(plt, cl_platform_info(param), resultSize,
                                     result.get(), NULL));
    return result.get();
  }
};

template <>
struct get_platform_info_cl<vector_class<string_class>,
                            info::platform::extensions> {
  static vector_class<string_class> _(cl_platform_id plt) {
    string_class result =
        get_platform_info_cl<string_class, info::platform::extensions>::_(plt);
    return split_string(result, ' ');
  }
};

// Host platform information methods
template <info::platform param>
typename info::param_traits<info::platform, param>::return_type
get_platform_info_host() = delete;

template <> string_class get_platform_info_host<info::platform::profile>();

template <> string_class get_platform_info_host<info::platform::version>();

template <> string_class get_platform_info_host<info::platform::name>();

template <> string_class get_platform_info_host<info::platform::vendor>();

template <>
vector_class<string_class> get_platform_info_host<info::platform::extensions>();

} // namespace detail
} // namespace sycl
} // namespace cl
