//==-------- kernel_info.hpp - SYCL kernel info methods --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/common_info.hpp>
#include <CL/sycl/info/info_desc.hpp>

namespace cl {
namespace sycl {
namespace detail {

// OpenCL kernel information methods
template <typename T, info::kernel Param> struct get_kernel_info_cl {};

template <info::kernel Param> struct get_kernel_info_cl<string_class, Param> {
  static string_class _(cl_kernel ClKernel) {
    size_t ResultSize;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetKernelInfo(ClKernel, cl_kernel_info(Param), 0, nullptr,
                                   &ResultSize));
    if (ResultSize == 0) {
      return "";
    }
    vector_class<char> Result(ResultSize);
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetKernelInfo(ClKernel, cl_kernel_info(Param), ResultSize,
                                   Result.data(), nullptr));
    return string_class(Result.data());
  }
};

template <info::kernel Param> struct get_kernel_info_cl<cl_uint, Param> {
  static cl_uint _(cl_kernel ClKernel) {
    cl_uint Result;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetKernelInfo(ClKernel, cl_kernel_info(Param),
                                   sizeof(cl_uint), &Result, nullptr));
    return Result;
  }
};

// OpenCL kernel work-group methods

template <typename T, info::kernel_work_group Param>
struct get_kernel_work_group_info_cl {
  static T _(cl_kernel ClKernel, cl_device_id ClDevice) {
    T Result;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetKernelWorkGroupInfo(ClKernel, ClDevice,
                                            cl_kernel_work_group_info(Param),
                                            sizeof(T), &Result, nullptr));
    return Result;
  }
};

template <info::kernel_work_group Param>
struct get_kernel_work_group_info_cl<cl::sycl::range<3>, Param> {
  static cl::sycl::range<3> _(cl_kernel ClKernel, cl_device_id ClDevice) {
    size_t Result[3];
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetKernelWorkGroupInfo(
        ClKernel, ClDevice, cl_kernel_work_group_info(Param),
        sizeof(size_t) * 3, Result, nullptr));
    return cl::sycl::range<3>(Result[0], Result[1], Result[2]);
  }
};

template <info::kernel_work_group Param>
typename info::param_traits<info::kernel_work_group, Param>::return_type
get_kernel_work_group_info_host(const cl::sycl::device &Device);

template <>
cl::sycl::range<3>
get_kernel_work_group_info_host<info::kernel_work_group::global_work_size>(
    const cl::sycl::device &Device);

template <>
size_t
get_kernel_work_group_info_host<info::kernel_work_group::work_group_size>(
    const cl::sycl::device &Device);

template <>
cl::sycl::range<3> get_kernel_work_group_info_host<
    info::kernel_work_group::compile_work_group_size>(
    const cl::sycl::device &Device);

template <>
size_t get_kernel_work_group_info_host<
    info::kernel_work_group::preferred_work_group_size_multiple>(
    const cl::sycl::device &Device);

template <>
cl_ulong
get_kernel_work_group_info_host<info::kernel_work_group::private_mem_size>(
    const cl::sycl::device &Device);

// OpenCL kernel sub-group methods

template <typename TOut, info::kernel_sub_group Param>
struct get_kernel_sub_group_info_cl {
  static TOut _(cl_kernel ClKernel, cl_device_id ClDevice) {
    TOut Result;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetKernelSubGroupInfo(
        ClKernel, ClDevice, cl_kernel_sub_group_info(Param), 0, nullptr,
        sizeof(TOut), &Result, nullptr));
    return Result;
  }
};

template <typename TOut, info::kernel_sub_group Param, typename TIn>
struct get_kernel_sub_group_info_with_input_cl {
  static TOut _(cl_kernel ClKernel, cl_device_id ClDevice, TIn In) {
    TOut Result;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetKernelSubGroupInfo(
        ClKernel, ClDevice, cl_kernel_sub_group_info(Param), sizeof(TIn), &In,
        sizeof(TOut), &Result, nullptr));
    return Result;
  }
};

template <info::kernel_sub_group Param>
struct get_kernel_sub_group_info_with_input_cl<cl::sycl::range<3>, Param,
                                               size_t> {
  static cl::sycl::range<3> _(cl_kernel ClKernel, cl_device_id ClDevice,
                              size_t In) {
    size_t Result[3];
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetKernelSubGroupInfo(
        ClKernel, ClDevice, cl_kernel_sub_group_info(Param), sizeof(size_t),
        &In, sizeof(size_t) * 3, Result, nullptr));
    return cl::sycl::range<3>(Result[0], Result[1], Result[2]);
  }
};

template <info::kernel_sub_group Param>
struct get_kernel_sub_group_info_with_input_cl<size_t, Param,
                                               cl::sycl::range<3>> {
  static size_t _(cl_kernel ClKernel, cl_device_id ClDevice,
                              cl::sycl::range<3> In) {
    size_t Input[3] = {In[0], In[1], In[2]};
    size_t Result;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetKernelSubGroupInfo(
        ClKernel, ClDevice, cl_kernel_sub_group_info(Param), sizeof(size_t) * 3,
        Input, sizeof(size_t), &Result, nullptr));
    return Result;
  }
};
} // namespace detail
} // namespace sycl
} // namespace cl
