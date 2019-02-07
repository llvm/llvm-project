//==---------------- context_info.hpp - SYCL context -----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/info/info_desc.hpp>

namespace cl {
namespace sycl {
namespace detail {

template <info::context param> struct get_context_info_cl {
  using RetType =
      typename info::param_traits<info::context, param>::return_type;

  static RetType _(cl_context ctx) {
    RetType Result = 0;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetContextInfo(ctx, cl_context_info(param), sizeof(Result),
                                    &Result, nullptr));
    return Result;
  }
};

} // namespace detail
} // namespace sycl
} // namespace cl
