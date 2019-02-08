//==----- program_impl.cpp --- SYCL program implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/program_impl.hpp>

namespace cl {
namespace sycl {
namespace detail {
template <>
cl_uint program_impl::get_info<info::program::reference_count>() const {
  if (is_host()) {
    throw invalid_object_error("This instance of program is a host instance");
  }
  cl_uint result;
  clGetProgramInfo(ClProgram, CL_PROGRAM_REFERENCE_COUNT, sizeof(cl_uint),
                   &result, nullptr);
  return result;
}

template <> context program_impl::get_info<info::program::context>() const {
  return get_context();
}

template <>
vector_class<device> program_impl::get_info<info::program::devices>() const {
  return get_devices();
}

} // namespace detail
} // namespace sycl
} // namespace cl
