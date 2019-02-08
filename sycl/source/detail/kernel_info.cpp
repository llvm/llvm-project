//==-------- kernel_info.cpp - SYCL kernel info methods --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/kernel_info.hpp>
#include <CL/sycl/device.hpp>

namespace cl {
namespace sycl {
namespace detail {
template <>
cl::sycl::range<3>
get_kernel_work_group_info_host<info::kernel_work_group::global_work_size>(
    const cl::sycl::device &Dev) {
  throw invalid_object_error("This instance of kernel is a host instance");
}

template <>
size_t
get_kernel_work_group_info_host<info::kernel_work_group::work_group_size>(
    const cl::sycl::device &Dev) {
  return Dev.get_info<info::device::max_work_group_size>();
}

template <>
cl::sycl::range<3> get_kernel_work_group_info_host<
    info::kernel_work_group::compile_work_group_size>(
    const cl::sycl::device &Dev) {
  return {0, 0, 0};
}

template <>
size_t get_kernel_work_group_info_host<
    info::kernel_work_group::preferred_work_group_size_multiple>(
    const cl::sycl::device &Dev) {
  return get_kernel_work_group_info_host<
      info::kernel_work_group::work_group_size>(Dev);
}

template <>
cl_ulong
get_kernel_work_group_info_host<info::kernel_work_group::private_mem_size>(
    const cl::sycl::device &Dev) {
  return 0;
}

} // namespace detail
} // namespace sycl
} // namespace cl
