//==------------------ queue_impl.cpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/device.hpp>

namespace cl {
namespace sycl {
namespace detail {
template <> cl_uint queue_impl::get_info<info::queue::reference_count>() const {
  cl_uint result = 0;
  if (!is_host())
    CHECK_OCL_CODE(clGetCommandQueueInfo(m_CommandQueue,
                                         CL_QUEUE_REFERENCE_COUNT,
                                         sizeof(result), &result, nullptr));
  return result;
}

template <> context queue_impl::get_info<info::queue::context>() const {
  return get_context();
}

template <> device queue_impl::get_info<info::queue::device>() const {
  return get_device();
}
} // namespace detail
} // namespace sycl
} // namespace cl
