//==------------------ queue_impl.cpp - SYCL queue -------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  CHECK_OCL_CODE(clGetCommandQueueInfo(m_CommandQueue, CL_QUEUE_REFERENCE_COUNT,
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
