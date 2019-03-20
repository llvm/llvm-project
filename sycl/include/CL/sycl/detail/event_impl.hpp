//==---------------- event_impl.hpp - SYCL event ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/event_info.hpp>
#include <CL/sycl/stl.hpp>

#include <cassert>

namespace cl {
namespace sycl {
class context;
namespace detail {
class context_impl;
using ContextImplPtr = std::shared_ptr<cl::sycl::detail::context_impl>;

class event_impl {
public:
  event_impl() = default;
  event_impl(cl_event CLEvent, const context &SyclContext);

  // Threat all devices that don't support interoperability as host devices to
  // avoid attempts to call method get on such events.
  bool is_host() const;

  cl_event get() const;

  // Self is needed in order to pass shared_ptr to Scheduler.
  void wait(std::shared_ptr<cl::sycl::detail::event_impl> Self) const;

  void wait_and_throw(std::shared_ptr<cl::sycl::detail::event_impl> Self);

  template <info::event_profiling param>
  typename info::param_traits<info::event_profiling, param>::return_type
  get_profiling_info() const;

  template <info::event param>
  typename info::param_traits<info::event, param>::return_type get_info() const;

  ~event_impl();

  void waitInternal() const;

  // Warning. Returned reference will be invalid if event_impl was destroyed.
  cl_event &getHandleRef();

  const ContextImplPtr &getContextImpl();

  // Warning. Provided cl_context inside ContextImplPtr must be associated
  // with the cl_event object stored in this class
  void setContextImpl(const ContextImplPtr &Context);

private:
  cl_event m_Event = nullptr;
  ContextImplPtr m_Context;
  bool m_OpenCLInterop = false;
  bool m_HostEvent = true;
};

} // namespace detail
} // namespace sycl
} // namespace cl
