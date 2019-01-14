//==---------------- event_impl.hpp - SYCL event ---------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/event_info.hpp>
#include <CL/sycl/stl.hpp>

#include <cassert>

namespace cl {
namespace sycl {
namespace detail {

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

  template <info::event_profiling param>
  typename info::param_traits<info::event_profiling, param>::return_type
  get_profiling_info() const;

  template <info::event param>
  typename info::param_traits<info::event, param>::return_type get_info() const;

  ~event_impl();

  void waitInternal() const;

  cl_event &getHandleRef();

  void setIsHostEvent(bool Value);

private:
  cl_event m_Event = nullptr;
  bool m_OpenCLInterop = false;
  bool m_HostEvent = true;
};

} // namespace detail
} // namespace sycl
} // namespace cl
