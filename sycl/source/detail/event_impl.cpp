//==---------------- event_impl.cpp - SYCL event ---------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/scheduler/scheduler.h>

namespace cl {
namespace sycl {
namespace detail {

// Threat all devices that don't support interoperability as host devices to
// avoid attempts to call method get on such events.
bool event_impl::is_host() const { return m_HostEvent || !m_OpenCLInterop; }

cl_event event_impl::get() const {
  if (m_OpenCLInterop) {
    CHECK_OCL_CODE(clRetainEvent(m_Event));
    return m_Event;
  }
  throw invalid_object_error(
      "This instance of event doesn't support OpenCL interoperability.");
}

event_impl::~event_impl() {
  if (!m_HostEvent) {
    CHECK_OCL_CODE_NO_EXC(clReleaseEvent(m_Event));
  }
}

void event_impl::waitInternal() const {
  if (!m_HostEvent) {
    CHECK_OCL_CODE(clWaitForEvents(1, &m_Event));
  }
  // Waiting of host events is NOP so far as all operations on host device
  // are blocking.
}

cl_event &event_impl::getHandleRef() { return m_Event; }

void event_impl::setIsHostEvent(bool Value) {
  m_HostEvent = Value;
  m_OpenCLInterop = !Value;
}

event_impl::event_impl(cl_event CLEvent, const context &SyclContext)
    : m_Event(CLEvent), m_OpenCLInterop(true), m_HostEvent(false) {
  CHECK_OCL_CODE(clRetainEvent(m_Event));
  // TODO: Add check that CLEvent is bound to cl_context encapsulated in
  // SyclContext.
  if (SyclContext.is_host()) {
    throw cl::sycl::invalid_parameter_error(
        "The syclContext must match the OpenCL context associated with the "
        "clEvent.");
  }
}

void event_impl::wait(
    std::shared_ptr<cl::sycl::detail::event_impl> Self) const {

  if (m_Event || m_HostEvent)
    // presence of m_Event means the command has been enqueued, so no need to
    // go via the slow path event waiting in the scheduler
    waitInternal();
  else
    simple_scheduler::Scheduler::getInstance().waitForEvent(Self);
}

template <>
cl_ulong
event_impl::get_profiling_info<info::event_profiling::command_submit>() const {
  if (!m_HostEvent) {
    return get_event_profiling_info_cl<
        info::event_profiling::command_submit>::_(this->get());
  }
  assert(!"Not implemented for host device.");
  return (cl_ulong)0;
}

template <>
cl_ulong
event_impl::get_profiling_info<info::event_profiling::command_start>() const {
  if (!m_HostEvent) {
    return get_event_profiling_info_cl<info::event_profiling::command_start>::_(
        this->get());
  }
  assert(!"Not implemented for host device.");
  return (cl_ulong)0;
}

template <>
cl_ulong
event_impl::get_profiling_info<info::event_profiling::command_end>() const {
  if (!m_HostEvent) {
    return get_event_profiling_info_cl<info::event_profiling::command_end>::_(
        this->get());
  }
  assert(!"Not implemented for host device.");
  return (cl_ulong)0;
}

template <> cl_uint event_impl::get_info<info::event::reference_count>() const {
  if (!m_HostEvent) {
    return get_event_info_cl<info::event::reference_count>::_(this->get());
  }
  assert(!"Not implemented for host device.");
  return (cl_ulong)0;
}

template <>
info::event_command_status
event_impl::get_info<info::event::command_execution_status>() const {
  if (!m_HostEvent) {
    return get_event_info_cl<info::event::command_execution_status>::_(
        this->get());
  }
  assert(!"Not implemented for host device.");
  return info::event_command_status::complete;
}

} // namespace detail
} // namespace sycl
} // namespace cl
