//==---------------- event.cpp --- SYCL event ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/detail/scheduler/scheduler.h>

#include <CL/sycl/stl.hpp>

#include <memory>
#include <unordered_set>

namespace cl {
namespace sycl {

event::event() : impl(std::make_shared<detail::event_impl>()) {}

event::event(cl_event clEvent, const context &syclContext)
    : impl(std::make_shared<detail::event_impl>(clEvent, syclContext)) {}

bool event::operator==(const event &rhs) const { return rhs.impl == impl; }

bool event::operator!=(const event &rhs) const { return !(*this == rhs); }

cl_event event::get() { return impl->get(); }

bool event::is_host() const { return impl->is_host(); }

void event::wait() { impl->wait(impl); }

void event::wait(const vector_class<event> &EventList) {
  for (auto E : EventList) {
    E.wait();
  }
}

void event::wait_and_throw() { impl->wait_and_throw(impl); }

void event::wait_and_throw(const vector_class<event> &EventList) {
  for (auto E : EventList) {
    E.wait_and_throw();
  }
}

vector_class<event> event::get_wait_list() {
  return cl::sycl::simple_scheduler::Scheduler::getInstance().
      getDepEvents(impl);
}

event::event(std::shared_ptr<detail::event_impl> event_impl)
    : impl(event_impl) {}

template <> cl_uint event::get_info<info::event::reference_count>() const {
  return impl->get_info<info::event::reference_count>();
}

template <>
info::event_command_status
event::get_info<info::event::command_execution_status>() const {
  return impl->get_info<info::event::command_execution_status>();
}

template <>
cl_ulong
event::get_profiling_info<info::event_profiling::command_submit>() const {
  impl->wait(impl);
  return impl->get_profiling_info<info::event_profiling::command_submit>();
}
template <>
cl_ulong
event::get_profiling_info<info::event_profiling::command_start>() const {
  impl->wait(impl);
  return impl->get_profiling_info<info::event_profiling::command_start>();
}

template <>
cl_ulong event::get_profiling_info<info::event_profiling::command_end>() const {
  impl->wait(impl);
  return impl->get_profiling_info<info::event_profiling::command_end>();
}

} // namespace sycl
} // namespace cl
