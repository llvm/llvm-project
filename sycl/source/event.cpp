//==---------------- event.cpp --- SYCL event ------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/stl.hpp>

#include <memory>

namespace cl {
namespace sycl {

event::event() : impl(std::make_shared<detail::event_impl>()) {}

event::event(cl_event clEvent, const context &syclContext)
    : impl(std::make_shared<detail::event_impl>(clEvent, syclContext)) {}

bool event::operator==(const event &rhs) const { return rhs.impl == impl; }

bool event::operator!=(const event &rhs) const { return !(*this == rhs); }

cl_event event::get() { return impl->get(); }

bool event::is_host() const { return impl->is_host(); }

void event::wait() const { impl->wait(impl); }

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
  return impl->get_profiling_info<info::event_profiling::command_submit>();
}
template <>
cl_ulong
event::get_profiling_info<info::event_profiling::command_start>() const {
  return impl->get_profiling_info<info::event_profiling::command_start>();
}

template <>
cl_ulong event::get_profiling_info<info::event_profiling::command_end>() const {
  return impl->get_profiling_info<info::event_profiling::command_end>();
}

} // namespace sycl
} // namespace cl
