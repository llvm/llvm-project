//==---------- device_event.hpp --- SYCL device event ---------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_types.hpp>

namespace cl {
namespace sycl {

class device_event {
private:
  cl::__spirv::OpTypeEvent *m_Event;

public:
  device_event(const device_event &rhs) = default;
  device_event(device_event &&rhs) = default;
  device_event &operator=(const device_event &rhs) = default;
  device_event &operator=(device_event &&rhs) = default;

  device_event(cl::__spirv::OpTypeEvent *Event) : m_Event(Event) {}

  void wait() {
    cl::__spirv::OpGroupWaitEvents(cl::__spirv::Scope::Workgroup, 1,
                                   &m_Event);
  }
};

} // namespace sycl
} // namespace cl
