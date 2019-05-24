//==---------- device_event.hpp --- SYCL device event ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_types.hpp>

namespace cl {
namespace sycl {

class device_event {
private:
  __ocl_event_t *m_Event;

public:
  device_event(const device_event &rhs) = default;
  device_event(device_event &&rhs) = default;
  device_event &operator=(const device_event &rhs) = default;
  device_event &operator=(device_event &&rhs) = default;

  device_event(__ocl_event_t *Event) : m_Event(Event) {}

  void wait() {
    __spirv_GroupWaitEvents(Scope::Workgroup, 1,
                                   m_Event);
  }
};

} // namespace sycl
} // namespace cl
