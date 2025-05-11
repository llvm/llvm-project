//===-- MemoryMonitorMacOSX.mm --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/MemoryMonitor.h"
#include <cassert>
#include <dispatch/dispatch.h>

using namespace lldb_private;

class MemoryMonitorMacOSX : public MemoryMonitor {
  using MemoryMonitor::MemoryMonitor;
  void Start() override {
    m_memory_pressure_source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));

    if (!m_memory_pressure_source)
      return;

    dispatch_source_set_event_handler(m_memory_pressure_source, ^{
      dispatch_source_memorypressure_flags_t pressureLevel =
          dispatch_source_get_data(m_memory_pressure_source);
      if (pressureLevel &
          (DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL)) {
        m_callback();
      }
    });
    dispatch_activate(m_memory_pressure_source);
  }

  void Stop() override {
    if (m_memory_pressure_source) {
      dispatch_source_cancel(m_memory_pressure_source);
      dispatch_release(m_memory_pressure_source);
    }
  }

private:
  dispatch_source_t m_memory_pressure_source;
};

std::unique_ptr<MemoryMonitor> MemoryMonitor::Create(Callback callback) {
  return std::make_unique<MemoryMonitorMacOSX>(callback);
}
