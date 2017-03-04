//===-- NativeBreakpoint.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/common/NativeBreakpoint.h"

#include "lldb/Utility/Error.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-defines.h"

using namespace lldb_private;

NativeBreakpoint::NativeBreakpoint(lldb::addr_t addr)
    : m_addr(addr), m_ref_count(1), m_enabled(true) {
  assert(addr != LLDB_INVALID_ADDRESS && "breakpoint set for invalid address");
}

NativeBreakpoint::~NativeBreakpoint() {}

void NativeBreakpoint::AddRef() {
  ++m_ref_count;

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
  if (log)
    log->Printf("NativeBreakpoint::%s addr = 0x%" PRIx64
                " bumped up, new ref count %" PRIu32,
                __FUNCTION__, m_addr, m_ref_count);
}

int32_t NativeBreakpoint::DecRef() {
  --m_ref_count;

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
  if (log)
    log->Printf("NativeBreakpoint::%s addr = 0x%" PRIx64
                " ref count decremented, new ref count %" PRIu32,
                __FUNCTION__, m_addr, m_ref_count);

  return m_ref_count;
}

Error NativeBreakpoint::Enable() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));

  if (m_enabled) {
    // We're already enabled. Just log and exit.
    if (log)
      log->Printf("NativeBreakpoint::%s addr = 0x%" PRIx64
                  " already enabled, ignoring.",
                  __FUNCTION__, m_addr);
    return Error();
  }

  // Log and enable.
  if (log)
    log->Printf("NativeBreakpoint::%s addr = 0x%" PRIx64 " enabling...",
                __FUNCTION__, m_addr);

  Error error = DoEnable();
  if (error.Success()) {
    m_enabled = true;
    if (log)
      log->Printf("NativeBreakpoint::%s addr = 0x%" PRIx64 " enable SUCCESS.",
                  __FUNCTION__, m_addr);
  } else {
    if (log)
      log->Printf("NativeBreakpoint::%s addr = 0x%" PRIx64 " enable FAIL: %s",
                  __FUNCTION__, m_addr, error.AsCString());
  }

  return error;
}

Error NativeBreakpoint::Disable() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));

  if (!m_enabled) {
    // We're already disabled. Just log and exit.
    if (log)
      log->Printf("NativeBreakpoint::%s addr = 0x%" PRIx64
                  " already disabled, ignoring.",
                  __FUNCTION__, m_addr);
    return Error();
  }

  // Log and disable.
  if (log)
    log->Printf("NativeBreakpoint::%s addr = 0x%" PRIx64 " disabling...",
                __FUNCTION__, m_addr);

  Error error = DoDisable();
  if (error.Success()) {
    m_enabled = false;
    if (log)
      log->Printf("NativeBreakpoint::%s addr = 0x%" PRIx64 " disable SUCCESS.",
                  __FUNCTION__, m_addr);
  } else {
    if (log)
      log->Printf("NativeBreakpoint::%s addr = 0x%" PRIx64 " disable FAIL: %s",
                  __FUNCTION__, m_addr, error.AsCString());
  }

  return error;
}
