//===-- RegisterContextWindows.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Utility/Error.h"
#include "lldb/lldb-private-types.h"

#include "ProcessWindowsLog.h"
#include "RegisterContextWindows.h"
#include "TargetThreadWindows.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

const DWORD kWinContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;

//------------------------------------------------------------------
// Constructors and Destructors
//------------------------------------------------------------------
RegisterContextWindows::RegisterContextWindows(Thread &thread,
                                               uint32_t concrete_frame_idx)
    : RegisterContext(thread, concrete_frame_idx), m_context(),
      m_context_stale(true) {}

RegisterContextWindows::~RegisterContextWindows() {}

void RegisterContextWindows::InvalidateAllRegisters() {
  m_context_stale = true;
}

bool RegisterContextWindows::ReadAllRegisterValues(
    lldb::DataBufferSP &data_sp) {
  if (!CacheAllRegisterValues())
    return false;
  if (data_sp->GetByteSize() < sizeof(m_context)) {
    data_sp.reset(new DataBufferHeap(sizeof(CONTEXT), 0));
  }
  memcpy(data_sp->GetBytes(), &m_context, sizeof(m_context));
  return true;
}

bool RegisterContextWindows::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  assert(data_sp->GetByteSize() >= sizeof(m_context));
  memcpy(&m_context, data_sp->GetBytes(), sizeof(m_context));

  TargetThreadWindows &wthread = static_cast<TargetThreadWindows &>(m_thread);
  if (!::SetThreadContext(
          wthread.GetHostThread().GetNativeThread().GetSystemHandle(),
          &m_context))
    return false;

  return true;
}

uint32_t RegisterContextWindows::ConvertRegisterKindToRegisterNumber(
    lldb::RegisterKind kind, uint32_t num) {
  const uint32_t num_regs = GetRegisterCount();

  assert(kind < kNumRegisterKinds);
  for (uint32_t reg_idx = 0; reg_idx < num_regs; ++reg_idx) {
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex(reg_idx);

    if (reg_info->kinds[kind] == num)
      return reg_idx;
  }

  return LLDB_INVALID_REGNUM;
}

//------------------------------------------------------------------
// Subclasses can these functions if desired
//------------------------------------------------------------------
uint32_t RegisterContextWindows::NumSupportedHardwareBreakpoints() {
  // Support for hardware breakpoints not yet implemented.
  return 0;
}

uint32_t RegisterContextWindows::SetHardwareBreakpoint(lldb::addr_t addr,
                                                       size_t size) {
  return 0;
}

bool RegisterContextWindows::ClearHardwareBreakpoint(uint32_t hw_idx) {
  return false;
}

uint32_t RegisterContextWindows::NumSupportedHardwareWatchpoints() {
  // Support for hardware watchpoints not yet implemented.
  return 0;
}

uint32_t RegisterContextWindows::SetHardwareWatchpoint(lldb::addr_t addr,
                                                       size_t size, bool read,
                                                       bool write) {
  return 0;
}

bool RegisterContextWindows::ClearHardwareWatchpoint(uint32_t hw_index) {
  return false;
}

bool RegisterContextWindows::HardwareSingleStep(bool enable) { return false; }

bool RegisterContextWindows::CacheAllRegisterValues() {
  if (!m_context_stale)
    return true;

  TargetThreadWindows &wthread = static_cast<TargetThreadWindows &>(m_thread);
  memset(&m_context, 0, sizeof(m_context));
  m_context.ContextFlags = kWinContextFlags;
  if (!::GetThreadContext(
          wthread.GetHostThread().GetNativeThread().GetSystemHandle(),
          &m_context)) {
    WINERR_IFALL(
        WINDOWS_LOG_REGISTERS,
        "GetThreadContext failed with error %lu while caching register values.",
        ::GetLastError());
    return false;
  }
  WINLOG_IFALL(WINDOWS_LOG_REGISTERS,
               "GetThreadContext successfully updated the register values.");
  m_context_stale = false;
  return true;
}
