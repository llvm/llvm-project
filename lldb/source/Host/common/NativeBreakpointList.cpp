//===-- NativeBreakpointList.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/common/NativeBreakpointList.h"

#include "lldb/Utility/Log.h"

#include "lldb/Host/common/NativeBreakpoint.h"
#include "lldb/Host/common/SoftwareBreakpoint.h"

using namespace lldb;
using namespace lldb_private;

NativeBreakpointList::NativeBreakpointList() : m_mutex() {}

Status NativeBreakpointList::AddRef(lldb::addr_t addr, size_t size_hint,
                                    bool hardware,
                                    CreateBreakpointFunc create_func) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
  if (log)
    log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64
                ", size_hint = %zu, hardware = %s",
                __FUNCTION__, addr, size_hint, hardware ? "true" : "false");

  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  // Check if the breakpoint is already set.
  auto iter = m_breakpoints.find(addr);
  if (iter != m_breakpoints.end()) {
    // Yes - bump up ref count.
    if (log)
      log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64
                  " -- already enabled, upping ref count",
                  __FUNCTION__, addr);

    iter->second->AddRef();
    return Status();
  }

  // Create a new breakpoint using the given create func.
  if (log)
    log->Printf(
        "NativeBreakpointList::%s creating breakpoint for addr = 0x%" PRIx64
        ", size_hint = %zu, hardware = %s",
        __FUNCTION__, addr, size_hint, hardware ? "true" : "false");

  NativeBreakpointSP breakpoint_sp;
  Status error = create_func(addr, size_hint, hardware, breakpoint_sp);
  if (error.Fail()) {
    if (log)
      log->Printf(
          "NativeBreakpointList::%s creating breakpoint for addr = 0x%" PRIx64
          ", size_hint = %zu, hardware = %s -- FAILED: %s",
          __FUNCTION__, addr, size_hint, hardware ? "true" : "false",
          error.AsCString());
    return error;
  }

  // Remember the breakpoint.
  assert(breakpoint_sp && "NativeBreakpoint create function succeeded but "
                          "returned NULL breakpoint");
  m_breakpoints.insert(BreakpointMap::value_type(addr, breakpoint_sp));

  return error;
}

Status NativeBreakpointList::DecRef(lldb::addr_t addr) {
  Status error;

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
  if (log)
    log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64, __FUNCTION__,
                addr);

  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  // Check if the breakpoint is already set.
  auto iter = m_breakpoints.find(addr);
  if (iter == m_breakpoints.end()) {
    // Not found!
    if (log)
      log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64 " -- NOT FOUND",
                  __FUNCTION__, addr);
    error.SetErrorString("breakpoint not found");
    return error;
  }

  // Decrement ref count.
  const int32_t new_ref_count = iter->second->DecRef();
  assert(new_ref_count >= 0 && "NativeBreakpoint ref count went negative");

  if (new_ref_count > 0) {
    // Still references to this breakpoint.  Leave it alone.
    if (log)
      log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64
                  " -- new breakpoint ref count %" PRIu32,
                  __FUNCTION__, addr, new_ref_count);
    return error;
  }

  // Breakpoint has no more references.  Disable it if it's not already
  // disabled.
  if (log)
    log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64
                " -- removing due to no remaining references",
                __FUNCTION__, addr);

  // If it's enabled, we need to disable it.
  if (iter->second->IsEnabled()) {
    if (log)
      log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64
                  " -- currently enabled, now disabling",
                  __FUNCTION__, addr);
    error = iter->second->Disable();
    if (error.Fail()) {
      if (log)
        log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64
                    " -- removal FAILED: %s",
                    __FUNCTION__, addr, error.AsCString());
      // Continue since we still want to take it out of the breakpoint list.
    }
  } else {
    if (log)
      log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64
                  " -- already disabled, nothing to do",
                  __FUNCTION__, addr);
  }

  // Take the breakpoint out of the list.
  if (log)
    log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64
                " -- removed from breakpoint map",
                __FUNCTION__, addr);

  m_breakpoints.erase(iter);
  return error;
}

Status NativeBreakpointList::EnableBreakpoint(lldb::addr_t addr) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
  if (log)
    log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64, __FUNCTION__,
                addr);

  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  // Ensure we have said breakpoint.
  auto iter = m_breakpoints.find(addr);
  if (iter == m_breakpoints.end()) {
    // Not found!
    if (log)
      log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64 " -- NOT FOUND",
                  __FUNCTION__, addr);
    return Status("breakpoint not found");
  }

  // Enable it.
  return iter->second->Enable();
}

Status NativeBreakpointList::DisableBreakpoint(lldb::addr_t addr) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
  if (log)
    log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64, __FUNCTION__,
                addr);

  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  // Ensure we have said breakpoint.
  auto iter = m_breakpoints.find(addr);
  if (iter == m_breakpoints.end()) {
    // Not found!
    if (log)
      log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64 " -- NOT FOUND",
                  __FUNCTION__, addr);
    return Status("breakpoint not found");
  }

  // Disable it.
  return iter->second->Disable();
}

Status NativeBreakpointList::GetBreakpoint(lldb::addr_t addr,
                                           NativeBreakpointSP &breakpoint_sp) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
  if (log)
    log->Printf("NativeBreakpointList::%s addr = 0x%" PRIx64, __FUNCTION__,
                addr);

  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  // Ensure we have said breakpoint.
  auto iter = m_breakpoints.find(addr);
  if (iter == m_breakpoints.end()) {
    // Not found!
    breakpoint_sp.reset();
    return Status("breakpoint not found");
  }

  // Disable it.
  breakpoint_sp = iter->second;
  return Status();
}

Status NativeBreakpointList::RemoveTrapsFromBuffer(lldb::addr_t addr, void *buf,
                                                   size_t size) const {
  for (const auto &map : m_breakpoints) {
    lldb::addr_t bp_addr = map.first;
    // Breapoint not in range, ignore
    if (bp_addr < addr || addr + size <= bp_addr)
      continue;
    const auto &bp_sp = map.second;
    // Not software breakpoint, ignore
    if (!bp_sp->IsSoftwareBreakpoint())
      continue;
    auto software_bp_sp = std::static_pointer_cast<SoftwareBreakpoint>(bp_sp);
    auto opcode_addr = static_cast<char *>(buf) + bp_addr - addr;
    auto saved_opcodes = software_bp_sp->m_saved_opcodes;
    auto opcode_size = software_bp_sp->m_opcode_size;
    ::memcpy(opcode_addr, saved_opcodes, opcode_size);
  }
  return Status();
}
