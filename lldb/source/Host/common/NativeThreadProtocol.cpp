//===-- NativeThreadProtocol.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/common/NativeThreadProtocol.h"

#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Host/common/SoftwareBreakpoint.h"

using namespace lldb;
using namespace lldb_private;

NativeThreadProtocol::NativeThreadProtocol(NativeProcessProtocol *process,
                                           lldb::tid_t tid)
    : m_process_wp(process->shared_from_this()), m_tid(tid) {}

Error NativeThreadProtocol::ReadRegister(uint32_t reg,
                                         RegisterValue &reg_value) {
  NativeRegisterContextSP register_context_sp = GetRegisterContext();
  if (!register_context_sp)
    return Error("no register context");

  const RegisterInfo *const reg_info =
      register_context_sp->GetRegisterInfoAtIndex(reg);
  if (!reg_info)
    return Error("no register info for reg num %" PRIu32, reg);

  return register_context_sp->ReadRegister(reg_info, reg_value);
  ;
}

Error NativeThreadProtocol::WriteRegister(uint32_t reg,
                                          const RegisterValue &reg_value) {
  NativeRegisterContextSP register_context_sp = GetRegisterContext();
  if (!register_context_sp)
    return Error("no register context");

  const RegisterInfo *const reg_info =
      register_context_sp->GetRegisterInfoAtIndex(reg);
  if (!reg_info)
    return Error("no register info for reg num %" PRIu32, reg);

  return register_context_sp->WriteRegister(reg_info, reg_value);
}

Error NativeThreadProtocol::SaveAllRegisters(lldb::DataBufferSP &data_sp) {
  NativeRegisterContextSP register_context_sp = GetRegisterContext();
  if (!register_context_sp)
    return Error("no register context");
  return register_context_sp->WriteAllRegisterValues(data_sp);
}

Error NativeThreadProtocol::RestoreAllRegisters(lldb::DataBufferSP &data_sp) {
  NativeRegisterContextSP register_context_sp = GetRegisterContext();
  if (!register_context_sp)
    return Error("no register context");
  return register_context_sp->ReadAllRegisterValues(data_sp);
}

NativeProcessProtocolSP NativeThreadProtocol::GetProcess() {
  return m_process_wp.lock();
}
