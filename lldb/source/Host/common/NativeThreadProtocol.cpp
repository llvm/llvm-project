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

NativeThreadProtocol::NativeThreadProtocol(NativeProcessProtocol &process,
                                           lldb::tid_t tid)
    : m_process(process), m_tid(tid) {}

Status NativeThreadProtocol::ReadRegister(uint32_t reg,
                                          RegisterValue &reg_value) {
  NativeRegisterContext &register_context = GetRegisterContext();

  const RegisterInfo *const reg_info =
      register_context.GetRegisterInfoAtIndex(reg);
  if (!reg_info)
    return Status("no register info for reg num %" PRIu32, reg);

  return register_context.ReadRegister(reg_info, reg_value);
  ;
}

Status NativeThreadProtocol::WriteRegister(uint32_t reg,
                                           const RegisterValue &reg_value) {
  NativeRegisterContext& register_context = GetRegisterContext();

  const RegisterInfo *const reg_info =
      register_context.GetRegisterInfoAtIndex(reg);
  if (!reg_info)
    return Status("no register info for reg num %" PRIu32, reg);

  return register_context.WriteRegister(reg_info, reg_value);
}

Status NativeThreadProtocol::SaveAllRegisters(lldb::DataBufferSP &data_sp) {
  return GetRegisterContext().WriteAllRegisterValues(data_sp);
}

Status NativeThreadProtocol::RestoreAllRegisters(lldb::DataBufferSP &data_sp) {
  return GetRegisterContext().ReadAllRegisterValues(data_sp);
}
