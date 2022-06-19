//===-- NativeRegisterContextLinux.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextLinux.h"

#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/Host/linux/Ptrace.h"
#include "lldb/Utility/RegisterValue.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"

using namespace lldb_private;
using namespace lldb_private::process_linux;

lldb::ByteOrder NativeRegisterContextLinux::GetByteOrder() const {
  return m_thread.GetProcess().GetByteOrder();
}

Status NativeRegisterContextLinux::ReadRegisterRaw(uint32_t reg_index,
                                                   RegisterValue &reg_value) {
  const RegisterInfo *const reg_info = GetRegisterInfoAtIndex(reg_index);
  if (!reg_info)
    return Status("register %" PRIu32 " not found", reg_index);

  return DoReadRegisterValue(GetPtraceOffset(reg_index), reg_info->name,
                             reg_info->byte_size, reg_value);
}

Status
NativeRegisterContextLinux::WriteRegisterRaw(uint32_t reg_index,
                                             const RegisterValue &reg_value) {
  uint32_t reg_to_write = reg_index;
  RegisterValue value_to_write = reg_value;

  // Check if this is a subregister of a full register.
  const RegisterInfo *reg_info = GetRegisterInfoAtIndex(reg_index);
  if (reg_info->invalidate_regs &&
      (reg_info->invalidate_regs[0] != LLDB_INVALID_REGNUM)) {
    Status error;

    RegisterValue full_value;
    uint32_t full_reg = reg_info->invalidate_regs[0];
    const RegisterInfo *full_reg_info = GetRegisterInfoAtIndex(full_reg);

    // Read the full register.
    error = ReadRegister(full_reg_info, full_value);
    if (error.Fail())
      return error;

    lldb::ByteOrder byte_order = GetByteOrder();
    uint8_t dst[RegisterValue::kMaxRegisterByteSize];

    // Get the bytes for the full register.
    const uint32_t dest_size = full_value.GetAsMemoryData(
        full_reg_info, dst, sizeof(dst), byte_order, error);
    if (error.Success() && dest_size) {
      uint8_t src[RegisterValue::kMaxRegisterByteSize];

      // Get the bytes for the source data.
      const uint32_t src_size = reg_value.GetAsMemoryData(
          reg_info, src, sizeof(src), byte_order, error);
      if (error.Success() && src_size && (src_size < dest_size)) {
        // Copy the src bytes to the destination.
        memcpy(dst + (reg_info->byte_offset & 0x1), src, src_size);
        // Set this full register as the value to write.
        value_to_write.SetBytes(dst, full_value.GetByteSize(), byte_order);
        value_to_write.SetType(full_reg_info);
        reg_to_write = full_reg;
      }
    }
  }

  const RegisterInfo *const register_to_write_info_p =
      GetRegisterInfoAtIndex(reg_to_write);
  assert(register_to_write_info_p &&
         "register to write does not have valid RegisterInfo");
  if (!register_to_write_info_p)
    return Status("NativeRegisterContextLinux::%s failed to get RegisterInfo "
                  "for write register index %" PRIu32,
                  __FUNCTION__, reg_to_write);

  return DoWriteRegisterValue(GetPtraceOffset(reg_index), reg_info->name,
                              reg_value);
}

Status NativeRegisterContextLinux::ReadGPR() {
  return NativeProcessLinux::PtraceWrapper(
      PTRACE_GETREGS, m_thread.GetID(), nullptr, GetGPRBuffer(), GetGPRSize());
}

Status NativeRegisterContextLinux::WriteGPR() {
  return NativeProcessLinux::PtraceWrapper(
      PTRACE_SETREGS, m_thread.GetID(), nullptr, GetGPRBuffer(), GetGPRSize());
}

Status NativeRegisterContextLinux::ReadFPR() {
  return NativeProcessLinux::PtraceWrapper(PTRACE_GETFPREGS, m_thread.GetID(),
                                           nullptr, GetFPRBuffer(),
                                           GetFPRSize());
}

Status NativeRegisterContextLinux::WriteFPR() {
  return NativeProcessLinux::PtraceWrapper(PTRACE_SETFPREGS, m_thread.GetID(),
                                           nullptr, GetFPRBuffer(),
                                           GetFPRSize());
}

Status NativeRegisterContextLinux::ReadRegisterSet(void *buf, size_t buf_size,
                                                   unsigned int regset) {
  return NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET, m_thread.GetID(),
                                           static_cast<void *>(&regset), buf,
                                           buf_size);
}

Status NativeRegisterContextLinux::WriteRegisterSet(void *buf, size_t buf_size,
                                                    unsigned int regset) {
  return NativeProcessLinux::PtraceWrapper(PTRACE_SETREGSET, m_thread.GetID(),
                                           static_cast<void *>(&regset), buf,
                                           buf_size);
}

Status NativeRegisterContextLinux::DoReadRegisterValue(uint32_t offset,
                                                       const char *reg_name,
                                                       uint32_t size,
                                                       RegisterValue &value) {
  Log *log = GetLog(POSIXLog::Registers);

  long data;
  Status error = NativeProcessLinux::PtraceWrapper(
      PTRACE_PEEKUSER, m_thread.GetID(), reinterpret_cast<void *>(offset),
      nullptr, 0, &data);

  if (error.Success())
    // First cast to an unsigned of the same size to avoid sign extension.
    value.SetUInt(static_cast<unsigned long>(data), size);

  LLDB_LOG(log, "{0}: {1:x}", reg_name, data);
  return error;
}

Status NativeRegisterContextLinux::DoWriteRegisterValue(
    uint32_t offset, const char *reg_name, const RegisterValue &value) {
  Log *log = GetLog(POSIXLog::Registers);

  void *buf = reinterpret_cast<void *>(value.GetAsUInt64());
  LLDB_LOG(log, "{0}: {1}", reg_name, buf);

  return NativeProcessLinux::PtraceWrapper(
      PTRACE_POKEUSER, m_thread.GetID(), reinterpret_cast<void *>(offset), buf);
}
