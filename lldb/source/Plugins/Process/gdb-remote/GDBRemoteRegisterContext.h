//===-- GDBRemoteRegisterContext.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_GDBREMOTEREGISTERCONTEXT_H
#define LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_GDBREMOTEREGISTERCONTEXT_H

#include <vector>

#include "lldb/Register/RegisterDefinition.h"
#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private.h"

#include "GDBRemoteCommunicationClient.h"

class StringExtractor;

namespace lldb_private {
namespace process_gdb_remote {

class ThreadGDBRemote;
class ProcessGDBRemote;

class GDBRemoteRegisterContext : public ThreadRegisterContext {
public:
  GDBRemoteRegisterContext(ThreadGDBRemote &thread, uint32_t concrete_frame_idx,
                           std::shared_ptr<const RegisterLayout> layout_sp,
                           bool read_all_at_once, bool write_all_at_once);

  ~GDBRemoteRegisterContext() override;

  void InvalidateAllRegisters() override { SetAllRegistersUnfetched(); }

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error WriteRegister(lldb_private::RegisterID id,
                            const lldb_private::RegisterValue &value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  bool ReadAllRegisterValues(RegisterCheckpoint &reg_checkpoint) override;

  bool WriteAllRegisterValues(const RegisterCheckpoint &reg_checkpoint) override;

  bool RegisterWriteCausesReconfigure(const llvm::StringRef name) override;

  bool ReconfigureRegisterLayout() override;

protected:
  friend class ThreadGDBRemote;

  bool ReadRegisterBytes(const RegisterDescriptor *reg_info);

  bool PrivateSetRegisterValue(RegisterID reg, llvm::ArrayRef<uint8_t> data);

  bool PrivateSetRegisterValue(RegisterID reg, uint64_t val);

  void SetAllRegistersValid() { SetAllRegistersValidState(eLazyBoolYes); }
  void SetAllRegistersUnfetched() {
    SetAllRegistersValidState(eLazyBoolCalculate);
  }

  bool GetRegisterIsValid(uint32_t reg) const {
    return GetRegisterValidState(reg) == eLazyBoolYes;
  }
  bool GetRegisterIsUnavailable(uint32_t reg) const {
    return GetRegisterValidState(reg) == eLazyBoolNo;
  }
  bool GetRegisterIsUnfetched(uint32_t reg) const {
    return GetRegisterValidState(reg) == eLazyBoolCalculate;
  }

  void SetRegisterIsValid(const RegisterDescriptor *reg_info) {
    SetRegisterIsValidState(reg_info, eLazyBoolYes);
  }
  void SetRegisterIsUnavailable(const RegisterDescriptor *reg_info) {
    SetRegisterIsValidState(reg_info, eLazyBoolNo);
  }
  void SetRegisterIsUnfetched(const RegisterDescriptor *reg_info) {
    SetRegisterIsValidState(reg_info, eLazyBoolCalculate);
  }
  void SetRegisterIsValid(uint32_t reg) {
    SetRegisterIsValidState(reg, eLazyBoolYes);
  }
  void SetRegisterIsUnavailable(uint32_t reg) {
    SetRegisterIsValidState(reg, eLazyBoolNo);
  }
  void SetRegisterIsUnfetched(uint32_t reg) {
    SetRegisterIsValidState(reg, eLazyBoolCalculate);
  }

  std::shared_ptr<const RegisterLayout> m_register_layout_sp;

  /// eLazyBoolYes - we have the bytes for this register locally.
  /// eLazyBoolCalculate - we have not yet tried to get the bytes.
  /// eLazyBoolNo - the bytes are unreadable at this stop.
  std::vector<LazyBool> m_reg_valid;

  DataExtractor m_reg_data;
  bool m_read_all_at_once;
  bool m_write_all_at_once;
  bool m_gpacket_cached;

private:
  LazyBool GetRegisterValidState(uint32_t reg) const;
  void SetAllRegistersValidState(LazyBool valid);
  void SetRegisterIsValidState(const RegisterDescriptor *reg_info,
                               LazyBool valid);
  void SetRegisterIsValidState(uint32_t reg, LazyBool valid);

  // Helper function for ReadRegisterBytes().
  bool GetPrimordialRegister(const RegisterDescriptor *reg_info,
                             GDBRemoteCommunicationClient &gdb_comm);
  // Send one primary value from the real packet image.
  bool SetPrimordialRegister(const RegisterDescriptor *reg_info,
                             GDBRemoteCommunicationClient &gdb_comm);

  GDBRemoteRegisterContext(const GDBRemoteRegisterContext &) = delete;
  const GDBRemoteRegisterContext &
  operator=(const GDBRemoteRegisterContext &) = delete;
};

} // namespace process_gdb_remote
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_GDBREMOTEREGISTERCONTEXT_H
