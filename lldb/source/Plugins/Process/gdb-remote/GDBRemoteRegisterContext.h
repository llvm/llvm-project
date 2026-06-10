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

#include "lldb/Target/DynamicRegisterInfo.h"
#include "lldb/Target/RegisterContext.h"
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
class GDBRemoteDynamicRegisterInfo;

typedef std::shared_ptr<GDBRemoteDynamicRegisterInfo>
    GDBRemoteDynamicRegisterInfoSP;

class GDBRemoteDynamicRegisterInfo final : public DynamicRegisterInfo {
public:
  GDBRemoteDynamicRegisterInfo() : DynamicRegisterInfo() {}

  ~GDBRemoteDynamicRegisterInfo() override = default;

  void UpdateARM64SVERegistersInfos(uint64_t vg);
  void UpdateARM64SMERegistersInfos(uint64_t svg);
};

class GDBRemoteRegisterContext : public RegisterContext {
public:
  GDBRemoteRegisterContext(ThreadGDBRemote &thread, uint32_t concrete_frame_idx,
                           GDBRemoteDynamicRegisterInfoSP reg_info_sp,
                           bool read_all_at_once, bool write_all_at_once);

  ~GDBRemoteRegisterContext() override;

  void InvalidateAllRegisters() override { SetAllRegistersUnfetched(); }

  size_t GetRegisterCount() override;

  const RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override;

  const RegisterSet *GetRegisterSet(size_t reg_set) override;

  bool ReadRegister(const RegisterInfo *reg_info,
                    RegisterValue &value) override;

  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  bool ReadAllRegisterValues(RegisterCheckpoint &reg_checkpoint) override;

  bool
  WriteAllRegisterValues(const RegisterCheckpoint &reg_checkpoint) override;

  uint32_t ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind,
                                               uint32_t num) override;

  bool RegisterWriteCausesReconfigure(const llvm::StringRef name) override;

  bool ReconfigureRegisterInfo() override;

protected:
  friend class ThreadGDBRemote;

  bool ReadRegisterBytes(const RegisterInfo *reg_info);

  bool WriteRegisterBytes(const RegisterInfo *reg_info, DataExtractor &data,
                          uint32_t data_offset);

  bool PrivateSetRegisterValue(uint32_t reg, llvm::ArrayRef<uint8_t> data);

  bool PrivateSetRegisterValue(uint32_t reg, uint64_t val);

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

  void SetRegisterIsValid(const RegisterInfo *reg_info) {
    SetRegisterIsValidState(reg_info, eLazyBoolYes);
  }
  void SetRegisterIsUnavailable(const RegisterInfo *reg_info) {
    SetRegisterIsValidState(reg_info, eLazyBoolNo);
  }
  void SetRegisterIsUnfetched(const RegisterInfo *reg_info) {
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

  GDBRemoteDynamicRegisterInfoSP m_reg_info_sp;

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
  void SetRegisterIsValidState(const RegisterInfo *reg_info, LazyBool valid);
  void SetRegisterIsValidState(uint32_t reg, LazyBool valid);

  // Helper function for ReadRegisterBytes().
  bool GetPrimordialRegister(const RegisterInfo *reg_info,
                             GDBRemoteCommunicationClient &gdb_comm);
  // Helper function for WriteRegisterBytes().
  bool SetPrimordialRegister(const RegisterInfo *reg_info,
                             GDBRemoteCommunicationClient &gdb_comm);

  GDBRemoteRegisterContext(const GDBRemoteRegisterContext &) = delete;
  const GDBRemoteRegisterContext &
  operator=(const GDBRemoteRegisterContext &) = delete;
};

} // namespace process_gdb_remote
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_GDBREMOTEREGISTERCONTEXT_H
