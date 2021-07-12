//===-- NativeRegisterContextLinux_arm64.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__arm64__) || defined(__aarch64__)

#ifndef lldb_NativeRegisterContextLinux_arm64_h
#define lldb_NativeRegisterContextLinux_arm64_h

#include "Plugins/Process/Linux/NativeRegisterContextLinux.h"
#include "Plugins/Process/Utility/LinuxPTraceDefines_arm64sve.h"
#include "Plugins/Process/Utility/NativeRegisterContextDBReg_arm64.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_arm64.h"

#include <asm/ptrace.h>

namespace lldb_private {
namespace process_linux {

class NativeProcessLinux;

class NativeRegisterContextLinux_arm64
    : public NativeRegisterContextLinux,
      public NativeRegisterContextDBReg_arm64 {
public:
  NativeRegisterContextLinux_arm64(
      const ArchSpec &target_arch, NativeThreadProtocol &native_thread,
      std::unique_ptr<RegisterInfoPOSIX_arm64> register_info_up);

  uint32_t GetRegisterSetCount() const override;

  uint32_t GetUserRegisterCount() const override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;

  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::DataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  void InvalidateAllRegisters() override;

  std::vector<uint32_t>
  GetExpeditedRegisters(ExpeditedRegs expType) const override;

  bool RegisterOffsetIsDynamic() const override { return true; }

  llvm::Expected<MemoryTaggingDetails>
  GetMemoryTaggingDetails(int32_t type) override;

protected:
  Status ReadGPR() override;

  Status WriteGPR() override;

  Status ReadFPR() override;

  Status WriteFPR() override;

  void *GetGPRBuffer() override { return &m_gpr_arm64; }

  // GetGPRBufferSize returns sizeof arm64 GPR ptrace buffer, it is different
  // from GetGPRSize which returns sizeof RegisterInfoPOSIX_arm64::GPR.
  size_t GetGPRBufferSize() { return sizeof(m_gpr_arm64); }

  void *GetFPRBuffer() override { return &m_fpr; }

  size_t GetFPRSize() override { return sizeof(m_fpr); }

  lldb::addr_t FixWatchpointHitAddress(lldb::addr_t hit_addr) override;

private:
  bool m_gpr_is_valid;
  bool m_fpu_is_valid;
  bool m_sve_buffer_is_valid;
  bool m_mte_ctrl_is_valid;

  bool m_sve_header_is_valid;
  bool m_pac_mask_is_valid;

  struct user_pt_regs m_gpr_arm64; // 64-bit general purpose registers.

  RegisterInfoPOSIX_arm64::FPU
      m_fpr; // floating-point registers including extended register sets.

  SVEState m_sve_state;
  struct sve::user_sve_header m_sve_header;
  std::vector<uint8_t> m_sve_ptrace_payload;

  bool m_refresh_hwdebug_info;

  struct user_pac_mask {
    uint64_t data_mask;
    uint64_t insn_mask;
  };

  struct user_pac_mask m_pac_mask;

  uint64_t m_mte_ctrl_reg;

  bool IsGPR(unsigned reg) const;

  bool IsFPR(unsigned reg) const;

  Status ReadAllSVE();

  Status WriteAllSVE();

  Status ReadSVEHeader();

  Status WriteSVEHeader();

  Status ReadPAuthMask();

  Status ReadMTEControl();

  Status WriteMTEControl();

  bool IsSVE(unsigned reg) const;
  bool IsPAuth(unsigned reg) const;
  bool IsMTE(unsigned reg) const;

  uint64_t GetSVERegVG() { return m_sve_header.vl / 8; }

  void SetSVERegVG(uint64_t vg) { m_sve_header.vl = vg * 8; }

  void *GetSVEHeader() { return &m_sve_header; }

  void *GetPACMask() { return &m_pac_mask; }

  void *GetMTEControl() { return &m_mte_ctrl_reg; }

  void *GetSVEBuffer();

  size_t GetSVEHeaderSize() { return sizeof(m_sve_header); }

  size_t GetPACMaskSize() { return sizeof(m_pac_mask); }

  size_t GetSVEBufferSize() { return m_sve_ptrace_payload.size(); }

  size_t GetMTEControlSize() { return sizeof(m_mte_ctrl_reg); }

  llvm::Error ReadHardwareDebugInfo() override;

  llvm::Error WriteHardwareDebugRegs(DREGType hwbType) override;

  uint32_t CalculateFprOffset(const RegisterInfo *reg_info) const;

  RegisterInfoPOSIX_arm64 &GetRegisterInfo() const;

  void ConfigureRegisterContext();

  uint32_t CalculateSVEOffset(const RegisterInfo *reg_info) const;
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_arm64_h

#endif // defined (__arm64__) || defined (__aarch64__)
