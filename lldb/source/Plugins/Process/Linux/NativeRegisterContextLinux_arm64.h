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

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  void InvalidateAllRegisters() override;

  std::vector<uint32_t>
  GetExpeditedRegisters(ExpeditedRegs expType) const override;

  bool RegisterOffsetIsDynamic() const override { return true; }

  llvm::Expected<MemoryTaggingDetails>
  GetMemoryTaggingDetails(int32_t type) override;

protected:
  void *GetGPRBuffer() override { return GetSetBuffer(RegisterKind::GPR); }

  void *GetFPRBuffer() override { return GetSetBuffer(RegisterKind::FPR); }

  size_t GetFPRSize() override { return GetSetSize(RegisterKind::FPR); }

  lldb::addr_t FixWatchpointHitAddress(lldb::addr_t hit_addr) override;

private:
  enum class RegisterKind : uint32_t {
    GPR = 1 << 0, // General purpose registers.
    FPR = 1 << 1, // When there is no SVE, or SVE in FPSIMD mode, or streaming
                  // only SVE that is in non-streaming mode.
    SVE = 1 << 2, // Used for SVE registers in streaming or non-streaming mode.
    SVE_HEADER = 1 << 3, // Only the ptrace header for SVE.
    PAC = 1 << 4,        // Pointer authentication mask registers.
    MTE = 1 << 5,        // Memory tagging control registers.
    TLS = 1 << 6,        // Thread local storage registers.
    ZA = 1 << 7,         // ZA only, because SVCR and SVG are pseudo registers.
    ZA_HEADER = 1 << 8,  // Only the ptrace header for ZA.
    ZT = 1 << 9,         // ZT only.
    FPMR = 1 << 10,      // Floating point mode control registers.
    GCS = 1 << 11,       // Guarded Control Stack registers.
    POE = 1 << 12,       // Permission Overlay registers.
  };

  // Registers that can be handled by looking up ptrace number, set buffer
  // and set size without any complicated indirection.
  static constexpr std::array SimpleRegisterSets{
      RegisterKind::GPR, RegisterKind::PAC,  RegisterKind::MTE,
      RegisterKind::TLS, RegisterKind::FPMR, RegisterKind::GCS,
      RegisterKind::POE,
  };

  class CacheValidity {
  private:
    using Storage = std::underlying_type_t<RegisterKind>;
    Storage m_valid_flags = 0;

  public:
    void Invalidate(RegisterKind set) {
      m_valid_flags &= ~static_cast<Storage>(set);
    }

    void Invalidate(const std::vector<RegisterKind> &sets) {
      for (auto set : sets)
        Invalidate(set);
    }

    void MakeValid(RegisterKind set) {
      m_valid_flags |= static_cast<Storage>(set);
    }

    bool IsValid(RegisterKind set) {
      return (m_valid_flags & static_cast<Storage>(set)) != 0;
    }
  } m_validity;

  unsigned int GetPtraceSet(RegisterKind set);
  size_t GetSetSize(RegisterKind set);
  std::vector<RegisterKind> GetWriteInvalidates(RegisterKind set);
  void *GetSetBuffer(RegisterKind set);
  uint32_t GetSimpleSetOffset(RegisterKind set);
  bool IsSimpleSetPresent(RegisterKind set);

  static uint8_t *AddRegisterKind(uint8_t *dst, RegisterKind register_set_type);

  uint8_t *AddSavedRegisters(uint8_t *dst, RegisterKind register_set_type,
                             std::optional<size_t> size = std::nullopt);

  Status RestoreRegisters(const uint8_t **src, const RegisterKind set);

  Status ReadRegisterKind(RegisterKind set);
  Status WriteRegisterKind(RegisterKind set);

  std::optional<Status> WriteSimpleRegisterSet(uint32_t reg,
                                               const RegisterInfo &reg_info,
                                               const RegisterValue &reg_value);
  std::optional<Status> ReadSimpleRegisterSet(uint32_t reg,
                                              const RegisterInfo &reg_info,
                                              RegisterValue &reg_value);

  struct user_pt_regs m_gpr_arm64; // 64-bit general purpose registers.

  RegisterInfoPOSIX_arm64::FPU
      m_fpr; // floating-point registers including extended register sets.

  SVEState m_sve_state = SVEState::Unknown;
  struct sve::user_sve_header m_sve_header;
  std::vector<uint8_t> m_sve_ptrace_payload;

  sve::user_za_header m_za_header;
  std::vector<uint8_t> m_za_ptrace_payload;

  bool m_refresh_hwdebug_info = true;

  struct user_pac_mask {
    uint64_t data_mask = 0;
    uint64_t insn_mask = 0;
  } m_pac_mask;

  uint64_t m_mte_ctrl_reg = 0;

  struct sme_pseudo_regs {
    uint64_t ctrl_reg = 0;
    uint64_t svg_reg = 0;
  } m_sme_pseudo_regs;

  size_t m_tls_size = 0;

  struct tls_regs {
    uint64_t tpidr_reg = 0;
    // Only valid when SME is present.
    uint64_t tpidr2_reg = 0;
  } m_tls_regs;

  // SME2's ZT is a 512 bit register.
  std::array<uint8_t, 64> m_zt_reg;

  uint64_t m_fpmr_reg = 0;

  struct poe_regs {
    uint64_t por_reg = 0;
  } m_poe_regs;

  struct gcs_regs {
    uint64_t features_enabled = 0;
    uint64_t features_locked = 0;
    uint64_t gcspr_e0 = 0;
  } m_gcs_regs;

  bool IsGPR(unsigned reg) const;

  bool IsFPR(unsigned reg) const;

  Status ReadSMESVG();
  uint64_t GetSVERegVG() { return m_sve_header.vl / 8; }
  void SetSVERegVG(uint64_t vg) { m_sve_header.vl = vg * 8; }

  // SVCR is a pseudo register and we do not allow writes to it.
  Status ReadSMEControl();
  void *GetSMEPseudoBuffer() { return &m_sme_pseudo_regs; }
  size_t GetSMEPseudoBufferSize() { return sizeof(m_sme_pseudo_regs); }

  llvm::Error ReadHardwareDebugInfo() override;

  llvm::Error WriteHardwareDebugRegs(DREGType hwbType) override;

  uint32_t CalculateFprOffset(const RegisterInfo *reg_info,
                              bool streaming_fpsimd) const;

  RegisterInfoPOSIX_arm64 &GetRegisterInfo() const;

  void ConfigureRegisterContext();

  uint32_t CalculateSVEOffset(const RegisterInfo *reg_info) const;

  Status CacheAllRegisters(uint32_t &cached_size);
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_arm64_h

#endif // defined (__arm64__) || defined (__aarch64__)
