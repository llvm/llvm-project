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
  // Bit mask enum used to refer to the types of registers we support. Currently
  // used for tracking cache validity and ReadAll/WriteAllRegister data. Will
  // be used for much more in future.
  enum RegisterSetType : uint32_t {
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

  // This single object manages all tracking of whether register value caches
  // are valid. Having a single object makes it easy to reset without missing
  // anything.
  class CacheValidity {
  private:
    using Storage = std::underlying_type_t<RegisterSetType>;
    Storage m_valid_flags = 0;

  public:
    void Invalidate(RegisterSetType set) {
      m_valid_flags &= ~static_cast<Storage>(set);
    }

    template <typename... Ts>
    void Invalidate(RegisterSetType first, Ts... rest) {
      static_assert((std::is_same_v<Ts, RegisterSetType> && ...));
      Invalidate(first);
      (Invalidate(rest), ...);
    }

    void MakeValid(RegisterSetType set) {
      m_valid_flags |= static_cast<Storage>(set);
    }
    bool IsValid(RegisterSetType set) {
      return (m_valid_flags & static_cast<Storage>(set)) != 0;
    }
  } m_validity;

  static uint8_t *AddRegisterSetType(uint8_t *dst,
                                     RegisterSetType register_set_type);

  static uint8_t *AddSavedRegisters(uint8_t *dst,
                                    RegisterSetType register_set_type,
                                    void *src, size_t size);

  Status RestoreRegisters(void *buffer, const uint8_t **src, size_t len,
                          const RegisterSetType set,
                          std::function<Status()> writer);

  size_t m_tls_size = 0;

  struct user_pt_regs m_gpr_arm64; // 64-bit general purpose registers.

  RegisterInfoPOSIX_arm64::FPU
      m_fpr; // floating-point registers including extended register sets.

  SVEState m_sve_state = SVEState::Unknown;
  struct sve::user_sve_header m_sve_header;
  std::vector<uint8_t> m_sve_ptrace_payload;

  sve::user_za_header m_za_header;
  std::vector<uint8_t> m_za_ptrace_payload;

  bool m_refresh_hwdebug_info;

  struct user_pac_mask {
    uint64_t data_mask;
    uint64_t insn_mask;
  };

  struct user_pac_mask m_pac_mask;

  uint64_t m_mte_ctrl_reg;

  struct sme_pseudo_regs {
    uint64_t ctrl_reg;
    uint64_t svg_reg;
  };

  struct sme_pseudo_regs m_sme_pseudo_regs;

  struct tls_regs {
    uint64_t tpidr_reg;
    // Only valid when SME is present.
    uint64_t tpidr2_reg;
  };

  struct tls_regs m_tls_regs;

  // SME2's ZT is a 512 bit register.
  std::array<uint8_t, 64> m_zt_reg;

  uint64_t m_fpmr_reg;

  struct poe_regs {
    uint64_t por_el0_reg;
  };

  struct poe_regs m_poe_regs;

  struct gcs_regs {
    uint64_t features_enabled;
    uint64_t features_locked;
    uint64_t gcspr_e0;
  } m_gcs_regs;

  bool IsGPR(unsigned reg) const;

  bool IsFPR(unsigned reg) const;

  Status ReadAllSVE();

  Status WriteAllSVE();

  Status ReadSVEHeader();

  Status WriteSVEHeader();

  Status ReadPAuthMask();

  Status ReadMTEControl();

  Status WriteMTEControl();

  Status ReadTLS();

  Status WriteTLS();

  Status ReadSMESVG();

  Status ReadZAHeader();

  Status ReadZA();

  Status WriteZA();

  Status ReadGCS();

  Status WriteGCS();

  // No WriteZAHeader because writing only the header will disable ZA.
  // Instead use WriteZA and ensure you have the correct ZA buffer size set
  // beforehand if you wish to disable it.

  Status ReadZT();

  Status WriteZT();

  // SVCR is a pseudo register and we do not allow writes to it.
  Status ReadSMEControl();

  Status ReadFPMR();

  Status WriteFPMR();

  Status ReadPOE();

  Status WritePOE();

  bool IsSVE(unsigned reg) const;
  bool IsSME(unsigned reg) const;
  bool IsPAuth(unsigned reg) const;
  bool IsMTE(unsigned reg) const;
  bool IsTLS(unsigned reg) const;
  bool IsFPMR(unsigned reg) const;
  bool IsGCS(unsigned reg) const;
  bool IsPOE(unsigned reg) const;

  uint64_t GetSVERegVG() { return m_sve_header.vl / 8; }

  void SetSVERegVG(uint64_t vg) { m_sve_header.vl = vg * 8; }

  void *GetSVEHeader() { return &m_sve_header; }

  void *GetZAHeader() { return &m_za_header; }

  size_t GetZAHeaderSize() { return sizeof(m_za_header); }

  void *GetPACMask() { return &m_pac_mask; }

  void *GetMTEControl() { return &m_mte_ctrl_reg; }

  void *GetTLSBuffer() { return &m_tls_regs; }

  void *GetSMEPseudoBuffer() { return &m_sme_pseudo_regs; }

  void *GetZTBuffer() { return m_zt_reg.data(); }

  void *GetSVEBuffer() { return m_sve_ptrace_payload.data(); }

  void *GetFPMRBuffer() { return &m_fpmr_reg; }

  void *GetGCSBuffer() { return &m_gcs_regs; }

  void *GetPOEBuffer() { return &m_poe_regs; }

  size_t GetSVEHeaderSize() { return sizeof(m_sve_header); }

  size_t GetPACMaskSize() { return sizeof(m_pac_mask); }

  size_t GetSVEBufferSize() { return m_sve_ptrace_payload.size(); }

  unsigned GetSVERegSet();

  void *GetZABuffer() { return m_za_ptrace_payload.data(); };

  size_t GetZABufferSize() { return m_za_ptrace_payload.size(); }

  size_t GetMTEControlSize() { return sizeof(m_mte_ctrl_reg); }

  size_t GetTLSBufferSize() { return m_tls_size; }

  size_t GetSMEPseudoBufferSize() { return sizeof(m_sme_pseudo_regs); }

  size_t GetZTBufferSize() { return m_zt_reg.size(); }

  size_t GetFPMRBufferSize() { return sizeof(m_fpmr_reg); }

  size_t GetGCSBufferSize() { return sizeof(m_gcs_regs); }

  size_t GetPOEBufferSize() { return sizeof(m_poe_regs); }

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
