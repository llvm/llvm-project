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

#include "llvm/ADT/BitmaskEnum.h"

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
  enum class RegisterSetType : uint32_t {
    // General purpose registers.
    GPR = 1 << 0,
    // When there is no SVE, or SVE in FPSIMD mode, or streaming only SVE that
    // is in non-streaming mode.
    FPR = 1 << 1,
    // Used for SVE registers in streaming or non-streaming mode.
    SVE = 1 << 2,
    // Only the ptrace header for SVE.
    SVE_HEADER = 1 << 3,
    // Pointer authentication mask registers.
    PAC = 1 << 4,
    // Memory tagging control registers.
    MTE = 1 << 5,
    // Thread local storage registers.
    TLS = 1 << 6,
    // ZA only, because SVCR and SVG are pseudo registers.
    ZA = 1 << 7,
    // Only the ptrace header for ZA.
    ZA_HEADER = 1 << 8,
    // ZT only.
    ZT = 1 << 9,
    // Floating point mode control registers.
    FPMR = 1 << 10,
    // Guarded Control Stack registers.
    GCS = 1 << 11,
    // Permission Overlay registers.
    POE = 1 << 12,
    LLVM_MARK_AS_BITMASK_ENUM(POE),
  };

  RegisterSetType m_validity = static_cast<RegisterSetType>(0);

  // Returns the ptrace register set number for the given register set.
  unsigned int GetPtraceSet(RegisterSetType set) const;

  size_t GetSetSize(RegisterSetType set) const;

  void *GetSetBuffer(RegisterSetType set);

  void MakeValid(RegisterSetType set) { m_validity |= set; }

  [[nodiscard]] bool IsValid(RegisterSetType set) const {
    return any(m_validity & set);
  }

  /// Returns the mask of sets that would be invalidated if the given set was
  /// invalidated. That is, the set itself and any sets that depend on it.
  ///
  /// If you need anything more complex such as only invalidating during certain
  /// modes, put that logic in the function that calls Invalidate().
  RegisterSetType GetInvalidationMask(const RegisterSetType set) const;

  /// Invalidate our saved copies of the given register sets and any sets that
  /// depend on those sets.
  template <typename... Ts> void Invalidate(RegisterSetType first, Ts... rest) {
    static_assert((std::is_same_v<Ts, RegisterSetType> && ...));
    m_validity &=
        ~(GetInvalidationMask(first) | ... | GetInvalidationMask(rest));
  }

  Status RestoreRegisters(void *buffer, const uint8_t **src, size_t len,
                          const RegisterSetType set,
                          std::function<Status()> writer);

  size_t m_tls_size = 0;

  /// 64-bit general purpose registers.
  struct user_pt_regs m_gpr_arm64{};

  /// Floating-point registers including extended register sets.
  RegisterInfoPOSIX_arm64::FPU m_fpr{};

  SVEState m_sve_state = SVEState::Unknown;
  struct sve::user_sve_header m_sve_header{};
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

  struct tls_regs {
    uint64_t tpidr_reg = 0;
    // Only valid when SME is present.
    uint64_t tpidr2_reg = 0;
  } m_tls_regs;

  // SME2's ZT is a 512 bit register.
  std::array<uint8_t, 64> m_zt_reg{};

  uint64_t m_fpmr_reg = 0;

  struct poe_regs {
    uint64_t por_el0_reg = 0;
  } m_poe_regs;

  struct gcs_regs {
    uint64_t features_enabled = 0;
    uint64_t features_locked = 0;
    uint64_t gcspr_e0 = 0;
  } m_gcs_regs;

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

  uint64_t GetSVERegVG() { return m_sve_header.vl / 8; }

  void SetSVERegVG(uint64_t vg) { m_sve_header.vl = vg * 8; }

  void *GetSVEHeader() { return &m_sve_header; }

  void *GetZAHeader() { return &m_za_header; }

  void *GetPACMask() { return &m_pac_mask; }

  void *GetMTEControl() { return &m_mte_ctrl_reg; }

  void *GetTLSBuffer() { return &m_tls_regs; }

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

  uint8_t *AddRegisterSetType(uint8_t *dst, RegisterSetType register_set_type);

  uint8_t *AddSavedRegisters(uint8_t *dst, RegisterSetType register_set_type,
                             void *src, size_t size);
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_arm64_h

#endif // defined (__arm64__) || defined (__aarch64__)
