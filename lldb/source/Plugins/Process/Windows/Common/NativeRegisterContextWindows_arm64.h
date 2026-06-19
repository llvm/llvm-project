//===-- NativeRegisterContextWindows_arm64.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__aarch64__) || defined(_M_ARM64)
#ifndef liblldb_NativeRegisterContextWindows_arm64_h_
#define liblldb_NativeRegisterContextWindows_arm64_h_

#include "NativeRegisterContextWindows.h"

#include "Plugins/Process/Utility/NativeRegisterContextDBReg_arm64.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_arm64.h"
#include "Plugins/Process/Utility/lldb-arm64-register-enums.h"

#include "lldb/Host/windows/windows.h"

#if defined(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
#include "Plugins/Process/Utility/LinuxPTraceDefines_arm64sve.h"
#endif

namespace lldb_private {

class NativeThreadWindows;

class NativeRegisterContextWindows_arm64
    : public NativeRegisterContextWindows,
      public NativeRegisterContextDBReg_arm64 {
public:
  NativeRegisterContextWindows_arm64(
      const ArchSpec &target_arch, NativeThreadProtocol &native_thread,
      std::unique_ptr<RegisterInfoPOSIX_arm64> register_info_up);

  uint32_t GetRegisterSetCount() const override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;

  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  void InvalidateAllRegisters() override;

protected:
  Status GPRRead(const uint32_t reg, RegisterValue &reg_value);

  Status GPRWrite(const uint32_t reg, const RegisterValue &reg_value);

  Status FPRRead(const uint32_t reg, RegisterValue &reg_value);

  Status FPRWrite(const uint32_t reg, const RegisterValue &reg_value);

#if defined(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
  Status SVERead(const uint32_t reg, RegisterValue &reg_value);

  Status SVEWrite(const uint32_t reg, const RegisterValue &reg_value);
#endif

private:
  PCONTEXT m_context;
  std::shared_ptr<DataBufferHeap> m_context_buffer;

#if defined(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
  XSAVE_ARM64_SVE_HEADER *m_sve_header;
  bool m_sve_header_is_valid;
  SVEState m_sve_state;
  std::shared_ptr<DataBufferHeap> m_sve_z_buffer;
  bool m_sve_z_buffer_is_valid;

  static constexpr uint32_t k_z_low_bits_size = sizeof(ARM64_NT_NEON128);
#endif

  bool IsGPR(uint32_t reg_index) const;

  bool IsFPR(uint32_t reg_index) const;

  llvm::Error ReadHardwareDebugInfo() override;

  llvm::Error WriteHardwareDebugRegs(DREGType hwbType) override;

  Status CacheAllRegisterValues();

  RegisterInfoPOSIX_arm64 &GetRegisterInfo() const;

#if defined(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
  bool IsSVE(uint32_t reg_index) const;

  uint32_t GetSVERegVG() const { return m_sve_header->VectorLength / 8; }

  void ConfigureRegisterContext();

  Status ReadSVEHeader();

  Status CacheSVEZRegisters();
#endif
};

} // namespace lldb_private

#endif // liblldb_NativeRegisterContextWindows_arm64_h_
#endif // defined(__aarch64__) || defined(_M_ARM64)
