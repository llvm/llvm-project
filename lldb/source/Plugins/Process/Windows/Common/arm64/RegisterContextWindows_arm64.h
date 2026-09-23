//===-- RegisterContextWindows_arm64.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextWindows_arm64_H_
#define liblldb_RegisterContextWindows_arm64_H_

#if defined(__aarch64__) || defined(_M_ARM64)

#include "RegisterContextWindows.h"

#include "Plugins/Process/Utility/lldb-arm64-register-enums.h"

#include "lldb/Host/windows/windows.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class Thread;

class RegisterContextWindows_arm64 : public RegisterContextWindows {
public:
  // Constructors and Destructors
  RegisterContextWindows_arm64(Thread &thread, uint32_t concrete_frame_idx);

  virtual ~RegisterContextWindows_arm64();

  // Subclasses must override these functions
  size_t GetRegisterCount() override;

  const RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override;

  const RegisterSet *GetRegisterSet(size_t reg_set) override;

  bool ReadRegister(const RegisterInfo *reg_info,
                    RegisterValue &reg_value) override;

  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &reg_value) override;

  void InvalidateAllRegisters() override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

protected:
  bool CacheAllRegisterValues() override;

  bool GPRRead(const uint32_t reg, RegisterValue &reg_value);

  bool GPRWrite(const uint32_t reg, const RegisterValue &reg_value);

  bool FPRRead(const uint32_t reg, RegisterValue &reg_value);

  bool FPRWrite(const uint32_t reg, const RegisterValue &reg_value);

private:
  PCONTEXT m_context_arm64;
  std::shared_ptr<DataBufferHeap> m_context_arm64_buffer;

  bool IsGPR(uint32_t reg) const;

  bool IsFPR(uint32_t reg) const;
};
} // namespace lldb_private

#endif // defined(__aarch64__) || defined(_M_ARM64)

#endif // #ifndef liblldb_RegisterContextWindows_arm64_H_
