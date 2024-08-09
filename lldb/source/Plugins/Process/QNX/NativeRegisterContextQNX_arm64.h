//===-- NativeRegisterContextQNX_arm64.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__aarch64__) && defined(__QNX__)

#ifndef lldb_NativeRegisterContextQNX_arm64_h
#define lldb_NativeRegisterContextQNX_arm64_h

#include <aarch64/context.h>

#include "Plugins/Process/QNX/NativeRegisterContextQNX.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_arm64.h"

namespace lldb_private {
namespace process_qnx {

class NativeProcessQNX;

class NativeRegisterContextQNX_arm64 : public NativeRegisterContextQNX {
public:
  NativeRegisterContextQNX_arm64(const ArchSpec &target_arch,
                                 NativeThreadProtocol &native_thread);

  uint32_t GetRegisterSetCount() const override;

  uint32_t GetUserRegisterCount() const override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;

  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

protected:
  Status ReadGPR();

  Status WriteGPR();

  Status ReadFPR();

  Status WriteFPR();

private:
  AARCH64_CPU_REGISTERS m_cpu_reg_data;
  AARCH64_FPU_REGISTERS m_fpu_reg_data;

  RegisterInfoPOSIX_arm64 &GetRegisterInfo() const;
};

} // namespace process_qnx
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextQNX_arm64_h

#endif // defined(__aarch64__) && defined(__QNX__)
