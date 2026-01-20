//===------ NativeRegisterContextAIX_ppc64.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__powerpc64__)

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVEREGISTERCONTEXTAIX_PPC64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVEREGISTERCONTEXTAIX_PPC64_H

#include "Plugins/Process/AIX/NativeRegisterContextAIX.h"
#include "Plugins/Process/Utility/lldb-ppc64-register-enums.h"

namespace lldb_private {
namespace process_aix {

class NativeProcessAIX;

class NativeRegisterContextAIX_ppc64 : public NativeRegisterContextAIX {
public:
  NativeRegisterContextAIX_ppc64(const ArchSpec &target_arch,
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

private:
  bool IsGPR(unsigned reg) const;

  bool IsFPR(unsigned reg) const;

  bool IsVMX(unsigned reg) const;

  bool IsVSX(unsigned reg) const;

  uint32_t CalculateFprOffset(const RegisterInfo *reg_info) const;

  uint32_t CalculateVmxOffset(const RegisterInfo *reg_info) const;

  uint32_t CalculateVsxOffset(const RegisterInfo *reg_info) const;
};

} // namespace process_aix
} // namespace lldb_private

#endif // #ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVEREGISTERCONTEXTAIX_H

#endif // defined(__powerpc64__)
