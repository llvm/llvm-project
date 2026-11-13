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

#include "lldb/Register/Arch/lldb-ppc64-dwarf-register-enums.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-types.h"
#include "llvm/Support/Compiler.h"
#include <cstddef>
#include <cstdint>

#include "lldb/Register/Arch/lldb-ppc64-register-enums.h"

#include "Plugins/Process/AIX/RegisterLayoutAIX_ppc64.h"

namespace lldb_private {

namespace process_aix {

class NativeProcessAIX;

class NativeRegisterContextAIX_ppc64 : public NativeRegisterContextAIX {
public:
  NativeRegisterContextAIX_ppc64(const ArchSpec &target_arch,
                                 NativeThreadProtocol &native_thread);

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                const lldb_private::RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

protected:
  void *GetGPRBuffer() override { return m_gpr; }

  void *GetFPRBuffer() override { return nullptr; }

  size_t GetFPRSize() override { return 0; }

private:
  bool IsGPR(unsigned reg) const;

  bool IsFPR(unsigned reg) const;

  bool IsVMX(unsigned reg) const;

  bool IsVSX(unsigned reg) const;

  uint32_t CalculateFprOffset(const RegisterDescriptor *reg_info) const;

  uint32_t CalculateVmxOffset(const RegisterDescriptor *reg_info) const;

  uint32_t CalculateVsxOffset(const RegisterDescriptor *reg_info) const;

  union GPRStorage {
    aix_ppc64::GPR_PPC gpr32;   // 32-bit general purpose registers.
    aix_ppc64::GPR_PPC64 gpr64; // 64-bit general purpose registers.
  } m_gpr_storage;

  void *m_gpr = nullptr;
};

} // namespace process_aix
} // namespace lldb_private

#endif // #ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVEREGISTERCONTEXTAIX_H

#endif // defined(__powerpc64__)
