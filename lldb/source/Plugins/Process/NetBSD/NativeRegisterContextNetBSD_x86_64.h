//===-- NativeRegisterContextNetBSD_x86_64.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__i386__) || defined(__x86_64__)

#ifndef lldb_NativeRegisterContextNetBSD_x86_64_h
#define lldb_NativeRegisterContextNetBSD_x86_64_h

// clang-format off
#include <sys/param.h>
#include <sys/types.h>
#include <sys/ptrace.h>
#include <machine/reg.h>
// clang-format on

#include "Plugins/Process/NetBSD/NativeRegisterContextNetBSD.h"
#include "Plugins/Process/Utility/RegisterContext_x86.h"
#include "Plugins/Process/Utility/NativeRegisterContextWatchpoint_x86.h"
#include "Plugins/Process/Utility/lldb-x86-register-enums.h"

namespace lldb_private {
namespace process_netbsd {

class NativeProcessNetBSD;

class NativeRegisterContextNetBSD_x86_64
    : public NativeRegisterContextNetBSD,
      public NativeRegisterContextWatchpoint_x86 {
public:
  NativeRegisterContextNetBSD_x86_64(const ArchSpec &target_arch,
                                     NativeThreadProtocol &native_thread);
  uint32_t GetRegisterSetCount() const override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;

  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::DataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  llvm::Error
  CopyHardwareWatchpointsFrom(NativeRegisterContextNetBSD &source) override;

private:
  // Private member types.
  enum { GPRegSet, XStateRegSet, DBRegSet };

  // Private member variables.
  struct reg m_gpr;
  struct xstate m_xstate;
  struct dbreg m_dbr;

  int GetSetForNativeRegNum(int reg_num) const;
  int GetDR(int num) const;

  Status ReadRegisterSet(uint32_t set);
  Status WriteRegisterSet(uint32_t set);
};

} // namespace process_netbsd
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextNetBSD_x86_64_h

#endif // defined(__x86_64__)
