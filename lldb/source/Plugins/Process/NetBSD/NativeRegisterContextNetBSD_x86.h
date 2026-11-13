//===-- NativeRegisterContextNetBSD_x86.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__i386__) || defined(__x86_64__)

#ifndef lldb_NativeRegisterContextNetBSD_x86_h
#define lldb_NativeRegisterContextNetBSD_x86_h

// clang-format off
#include <sys/param.h>
#include <sys/types.h>
#include <sys/ptrace.h>
#include <machine/reg.h>
// clang-format on

#include <array>
#include <optional>

#include "Plugins/Process/NetBSD/NativeRegisterContextNetBSD.h"
#include "Plugins/Register/Native/NativeRegisterContextDBReg_x86.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"
#include "lldb/Register/Common/RegisterInfo_x86.h"

namespace lldb_private {
namespace process_netbsd {

class NativeProcessNetBSD;

class NativeRegisterContextNetBSD_x86
    : public NativeRegisterContextNetBSD,
      public NativeRegisterContextDBReg_x86 {
public:
  NativeRegisterContextNetBSD_x86(const ArchSpec &target_arch,
                                     NativeThreadProtocol &native_thread);

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                const lldb_private::RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  llvm::Error
  CopyHardwareWatchpointsFrom(NativeRegisterContextNetBSD &source) override;

private:
  // Private member types.
  enum RegSetKind {
    GPRegSet,
    FPRegSet,
    DBRegSet,
    MaxRegularRegSet = DBRegSet,
    YMMRegSet,
    MPXRegSet,
    MaxRegSet = MPXRegSet,
  };

  // Private member variables.
  std::array<uint8_t, sizeof(struct reg)> m_gpr;
  std::array<uint8_t, sizeof(struct xstate)> m_xstate;
  std::array<uint8_t, sizeof(struct dbreg)> m_dbr;

  std::optional<RegSetKind> GetSetForNativeRegNum(uint32_t reg_num) const;

  Status ReadRegisterSet(RegSetKind set);
  Status WriteRegisterSet(RegSetKind set);

  struct YMMSplitPtr {
    void *xmm;
    void *ymm_hi;
  };
  std::optional<YMMSplitPtr> GetYMMSplitReg(uint32_t reg);
};

} // namespace process_netbsd
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextNetBSD_x86_h

#endif // defined(__i386__) || defined(__x86_64__)
