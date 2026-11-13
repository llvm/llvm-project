//===-- NativeRegisterContextFreeBSD_arm.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__arm__)

#ifndef lldb_NativeRegisterContextFreeBSD_arm_h
#define lldb_NativeRegisterContextFreeBSD_arm_h

// clang-format off
#include <sys/types.h>
#include <machine/reg.h>
#include <machine/vfp.h>
// clang-format on

#include "Plugins/Process/FreeBSD/NativeRegisterContextFreeBSD.h"
#include "lldb/Register/Common/RegisterInfo_arm.h"

#include <array>

namespace lldb_private {
namespace process_freebsd {

class NativeProcessFreeBSD;

class NativeRegisterContextFreeBSD_arm : public NativeRegisterContextFreeBSD {
public:
  NativeRegisterContextFreeBSD_arm(const ArchSpec &target_arch,
                                   NativeThreadFreeBSD &native_thread);

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                const lldb_private::RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  llvm::Error
  CopyHardwareWatchpointsFrom(NativeRegisterContextFreeBSD &source) override;

private:
  reg m_gpr{};
  vfp_state m_fpr{};

  Status ReadRegisterSet(uint32_t set);
  Status WriteRegisterSet(uint32_t set);
};

} // namespace process_freebsd
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextFreeBSD_arm_h

#endif // defined (__arm__)
