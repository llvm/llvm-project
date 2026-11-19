//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__powerpc__)

#ifndef LLDB_PLUGINS_PROCESS_FREEBSD_NATIVEREGISTERCONTEXTFREEBSD_PPC_H
#define LLDB_PLUGINS_PROCESS_FREEBSD_NATIVEREGISTERCONTEXTFREEBSD_PPC_H

// clang-format off
#include <sys/types.h>
#include <machine/reg.h>
// clang-format on

#include "NativeRegisterContextFreeBSD.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_ppc.h"
#include "llvm/ADT/SmallVector.h"

namespace lldb_private::process_freebsd {

/// Shared native bank access and checkpoint handling for FreeBSD PPC variants.
/// Leaves select the description/layout and initialize the virtual native base.
class NativeRegisterContextFreeBSD_ppc : public NativeRegisterContextFreeBSD {
public:
  llvm::Expected<RegisterValue> ReadRegister(RegisterID id) override;
  llvm::Error WriteRegister(RegisterID id, const RegisterValue &value) override;
  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data) override;
  Status WriteAllRegisterValues(const lldb::DataBufferSP &data) override;

  /// The architecture adapter supplies hardware-watchpoint policy. Keeping this
  /// hook abstract also leaves virtual native-base construction to that
  /// adapter.
  llvm::Error CopyHardwareWatchpointsFrom(
      NativeRegisterContextFreeBSD &source) override = 0;

protected:
  NativeRegisterContextFreeBSD_ppc(bool has_vmx, bool has_vsx)
      : m_has_vmx(has_vmx), m_has_vsx(has_vsx) {}

  static RegisterLayout SelectLayout(RegisterLayout layout, bool has_vmx,
                                     bool has_vsx);

private:
  enum RegSetKind { GPRegSet, FPRegSet, VMXRegSet, VSXRegSet };

  reg m_gpr{};
  fpreg m_fpr{};
  vmxreg m_vmx{};
  RegisterLayoutFreeBSD_ppc::VSX m_vsx{};
  bool m_has_vmx;
  bool m_has_vsx;

  llvm::SmallVector<MutableRegisterData, 4> GetRegisterBanks();
  bool HasRegisterSet(RegSetKind set) const;
  Status ReadRegisterSet(RegSetKind set);
  Status WriteRegisterSet(RegSetKind set);
};

} // namespace lldb_private::process_freebsd

#endif
#endif // defined(__powerpc__)
