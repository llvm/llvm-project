//===-- NativeRegisterContextFreeBSD_riscv64.h --------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__riscv) && __riscv_xlen == 64

#ifndef lldb_NativeRegisterContextFreeBSD_riscv64_h
#define lldb_NativeRegisterContextFreeBSD_riscv64_h

// clang-format off
#include <sys/types.h>
#include <sys/param.h>
#include <machine/reg.h>
// clang-format on

#include "Plugins/Process/FreeBSD/NativeRegisterContextFreeBSD.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_riscv64.h"

namespace lldb_private {
namespace process_freebsd {

class NativeRegisterContextFreeBSD_riscv64
    : public NativeRegisterContextFreeBSD {
public:
  NativeRegisterContextFreeBSD_riscv64(const ArchSpec &target_arch,
                                       NativeThreadFreeBSD &native_thread);

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

private:
  // FreeBSD's native register structures
  struct reg m_gpr;
  struct fpreg m_fpr;

  bool m_gpr_is_valid;
  bool m_fpr_is_valid;

  // Ptrace wrappers
  Status ReadGPR();
  Status WriteGPR();
  Status ReadFPR();
  Status WriteFPR();

  // Conversion functions between FreeBSD and POSIX layouts
  static void FreeBSDToPOSIXGPR(const struct reg &freebsd_gpr,
                                RegisterInfoPOSIX_riscv64::GPR &posix_gpr);
  static void POSIXToFreeBSDGPR(const RegisterInfoPOSIX_riscv64::GPR &posix_gpr,
                                struct reg &freebsd_gpr);
  static void FreeBSDToPOSIXFPR(const struct fpreg &freebsd_fpr,
                                RegisterInfoPOSIX_riscv64::FPR &posix_fpr);
  static void POSIXToFreeBSDFPR(const RegisterInfoPOSIX_riscv64::FPR &posix_fpr,
                                struct fpreg &freebsd_fpr);

  // Single register conversion helpers
  Status GetGPRValue(uint32_t reg_index, uint64_t &value) const;
  Status SetGPRValue(uint32_t reg_index, uint64_t value);

  RegisterInfoPOSIX_riscv64 &GetRegisterInfo() const;
};

} // namespace process_freebsd
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextFreeBSD_riscv64_h
#endif // defined(__riscv) && __riscv_xlen == 64
