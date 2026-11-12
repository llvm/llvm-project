//===-- NativeRegisterContextOpenBSD_arm64.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__aarch64__)

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVEREGISTERCONTEXTOPENBSD_ARM64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVEREGISTERCONTEXTOPENBSD_ARM64_H

// clang-format off
#include <sys/types.h>
#include <machine/reg.h>
// clang-format on

#include <optional>

#include "Plugins/Process/OpenBSD/NativeRegisterContextOpenBSD.h"
#include "Register/Common/RegisterInfo_arm64.h"

namespace lldb_private {
namespace process_openbsd {

class NativeProcessOpenBSD;

class NativeRegisterContextOpenBSD_arm64 : public NativeRegisterContextOpenBSD {
public:
  NativeRegisterContextOpenBSD_arm64(const ArchSpec &target_arch,
                                     NativeThreadProtocol &native_thread);

  uint32_t GetUserRegisterCount() const override;
  uint32_t GetRegisterSetCount() const override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;

  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

protected:
  void *GetGPRBuffer() override { return &m_gpr; }
  size_t GetGPRSize() override { return sizeof(m_gpr); }
  void *GetFPRBuffer() override { return &m_fpr; }
  size_t GetFPRSize() override { return sizeof(m_fpr); }

private:
  // Private member types.
  enum RegSetKind {
    GPRegSet,
    FPRegSet,
    PACMaskRegSet,
    TLSRegSet,
  };

  // Private member variables.
  //
  // OpenBSD's struct reg and struct fpreg begin with the same layout as
  // RegisterInfo_arm64::GPR and RegisterInfo_arm64::FPU, so registers are
  // accessed through their byte_offset. struct reg is longer than GPR (it
  // also holds r_tpidr, which is the tpidr register of the TLS set), so the
  // two sets are kept in separate buffers rather than one buffer laid out
  // like the register infos.
  struct reg m_gpr;
  struct fpreg m_fpr;
  register_t m_pacmask[2];

  std::optional<RegSetKind> GetSetForNativeRegNum(uint32_t reg_num) const;

  Status ReadRegisterSet(RegSetKind set);
  Status WriteRegisterSet(RegSetKind set);

  uint8_t *GetRegisterData(RegSetKind set, const RegisterInfo &reg_info);

  RegisterInfo_arm64 &GetRegisterInfo() const;
};

} // namespace process_openbsd
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVEREGISTERCONTEXTOPENBSD_ARM64_H

#endif // defined(__aarch64__)
