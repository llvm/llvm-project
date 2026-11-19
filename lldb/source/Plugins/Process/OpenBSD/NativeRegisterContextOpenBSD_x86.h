//===-- NativeRegisterContextOpenBSD_x86.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__i386__) || defined(__x86_64__)

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVEREGISTERCONTEXTOPENBSD_X86_H
#define LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVEREGISTERCONTEXTOPENBSD_X86_H

// clang-format off
#include <sys/types.h>
#include <machine/reg.h>
// clang-format on

#include <array>
#include <optional>
#include <vector>

#include "Plugins/Process/OpenBSD/NativeRegisterContextOpenBSD.h"
#include "Register/Arch/RegisterContext_x86.h"
#include "Register/Arch/lldb-x86-register-enums.h"

namespace lldb_private {
namespace process_openbsd {

class NativeProcessOpenBSD;

class NativeRegisterContextOpenBSD_x86 : public NativeRegisterContextOpenBSD {
public:
  NativeRegisterContextOpenBSD_x86(const ArchSpec &target_arch,
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
  void *GetGPRBuffer() override { return m_gpr.data(); }
  size_t GetGPRSize() override { return m_gpr.size(); }
  void *GetFPRBuffer() override { return m_fpr.data(); }
  size_t GetFPRSize() override { return m_fpr.size(); }

  Status DoReadFPR(void *buf) override;
  Status DoWriteFPR(void *buf) override;

private:
  // Private member types.
  enum RegSetKind {
    GPRegSet,
    FPRegSet,
    YMMRegSet,
    MaxRegSet = YMMRegSet,
  };

  // Private member variables.
  std::array<uint8_t, sizeof(struct reg)> m_gpr;
  std::array<uint8_t, sizeof(FXSAVE)> m_fpr;
  std::vector<uint8_t> m_xsave;
  std::array<uint32_t, MaxRegSet + 1> m_xsave_offsets;
  std::array<size_t, MaxRegSet + 1> m_regset_offsets;

  std::optional<RegSetKind> GetSetForNativeRegNum(uint32_t reg_num) const;

  Status ReadRegisterSet(RegSetKind set);
  Status WriteRegisterSet(RegSetKind set);

  uint8_t *GetOffsetRegSetData(RegSetKind set, size_t reg_offset);

  struct YMMSplitPtr {
    void *xmm;
    void *ymm_hi;
  };
  std::optional<YMMSplitPtr> GetYMMSplitReg(uint32_t reg);
};

} // namespace process_openbsd
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVEREGISTERCONTEXTOPENBSD_X86_H

#endif // defined(__i386__) || defined(__x86_64__)
