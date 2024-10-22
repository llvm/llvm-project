//===-- RegisterInfoPOSIX_riscv64.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERINFOPOSIX_RISCV64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERINFOPOSIX_RISCV64_H

#include "RegisterInfoAndSetInterface.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/Flags.h"
#include "lldb/lldb-private.h"
#include <map>

class RegisterInfoPOSIX_riscv64
    : public lldb_private::RegisterInfoAndSetInterface {
public:
  enum { GPRegSet = 0 };

  // RISC-V64 register set mask value
  enum {
    eRegsetMaskDefault = 0,
    eRegsetMaskFP = 1,
    eRegsetMaskAll = -1,
  };

  struct GPR {
    // note: gpr[0] is pc, not x0
    uint64_t gpr[32];
  };

  struct FPR {
    uint64_t fpr[32];
    uint32_t fcsr;
  };

  struct VPR {
    // The size should be VLEN*32 in bits, but we don't have VLEN here.
    void *vpr;
  };

  RegisterInfoPOSIX_riscv64(const lldb_private::ArchSpec &target_arch,
                            lldb_private::Flags opt_regsets);

  void AddRegSetGP();

  void AddRegSetFP();

  size_t GetGPRSize() const override;

  size_t GetFPRSize() const override;

  const lldb_private::RegisterInfo *GetRegisterInfo() const override;

  uint32_t GetRegisterCount() const override;

  const lldb_private::RegisterSet *
  GetRegisterSet(size_t reg_set) const override;

  size_t GetRegisterSetCount() const override;

  size_t GetRegisterSetFromRegisterIndex(uint32_t reg_index) const override;

  bool IsFPPresent() const { return m_opt_regsets.AnySet(eRegsetMaskFP); }

  bool IsFPReg(unsigned reg) const;

private:
  std::vector<lldb_private::RegisterInfo> m_register_infos;

  std::vector<lldb_private::RegisterSet> m_register_sets;

  // Contains pair of [start, end] register numbers of a register set with start
  // and end included.
  std::map<uint32_t, std::pair<uint32_t, uint32_t>> m_per_regset_regnum_range;

  // Register collections to be stored as reference for m_register_sets items
  std::vector<uint32_t> m_fp_regnum_collection;

  lldb_private::Flags m_opt_regsets;
};

#endif
