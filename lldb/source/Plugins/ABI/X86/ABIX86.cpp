//===-- ABIX86.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIMacOSX_i386.h"
#include "ABISysV_i386.h"
#include "ABISysV_x86_64.h"
#include "ABIWindows_x86_64.h"
#include "ABIX86.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ABIX86)

void ABIX86::Initialize() {
  ABIMacOSX_i386::Initialize();
  ABISysV_i386::Initialize();
  ABISysV_x86_64::Initialize();
  ABIWindows_x86_64::Initialize();
}

void ABIX86::Terminate() {
  ABIMacOSX_i386::Terminate();
  ABISysV_i386::Terminate();
  ABISysV_x86_64::Terminate();
  ABIWindows_x86_64::Terminate();
}

enum class RegKind {
  GPR32 = 0,
  GPR16,
  GPR8h,
  GPR8,

  MM = 0,
};

typedef llvm::SmallDenseMap<llvm::StringRef,
                            llvm::SmallVector<llvm::StringRef, 4>, 16>
    RegisterMap;

static void addPartialRegisters(
    std::vector<DynamicRegisterInfo::Register> &regs,
    llvm::ArrayRef<uint32_t> base_reg_indices, const RegisterMap &reg_names,
    uint32_t base_size, RegKind name_index, lldb::Encoding encoding,
    lldb::Format format, uint32_t subreg_size, uint32_t subreg_offset = 0) {
  for (uint32_t base_index : base_reg_indices) {
    if (base_index == LLDB_INVALID_REGNUM)
      break;
    assert(base_index < regs.size());
    DynamicRegisterInfo::Register &full_reg = regs[base_index];
    llvm::StringRef subreg_name = reg_names.lookup(
        full_reg.name.GetStringRef())[static_cast<int>(name_index)];
    if (subreg_name.empty() || full_reg.byte_size != base_size)
      continue;

    lldb_private::DynamicRegisterInfo::Register subreg{
        lldb_private::ConstString(subreg_name),
        lldb_private::ConstString(),
        lldb_private::ConstString("supplementary registers"),
        subreg_size,
        LLDB_INVALID_INDEX32,
        encoding,
        format,
        LLDB_INVALID_REGNUM,
        LLDB_INVALID_REGNUM,
        LLDB_INVALID_REGNUM,
        LLDB_INVALID_REGNUM,
        {base_index},
        {},
        subreg_offset};

    addSupplementaryRegister(regs, subreg);
  }
}

void ABIX86::AugmentRegisterInfo(
    std::vector<DynamicRegisterInfo::Register> &regs) {
  MCBasedABI::AugmentRegisterInfo(regs);

  ProcessSP process_sp = GetProcessSP();
  if (!process_sp)
    return;

  uint32_t gpr_base_size =
      process_sp->GetTarget().GetArchitecture().GetAddressByteSize();
  bool is64bit = gpr_base_size == 8;

  typedef RegisterMap::value_type RegPair;
#define GPR_BASE(basename) (is64bit ? "r" basename : "e" basename)
  RegisterMap gpr_regs{{
      RegPair(GPR_BASE("ax"), {"eax", "ax", "ah", "al"}),
      RegPair(GPR_BASE("bx"), {"ebx", "bx", "bh", "bl"}),
      RegPair(GPR_BASE("cx"), {"ecx", "cx", "ch", "cl"}),
      RegPair(GPR_BASE("dx"), {"edx", "dx", "dh", "dl"}),
      RegPair(GPR_BASE("si"), {"esi", "si", "", "sil"}),
      RegPair(GPR_BASE("di"), {"edi", "di", "", "dil"}),
      RegPair(GPR_BASE("bp"), {"ebp", "bp", "", "bpl"}),
      RegPair(GPR_BASE("sp"), {"esp", "sp", "", "spl"}),
  }};
#undef GPR_BASE
  if (is64bit) {
#define R(base) RegPair(base, {base "d", base "w", "", base "l"})
    RegisterMap amd64_regs{{
        R("r8"),
        R("r9"),
        R("r10"),
        R("r11"),
        R("r12"),
        R("r13"),
        R("r14"),
        R("r15"),
    }};
#undef R
    gpr_regs.insert(amd64_regs.begin(), amd64_regs.end());
  }

  RegisterMap st_regs{{
      RegPair("st0", {"mm0"}),
      RegPair("st1", {"mm1"}),
      RegPair("st2", {"mm2"}),
      RegPair("st3", {"mm3"}),
      RegPair("st4", {"mm4"}),
      RegPair("st5", {"mm5"}),
      RegPair("st6", {"mm6"}),
      RegPair("st7", {"mm7"}),
  }};

  // regs from gpr_basenames, in list order
  std::vector<uint32_t> gpr_base_reg_indices;
  // st0..st7, in list order
  std::vector<uint32_t> st_reg_indices;
  // map used for fast register lookups
  llvm::SmallDenseSet<llvm::StringRef, 64> subreg_name_set;

  // put all subreg names into the lookup set
  for (const RegisterMap &regset : {gpr_regs, st_regs}) {
    for (const RegPair &kv : regset)
      subreg_name_set.insert(kv.second.begin(), kv.second.end());
  }

  for (const auto &x : llvm::enumerate(regs)) {
    llvm::StringRef reg_name = x.value().name.GetStringRef();
    // find expected base registers
    if (gpr_regs.find(reg_name) != gpr_regs.end())
      gpr_base_reg_indices.push_back(x.index());
    else if (st_regs.find(reg_name) != st_regs.end())
      st_reg_indices.push_back(x.index());
    // abort if at least one sub-register is already present
    else if (llvm::is_contained(subreg_name_set, reg_name))
      return;
  }

  if (is64bit)
    addPartialRegisters(regs, gpr_base_reg_indices, gpr_regs, gpr_base_size,
                        RegKind::GPR32, eEncodingUint, eFormatHex, 4);
  addPartialRegisters(regs, gpr_base_reg_indices, gpr_regs, gpr_base_size,
                      RegKind::GPR16, eEncodingUint, eFormatHex, 2);
  addPartialRegisters(regs, gpr_base_reg_indices, gpr_regs, gpr_base_size,
                      RegKind::GPR8h, eEncodingUint, eFormatHex, 1, 1);
  addPartialRegisters(regs, gpr_base_reg_indices, gpr_regs, gpr_base_size,
                      RegKind::GPR8, eEncodingUint, eFormatHex, 1);

  addPartialRegisters(regs, st_reg_indices, st_regs, 10, RegKind::MM,
                      eEncodingUint, eFormatHex, 8);
}
