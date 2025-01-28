//=- LoongArchBaseInfo.h - Top level definitions for LoongArch MC -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone enum definitions and helper function
// definitions for the LoongArch target useful for the compiler back-end and the
// MC libraries.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHBASEINFO_H
#define LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHBASEINFO_H

#include "MCTargetDesc/LoongArchMCTargetDesc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/TargetParser/SubtargetFeature.h"

namespace llvm {

// This namespace holds all of the target specific flags that instruction info
// tracks.
namespace LoongArchII {
enum {
  MO_None,
  MO_CALL,
  MO_CALL_PLT,
  MO_PCREL_HI,
  MO_PCREL_LO,
  MO_PCREL64_LO,
  MO_PCREL64_HI,
  MO_GOT_PC_HI,
  MO_GOT_PC_LO,
  MO_GOT_PC64_LO,
  MO_GOT_PC64_HI,
  MO_LE_HI,
  MO_LE_LO,
  MO_LE64_LO,
  MO_LE64_HI,
  MO_IE_PC_HI,
  MO_IE_PC_LO,
  MO_IE_PC64_LO,
  MO_IE_PC64_HI,
  MO_LD_PC_HI,
  MO_GD_PC_HI,
  MO_CALL36,
  MO_DESC_PC_HI,
  MO_DESC_PC_LO,
  MO_DESC64_PC_HI,
  MO_DESC64_PC_LO,
  MO_DESC_LD,
  MO_DESC_CALL,
  MO_LE_HI_R,
  MO_LE_ADD_R,
  MO_LE_LO_R,
  // TODO: Add more flags.

  // Used to differentiate between target-specific "direct" flags and "bitmask"
  // flags. A machine operand can only have one "direct" flag, but can have
  // multiple "bitmask" flags.
  MO_DIRECT_FLAG_MASK = 0x3f,

  MO_RELAX = 0x40
};

// Given a MachineOperand that may carry out "bitmask" flags, such as MO_RELAX,
// return LoongArch target-specific "direct" flags.
static inline unsigned getDirectFlags(const MachineOperand &MO) {
  return MO.getTargetFlags() & MO_DIRECT_FLAG_MASK;
}

// Add MO_RELAX "bitmask" flag when FeatureRelax is enabled.
static inline unsigned encodeFlags(unsigned Flags, bool Relax) {
  return Flags | (Relax ? MO_RELAX : 0);
}

// \returns true if the given MachineOperand has MO_RELAX "bitmask" flag.
static inline bool hasRelaxFlag(const MachineOperand &MO) {
  return MO.getTargetFlags() & MO_RELAX;
}

// Target-specific flags of LAInst.
// All definitions must match LoongArchInstrFormats.td.
enum {
  // Whether the instruction's rd is normally required to differ from rj and
  // rk, in the way the 3-register atomic memory operations behave
  // (Section 2.2.7.1 and 2.2.7.2, LoongArch Reference Manual Volume 1 v1.10;
  // while Section 2.2.7.3 lacked similar description for the AMCAS
  // instructions, at least the INE exception is still signaled on Loongson
  // 3A6000 when its rd == rj).
  //
  // Used for generating diagnostics for assembler input that violate the
  // constraint. As described on the manual, the covered instructions require
  // rd != rj && rd != rk to work as intended.
  IsSubjectToAMORdConstraintShift = 0,
  IsSubjectToAMORdConstraintMask = 1 << IsSubjectToAMORdConstraintShift,

  // Whether the instruction belongs to the AMCAS family.
  IsAMCASShift = IsSubjectToAMORdConstraintShift + 1,
  IsAMCASMask = 1 << IsAMCASShift,
};

/// \returns true if this instruction's rd is normally required to differ
/// from rj and rk, in the way 3-register atomic memory operations behave.
static inline bool isSubjectToAMORdConstraint(uint64_t TSFlags) {
  return TSFlags & IsSubjectToAMORdConstraintMask;
}

/// \returns true if this instruction belongs to the AMCAS family.
static inline bool isAMCAS(uint64_t TSFlags) { return TSFlags & IsAMCASMask; }
} // end namespace LoongArchII

namespace LoongArchABI {
enum ABI {
  ABI_ILP32S,
  ABI_ILP32F,
  ABI_ILP32D,
  ABI_LP64S,
  ABI_LP64F,
  ABI_LP64D,
  ABI_Unknown
};

ABI computeTargetABI(const Triple &TT, const FeatureBitset &FeatureBits,
                     StringRef ABIName);
ABI getTargetABI(StringRef ABIName);

// Returns the register used to hold the stack pointer after realignment.
MCRegister getBPReg();
} // end namespace LoongArchABI

} // end namespace llvm

#endif // LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHBASEINFO_H
