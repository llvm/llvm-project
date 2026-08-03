//===-- PISAUtils.h ---- PISA Utility Functions ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAUTILS_H
#define LLVM_LIB_TARGET_PISA_PISAUTILS_H

#include "MCTargetDesc/PISABaseInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/IR/IRBuilder.h"
#include <string>

namespace llvm {
class MachineInstr;
class MCInst;
class MachineInstrBuilder;
class StringRef;

namespace PISA {
// Similar to getDefIgnoringCopies, but skips any bitcast instructions.
// Return defining instruction (or nil) and index of the (bitcasted) Reg
// within defining instruction.
//
// - when NoVectors is true, stop search if G_BITCAST src/dst is a vector type
std::tuple<MachineInstr *, unsigned>
getDefIgnoringBitcasts(Register Reg, const MachineRegisterInfo &MRI,
                       bool NoVectors = false);

} // namespace PISA
} // namespace llvm
#endif // LLVM_LIB_TARGET_PISA_PISAUTILS_H
