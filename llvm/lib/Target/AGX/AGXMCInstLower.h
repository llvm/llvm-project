//===-- AGXMCInstLower.h - Lower MachineInstr to MCInst ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGXMCINSTLOWER_H
#define LLVM_LIB_TARGET_AGX_AGXMCINSTLOWER_H

namespace llvm {

class AGXSubtarget;
class MCContext;
class MCInst;
class MCOperand;
class MachineInstr;
class MachineOperand;

/// Lowers code generation pseudo instructions to the real instructions of
/// the subtarget's encoding family.
class AGXMCInstLower {
  MCContext &Ctx;
  const AGXSubtarget &ST;

public:
  AGXMCInstLower(MCContext &Ctx, const AGXSubtarget &ST) : Ctx(Ctx), ST(ST) {}

  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp) const;

  /// Lower \p MI. Returns false if the instruction has no encoding for the
  /// subtarget.
  bool lower(const MachineInstr *MI, MCInst &OutMI) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGXMCINSTLOWER_H
