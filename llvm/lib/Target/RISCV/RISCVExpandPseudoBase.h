//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the base class used by the several RISC-V Pseudo
// Instruction Expansion passes. This avoids having to re-implement some of the
// boilerplate needed in these passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVEXPANDPSEUDOBASE_H
#define LLVM_LIB_TARGET_RISCV_RISCVEXPANDPSEUDOBASE_H

#include "RISCV.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class RISCVSubtarget;
class RISCVInstrInfo;

class RISCVExpandPseudoImplBase {
public:
  bool run(MachineFunction &MF);

  virtual ~RISCVExpandPseudoImplBase() = default;

protected:
  const RISCVSubtarget *STI;
  const RISCVInstrInfo *TII;

  virtual bool expandMI(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator MBBI,
                        MachineBasicBlock::iterator &NextMBBI) const;

private:
  bool expandMBB(MachineBasicBlock &MBB) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVEXPANDPSEUDOBASE_H
