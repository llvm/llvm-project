//===-- RISCVInsertReadWriteCSR.cpp - Insert Read/Write of RISC-V CSR -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements the machine function pass to insert read/write of CSR-s
// of the RISC-V instructions.
//
// Currently the pass implements naive insertion of a write to vxrm before an
// RVV fixed-point instruction.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

#define DEBUG_TYPE "riscv-insert-read-write-csr"
#define RISCV_INSERT_READ_WRITE_CSR_NAME "RISC-V Insert Read/Write CSR Pass"

namespace {

class RISCVInsertReadWriteCSR : public MachineFunctionPass {
  const TargetInstrInfo *TII;

public:
  static char ID;

  RISCVInsertReadWriteCSR() : MachineFunctionPass(ID) {
    initializeRISCVInsertReadWriteCSRPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return RISCV_INSERT_READ_WRITE_CSR_NAME;
  }

private:
  bool emitWriteVXRM(MachineBasicBlock &MBB);
  std::optional<unsigned> getRoundModeIdx(const MachineInstr &MI);
};

} // end anonymous namespace

char RISCVInsertReadWriteCSR::ID = 0;

INITIALIZE_PASS(RISCVInsertReadWriteCSR, DEBUG_TYPE,
                RISCV_INSERT_READ_WRITE_CSR_NAME, false, false)

// This function returns the index to the rounding mode immediate value if any,
// otherwise the function will return None.
std::optional<unsigned>
RISCVInsertReadWriteCSR::getRoundModeIdx(const MachineInstr &MI) {
  uint64_t TSFlags = MI.getDesc().TSFlags;
  if (!RISCVII::hasRoundModeOp(TSFlags))
    return std::nullopt;

  // The operand order
  // -------------------------------------
  // | n-1 (if any)   | n-2  | n-3 | n-4 |
  // | policy         | sew  | vl  | rm  |
  // -------------------------------------
  return MI.getNumExplicitOperands() - RISCVII::hasVecPolicyOp(TSFlags) - 3;
}

// This function inserts a write to vxrm when encountering an RVV fixed-point
// instruction.
bool RISCVInsertReadWriteCSR::emitWriteVXRM(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineInstr &MI : MBB) {
    if (auto RoundModeIdx = getRoundModeIdx(MI)) {
      Changed = true;

      unsigned VXRMImm = MI.getOperand(*RoundModeIdx).getImm();
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(RISCV::WriteVXRMImm))
          .addImm(VXRMImm);
      MI.addOperand(MachineOperand::CreateReg(RISCV::VXRM, /*IsDef*/ false,
                                              /*IsImp*/ true));
    }
  }
  return Changed;
}

bool RISCVInsertReadWriteCSR::runOnMachineFunction(MachineFunction &MF) {
  // Skip if the vector extension is not enabled.
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVInstructions())
    return false;

  TII = ST.getInstrInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF)
    Changed |= emitWriteVXRM(MBB);

  return Changed;
}

FunctionPass *llvm::createRISCVInsertReadWriteCSRPass() {
  return new RISCVInsertReadWriteCSR();
}
