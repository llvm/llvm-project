//===--- SIInsertNopsPass.cpp - Use predicates for control flow -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Insert two S_NOP instructions for every high level source statement.
///
/// Tools, such as debugger, need to pause execution based on user input (i.e.
/// breakpoint). In order to do this, two S_NOP instructions are inserted for
/// each high level source statement: one before first isa instruction of high
/// level source statement, and one after last isa instruction of high level
/// source statement. Further, debugger may replace S_NOP instructions with
/// S_TRAP instructions based on user input.
//
//===----------------------------------------------------------------------===//

#include "SIInstrInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "si-insert-nops"
#define PASS_NAME "SI Insert Nops"

namespace {

class SIInsertNops : public MachineFunctionPass {
public:
  static char ID;

  SIInsertNops() : MachineFunctionPass(ID) { }
  const char *getPassName() const override { return PASS_NAME; }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // anonymous namespace

INITIALIZE_PASS(SIInsertNops, DEBUG_TYPE, PASS_NAME, false, false)

char SIInsertNops::ID = 0;
char &llvm::SIInsertNopsID = SIInsertNops::ID;

FunctionPass *llvm::createSIInsertNopsPass() {
  return new SIInsertNops();
}

bool SIInsertNops::runOnMachineFunction(MachineFunction &MF) {
  const SIInstrInfo *TII =
    static_cast<const SIInstrInfo*>(MF.getSubtarget().getInstrInfo());

  DenseMap<unsigned, MachineBasicBlock::iterator> LineToInst;
  for (auto MBB = MF.begin(); MBB != MF.end(); ++MBB) {
    for (auto MI = MBB->begin(); MI != MBB->end(); ++MI) {
      if (MI->isDebugValue() || !MI->getDebugLoc()) {
        continue;
      }
      auto DL = MI->getDebugLoc();
      auto CL = DL.getLine();
      auto LineToInstEntry = LineToInst.find(CL);
      if (LineToInstEntry == LineToInst.end()) {
        BuildMI(*MBB, *MI, DL, TII->get(AMDGPU::S_NOP))
          .addImm(0);
        LineToInst.insert(std::make_pair(CL, MI));
      } else {
        LineToInstEntry->second = MI;
      }
    }
  }
  for (auto LineToInstEntry = LineToInst.begin();
         LineToInstEntry != LineToInst.end(); ++LineToInstEntry) {
    auto MBB = LineToInstEntry->second->getParent();
    auto DL = LineToInstEntry->second->getDebugLoc();
    MachineBasicBlock::iterator MI = LineToInstEntry->second;
    ++MI;
    if (MI != MBB->end()) {
      BuildMI(*MBB, *MI, DL, TII->get(AMDGPU::S_NOP))
        .addImm(0);
    }
  }
  MachineBasicBlock &MBB = MF.front();
  MachineInstr &MI = MBB.front();
  BuildMI(MBB, MI, DebugLoc(), TII->get(AMDGPU::S_NOP))
    .addImm(0);

  return true;
}
