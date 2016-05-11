//===--- SIDebuggerInsertNops.cpp - Inserts nops for debugger usage -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Inserts two nop instructions for each high level source statement for
/// debugger usage.
///
/// Tools, such as debugger, need to pause execution based on user input (i.e.
/// breakpoint). In order to do this, two nop instructions are inserted for each
/// high level source statement: one before first isa instruction of high level
/// source statement, and one after last isa instruction of high level source
/// statement. Further, debugger may replace nop instructions with trap
/// instructions based on user input.
//
//===----------------------------------------------------------------------===//

#include "SIInstrInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
using namespace llvm;

#define DEBUG_TYPE "si-debugger-insert-nops"
#define PASS_NAME "SI Debugger Insert Nops"

namespace {

class SIDebuggerInsertNops : public MachineFunctionPass {
public:
  static char ID;

  SIDebuggerInsertNops() : MachineFunctionPass(ID) { }
  const char *getPassName() const override { return PASS_NAME; }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // anonymous namespace

INITIALIZE_PASS(SIDebuggerInsertNops, DEBUG_TYPE, PASS_NAME, false, false)

char SIDebuggerInsertNops::ID = 0;
char &llvm::SIDebuggerInsertNopsID = SIDebuggerInsertNops::ID;

FunctionPass *llvm::createSIDebuggerInsertNopsPass() {
  return new SIDebuggerInsertNops();
}

bool SIDebuggerInsertNops::runOnMachineFunction(MachineFunction &MF) {
  // Skip this pass if "amdgpu-debugger-insert-nops" attribute was not
  // specified.
  const AMDGPUSubtarget &ST = MF.getSubtarget<AMDGPUSubtarget>();
  if (!ST.debuggerInsertNops())
    return false;

  // Skip machine functions without debug info.
  if (!MF.getMMI().hasDebugInfo())
    return false;

  // Target instruction info.
  const SIInstrInfo *TII =
    static_cast<const SIInstrInfo*>(MF.getSubtarget().getInstrInfo());

  // Mapping from high level source statement line number to last corresponding
  // isa instruction.
  DenseMap<unsigned, MachineBasicBlock::iterator> LineToInst;
  // Insert nop instruction before first isa instruction of each high level
  // source statement and collect last isa instruction for each high level
  // source statement.
  for (auto &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
      if (MI->isDebugValue() || !MI->getDebugLoc())
        continue;

      auto DL = MI->getDebugLoc();
      auto CL = DL.getLine();
      auto LineToInstEntry = LineToInst.find(CL);
      if (LineToInstEntry == LineToInst.end()) {
        BuildMI(MBB, *MI, DL, TII->get(AMDGPU::S_NOP))
          .addImm(0);
        LineToInst.insert(std::make_pair(CL, MI));
      } else {
        LineToInstEntry->second = MI;
      }
    }
  }
  // Insert nop instruction after last isa instruction of each high level source
  // statement.
  for (auto const &LineToInstEntry : LineToInst) {
    auto MBB = LineToInstEntry.second->getParent();
    auto DL = LineToInstEntry.second->getDebugLoc();
    MachineBasicBlock::iterator MI = LineToInstEntry.second;
    if (MI->getOpcode() != AMDGPU::S_ENDPGM)
      BuildMI(*MBB, *(++MI), DL, TII->get(AMDGPU::S_NOP))
        .addImm(0);
  }
  // Insert nop instruction before prologue.
  MachineBasicBlock &MBB = MF.front();
  MachineInstr &MI = MBB.front();
  BuildMI(MBB, MI, DebugLoc(), TII->get(AMDGPU::S_NOP))
    .addImm(0);

  return true;
}
