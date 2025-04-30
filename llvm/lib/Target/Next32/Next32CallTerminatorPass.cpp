//===-- Next32CallTerminatorPass.cpp - use or replace LEA instructions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass that finds Next32::CALL_TERMINATOR instructions
// that are mark as dead (Can happened when the one of the return values are
// not in used), mark it as alive and reallocate new physical register that is
// not in used by other feeders.
//
//===----------------------------------------------------------------------===//

#include "Next32.h"
#include "Next32InstrInfo.h"
#include "Next32PassTrace.h"
#include "Next32Subtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
using namespace llvm;

namespace llvm {
void initializeNext32CallTerminatorPassPass(PassRegistry &);
}

#define NEXT32CALLTERMINATOR_DESC "Next32 CallTerminator Fixups"
#define NEXT32CALLTERMINATOR_NAME "Next32-call-terminator"

#define DEBUG_TYPE NEXT32CALLTERMINATOR_NAME

namespace {
class Next32CallTerminatorPass : public MachineFunctionPass {

public:
  static char ID;

  StringRef getPassName() const override { return NEXT32CALLTERMINATOR_DESC; }

  Next32CallTerminatorPass() : MachineFunctionPass(ID) {
    initializeNext32CallTerminatorPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &Func) override;

private:
  RegisterClassInfo RCI;
  void CalculatePhysicalRegistersMap(MachineBasicBlock &MBB,
                                     DenseMap<unsigned, bool> &Physicals);
  bool ProcessBasicBlock(MachineBasicBlock &MBB);
};
} // namespace

char Next32CallTerminatorPass::ID = 0;

INITIALIZE_PASS(Next32CallTerminatorPass, NEXT32CALLTERMINATOR_NAME,
                NEXT32CALLTERMINATOR_DESC, false, false)

FunctionPass *llvm::createNext32CallTerminators() {
  return new Next32CallTerminatorPass();
}

bool Next32CallTerminatorPass::runOnMachineFunction(MachineFunction &Func) {
  Next32PassTrace TFunc(DEBUG_TYPE, Func);
  bool Changed = false;
  RCI.runOnMachineFunction(Func);
  for (auto &MBB : TFunc)
    if (!MBB.empty())
      Changed |= ProcessBasicBlock(MBB);
  return Changed;
}

void Next32CallTerminatorPass::CalculatePhysicalRegistersMap(
    MachineBasicBlock &MBB, DenseMap<unsigned, bool> &Physicals) {
  // Go over all the live-ins of a function, including all return values
  // and mark the physical register as used
  for (auto &LiveIn : MBB.liveins())
    Physicals.insert(std::make_pair(LiveIn.PhysReg, true));

  for (auto &I : MBB) {
    if (I.getOpcode() == Next32::CALL_TERMINATOR) {
      for (unsigned int i = 0; i < I.getNumDefs(); i++) {
        if (!I.getOperand(i).isDead())
          Physicals.insert(std::make_pair(I.getOperand(i).getReg(), true));
      }
    }
    if (I.getOpcode() == Next32::MOVL || I.getOpcode() == Next32::DUP)
      Physicals.insert(std::make_pair(I.getOperand(0).getReg(), true));
  }
}

bool Next32CallTerminatorPass::ProcessBasicBlock(MachineBasicBlock &MBB) {
  bool Ret = false;
  // Save all Physical live-ins
  DenseMap<unsigned, bool> Physicals;
  ArrayRef<MCPhysReg> allocationOrder = RCI.getOrder(&Next32::GPR32RegClass);

  CalculatePhysicalRegistersMap(MBB, Physicals);

  for (auto &I : MBB) {
    if (I.getOpcode() != Next32::CALL_TERMINATOR)
      continue;

    for (unsigned int i = 0; i < I.getNumDefs(); i++) {
      if (!I.getOperand(i).isDead())
        continue;

      // Mark that the operand isn't dead
      Ret |= true;
      I.getOperand(i).setIsDead(false);

      // If the current allocated physical register is not used there is no need
      // to allocate a new register
      if (Physicals.find(I.getOperand(i).getReg()) == Physicals.end())
        continue;

      // Assign a new physical register allocation
      MCPhysReg NewReg = 0; // NO_PHYS_REG
      for (auto r : allocationOrder)
        if (Physicals.find(r) == Physicals.end()) {
          NewReg = r;
          break;
        }

      if (NewReg == 0) // NO_PHYS_REG
        report_fatal_error(
            "ran out of registers during next32 terminator pass");

      I.getOperand(i).setReg(NewReg);
    }
  }
  return Ret;
}
