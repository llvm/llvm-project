//===----------------------------------------------------------------------===//
// Instruction Selector Subtarget Control
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// This file defines a pass used to change the subtarget for the
// Mips Instruction selector.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsTargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "mips-isel"

namespace {
  class MipsModuleDAGToDAGISel : public MachineFunctionPass {
  public:
    static char ID;

    explicit MipsModuleDAGToDAGISel(MipsTargetMachine &TM_)
      : MachineFunctionPass(ID), TM(TM_) {}

    // Pass Name
    StringRef getPassName() const override {
      return "MIPS DAG->DAG Pattern Instruction Selection";
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

  protected:
    MipsTargetMachine &TM;
  };

  char MipsModuleDAGToDAGISel::ID = 0;
}

bool MipsModuleDAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(errs() << "In MipsModuleDAGToDAGISel::runMachineFunction\n");
  TM.resetSubtarget(&MF);
  return false;
}

llvm::FunctionPass *llvm::createMipsModuleISelDagPass(MipsTargetMachine &TM) {
  return new MipsModuleDAGToDAGISel(TM);
}
