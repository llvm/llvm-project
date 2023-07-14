#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

/* namespace llvm */
namespace llvm {
  FunctionPass *createX86BasicBlockTraverse();
  void initializeX86BasicBlockTraversePass(PassRegistry &);
}

/* namespace anonymous */
namespace {
  struct X86BasicBlockTraverse : public MachineFunctionPass {
    static char ID; // Pass identification, replacement for typeid

    X86BasicBlockTraverse() : MachineFunctionPass(ID) {
      initializeX86BasicBlockTraversePass(*PassRegistry::getPassRegistry());
    }
    StringRef getPassName() const override { return "basic block traverse pass"; }

    virtual bool runOnMachineFunction(MachineFunction &MF) override;
  };
}

char X86BasicBlockTraverse::ID = 0;
INITIALIZE_PASS_BEGIN(X86BasicBlockTraverse, "basic block traverse", "basic block traverse pass", true, true)
INITIALIZE_PASS_END(X86BasicBlockTraverse, "basic block traverse", "basic block traverse pass", true, true)

FunctionPass *llvm::createX86BasicBlockTraverse() { return new X86BasicBlockTraverse(); }

bool X86BasicBlockTraverse::runOnMachineFunction(MachineFunction &MF) {
  for (auto &MBB : MF) {
        outs() << "Contents of MachineBasicBlock:\n";
        outs() << MBB << "\n";
        const BasicBlock *BB = MBB.getBasicBlock();
        outs() << "Contents of BasicBlock corresponding to MachineBasicBlock:\n";
        outs() << BB << "\n";
    }
    return false;
}