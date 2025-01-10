#include "X86.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "X86MatchJumptablePass.h"

#define DEBUG_TYPE "x86-my-pass"

using namespace llvm;

namespace {
  class X86MatchJumptablePass : public MachineFunctionPass {
  public:
    static char ID;

    X86MatchJumptablePass() : MachineFunctionPass(ID) {}

    bool runOnMachineFunction(MachineFunction &MF) override {
      LLVM_DEBUG(dbgs() << "Running X86MyBackendPass on function: "
                        << MF.getName() << "\n");

      // Example: Iterate through instructions
      for (auto &MBB : MF) {
        for (auto &MI : MBB) {
          // Process instructions here
          LLVM_DEBUG(dbgs() << "Instruction: " << MI << "\n");
        }
      }

      return false; // Return true if the pass modifies the function
    }

    StringRef getPassName() const override {
      return "X86 My Backend Pass";
    }
  };
}

char X86MatchJumptablePass::ID = 0;

// Ensure the function is in the llvm namespace
namespace llvm {
  
  // Define the pass
  FunctionPass *createX86MatchJumptablePass() {
    return new X86MatchJumptablePass();
  }

} // end llvm namespace
