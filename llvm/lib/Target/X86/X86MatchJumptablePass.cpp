// #include "X86.h"
// #include "llvm/CodeGen/MachineFunctionPass.h"
// #include "llvm/CodeGen/MachineInstr.h"
// #include "llvm/CodeGen/MachineRegisterInfo.h"
// #include "llvm/Support/Debug.h"
// #include "llvm/Support/raw_ostream.h"
// #include "X86MatchJumptablePass.h"
// #include "llvm/CodeGen/MachineJumpTableInfo.h"

// #define DEBUG_TYPE "match-jump-table"

// using namespace llvm;

// namespace {
//   class X86MatchJumptablePass : public MachineFunctionPass {
//   public:
//     static char ID;

//     X86MatchJumptablePass() : MachineFunctionPass(ID) {}

//     bool runOnMachineFunction(MachineFunction &MF) override {
//     LLVM_DEBUG(dbgs() << "Analyzing jump tables in function: " << MF.getName() << "\n");

//     // Get jump table information
//     MachineJumpTableInfo *JumpTableInfo = MF.getJumpTableInfo();
//     if (!JumpTableInfo) {
//       LLVM_DEBUG(dbgs() << "No jump tables in this function.\n");
//       return false;
//     }
//     // Assuming JumpTableInfo is available
//     for (unsigned JTIndex = 0; JTIndex < JumpTableInfo->getJumpTables().size(); ++JTIndex) {
//   const MachineJumpTableEntry &JTEntry = JumpTableInfo->getJumpTables()[JTIndex];
  
//   LLVM_DEBUG(dbgs() << "Jump Table #" << JTIndex << " Base Address: " << JTEntry.BaseAddress << "\n");

//   // Iterate through the entries (target basic blocks) in this jump table
//   for (auto *MBB : JTEntry.MBBs) {
//     LLVM_DEBUG(dbgs() << "  Target BasicBlock: " << MBB->getName() << " Address: " << MBB->getAddress() << "\n");
//      }
    

//       // Trace potential indirect jumps related to this jump table
//       traceIndirectJumps(MF, JTIndex, JumpTableInfo);
//     }
//     return false;

// }

//   void traceIndirectJumps(MachineFunction &MF, unsigned JTIndex, MachineJumpTableInfo *JumpTableInfo) {
//     const MachineJumpTableEntry &JTEntry = JumpTableInfo->getJumpTables()[JTIndex];

//     for (auto &MBB : MF) {
//       for (auto &MI : MBB) {
//         if (MI.isIndirectBranch()) {
//           LLVM_DEBUG(dbgs() << "Found indirect jump: " << MI << "\n");

//           // Analyze data flow to check if this jump is related to the jump table
//           if (isJumpTableRelated(MI, JTEntry, MF)) {
//             LLVM_DEBUG(dbgs() << "This indirect jump is related to Jump Table #" << JTIndex << "\n");
//           }
//         }
//       }
//     }
//   }

//   bool isJumpTableRelated(MachineInstr &MI, const MachineJumpTableEntry &JTEntry, MachineFunction &MF) {
//   for (unsigned OpIdx = 0; OpIdx < MI.getNumOperands(); ++OpIdx) {
//     const MachineOperand &Op = MI.getOperand(OpIdx);

//     if (Op.isReg()) {
//       Register Reg = Op.getReg();
//       MachineRegisterInfo &MRI = MF.getRegInfo();
//       // Check if any of the definitions of the register are related to a jump table load
//       for (MachineInstr &DefMI : MRI.def_instructions(Reg)) {
//         if (isJumpTableLoad(DefMI, JTEntry)) {
//           return true;
//         }
//       }
//     } else if (Op.isImm()) {
//       // Check if the immediate operand might be an offset/index into the jump table
//       int64_t ImmValue = Op.getImm();
      
//       // For example, if the jump table has 10 entries, check if the immediate is between 0 and 9
//       if (ImmValue >= 0 && ImmValue < JTEntry.MBBs.size()) {
//         // This immediate value could be an index into the jump table
//         LLVM_DEBUG(dbgs() << "Immediate operand is a possible jump table index: " << ImmValue << "\n");
//         return true;
//       }
//     }
//   }
//   return false;
// }

// bool isJumpTableLoad(MachineInstr &MI, const MachineJumpTableEntry &JTEntry) {
//     if (MI.mayLoad()) {
//         for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
//             const MachineOperand &Op = MI.getOperand(i);
//             if (Op.isGlobal() && Op.getGlobal() == JTEntry.BaseAddress) {
//                 return true;
//             }
//         }
//     }
//     return false;
// }

//   StringRef getPassName() const override {
//     return "Match Jump Table Pass";
//   }

//     // StringRef getPassName() const override {
//     //   return "X86 My Backend Pass";
//     // }
//   };
// }

// char X86MatchJumptablePass::ID = 0;

// // Ensure the function is in the llvm namespace
// namespace llvm {
  
//   // Define the pass
//   FunctionPass *createX86MatchJumptablePass() {
//     return new X86MatchJumptablePass();
//   }

// } // end llvm namespace

// static RegisterPass<X86MatchJumptablePass> X("match-jump-table", "Match Jump Table Pass", false, false);


#include "X86.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"

#define DEBUG_TYPE "match-jump-table"

using namespace llvm;

namespace {
class X86MatchJumptablePass : public MachineFunctionPass {
public:
  static char ID;

  X86MatchJumptablePass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    LLVM_DEBUG(dbgs() << "Analyzing jump tables in function: " << MF.getName() << "\n");

    // Get jump table information
    MachineJumpTableInfo *JumpTableInfo = MF.getJumpTableInfo();
    if (!JumpTableInfo) {
      LLVM_DEBUG(dbgs() << "No jump tables in this function.\n");
      return false;
    }

    for (unsigned JTIndex = 0; JTIndex < JumpTableInfo->getJumpTables().size(); ++JTIndex) {
      const MachineJumpTableEntry &JTEntry = JumpTableInfo->getJumpTables()[JTIndex];

      LLVM_DEBUG(dbgs() << "Jump Table #" << JTIndex << " contains " << JTEntry.MBBs.size()
                        << " entries.\n");

      // Iterate through the entries (target basic blocks) in this jump table
      for (auto *MBB : JTEntry.MBBs) {
        if (MBB) {
          LLVM_DEBUG(dbgs() << "  Target BasicBlock: " << MBB->getName() << "\n");
        }
      }

      // Assuming you have access to MF, JTIndex, and JumpTableInfo
    MachineInstr* indirectJumpInstr = traceIndirectJumps(MF, JTIndex, JumpTableInfo);

      if (indirectJumpInstr) {
          // Handle the found indirect jump instruction
          dbgs() << "Indirect jump found at address: " << indirectJumpInstr << "\n";
          for (auto &MBB : JTEntry.MBBs) {
          LLVM_DEBUG(dbgs() << "Address of MBB: " << &MBB << "\n"); // Print the address of the MBB
          // Optionally print the name and instructions inside the MBB
         // LLVM_DEBUG(dbgs() << "MBB: " << MBB.getName() << "\n");
          // for (auto &MI : MBB) {
          //     LLVM_DEBUG(dbgs() << "  Instruction: " << MI << "\n");
          // }
        }
      } 
    }

    return false;
  }

  // void traceIndirectJumps(MachineFunction &MF, unsigned JTIndex, MachineJumpTableInfo *JumpTableInfo) {
  //   const MachineJumpTableEntry &JTEntry = JumpTableInfo->getJumpTables()[JTIndex];

  //   for (auto &MBB : MF) {
  //     for (auto &MI : MBB) {
  //       if (MI.isIndirectBranch()) {
  //         LLVM_DEBUG(dbgs() << "Found indirect jump: " << MI << "\n");

  //         // Analyze data flow to check if this jump is related to the jump table
  //         if (isJumpTableRelated(MI, JTEntry, MF)) {
  //           LLVM_DEBUG(dbgs() << "This indirect jump is related to Jump Table #" << JTIndex << "\n");
  //         }
  //       }
  //     }
  //   }
  // }
  MachineInstr* traceIndirectJumps(MachineFunction &MF, unsigned JTIndex, MachineJumpTableInfo *JumpTableInfo) {
    const MachineJumpTableEntry &JTEntry = JumpTableInfo->getJumpTables()[JTIndex];

    for (auto &MBB : MF) {
        for (auto &MI : MBB) {
            if (MI.isIndirectBranch()) {
                LLVM_DEBUG(dbgs() << "Found indirect jump: " << MI << "\n");

                // Analyze data flow to check if this jump is related to the jump table
                if (isJumpTableRelated(MI, JTEntry, MF)) {
                    LLVM_DEBUG(dbgs() << "This indirect jump is related to Jump Table #" << JTIndex << "\n");
                    
                    // Return the address of the indirect jump (MI)
                    return &MI;
                }
            }
        }
    }

    // Return nullptr if no indirect jump is found
    return nullptr;
}

  bool isJumpTableRelated(MachineInstr &MI, const MachineJumpTableEntry &JTEntry, MachineFunction &MF) {
    for (unsigned OpIdx = 0; OpIdx < MI.getNumOperands(); ++OpIdx) {
      const MachineOperand &Op = MI.getOperand(OpIdx);

      if (Op.isReg()) {
        Register Reg = Op.getReg();
        MachineRegisterInfo &MRI = MF.getRegInfo();
        // Check if any of the definitions of the register are related to a jump table load
        for (MachineInstr &DefMI : MRI.def_instructions(Reg)) {
          if (isJumpTableLoad(DefMI, JTEntry)) {
            return true;
          }
        }
      } else if (Op.isImm()) {
        // Check if the immediate operand might be an offset/index into the jump table
        int64_t ImmValue = Op.getImm();
        if (ImmValue >= 0 && ImmValue < JTEntry.MBBs.size()) {
          // This immediate value could be an index into the jump table
          LLVM_DEBUG(dbgs() << "Immediate operand is a possible jump table index: " << ImmValue
                            << "\n");
          return true;
        }
      }
    }
    return false;
  }

  bool isJumpTableLoad(MachineInstr &MI, const MachineJumpTableEntry &JTEntry) {
    if (MI.mayLoad()) {
      for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
        const MachineOperand &Op = MI.getOperand(i);
          if (Op.getType() == MachineOperand::MO_JumpTableIndex) {
          unsigned JTIndex = Op.getIndex();
          if (JTIndex < JTEntry.MBBs.size()) {
            LLVM_DEBUG(dbgs() << "Instruction loads from Jump Table index: " << JTIndex << "\n");
            return true;
          }
        }
      }
    }
    return false;
  }

  StringRef getPassName() const override { return "Match Jump Table Pass"; }
};

} // namespace

char X86MatchJumptablePass::ID = 0;

namespace llvm {

// Define the pass
FunctionPass *createX86MatchJumptablePass() { return new X86MatchJumptablePass(); }

} // namespace llvm

static RegisterPass<X86MatchJumptablePass> X("match-jump-table", "Match Jump Table Pass", false,
                                             false);
