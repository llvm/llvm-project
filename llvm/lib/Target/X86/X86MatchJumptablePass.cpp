// X86MatchJumptablePass.cpp
#include "X86MatchJumptablePass.h"
#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/PassRegistry.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "match-jump-table"

namespace llvm {

char X86MatchJumptablePass::ID = 0;

void initializeX86MatchJumptablePassPass(PassRegistry &Registry) {
  RegisterPass<X86MatchJumptablePass> X("match-jump-table", "Match Jump Table Pass", false, false);
}

void X86MatchJumptablePass::insertIdentifyingMarker(MachineInstr* MI, 
                                                   MachineFunction &MF, 
                                                   unsigned JTIndex) {
  // Create a unique identifier for this indirect jump
  std::string marker = "IJUMP_" + MF.getName().str() + "_" + std::to_string(JTIndex);
  
  // Get target instruction info
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  
  // Insert a special no-op instruction before the indirect jump with debug info
  BuildMI(*MI->getParent(), MI, MI->getDebugLoc(), 
          TII->get(X86::NOOP))
          .addExternalSymbol(marker.c_str());
}

void X86MatchJumptablePass::recordJumpLocation(MachineInstr* MI, 
                                              MachineFunction &MF, 
                                              unsigned JTIndex) {
  // Create a new section entry
  auto &Context = MF.getFunction().getContext();
  auto *M = MF.getFunction().getParent();
  
  // Create mapping data
  std::string sectionName = ".jumptable.map";
  std::string data = MF.getName().str() + "," + 
                    std::to_string(JTIndex) + "\n";
  
  // Add the data to a special section
  auto *GV = new GlobalVariable(
    *M,
    ArrayType::get(Type::getInt8Ty(Context), data.size() + 1),
    true,
    GlobalValue::PrivateLinkage,
    ConstantDataArray::getString(Context, data),
    "jump_map_entry"
  );
  
  GV->setSection(sectionName);
}

bool X86MatchJumptablePass::runOnMachineFunction(MachineFunction &MF) {
    static int PassRunCount = 0;
    PassRunCount++;
    
    LLVM_DEBUG(dbgs() << "\n=== Pass Run #" << PassRunCount << " ===\n");
    LLVM_DEBUG(dbgs() << "Current optimization phase: " 
               << MF.getFunction().getParent()->getSourceFileName() << "\n");
    LLVM_DEBUG(dbgs() << "Analyzing jump tables in function: " << MF.getName() << "\n");

    // Get jump table information
    MachineJumpTableInfo *JumpTableInfo = MF.getJumpTableInfo();
    if (!JumpTableInfo) {
        LLVM_DEBUG(dbgs() << "No jump tables in this function.\n");
        return false;
    }

    bool Modified = false;
    for (unsigned JTIndex = 0; JTIndex < JumpTableInfo->getJumpTables().size(); ++JTIndex) {
        const MachineJumpTableEntry &JTEntry = JumpTableInfo->getJumpTables()[JTIndex];

        LLVM_DEBUG(dbgs() << "Jump Table #" << JTIndex << " contains " 
                         << JTEntry.MBBs.size() << " entries.\n");

        // Print detailed information about jump table entries
        LLVM_DEBUG(dbgs() << "Jump table entries:\n");
        for (unsigned i = 0; i < JTEntry.MBBs.size(); ++i) {
            if (JTEntry.MBBs[i]) {
                LLVM_DEBUG(dbgs() << "  [" << i << "] -> BB#" 
                           << JTEntry.MBBs[i]->getNumber() 
                           << " (" << JTEntry.MBBs[i]->getName() << ")\n");
            }
        }

        // Find and process indirect jumps
        MachineInstr* indirectJumpInstr = traceIndirectJumps(MF, JTIndex, JumpTableInfo);

        if (indirectJumpInstr) {
            // Print more detailed information about the indirect jump
            LLVM_DEBUG(dbgs() << "Found indirect jump: " << *indirectJumpInstr << "\n");
            LLVM_DEBUG(dbgs() << "Parent basic block: " << indirectJumpInstr->getParent()->getName() << "\n");
            
            // Print operand information
            LLVM_DEBUG(dbgs() << "Jump instruction operands:\n");
            for (unsigned i = 0; i < indirectJumpInstr->getNumOperands(); ++i) {
                LLVM_DEBUG(dbgs() << "  Operand " << i << ": " << indirectJumpInstr->getOperand(i) << "\n");
            }

            // Insert marker and record location
            insertIdentifyingMarker(indirectJumpInstr, MF, JTIndex);
            recordJumpLocation(indirectJumpInstr, MF, JTIndex);
            Modified = true;
        }
    }

    return Modified;
}

MachineInstr* X86MatchJumptablePass::traceIndirectJumps(MachineFunction &MF, 
                                                        unsigned JTIndex,
                                                        MachineJumpTableInfo *JumpTableInfo) {
    const MachineJumpTableEntry &JTEntry = JumpTableInfo->getJumpTables()[JTIndex];

    LLVM_DEBUG(dbgs() << "Tracing indirect jumps:\n");
    for (auto &MBB : MF) {
        LLVM_DEBUG(dbgs() << "  Checking BB: " << MBB.getName() << "\n");
        for (auto &MI : MBB) {
            LLVM_DEBUG(dbgs() << "    Checking instruction: " << MI << "\n");
            if (MI.isIndirectBranch()) {
                LLVM_DEBUG(dbgs() << "    Found indirect jump: " << MI << "\n");

                if (isJumpTableRelated(MI, JTEntry, MF)) {
                    LLVM_DEBUG(dbgs() << "    This indirect jump is related to Jump Table #" 
                              << JTIndex << "\n");
                    return &MI;
                } else {
                    LLVM_DEBUG(dbgs() << "    Jump is not related to this jump table\n");
                }
            }
        }
    }

    LLVM_DEBUG(dbgs() << "  No related indirect jump found\n");
    return nullptr;
}

bool X86MatchJumptablePass::isJumpTableLoad(MachineInstr &MI, const MachineJumpTableEntry &JTEntry) {
    LLVM_DEBUG(dbgs() << "\nAnalyzing potential jump table load instruction: " << MI << "\n");

    // First check memory operands for jump table metadata
    for (const MachineMemOperand *MMO : MI.memoperands()) {
        LLVM_DEBUG(dbgs() << "  Checking memory operand flags: " << MMO->getFlags() << "\n");
        if (MMO->getValue()) {
            StringRef ValueName = MMO->getValue()->getName();
            LLVM_DEBUG(dbgs() << "    Memory value name: '" << ValueName << "'\n");
            if (ValueName.contains("jump-table")) {
                LLVM_DEBUG(dbgs() << "    Found jump table in memory value name\n");
                return true;
            }
        }

        // Check if this is a jump table load directly from memory operand comments
        if (MI.getDesc().mayLoad() && MI.hasOneMemOperand()) {
            // Look for jump table reference in the instruction's debug info or comments
            if (MI.getDebugLoc()) {
                std::string Comment;
                raw_string_ostream OS(Comment);
                MI.print(OS);
                if (Comment.find("jump-table") != std::string::npos) {
                    LLVM_DEBUG(dbgs() << "    Found jump table reference in instruction comment\n");
                    return true;
                }
            }
        }
    }

    // Check for the MOVSX pattern
    if (MI.getOpcode() == X86::MOVSX64rm32) {
        LLVM_DEBUG(dbgs() << "  Found MOVSX64rm32 instruction\n");
        Register BaseReg;
        
        // Find base register
        for (const MachineOperand &MO : MI.operands()) {
            if (MO.isReg() && MO.isUse()) {
                BaseReg = MO.getReg();
                LLVM_DEBUG(dbgs() << "    Found base register: " << printReg(BaseReg, nullptr) << "\n");
                break;
            }
        }

        if (BaseReg) {
            // Look for preceding LEA
            MachineBasicBlock::iterator MBBI = MI;
            const MachineBasicBlock *MBB = MI.getParent();
            
            LLVM_DEBUG(dbgs() << "    Looking for LEA defining register: " << printReg(BaseReg, nullptr) << "\n");
            
            while (MBBI != MBB->begin()) {
                --MBBI;
                LLVM_DEBUG(dbgs() << "      Checking: " << *MBBI << "\n");
                
                if (MBBI->getOpcode() == X86::LEA64r) {
                    LLVM_DEBUG(dbgs() << "      Found LEA64r\n");
                    
                    // Verify this LEA defines our base register
                    const MachineOperand &DefReg = MBBI->getOperand(0);
                    if (!DefReg.isReg() || DefReg.getReg() != BaseReg) {
                        LLVM_DEBUG(dbgs() << "      LEA defines different register\n");
                        continue;
                    }
                    
                    // Check for jump table symbol
                    for (const MachineOperand &MO : MBBI->operands()) {
                        if (MO.isSymbol()) {
                            StringRef SymName = MO.getSymbolName();
                            LLVM_DEBUG(dbgs() << "      Checking symbol: '" << SymName << "'\n");
                            if (SymName.contains("jump-table")) {
                                LLVM_DEBUG(dbgs() << "      Found jump table symbol!\n");
                                return true;
                            }
                        }
                    }
                }
            }
            LLVM_DEBUG(dbgs() << "    No matching LEA found\n");
        }
    }

    return false;
}

bool X86MatchJumptablePass::isJumpTableRelated(MachineInstr &MI, 
                                              const MachineJumpTableEntry &JTEntry,
                                              MachineFunction &MF) {
    if (!MI.isIndirectBranch()) {
        LLVM_DEBUG(dbgs() << "Not an indirect branch, skipping\n");
        return false;
    }

    LLVM_DEBUG(dbgs() << "\nAnalyzing indirect jump: " << MI << "\n");

    // Get jump register
    Register JumpReg;
    for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isUse()) {
            JumpReg = MO.getReg();
            LLVM_DEBUG(dbgs() << "Found jump register: " << printReg(JumpReg, nullptr) << "\n");
            break;
        }
    }

    if (!JumpReg) {
        LLVM_DEBUG(dbgs() << "No jump register found\n");
        return false;
    }

    SmallVector<MachineInstr*, 8> Worklist;
    SmallPtrSet<MachineInstr*, 16> Visited;
    
    LLVM_DEBUG(dbgs() << "Starting backward analysis from register " << printReg(JumpReg, nullptr) << "\n");

    for (MachineInstr &DefMI : MF.getRegInfo().def_instructions(JumpReg)) {
        Worklist.push_back(&DefMI);
        LLVM_DEBUG(dbgs() << "Added to worklist: " << DefMI << "\n");
    }

    while (!Worklist.empty()) {
        MachineInstr *CurrMI = Worklist.pop_back_val();
        if (!Visited.insert(CurrMI).second) {
            LLVM_DEBUG(dbgs() << "Already visited: " << *CurrMI << "\n");
            continue;
        }

        LLVM_DEBUG(dbgs() << "Analyzing instruction: " << *CurrMI << "\n");

        if (isJumpTableLoad(*CurrMI, JTEntry)) {
            LLVM_DEBUG(dbgs() << "Found jump table load!\n");
            return true;
        }

        if (CurrMI->getOpcode() == X86::ADD64rr) {
            LLVM_DEBUG(dbgs() << "Found ADD64rr, checking operands\n");
            for (const MachineOperand &MO : CurrMI->operands()) {
                if (MO.isReg() && MO.isUse()) {
                    LLVM_DEBUG(dbgs() << "Checking register operand: " << printReg(MO.getReg(), nullptr) << "\n");
                    for (MachineInstr &DefMI : MF.getRegInfo().def_instructions(MO.getReg())) {
                        if (isJumpTableLoad(DefMI, JTEntry)) {
                            LLVM_DEBUG(dbgs() << "Found jump table load via ADD operand!\n");
                            return true;
                        }
                    }
                }
            }
        }

        // Add uses to worklist
        for (const MachineOperand &MO : CurrMI->operands()) {
            if (MO.isReg() && MO.isUse()) {
                LLVM_DEBUG(dbgs() << "Adding definitions of register " << printReg(MO.getReg(), nullptr) << " to worklist\n");
                for (MachineInstr &DefMI : MF.getRegInfo().def_instructions(MO.getReg())) {
                    if (!Visited.count(&DefMI)) {
                        Worklist.push_back(&DefMI);
                        LLVM_DEBUG(dbgs() << "Added to worklist: " << DefMI << "\n");
                    }
                }
            }
        }
    }

    LLVM_DEBUG(dbgs() << "No jump table relation found\n");
    return false;
}

FunctionPass *createX86MatchJumptablePass() { 
    return new X86MatchJumptablePass();
}

} // end namespace llvm