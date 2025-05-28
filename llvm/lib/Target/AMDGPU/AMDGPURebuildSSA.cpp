#include "AMDGPU.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"
#include "GCNSubtarget.h"

#include <stack>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-rebuild-ssa"

namespace {

class AMDGPURebuildSSALegacy : public MachineFunctionPass {
  LiveIntervals *LIS;
  MachineDominatorTree *MDT;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;

  SetVector<unsigned> CrossBlockVRegs;
  DenseMap<unsigned, SmallPtrSet<MachineBasicBlock *, 8>> DefBlocks;
  DenseMap<unsigned, SmallPtrSet<MachineBasicBlock *, 8>> LiveInBlocks;
  DenseMap<unsigned, SmallSet<unsigned, 4>> PHINodes;
  DenseMap<MachineInstr *, unsigned> PHIMap;
  DenseMap<unsigned, std::stack<unsigned>> VregNames;
  DenseSet<unsigned> DefSeen;

  void collectCrossBlockVRegs(MachineFunction &MF);
  void findPHINodesPlacement(const SmallPtrSetImpl<MachineBasicBlock *> &LiveInBlocks,
                          const SmallPtrSetImpl<MachineBasicBlock *> &DefBlocks,
                          SmallVectorImpl<MachineBasicBlock *> &PHIBlocks) {
    
    IDFCalculatorBase<MachineBasicBlock, false> IDF(MDT->getBase());

    IDF.setLiveInBlocks(LiveInBlocks);
    IDF.setDefiningBlocks(DefBlocks);
    IDF.calculate(PHIBlocks);
  }

  void renameVRegs(MachineBasicBlock &MBB) {
    for (auto &PHI : MBB.phis()) {
      Register Res = PHI.getOperand(0).getReg();
      const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, Res);
      Register NewVReg = MRI->createVirtualRegister(RC);
      PHI.getOperand(0).setReg(NewVReg);
      VregNames[Res].push(NewVReg);
      DefSeen.insert(NewVReg);
    }
    for (auto &I : make_range(MBB.getFirstNonPHI(), MBB.end())) {
     
      for (auto &Op : I.uses()) {
        if (Op.isReg() && Op.getReg().isVirtual()) {
          unsigned VReg = Op.getReg();
          if (VregNames[VReg].empty()) {
            // If no new name is available, use the original VReg.
            continue;
          }
          unsigned NewVReg = VregNames[VReg].top();
          //VregNames[VReg].pop();
          Op.setReg(NewVReg);
        }
      }

      for (auto &Op : I.defs()) {
        if (Op.getReg().isVirtual()) {
          unsigned VReg = Op.getReg();
          if (DefSeen.contains(VReg)) {
            const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, VReg);
            Register NewVReg = MRI->createVirtualRegister(RC);
            Op.setReg(NewVReg);
            VregNames[VReg].push(NewVReg);
          } else {
            DefSeen.insert(VReg);
          }
        }
      }
    }

    for (auto Succ : successors(&MBB)) {
      for (auto &PHI : Succ->phis()) {
        Register Res = PHIMap[&PHI];
        if (VregNames[Res].empty()) {
          PHI.addOperand(MachineOperand::CreateReg(Res, false));
        } else {
          PHI.addOperand(
              MachineOperand::CreateReg(VregNames[Res].top(), false));
        }
        PHI.addOperand(MachineOperand::CreateMBB(&MBB));
      }
    }
    // recurse to the succs in DomTree
    DomTreeNodeBase<MachineBasicBlock> *Node = MDT->getNode(&MBB);
    for (auto *Child : Node->children()) {
      MachineBasicBlock *ChildMBB = Child->getBlock();
      // Process child in the dominator tree
      renameVRegs(*ChildMBB);
    }

    for (auto &I : MBB) {
      for (auto Op : I.defs()) {
        if (Op.getReg().isVirtual()) {
          Register VReg = Op.getReg();
          if (!VregNames[VReg].empty())
            VregNames[VReg].pop();
        }
      }
    }
  }

public:
  static char ID;
  AMDGPURebuildSSALegacy() : MachineFunctionPass(ID) {
    initializeAMDGPURebuildSSALegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitiveID(MachineDominatorsID);
    AU.addPreservedID(MachineDominatorsID);
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

void AMDGPURebuildSSALegacy::collectCrossBlockVRegs(MachineFunction &MF) {
  for (auto &MBB : MF) {
    SetVector<unsigned> Killed;
    for (auto &I : MBB) {
      for (auto Op : I.uses()) {
        if (Op.isReg() && Op.getReg().isVirtual() &&
            !Killed.contains(Op.getReg())) {
          CrossBlockVRegs.insert(Op.getReg());
          LiveInBlocks[Op.getReg()].insert(&MBB);
        }
      }
      for (auto Op : I.defs()) {
        if (Op.isReg() && Op.getReg().isVirtual()) {
          Killed.insert(Op.getReg());
          DefBlocks[Op.getReg()].insert(&MBB);
        }
      }
    }
  }
}

bool AMDGPURebuildSSALegacy::runOnMachineFunction(MachineFunction &MF) {
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();

  CrossBlockVRegs.clear();
  DefBlocks.clear();
  LiveInBlocks.clear();
  PHINodes.clear();
  VregNames.clear();
  DefSeen.clear();
  //   for (auto &MBB : MF) {
  //     PHINodes[MBB.getNumber()] = SmallSet<unsigned, 4>();
  //   }
  // Collect all cross-block virtual registers.
  // This includes registers that are live-in to the function, and registers
  // that are defined in multiple blocks.
  // We will insert PHI nodes for these registers.
  collectCrossBlockVRegs(MF);

  LLVM_DEBUG(dbgs() << "##### Virt regs live cross block ##################\n";
             for (auto VReg : CrossBlockVRegs) {
               dbgs() << Register::virtReg2Index(VReg) << " ";
             } dbgs()
             << "\n");

  for (auto VReg : CrossBlockVRegs) {
    SmallVector<MachineBasicBlock *> PHIBlocks;

    LLVM_DEBUG(
        dbgs() << "findPHINodesPlacement input:\nVreg: "
               << Register::virtReg2Index(VReg) << "\n";
        dbgs() << "Def Blocks: \n"; for (auto MBB : DefBlocks[VReg]) {
          dbgs() << MBB->getName() << "." << MBB->getNumber() << " ";
        } dbgs() << "\nLiveIn Blocks: \n";
        for (auto MBB : LiveInBlocks[VReg]) {
          dbgs() << MBB->getName() << "." << MBB->getNumber() << " ";
        } dbgs()
        << "\n");

    findPHINodesPlacement(LiveInBlocks[VReg], DefBlocks[VReg], PHIBlocks);
    LLVM_DEBUG(dbgs() << "\nBlocks to insert PHI nodes:\n";
               for (auto MBB : PHIBlocks) {
                 dbgs() << MBB->getName() << "." << MBB->getNumber() << " ";
               } dbgs()
               << "\n");
    for (auto MBB : PHIBlocks) {
      if (!PHINodes[MBB->getNumber()].contains(VReg)) {
        // Insert PHI for VReg. Don't use new VReg here as we'll replace them
        // in renaming phase.
        auto PHINode = BuildMI(*MBB, MBB->begin(), DebugLoc(), TII->get(TargetOpcode::PHI))
            .addReg(VReg, RegState::Define);
        PHINodes[MBB->getNumber()].insert(VReg);
        PHIMap[PHINode] = VReg;
      }
    }
  }

    // Rename virtual registers in the basic block.
  renameVRegs(MF.front());
  LLVM_DEBUG(dbgs() << "##### Vreg names after renaming ##################\n";
             for (auto &Pair : VregNames) {
               dbgs() << Register::virtReg2Index(Pair.first) << ": ";
               if (Pair.second.empty()) {
                 dbgs() << "empty";
               } else {
                 dbgs() << Pair.second.top();
               }
               dbgs() << "\n";
             } dbgs()
             << "\n");

  return false;
}

char AMDGPURebuildSSALegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                    false, false)

// Legacy PM registration
FunctionPass *llvm::createAMDGPURebuildSSALegacyPass() {
  return new AMDGPURebuildSSALegacy();
}

PreservedAnalyses
llvm::AMDGPURebuildSSAPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  AMDGPURebuildSSALegacy Impl;
  bool Changed = Impl.runOnMachineFunction(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  // TODO: We could detect CFG changed.
  auto PA = getMachineFunctionPassPreservedAnalyses();
  return PA;
}

llvm::PassPluginLibraryInfo getAMDGPURebuildSSAPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AMDGPURebuildSSA", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, MachineFunctionPassManager &PM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "amdgpu-rebuild-ssa") {
                    PM.addPass(AMDGPURebuildSSAPass());
                    return true;
                  }
                  return false;
                });
          }};
}

// Expose the pass to LLVM’s pass manager infrastructure
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getAMDGPURebuildSSAPassPluginInfo();
}