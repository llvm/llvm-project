
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/InitializePasses.h"
#include "llvm/Passes/PassBuilder.h"

#include "AMDGPU.h"

#include "AMDGPUNextUseAnalysis.h"

#define DEBUG_TYPE "amdgpu-next-use"

using namespace llvm;

//namespace {


void NextUseResult::init(const MachineFunction &MF) {

  for (auto L : LI->getLoopsInPreorder()) {
    SmallVector<MachineBasicBlock *> Exiting;
    L->getExitingBlocks(Exiting);
    for (auto B : Exiting) {
      for (auto S : successors(B)) {
        if (!L->contains(S)) {
          EdgeWeigths[B->getNumber()] = S->getNumber();
        }
      }
    }
  }
}

void NextUseResult::analyze(const MachineFunction &MF) {
  bool Changed = true;
  while(Changed) {
    Changed = false;
    for (auto MBB : post_order(&MF)) {
      SlotIndex Begin = Indexes->getMBBStartIdx(MBB);
      VRegDistances Curr, Prev;
      if (auto CurrMapRef = getVRegMap(MBB)) {
        Prev = CurrMapRef.value();
      }


      for (auto Succ : successors(MBB)) {
        auto SuccMapRef = getVRegMap(Succ);

        if (SuccMapRef) {
          // Check if the edge from MBB to Succ goes out of the Loop
          unsigned Weight = 0;
          if (EdgeWeigths.contains(MBB->getNumber())) {
            int SuccNum = EdgeWeigths[MBB->getNumber()];
            if (Succ->getNumber() == SuccNum)
              Weight = Infinity;
          }
          mergeDistances(Curr, SuccMapRef.value(), Weight);
        }
      }
      unsigned MBBLen =
          Begin.distance(Indexes->getMBBEndIdx(MBB)) / SlotIndex::InstrDist;
      for (auto &P : Curr) {
        P.second += MBBLen;
      }

      NextUseMap[MBB->getNumber()] = std::move(Curr);

      for (auto &MI : make_range(MBB->rbegin(), MBB->rend())) {
        for (auto &MO : MI.operands()) {
          if (MO.isReg() && MO.getReg().isVirtual() && MO.isUse()) {
            Register VReg = MO.getReg();
            MachineInstr *Def = MRI->getVRegDef(VReg);
            if (Def->getParent() == MBB)
              // defined in block - skip it
              continue;
            unsigned Distance =
                Begin.distance(Indexes->getInstructionIndex(MI)) /
                SlotIndex::InstrDist;
            setNextUseDistance(MBB, VReg, Distance);
            UsedInBlock[MBB->getNumber()].insert(VReg);
          }
        }
      }
      VRegDistances &Next = NextUseMap[MBB->getNumber()];
      dbgs() << "MBB_" << MBB->getNumber() << "\n";
      printVregDistancesD(Next);
      bool Changed4MBB = diff(Prev, Next);

      Changed |= Changed4MBB;
    }
  }
}

unsigned NextUseResult::getNextUseDistance(const MachineInstr &MI, const Register Vreg) {
  unsigned Dist = Infinity;
  const MachineBasicBlock *MBB = MI.getParent();
  SlotIndex Begin = Indexes->getMBBStartIdx(MBB->getNumber());
  SlotIndex Idx = Indexes->getInstructionIndex(MI);
  int IDist = Begin.distance(Idx)/SlotIndex::InstrDist;
  if (auto VMapRef = getVRegMap(MBB)) {
    VRegDistances &VRegs = VMapRef.value();
    if (VRegs.contains(Vreg)) {
      int UseDist = VRegs[Vreg];
      if ((UseDist - IDist) < 0) {
        for (auto Succ : successors(MBB)) {
          if (auto SuccVMapRef = getVRegMap(Succ)) {
            VRegDistances &SuccVRegs = SuccVMapRef.value();
            if (SuccVRegs.contains(Vreg)) {
              Dist = std::min(Dist, SuccVRegs[Vreg]);
            }
          }
        }
      } else {
        Dist = UseDist - IDist;
      }
      return Dist;
    }
  }
  return Infinity;
}

AMDGPUNextUseAnalysis::Result
AMDGPUNextUseAnalysis::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &MFAM) {
  return AMDGPUNextUseAnalysis::Result(MF,
                                       MFAM.getResult<SlotIndexesAnalysis>(MF),
                                       MFAM.getResult<MachineLoopAnalysis>(MF));
}

AnalysisKey AMDGPUNextUseAnalysis::Key;

//} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AMDGPUNextUseAnalysisPass",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerAnalysisRegistrationCallback(
                [](MachineFunctionAnalysisManager &MFAM) {
                  MFAM.registerPass([] { return AMDGPUNextUseAnalysis(); });
                });
          }};
}

char AMDGPUNextUseAnalysisWrapper::ID = 0;
char &llvm::AMDGPUNextUseAnalysisID = AMDGPUNextUseAnalysisWrapper::ID;
INITIALIZE_PASS_BEGIN(AMDGPUNextUseAnalysisWrapper, "amdgpu-next-use",
                      "AMDGPU Next Use Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUNextUseAnalysisWrapper, "amdgpu-next-use",
                    "AMDGPU Next Use Analysis", false, false)

bool AMDGPUNextUseAnalysisWrapper::runOnMachineFunction(
    MachineFunction &MF) {
  NU.Indexes = &getAnalysis<SlotIndexesWrapperPass>().getSI();
  NU.LI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  NU.MRI = &MF.getRegInfo();
  NU.init(MF);
  NU.analyze(MF);
  LLVM_DEBUG(NU.dump());
  return false;
}

void AMDGPUNextUseAnalysisWrapper::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequiredTransitiveID(MachineLoopInfoID);
  AU.addPreservedID(MachineLoopInfoID);
  AU.addRequiredTransitiveID(MachineDominatorsID);
  AU.addPreservedID(MachineDominatorsID);
  AU.addPreserved<SlotIndexesWrapperPass>();
  AU.addRequiredTransitive<SlotIndexesWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

AMDGPUNextUseAnalysisWrapper::AMDGPUNextUseAnalysisWrapper()
    : MachineFunctionPass(ID) {
  initializeAMDGPUNextUseAnalysisWrapperPass(*PassRegistry::getPassRegistry());
}