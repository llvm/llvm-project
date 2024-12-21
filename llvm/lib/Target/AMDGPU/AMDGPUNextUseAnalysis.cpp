

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
#include "llvm/Support/Timer.h"

#include "AMDGPU.h"

#include "AMDGPUNextUseAnalysis.h"

#define DEBUG_TYPE "amdgpu-next-use"

using namespace llvm;

//namespace {


void NextUseResult::init(const MachineFunction &MF) {
  TG = new TimerGroup("Next Use Analysis",
                      "Compilation Timers for Next Use Analysis");
  T1 = new Timer("Next Use Analysis", "Time spent in analyse()", *TG);
  T2 = new Timer("Next Use Analysis", "Time spent in computeNextUseDistance()",
                 *TG);
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
  // Upward-exposed distances are only necessary to convey the data flow from
  // the block to its predecessors. No need to store it beyond the analyze
  // function as the analysis users are only interested in the use distances
  // relatively to the given MI or the given block end.
  DenseMap<unsigned, VRegDistances> UpwardNextUses;
  T1->startTimer();
  bool Changed = true;
  while(Changed) {
    Changed = false;
    for (auto MBB : post_order(&MF)) {
      unsigned MBBNum = MBB->getNumber();
      VRegDistances Curr, Prev;
      if (UpwardNextUses.contains(MBBNum)) {
        Prev = UpwardNextUses[MBBNum];
      }


      for (auto Succ : successors(MBB)) {
        unsigned SuccNum = Succ->getNumber();

        if (UpwardNextUses.contains(SuccNum)) {
          VRegDistances SuccDist = UpwardNextUses[SuccNum];
          // Check if the edge from MBB to Succ goes out of the Loop
          unsigned Weight = 0;
          if (EdgeWeigths.contains(MBB->getNumber())) {
            int SuccNum = EdgeWeigths[MBB->getNumber()];
            if (Succ->getNumber() == SuccNum)
              Weight = Infinity;
          }
          mergeDistances(Curr, SuccDist, Weight);
        }
      }

      NextUseMap[MBBNum].Bottom = Curr;

      for (auto &MI : make_range(MBB->rbegin(), MBB->rend())) {
        
        for (auto &P : Curr) {
          P.second++;
        }

        for (auto &MO : MI.operands()) {
          if (MO.isReg() && MO.getReg().isVirtual()) {
            Register VReg = MO.getReg();
            if(MO.isUse()) {
              Curr[VReg] = 0;
              UsedInBlock[MBB->getNumber()].insert(VReg);
            } else if (MO.isDef()) {
              Curr.erase(VReg);
            }
          }
        }
        NextUseMap[MBBNum].InstrDist[&MI] = Curr;
        // printVregDistancesD(Curr);
      }

      UpwardNextUses[MBBNum] = std::move(Curr);

      bool Changed4MBB = diff(Prev, UpwardNextUses[MBBNum]);

      Changed |= Changed4MBB;
    }
  }
  T1->stopTimer();
  TG->print(llvm::errs());
}

unsigned NextUseResult::getNextUseDistance(const MachineBasicBlock::iterator I,
                                           const Register VReg) {
  unsigned Dist = Infinity;
  const MachineBasicBlock *MBB = I->getParent();
  unsigned MBBNum = MBB->getNumber();
  if (NextUseMap.contains(MBBNum) &&
      NextUseMap[MBBNum].InstrDist.contains(&*I) &&
      NextUseMap[MBBNum].InstrDist[&*I].contains(VReg))
    Dist = NextUseMap[MBBNum].InstrDist[&*I][VReg];
  return Dist;
}

unsigned NextUseResult::getNextUseDistance(const MachineBasicBlock &MBB,
                                           const Register VReg) {
  unsigned Dist = Infinity;
  unsigned MBBNum = MBB.getNumber();
  if (NextUseMap.contains(MBBNum))
    Dist = NextUseMap[MBBNum].Bottom[VReg];
  return Dist;
}

// unsigned NextUseResult::getNextUseDistance(const MachineInstr &MI,
//                                            const Register VReg) {
//   SlotIndex Idx = Indexes->getInstructionIndex(MI);
//   assert(Idx.isValid() && "Invalid Instruction index!");
//   if (InstrCache.contains(&Idx) && InstrCache[&Idx].contains(VReg)) {
//     return InstrCache[&Idx][VReg];
//   }
//   return computeNextUseDistance(*MI.getParent(), Idx, VReg);
// }

// unsigned NextUseResult::getNextUseDistance(const MachineBasicBlock &MBB,
//                                            Register VReg) {
//   SlotIndex Idx = Indexes->getMBBEndIdx(&MBB);
//   assert(Idx.isValid() && "Invalid Instruction index!");
//   if (InstrCache.contains(&Idx) && InstrCache[&Idx].contains(VReg)) {
//     return InstrCache[&Idx][VReg];
//   }
//   return computeNextUseDistance(MBB, Idx, VReg);
// }

// unsigned NextUseResult::computeNextUseDistance(const MachineBasicBlock &MBB,
//                                                const SlotIndex I,
//                                                Register VReg) {
//   //T2->startTimer();

//   unsigned Dist = Infinity;

//   SlotIndex Begin = Indexes->getMBBStartIdx(MBB.getNumber());
  
//   int IDist = Begin.distance(I)/SlotIndex::InstrDist;
//   if (auto VMapRef = getVRegMap(&MBB)) {
//     VRegDistances &VRegs = VMapRef.value();
//     if (VRegs.contains(VReg)) {
//       int UseDist = VRegs[VReg];
//       if ((UseDist - IDist) < 0) {

//         // FIXME:  VRegs contains only upward exposed info! In other words - the
//         // very first use in block!
//         // (UseDist - IDist) < 0 just means that our MI is later then the 1st
//         // use of the VReg.
//         // Function user (calls from outside: from SSASpiller) is interested in
//         // the next use in block after the MI!
//         // We need to scan for the uses in current block - from MI to the block
//         // end BEFORE checking the Succs!

//         // NOTE: Make sure that we don't spoil the info for Next Use analysis
//         // itself. If so, we need 2 different functions for querying
//         // nextUseDistance!
//         bool Done = false;
//         MachineInstr *Instr = Indexes->getInstructionFromIndex(I);
//         if (Instr) {
//           // we canot use SlotIndexes to compare positions because
//           // spills/reloads were not added in Instruction Index. So, just scan
//           // the BB.
//           unsigned D = 0;
//           MachineBasicBlock::iterator It(Instr);
//           while (It != MBB.end()) {
//             if (It->definesRegister(VReg, TRI)) {
//               // VReg is DEAD
//               Dist = Infinity;
//               Done = true;
//               break;
//             }
//             if (It->readsRegister(VReg, TRI)) {
//               Dist = D;
//               Done = true;
//               break;
//             }
//             D++;
//             It++;
//           }
//         }
//         if (!Done)
//           // The instruction of interest is after the first use  of the register
//           // in the block and the register has not been killed in block. Look
//           // for the next use in successors.
//           for (auto Succ : successors(&MBB)) {
//             if (auto SuccVMapRef = getVRegMap(Succ)) {
//               VRegDistances &SuccVRegs = SuccVMapRef.value();
//               if (SuccVRegs.contains(VReg)) {
//                 Dist = std::min(Dist, SuccVRegs[VReg]);
//               }
//             }
//           }
//       } else {
//         Dist = UseDist - IDist;
//       }
//     } else {
//       // We hit a case when the VReg is defined and used inside the block.
//       // Let's see if I is in between. Since we may be called from the broken
//       // SSA function we cannot rely on MRI.getVRegDef. The VReg Def in block
//       // may be reload, so we canot use SlotIndexes to compare positions because
//       // spills/reloads were not added in Instruction Index. So, just scan the
//       // BB.
//       MachineInstr *Instr = Indexes->getInstructionFromIndex(I);
//       if (Instr) {
//         bool DefSeen = false, InstrSeen = false;
//         unsigned D = 0;
//         for (auto &MI : MBB) {
//           if (InstrSeen)
//             D++;
//           if (Instr == &MI) {
//             if (!DefSeen)
//               break;
//             InstrSeen = true;
//           }

//           if (MI.definesRegister(VReg, TRI))
//             DefSeen = true;
//           if (MI.readsRegister(VReg, TRI) && InstrSeen) {
//             Dist = D;
//             break;
//           }
//         }
//       }

//       // MachineInstr *Def = MRI->getVRegDef(VReg);
//       // assert(Def && "Neither use distance no Def found for reg!");
//       // SlotIndex DefIdx = Indexes->getInstructionIndex(*Def);
//       // assert(DefIdx.isValid() && "Register Def not in the Index");
//       // if (SlotIndex::isEarlierInstr(DefIdx, I)) {
//       //   // "I" is after the Def
//       //   for (auto &U : MRI->use_instructions(VReg)) {
//       //     assert(U.getParent() == &MBB &&
//       //            "Use out of the block fount but distance was not recorded");
//       //     SlotIndex UIdx = Indexes->getInstructionIndex(U);
//       //     if (SlotIndex::isEarlierInstr(I, UIdx)) {
//       //       unsigned UDist = I.distance(UIdx)/SlotIndex::InstrDist;
//       //       if (UDist < Dist)
//       //         Dist = UDist;
//       //     }
//       //   }
//       // }
//     }
//     if (Dist != Infinity)
//       InstrCache[&I][VReg] = Dist;
//   }
//   //T2->stopTimer();
//   //TG->print(llvm::errs());
//   return Dist;
// }

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
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUNextUseAnalysisWrapper, "amdgpu-next-use",
                    "AMDGPU Next Use Analysis", false, false)

bool AMDGPUNextUseAnalysisWrapper::runOnMachineFunction(
    MachineFunction &MF) {
  NU.Indexes = &getAnalysis<SlotIndexesWrapperPass>().getSI();
  NU.LI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  NU.MRI = &MF.getRegInfo();
  NU.TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  assert(NU.MRI->isSSA());
  NU.init(MF);
  NU.analyze(MF);
//  LLVM_DEBUG(NU.dump());
  return false;
}

void AMDGPUNextUseAnalysisWrapper::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addRequired<SlotIndexesWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

AMDGPUNextUseAnalysisWrapper::AMDGPUNextUseAnalysisWrapper()
    : MachineFunctionPass(ID) {
  initializeAMDGPUNextUseAnalysisWrapperPass(*PassRegistry::getPassRegistry());
}