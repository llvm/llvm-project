//===-- SIWholeQuadMode.cpp - enter and suspend whole quad mode -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass adds instructions to enable whole quad mode for pixel
/// shaders, and whole wavefront mode for all programs.
///
/// Whole quad mode is required for derivative computations, but it interferes
/// with shader side effects (stores and atomics). This pass is run on the
/// scheduled machine IR but before register coalescing, so that machine SSA is
/// available for analysis. It ensures that WQM is enabled when necessary, but
/// disabled around stores and atomics.
///
/// When necessary, this pass creates a function prolog
///
///   S_MOV_B64 LiveMask, EXEC
///   S_WQM_B64 EXEC, EXEC
///
/// to enter WQM at the top of the function and surrounds blocks of Exact
/// instructions by
///
///   S_AND_SAVEEXEC_B64 Tmp, LiveMask
///   ...
///   S_MOV_B64 EXEC, Tmp
///
/// We also compute when a sequence of instructions requires Whole Wavefront
/// Mode (WWM) and insert instructions to save and restore it:
///
/// S_OR_SAVEEXEC_B64 Tmp, -1
/// ...
/// S_MOV_B64 EXEC, Tmp
///
/// In order to avoid excessive switching during sequences of Exact
/// instructions, the pass first analyzes which instructions must be run in WQM
/// (aka which instructions produce values that lead to derivative
/// computations).
///
/// Basic blocks are always exited in WQM as long as some successor needs WQM.
///
/// There is room for improvement given better control flow analysis:
///
///  (1) at the top level (outside of control flow statements, and as long as
///      kill hasn't been used), one SGPR can be saved by recovering WQM from
///      the LiveMask (this is implemented for the entry block).
///
///  (2) when entire regions (e.g. if-else blocks or entire loops) only
///      consist of exact and don't-care instructions, the switch only has to
///      be done at the entry and exit points rather than potentially in each
///      block of the region.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "si-wqm"

namespace {

enum {
  StateWQM = 0x1,
  StateWWM = 0x2,
  StateExact = 0x4,
};

struct PrintState {
public:
  int State;

  explicit PrintState(int State) : State(State) {}
};

#ifndef NDEBUG
static raw_ostream &operator<<(raw_ostream &OS, const PrintState &PS) {
  if (PS.State & StateWQM)
    OS << "WQM";
  if (PS.State & StateWWM) {
    if (PS.State & StateWQM)
      OS << '|';
    OS << "WWM";
  }
  if (PS.State & StateExact) {
    if (PS.State & (StateWQM | StateWWM))
      OS << '|';
    OS << "Exact";
  }

  return OS;
}
#endif

struct InstrInfo {
  char Needs = 0;
  char Disabled = 0;
  char OutNeeds = 0;
};

struct BlockInfo {
  char Needs = 0;
  char InNeeds = 0;
  char OutNeeds = 0;
  char InitialState = 0;
  unsigned LiveMaskIn = 0; // Initial live mask register
  unsigned LiveMaskOut = 0; // Outgoing live mask register
};

struct WorkItem {
  MachineBasicBlock *MBB = nullptr;
  MachineInstr *MI = nullptr;

  WorkItem() = default;
  WorkItem(MachineBasicBlock *MBB) : MBB(MBB) {}
  WorkItem(MachineInstr *MI) : MI(MI) {}
};

class SIWholeQuadMode : public MachineFunctionPass {
private:
  CallingConv::ID CallingConv;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  const GCNSubtarget *ST;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  MachineDominatorTree *MDT;
  MachinePostDominatorTree *PDT;

  DenseMap<const MachineInstr *, InstrInfo> Instructions;
  DenseMap<MachineBasicBlock *, BlockInfo> Blocks;

  // Tracks live mask output of instructions
  DenseMap<const MachineInstr *, unsigned> LiveMaskRegs;
  // Tracks state (WQM/WWM/Exact) after a given instruction
  DenseMap<const MachineInstr *, char> StateTransition;

  SmallVector<MachineInstr *, 2> LiveMaskQueries;
  SmallVector<MachineInstr *, 4> LowerToMovInstrs;
  SmallVector<MachineInstr *, 4> LowerToCopyInstrs;
  SmallVector<MachineInstr *, 4> DemoteInstrs;
  SmallSet<MachineInstr *, 32> NeedsDemoteCleanup;

  bool HasWaterfalls;

  void printInfo();

  void markInstruction(MachineInstr &MI, char Flag,
                       std::vector<WorkItem> &Worklist);
  void markInstructionUses(const MachineInstr &MI, char Flag,
                           std::vector<WorkItem> &Worklist);
  char scanInstructions(MachineFunction &MF, std::vector<WorkItem> &Worklist);
  void propagateInstruction(MachineInstr &MI, std::vector<WorkItem> &Worklist);
  void propagateBlock(MachineBasicBlock &MBB, std::vector<WorkItem> &Worklist);
  char analyzeFunction(MachineFunction &MF);

  void scanLiveLanes(MachineBasicBlock &MBB,
                     std::vector<MachineBasicBlock *> &Worklist);
  void analyzeLiveLanes(MachineFunction &MF);

  MachineBasicBlock::iterator saveSCC(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator Before);
  MachineBasicBlock::iterator
  prepareInsertion(MachineBasicBlock &MBB, MachineBasicBlock::iterator First,
                   MachineBasicBlock::iterator Last, bool PreferLast,
                   bool SaveSCC, bool CheckPhys);
  void toExact(MachineBasicBlock &MBB, MachineBasicBlock::iterator Before,
               unsigned SaveWQM, unsigned LiveMaskReg);
  void toWQM(MachineBasicBlock &MBB, MachineBasicBlock::iterator Before,
             unsigned SavedWQM);
  void toWWM(MachineBasicBlock &MBB, MachineBasicBlock::iterator Before,
             unsigned SaveOrig);
  void fromWWM(MachineBasicBlock &MBB, MachineBasicBlock::iterator Before,
               unsigned SavedOrig, char NonWWMState);

  bool canSplitBlockAt(MachineBasicBlock *BB, MachineInstr *MI);
  MachineBasicBlock *splitBlock(MachineBasicBlock *BB,
                                MachineInstr *TermMI);
  void lowerBlock(MachineBasicBlock &MBB);

  unsigned findLiveMaskReg(MachineBasicBlock &MBB, BlockInfo &BI,
                           MachineBasicBlock::iterator &Before);
  void processBlock(MachineBasicBlock &MBB, bool isEntry);

  bool lowerLiveMaskQueries(unsigned LiveMaskReg);
  void lowerCopyInstrs();
  bool lowerDemoteInstrs();

  void lowerLiveMaskQuery(MachineBasicBlock &MBB,
                          MachineInstr &MI,
                          unsigned LiveMaskReg,
                          bool isWQM);
  MachineInstr *lowerDemote(MachineBasicBlock &MBB, MachineInstr &MI,
                   unsigned LiveMaskIn, unsigned LiveMaskOut,
                   bool isWQM);
  MachineInstr *insertDemoteCleanup(MachineBasicBlock &MBB,
                                    MachineInstr *MI,
                                    MachineBasicBlock::iterator *Before,
                                    unsigned LiveMask);

public:
  static char ID;

  SIWholeQuadMode() :
    MachineFunctionPass(ID) { }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI Whole Quad Mode"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervals>();
    AU.addPreserved<LiveIntervals>();
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    AU.addRequired<MachinePostDominatorTree>();
    AU.addPreserved<MachinePostDominatorTree>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char SIWholeQuadMode::ID = 0;

INITIALIZE_PASS_BEGIN(SIWholeQuadMode, DEBUG_TYPE, "SI Whole Quad Mode", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_END(SIWholeQuadMode, DEBUG_TYPE, "SI Whole Quad Mode", false,
                    false)

char &llvm::SIWholeQuadModeID = SIWholeQuadMode::ID;

FunctionPass *llvm::createSIWholeQuadModePass() {
  return new SIWholeQuadMode;
}

#ifndef NDEBUG
LLVM_DUMP_METHOD void SIWholeQuadMode::printInfo() {
  for (const auto &BII : Blocks) {
    dbgs() << "\n"
           << printMBBReference(*BII.first) << ":\n"
           << "  InNeeds = " << PrintState(BII.second.InNeeds)
           << ", Needs = " << PrintState(BII.second.Needs)
           << ", OutNeeds = " << PrintState(BII.second.OutNeeds) << "\n\n";

    for (const MachineInstr &MI : *BII.first) {
      auto III = Instructions.find(&MI);
      if (III == Instructions.end())
        continue;

      dbgs() << "  " << MI << "    Needs = " << PrintState(III->second.Needs)
             << ", OutNeeds = " << PrintState(III->second.OutNeeds) << '\n';
    }
  }
}
#endif

static bool isWaterfallStart(unsigned Opcode) {
  switch (Opcode) {
    case AMDGPU::SI_WATERFALL_BEGIN_V1:
    case AMDGPU::SI_WATERFALL_BEGIN_V2:
    case AMDGPU::SI_WATERFALL_BEGIN_V4:
    case AMDGPU::SI_WATERFALL_BEGIN_V8:
      return true;
    default:
      return false;
  }
}

void SIWholeQuadMode::markInstruction(MachineInstr &MI, char Flag,
                                      std::vector<WorkItem> &Worklist) {
  InstrInfo &II = Instructions[&MI];

  assert(!(Flag & StateExact) && Flag != 0);

  // Remove any disabled states from the flag. The user that required it gets
  // an undefined value in the helper lanes. For example, this can happen if
  // the result of an atomic is used by instruction that requires WQM, where
  // ignoring the request for WQM is correct as per the relevant specs.
  Flag &= ~II.Disabled;

  // Ignore if the flag is already encompassed by the existing needs, or we
  // just disabled everything.
  if ((II.Needs & Flag) == Flag)
    return;

  II.Needs |= Flag;
  Worklist.push_back(&MI);
}

/// Mark all instructions defining the uses in \p MI with \p Flag.
void SIWholeQuadMode::markInstructionUses(const MachineInstr &MI, char Flag,
                                          std::vector<WorkItem> &Worklist) {
  for (const MachineOperand &Use : MI.uses()) {
    if (!Use.isReg() || !Use.isUse())
      continue;

    Register Reg = Use.getReg();

    // Handle physical registers that we need to track; this is mostly relevant
    // for VCC, which can appear as the (implicit) input of a uniform branch,
    // e.g. when a loop counter is stored in a VGPR.
    if (!Reg.isVirtual()) {
      if (Reg == AMDGPU::EXEC || Reg == AMDGPU::EXEC_LO)
        continue;

      for (MCRegUnitIterator RegUnit(Reg.asMCReg(), TRI); RegUnit.isValid();
           ++RegUnit) {
        LiveRange &LR = LIS->getRegUnit(*RegUnit);
        const VNInfo *Value = LR.Query(LIS->getInstructionIndex(MI)).valueIn();
        if (!Value)
          continue;

        // Since we're in machine SSA, we do not need to track physical
        // registers across basic blocks.
        if (Value->isPHIDef())
          continue;

        markInstruction(*LIS->getInstructionFromIndex(Value->def), Flag,
                        Worklist);
      }

      continue;
    }

    for (MachineInstr &DefMI : MRI->def_instructions(Use.getReg()))
      markInstruction(DefMI, Flag, Worklist);
  }
}

// Scan instructions to determine which ones require an Exact execmask and
// which ones seed WQM requirements.
char SIWholeQuadMode::scanInstructions(MachineFunction &MF,
                                       std::vector<WorkItem> &Worklist) {
  char GlobalFlags = 0;
  bool WQMOutputs = MF.getFunction().hasFnAttribute("amdgpu-ps-wqm-outputs");
  SmallVector<MachineInstr *, 4> SetInactiveInstrs;
  SmallVector<MachineInstr *, 4> SoftWQMInstrs;

  // We need to visit the basic blocks in reverse post-order so that we visit
  // defs before uses, in particular so that we don't accidentally mark an
  // instruction as needing e.g. WQM before visiting it and realizing it needs
  // WQM disabled.
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (auto BI = RPOT.begin(), BE = RPOT.end(); BI != BE; ++BI) {
    MachineBasicBlock &MBB = **BI;
    BlockInfo &BBI = Blocks[&MBB];
    bool HasDemoteInBlock = false;

    for (auto II = MBB.begin(), IE = MBB.end(); II != IE; ++II) {
      MachineInstr &MI = *II;
      InstrInfo &III = Instructions[&MI];
      unsigned Opcode = MI.getOpcode();
      char Flags = 0;

      if (TII->isWQM(Opcode)) {
        // Sampling instructions don't need to produce results for all pixels
        // in a quad, they just require all inputs of a quad to have been
        // computed for derivatives.
        markInstructionUses(MI, StateWQM, Worklist);
        GlobalFlags |= StateWQM;
        continue;
      } else if (Opcode == AMDGPU::WQM) {
        // The WQM intrinsic requires its output to have all the helper lanes
        // correct, so we need it to be in WQM.
        Flags = StateWQM;
        LowerToCopyInstrs.push_back(&MI);
      } else if (Opcode == AMDGPU::SOFT_WQM) {
        LowerToCopyInstrs.push_back(&MI);
        SoftWQMInstrs.push_back(&MI);
        continue;
      } else if (Opcode == AMDGPU::WWM) {
        // The WWM intrinsic doesn't make the same guarantee, and plus it needs
        // to be executed in WQM or Exact so that its copy doesn't clobber
        // inactive lanes.
        markInstructionUses(MI, StateWWM, Worklist);
        GlobalFlags |= StateWWM;
        LowerToMovInstrs.push_back(&MI);
        continue;
      } else if (Opcode == AMDGPU::V_SET_INACTIVE_B32 ||
                 Opcode == AMDGPU::V_SET_INACTIVE_B64) {
        III.Disabled = StateWWM;
        MachineOperand &Inactive = MI.getOperand(2);
        if (Inactive.isReg()) {
          if (Inactive.isUndef()) {
            LowerToCopyInstrs.push_back(&MI);
          } else {
            Register Reg = Inactive.getReg();
            if (Reg.isVirtual()) {
              for (MachineInstr &DefMI : MRI->def_instructions(Reg))
                markInstruction(DefMI, StateWWM, Worklist);
            }
          }
        }
        SetInactiveInstrs.push_back(&MI);
        continue;
      } else if (TII->isDisableWQM(MI)) {
        BBI.Needs |= StateExact;
        if (!(BBI.InNeeds & StateExact)) {
          BBI.InNeeds |= StateExact;
          Worklist.push_back(&MBB);
        }
        GlobalFlags |= StateExact;
        III.Disabled = StateWQM | StateWWM;
        continue;
      } else {
        if (Opcode == AMDGPU::SI_PS_LIVE || Opcode == AMDGPU::SI_WQM_HELPER) {
          LiveMaskQueries.push_back(&MI);
        } else if (Opcode == AMDGPU::SI_DEMOTE_I1) {
          // Only perform a demote dominance test once per block
          if (!HasDemoteInBlock) {
            SmallVector<MachineInstr *, 4> ControlFlowInstrs;
            bool DominatesAllReachable = true;

            // Simultaneously check if this demote is in control flow
            // (dominates all blocks) and find all control flow ends
            // which post dominate this block.
            for (MachineBasicBlock *Other : depth_first(&MBB)) {
              if (DominatesAllReachable && !MDT->dominates(&MBB, Other))
                DominatesAllReachable = false;
              if (PDT->dominates(Other, &MBB)) {
                auto FirstMI = Other->getFirstNonPHI();
                if ((FirstMI != Other->end()) &&
                    (FirstMI->getOpcode() == AMDGPU::SI_END_CF)) {
                  ControlFlowInstrs.push_back(&*FirstMI);
                }
              }
            }

            if (!DominatesAllReachable) {
              // Demote is in control flow hence we must mark all control
              // flow end instructions requiring clean up.
              for (MachineInstr *CF : ControlFlowInstrs)
                NeedsDemoteCleanup.insert(CF);
            }
          }

          DemoteInstrs.push_back(&MI);
          HasDemoteInBlock = true;
        } else if (Opcode == AMDGPU::SI_KILL_I1_TERMINATOR) {
          DemoteInstrs.push_back(&MI);
        } else if (isWaterfallStart(MI.getOpcode())) {
          HasWaterfalls = true;
        } else if (WQMOutputs) {
          // The function is in machine SSA form, which means that physical
          // VGPRs correspond to shader inputs and outputs. Inputs are
          // only used, outputs are only defined.
          for (const MachineOperand &MO : MI.defs()) {
            if (!MO.isReg())
              continue;

            Register Reg = MO.getReg();

            if (!Reg.isVirtual() &&
                TRI->hasVectorRegisters(TRI->getPhysRegClass(Reg))) {
              Flags = StateWQM;
              break;
            }
          }
        }

        if (!Flags)
          continue;
      }

      markInstruction(MI, Flags, Worklist);
      GlobalFlags |= Flags;
    }
  }

  // Mark sure that any SET_INACTIVE instructions are computed in WQM if WQM is
  // ever used anywhere in the function. This implements the corresponding
  // semantics of @llvm.amdgcn.set.inactive.
  // Similarly for SOFT_WQM instructions, implementing @llvm.amdgcn.softwqm.
  if (GlobalFlags & StateWQM) {
    for (MachineInstr *MI : SetInactiveInstrs)
      markInstruction(*MI, StateWQM, Worklist);
    for (MachineInstr *MI : SoftWQMInstrs)
      markInstruction(*MI, StateWQM, Worklist);
  }

  return GlobalFlags;
}

void SIWholeQuadMode::propagateInstruction(MachineInstr &MI,
                                           std::vector<WorkItem>& Worklist) {
  MachineBasicBlock *MBB = MI.getParent();
  InstrInfo II = Instructions[&MI]; // take a copy to prevent dangling references
  BlockInfo &BI = Blocks[MBB];

  // Control flow-type instructions and stores to temporary memory that are
  // followed by WQM computations must themselves be in WQM.
  if ((II.OutNeeds & StateWQM) && !(II.Disabled & StateWQM) &&
      (MI.isTerminator() || (TII->usesVM_CNT(MI) && MI.mayStore()))) {
    Instructions[&MI].Needs = StateWQM;
    II.Needs = StateWQM;
  }

  // Propagate to block level
  if (II.Needs & StateWQM) {
    BI.Needs |= StateWQM;
    if (!(BI.InNeeds & StateWQM)) {
      BI.InNeeds |= StateWQM;
      Worklist.push_back(MBB);
    }
  }

  // Propagate backwards within block
  if (MachineInstr *PrevMI = MI.getPrevNode()) {
    char InNeeds = (II.Needs & ~StateWWM) | II.OutNeeds;
    if (!PrevMI->isPHI()) {
      InstrInfo &PrevII = Instructions[PrevMI];
      if ((PrevII.OutNeeds | InNeeds) != PrevII.OutNeeds) {
        PrevII.OutNeeds |= InNeeds;
        Worklist.push_back(PrevMI);
      }
    }
  }

  // Propagate WQM flag to instruction inputs
  assert(!(II.Needs & StateExact));

  if (II.Needs != 0)
    markInstructionUses(MI, II.Needs, Worklist);

  // Ensure we process a block containing WWM, even if it does not require any
  // WQM transitions.
  if (II.Needs & StateWWM)
    BI.Needs |= StateWWM;
}

void SIWholeQuadMode::propagateBlock(MachineBasicBlock &MBB,
                                     std::vector<WorkItem>& Worklist) {
  BlockInfo BI = Blocks[&MBB]; // Make a copy to prevent dangling references.

  // Propagate through instructions
  if (!MBB.empty()) {
    MachineInstr *LastMI = &*MBB.rbegin();
    InstrInfo &LastII = Instructions[LastMI];
    if ((LastII.OutNeeds | BI.OutNeeds) != LastII.OutNeeds) {
      LastII.OutNeeds |= BI.OutNeeds;
      Worklist.push_back(LastMI);
    }
  }

  // Predecessor blocks must provide for our WQM/Exact needs.
  for (MachineBasicBlock *Pred : MBB.predecessors()) {
    BlockInfo &PredBI = Blocks[Pred];
    if ((PredBI.OutNeeds | BI.InNeeds) == PredBI.OutNeeds)
      continue;

    PredBI.OutNeeds |= BI.InNeeds;
    PredBI.InNeeds |= BI.InNeeds;
    Worklist.push_back(Pred);
  }

  // All successors must be prepared to accept the same set of WQM/Exact data.
  for (MachineBasicBlock *Succ : MBB.successors()) {
    BlockInfo &SuccBI = Blocks[Succ];
    if ((SuccBI.InNeeds | BI.OutNeeds) == SuccBI.InNeeds)
      continue;

    SuccBI.InNeeds |= BI.OutNeeds;
    Worklist.push_back(Succ);
  }
}

char SIWholeQuadMode::analyzeFunction(MachineFunction &MF) {
  std::vector<WorkItem> Worklist;
  char GlobalFlags = scanInstructions(MF, Worklist);

  while (!Worklist.empty()) {
    WorkItem WI = Worklist.back();
    Worklist.pop_back();

    if (WI.MI)
      propagateInstruction(*WI.MI, Worklist);
    else
      propagateBlock(*WI.MBB, Worklist);
  }

  return GlobalFlags;
}

// Trace live mask manipulate through block, creating new virtual registers.
// Additionally insert PHI nodes when block has multiple predecessors
// which manipulated the mask.
void SIWholeQuadMode::scanLiveLanes(MachineBasicBlock &MBB,
                                    std::vector<MachineBasicBlock *> &Worklist) {
  BlockInfo &BI = Blocks[&MBB];

  if (BI.LiveMaskIn && BI.LiveMaskOut)
    return; // Block has been fully traced already.

  if (!BI.LiveMaskIn) {
    // Find the incoming live mask, or insert PHI if there are multiple.
    unsigned LastPredReg = 0;
    unsigned Count = 0;
    bool Valid = true;

    // Find predecessor live masks while performing basic deduplication.
    for (MachineBasicBlock *Pred : MBB.predecessors()) {
      BlockInfo &PredBI = Blocks[Pred];
      if (!PredBI.LiveMaskOut) {
        Valid = false;
        break;
      }
      if (PredBI.LiveMaskOut != LastPredReg) {
        LastPredReg = PredBI.LiveMaskOut;
        Count++;
      }
    }

    if (Valid) {
      // All predecessors have live mask outputs.
      if (Count > 1) {
        BI.LiveMaskIn = MRI->createVirtualRegister(TRI->getBoolRC());
        MachineInstrBuilder PHI = BuildMI(MBB, MBB.begin(), DebugLoc(),
                                    TII->get(TargetOpcode::PHI),
                                    BI.LiveMaskIn);
        for (MachineBasicBlock *Pred : MBB.predecessors()) {
          BlockInfo &PredBI = Blocks[Pred];
          PHI.addReg(PredBI.LiveMaskOut);
          PHI.addMBB(Pred);
        }
        LIS->InsertMachineInstrInMaps(*PHI);
      } else {
        BI.LiveMaskIn = LastPredReg;
      }
    } else {
      // Not all predecessor blocks have live mask outputs,
      // so this block will need to be revisited.

      if (!BI.LiveMaskOut) {
        // Give this block a live mask output to ensure forward progress.
        BI.LiveMaskOut = MRI->createVirtualRegister(TRI->getBoolRC());
      }

      // Queue this block to be revisited and visit predecessors.
      Worklist.push_back(&MBB);
      for (MachineBasicBlock *Pred : MBB.predecessors()) {
        BlockInfo &PredBI = Blocks[Pred];
        if (!PredBI.LiveMaskOut)
          Worklist.push_back(Pred);
      }
      return;
    }
  }

  assert(BI.LiveMaskIn);

  // Now that the initial live mask register is known the block can
  // be traced and intermediate live mask registers assigned for instructions
  // which manipulate the mask.
  unsigned CurrentLive = BI.LiveMaskIn;
  auto II = MBB.getFirstNonPHI(), IE = MBB.end();
  while (II != IE) {
    MachineInstr &MI = *II;
    if ((MI.getOpcode() == AMDGPU::SI_DEMOTE_I1) ||
        (MI.getOpcode() == AMDGPU::SI_KILL_I1_TERMINATOR)) {
      unsigned NewLive = MRI->createVirtualRegister(TRI->getBoolRC());
      LiveMaskRegs[&MI] = NewLive;
      CurrentLive = NewLive;
    }
    II++;
  }

  // If an output register was assigned to guarantee forward progress
  // then it is possible the current live register will not become the output
  // live mask register.  This will be resolved during block lowering.
  if (!BI.LiveMaskOut) {
    BI.LiveMaskOut = CurrentLive;
  }
}

// Scan blocks for live mask manipulation operations in reverse post order
// to minimise rescans: a block will have to be rescanned if it's
// predecessors live mask output is not defined.
void SIWholeQuadMode::analyzeLiveLanes(MachineFunction &MF) {
  std::vector<MachineBasicBlock *> Worklist;

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (auto BI = RPOT.begin(), BE = RPOT.end(); BI != BE; ++BI) {
    MachineBasicBlock &MBB = **BI;
    scanLiveLanes(MBB, Worklist);
  }

  while (!Worklist.empty()) {
    MachineBasicBlock *MBB = Worklist.back();
    Worklist.pop_back();
    scanLiveLanes(*MBB, Worklist);
  }
}

MachineBasicBlock::iterator
SIWholeQuadMode::saveSCC(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator Before) {
  Register SaveReg = MRI->createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);

  MachineInstr *Save =
      BuildMI(MBB, Before, DebugLoc(), TII->get(AMDGPU::COPY), SaveReg)
          .addReg(AMDGPU::SCC);
  MachineInstr *Restore =
      BuildMI(MBB, Before, DebugLoc(), TII->get(AMDGPU::COPY), AMDGPU::SCC)
          .addReg(SaveReg);

  LIS->InsertMachineInstrInMaps(*Save);
  LIS->InsertMachineInstrInMaps(*Restore);
  LIS->createAndComputeVirtRegInterval(SaveReg);

  return Restore;
}

// Return an iterator in the (inclusive) range [First, Last] at which
// instructions can be safely inserted, keeping in mind that some of the
// instructions we want to add necessarily clobber SCC.
MachineBasicBlock::iterator SIWholeQuadMode::prepareInsertion(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator First,
    MachineBasicBlock::iterator Last, bool PreferLast, bool SaveSCC,
    bool CheckPhys) {
  if (!SaveSCC)
    return PreferLast ? Last : First;

  LiveRange &LR =
      LIS->getRegUnit(*MCRegUnitIterator(MCRegister::from(AMDGPU::SCC), TRI));
  auto MBBE = MBB.end();
  SlotIndex FirstIdx = First != MBBE ? LIS->getInstructionIndex(*First)
                                     : LIS->getMBBEndIdx(&MBB);
  SlotIndex LastIdx =
      Last != MBBE ? LIS->getInstructionIndex(*Last) : LIS->getMBBEndIdx(&MBB);
  SlotIndex Idx = PreferLast ? LastIdx : FirstIdx;
  const LiveRange::Segment *S;

  for (;;) {
    S = LR.getSegmentContaining(Idx);
    if (!S)
      break;

    if (PreferLast) {
      SlotIndex Next = S->start.getBaseIndex();
      if (Next < FirstIdx)
        break;
      Idx = Next;
    } else {
      SlotIndex Next = S->end.getNextIndex().getBaseIndex();
      if (Next > LastIdx)
        break;
      Idx = Next;
    }
  }

  MachineBasicBlock::iterator MBBI;

  if (MachineInstr *MI = LIS->getInstructionFromIndex(Idx)) {
    MBBI = MI;

    if (HasWaterfalls) {
      MachineBasicBlock::iterator I = MBBI;
      bool IsMaybeWaterfall = true;
      while ((I != Last) && IsMaybeWaterfall) {
        switch (I->getOpcode()) {
        case AMDGPU::SI_WATERFALL_BEGIN_V1:
        case AMDGPU::SI_WATERFALL_BEGIN_V2:
        case AMDGPU::SI_WATERFALL_BEGIN_V4:
        case AMDGPU::SI_WATERFALL_BEGIN_V8:
          // Waterfalls do not nest, so not inside waterfall
          IsMaybeWaterfall = false;
          break;
        case AMDGPU::SI_WATERFALL_END_V1:
        case AMDGPU::SI_WATERFALL_END_V2:
        case AMDGPU::SI_WATERFALL_END_V4:
        case AMDGPU::SI_WATERFALL_END_V8:
          // Waterfall has ended
          MBBI = I;
          MBBI++;
          IsMaybeWaterfall = false;
          break;
        default:
          break;
        }
        I++;
      }
    }

    if (CheckPhys) {
      // Make sure insertion point is after any COPY instructions
      // accessing physical live in registers.  This is ensures that
      // block splitting does not occur before all live ins have been copied.
      while (MBBI != Last) {
        if (MBBI->getOpcode() != AMDGPU::COPY)
          break;
        unsigned Src = MBBI->getOperand(1).getReg();
        if (!Register::isVirtualRegister(Src) && MBB.isLiveIn(Src)) {
          MBBI++;
        } else {
          break;
        }
      }
    }
  } else {
    assert(Idx == LIS->getMBBEndIdx(&MBB));
    MBBI = MBB.end();
  }

  if (S)
    MBBI = saveSCC(MBB, MBBI);

  return MBBI;
}

void SIWholeQuadMode::toExact(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator Before,
                              unsigned SaveWQM, unsigned LiveMaskReg) {
  MachineInstr *MI;

  if (SaveWQM) {
    MI = BuildMI(MBB, Before, DebugLoc(), TII->get(ST->isWave32() ?
                   AMDGPU::S_AND_SAVEEXEC_B32 : AMDGPU::S_AND_SAVEEXEC_B64),
                 SaveWQM)
             .addReg(LiveMaskReg);
  } else {
    unsigned Exec = ST->isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
    MI = BuildMI(MBB, Before, DebugLoc(), TII->get(ST->isWave32() ?
                   AMDGPU::S_AND_B32 : AMDGPU::S_AND_B64),
                 Exec)
             .addReg(Exec)
             .addReg(LiveMaskReg);
  }

  LIS->InsertMachineInstrInMaps(*MI);
  StateTransition[MI] = StateExact;
}

void SIWholeQuadMode::toWQM(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator Before,
                            unsigned SavedWQM) {
  MachineInstr *MI;

  unsigned Exec = ST->isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
  if (SavedWQM) {
    MI = BuildMI(MBB, Before, DebugLoc(), TII->get(AMDGPU::COPY), Exec)
             .addReg(SavedWQM);
  } else {
    MI = BuildMI(MBB, Before, DebugLoc(), TII->get(ST->isWave32() ?
                   AMDGPU::S_WQM_B32 : AMDGPU::S_WQM_B64),
                 Exec)
             .addReg(Exec);
  }

  LIS->InsertMachineInstrInMaps(*MI);
  StateTransition[MI] = StateWQM;
}

void SIWholeQuadMode::toWWM(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator Before,
                            unsigned SaveOrig) {
  MachineInstr *MI;

  assert(SaveOrig);
  MI = BuildMI(MBB, Before, DebugLoc(), TII->get(AMDGPU::ENTER_WWM), SaveOrig)
           .addImm(-1);
  LIS->InsertMachineInstrInMaps(*MI);
  StateTransition[MI] = StateWWM;
}

void SIWholeQuadMode::fromWWM(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator Before,
                              unsigned SavedOrig,
                              char NonWWMState) {
  MachineInstr *MI;

  assert(SavedOrig);
  MI = BuildMI(MBB, Before, DebugLoc(), TII->get(AMDGPU::EXIT_WWM),
               ST->isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC)
           .addReg(SavedOrig);
  LIS->InsertMachineInstrInMaps(*MI);
  StateTransition[MI] = NonWWMState;
}

void SIWholeQuadMode::lowerLiveMaskQuery(MachineBasicBlock &MBB,
                                         MachineInstr &MI,
                                         unsigned LiveMaskReg,
                                         bool isWQM) {
  const DebugLoc &DL = MI.getDebugLoc();
  unsigned Dest = MI.getOperand(0).getReg();
  MachineInstr *Copy =
      BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), Dest)
          .addReg(LiveMaskReg);
  LIS->ReplaceMachineInstrInMaps(MI, *Copy);
  MBB.remove(&MI);
}

MachineInstr *SIWholeQuadMode::insertDemoteCleanup(MachineBasicBlock &MBB,
                                  MachineInstr *MI,
                                  MachineBasicBlock::iterator *Before,
                                  unsigned LiveMask) {
  const DebugLoc &DL = DebugLoc();
  const unsigned TermOp = ST->isWave32() ?
    AMDGPU::SI_DEMOTE_CLEANUP_B32_TERMINATOR :
    AMDGPU::SI_DEMOTE_CLEANUP_B64_TERMINATOR;
  const unsigned WQMOp = ST->isWave32() ?
    AMDGPU::S_WQM_B32 : AMDGPU::S_WQM_B64;
  unsigned LiveMaskWQM = MRI->createVirtualRegister(TRI->getBoolRC());

  MachineInstr *LiveMaskMI =
    BuildMI(MBB, MI ? *MI : *Before, DL, TII->get(WQMOp), LiveMaskWQM)
      .addReg(LiveMask);
  MachineInstr *NewTerm =
    BuildMI(MBB, MI ? *MI : *Before, DL, TII->get(TermOp))
      .addReg(LiveMaskWQM);

  LIS->InsertMachineInstrInMaps(*LiveMaskMI);
  LIS->InsertMachineInstrInMaps(*NewTerm);

  return NewTerm;
}

// Lower an instruction which demotes lanes to helpers by adding
// appropriate live mask manipulation.  Note this is also applied to kills.
MachineInstr *SIWholeQuadMode::lowerDemote(MachineBasicBlock &MBB, MachineInstr &MI,
                                  unsigned LiveMaskIn, unsigned LiveMaskOut,
                                  bool isWQM) {
  const unsigned Exec = ST->isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
  const unsigned AndN2 =
    ST->isWave32() ? AMDGPU::S_ANDN2_B32 : AMDGPU::S_ANDN2_B64;
  const unsigned Xor =
    ST->isWave32() ? AMDGPU::S_XOR_B32 : AMDGPU::S_XOR_B64;

  const DebugLoc &DL = MI.getDebugLoc();
  MachineInstr *NewMI = nullptr;
  Register KillMaskReg = Register();

  const MachineOperand &Op = MI.getOperand(0);
  int64_t KillVal = MI.getOperand(1).getImm();
  if (Op.isImm()) {
    int64_t Imm = Op.getImm();
    if (Imm == KillVal) {
      NewMI = BuildMI(MBB, MI, DL,
                      TII->get(AndN2),
                      LiveMaskOut)
        .addReg(LiveMaskIn)
        .addReg(Exec);
    }
  } else {
    if (!KillVal) {
      const TargetRegisterClass *BoolRC = TRI->getBoolRC();
      KillMaskReg = MRI->createVirtualRegister(BoolRC);
      MachineInstr *ComputeKilledMaskMI =
        BuildMI(MBB, MI, DL,
                TII->get(Xor),
                KillMaskReg)
        .add(Op)
        .addReg(Exec);
      NewMI = BuildMI(MBB, MI, DL,
                      TII->get(AndN2),
                      LiveMaskOut)
        .addReg(LiveMaskIn)
        .addReg(KillMaskReg);
      LIS->InsertMachineInstrInMaps(*ComputeKilledMaskMI);
    } else {
      NewMI = BuildMI(MBB, MI, DL,
                      TII->get(AndN2),
                      LiveMaskOut)
        .addReg(LiveMaskIn)
        .add(Op);
    }
  }

  if (NewMI) {
    LIS->InsertMachineInstrInMaps(*NewMI);
    if (KillMaskReg)
      LIS->createAndComputeVirtRegInterval(KillMaskReg);
  }

  MachineInstr *TermMI = BuildMI(MBB, MI, DL, TII->get(AMDGPU::SI_EARLY_TERMINATE_SCC0));
  LIS->InsertMachineInstrInMaps(*TermMI);

  if (MI.getOpcode() == AMDGPU::SI_DEMOTE_I1) {
    if (isWQM) {
      // Inside WQM demotes are replaced with live mask manipulation
      // and a terminator which is later lowered to remove unused helpers
      MachineInstr *NewTerm = insertDemoteCleanup(MBB, &MI, nullptr, LiveMaskOut);
      LIS->RemoveMachineInstrFromMaps(MI);
      MBB.remove(&MI);
      return NewTerm;
    } else {
      // Outside WQM demotes become kills terminating the block
      MI.setDesc(TII->get(AMDGPU::SI_KILL_I1_TERMINATOR));
      return &MI;
    }
  }

  return nullptr;
}

bool SIWholeQuadMode::canSplitBlockAt(MachineBasicBlock *BB,
                                      MachineInstr *MI) {
  // Cannot split immediately before the epilog
  // because there are values in physical registers
  if (MI->getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG) {
    return false;
  }

  // Do not split inside a waterfall intrinsic pair
  MachineBasicBlock::iterator II = BB->getFirstNonPHI();
  MachineBasicBlock::iterator IE = BB->end();
  bool InWaterfall = false;
  while (II != IE) {
    if (&*II == MI)
      return !InWaterfall;

    switch (II->getOpcode()) {
    case AMDGPU::SI_WATERFALL_BEGIN_V1:
    case AMDGPU::SI_WATERFALL_BEGIN_V2:
    case AMDGPU::SI_WATERFALL_BEGIN_V4:
    case AMDGPU::SI_WATERFALL_BEGIN_V8:
      InWaterfall = true;
      break;
    case AMDGPU::SI_WATERFALL_END_V1:
    case AMDGPU::SI_WATERFALL_END_V2:
    case AMDGPU::SI_WATERFALL_END_V4:
    case AMDGPU::SI_WATERFALL_END_V8:
      InWaterfall = false;
      break;
    default:
      break;
    }
    II++;
  }

  return true;
}

MachineBasicBlock *SIWholeQuadMode::splitBlock(MachineBasicBlock *BB,
                                               MachineInstr *TermMI) {
  MachineBasicBlock::iterator SplitPoint(TermMI);
  SplitPoint++;

  LLVM_DEBUG(dbgs() << "Split block " << printMBBReference(*BB)
                    << " @ " << *TermMI << "\n");

  MachineBasicBlock *SplitBB = nullptr;

  // Only split the block if the split point is not
  // already the end of the block.
  if ((SplitPoint != BB->getFirstTerminator()) &&
      (SplitPoint != BB->end())) {
    MachineFunction *MF = BB->getParent();
    SplitBB = MF->CreateMachineBasicBlock(BB->getBasicBlock());

    MachineFunction::iterator MBBI(BB);
    ++MBBI;
    MF->insert(MBBI, SplitBB);

    SplitBB->splice(SplitBB->begin(), BB, SplitPoint, BB->end());
    SplitBB->transferSuccessorsAndUpdatePHIs(BB);
    BB->addSuccessor(SplitBB);

    // Update dominator trees
    using DomTreeT = DomTreeBase<MachineBasicBlock>;
    SmallVector<DomTreeT::UpdateType, 16> DTUpdates;
    for (MachineBasicBlock *Succ : SplitBB->successors()) {
      DTUpdates.push_back({DomTreeT::Insert, SplitBB, Succ});
      DTUpdates.push_back({DomTreeT::Delete, BB, Succ});
    }
    DTUpdates.push_back({DomTreeT::Insert, BB, SplitBB});
    if (MDT)
      MDT->getBase().applyUpdates(DTUpdates);
    if (PDT)
      PDT->getBase().applyUpdates(DTUpdates);

    // Update live intervals
    MachineInstr &InsertionPoint = SplitBB->front();
    LIS->insertMBBInMaps(SplitBB, &InsertionPoint);
  }

  // Convert last instruction in to a terminator.
  // Note: this only covers the expected patterns
  switch (TermMI->getOpcode()) {
  case AMDGPU::S_AND_B32:
    TermMI->setDesc(TII->get(AMDGPU::S_AND_B32_term));
    break;
  case AMDGPU::S_AND_B64:
    TermMI->setDesc(TII->get(AMDGPU::S_AND_B64_term));
    break;
  default:
    if (BB->getFirstTerminator() == BB->end()) {
      assert(SplitBB != nullptr);
      MachineInstr *MI = BuildMI(*BB, BB->end(), DebugLoc(), TII->get(AMDGPU::S_BRANCH))
        .addMBB(SplitBB);
      LIS->InsertMachineInstrInMaps(*MI);
    }
    break;
  }

  return SplitBB;
}

// Replace (or supplement) instructions accessing live mask.
// This can only happen once all the live mask registers have been created
// and the execute state (WQM/WWM/Exact) of instructions is known.
void SIWholeQuadMode::lowerBlock(MachineBasicBlock &MBB) {
  auto BII = Blocks.find(&MBB);
  if (BII == Blocks.end())
    return;

  LLVM_DEBUG(dbgs() << "\nLowering block " << printMBBReference(MBB)
                    << ":\n");

  const BlockInfo &BI = BII->second;

  SmallVector<MachineInstr *, 4> SplitPoints;
  unsigned LiveMaskReg = BI.LiveMaskIn;
  char State = BI.InitialState;

  auto II = MBB.getFirstNonPHI(), IE = MBB.end();
  while (II != IE) {
    auto Next = std::next(II);
    MachineInstr &MI = *II;

    if (StateTransition.count(&MI)) {
      // Mark transitions to Exact mode as split points so they become
      // block terminators.
      if (State != StateTransition[&MI] && StateTransition[&MI] == StateExact) {
        if (State != StateWWM && canSplitBlockAt(&MBB, &MI))
          SplitPoints.push_back(&MI);
      }
      State = StateTransition[&MI];
    }

    switch (MI.getOpcode()) {
    case AMDGPU::SI_PS_LIVE:
    case AMDGPU::SI_WQM_HELPER:
      lowerLiveMaskQuery(MBB, MI, LiveMaskReg, State == StateWQM);
      break;
    case AMDGPU::SI_KILL_I1_TERMINATOR:
      lowerDemote(MBB, MI, LiveMaskReg, LiveMaskRegs[&MI], State == StateWQM);
      break;
    case AMDGPU::SI_DEMOTE_I1: {
      MachineInstr *SplitPoint = lowerDemote(MBB, MI, LiveMaskReg,
                                   LiveMaskRegs[&MI],
                                   State == StateWQM);
      if (SplitPoint)
        SplitPoints.push_back(SplitPoint);
      break;
    }
    case AMDGPU::SI_END_CF:
      if ((State == StateWQM) && NeedsDemoteCleanup.count(&MI)) {
        MachineInstr *NewTerm = insertDemoteCleanup(MBB, nullptr, &Next, LiveMaskReg);
        SplitPoints.push_back(NewTerm);
      }
      break;
    default:
      break;
    }

    if (LiveMaskRegs.count(&MI))
      LiveMaskReg = LiveMaskRegs[&MI];

    II = Next;
  }

  if (BI.LiveMaskOut != LiveMaskReg) {
    // If the final live mask register does not match the expected
    // register of successor blocks then insert a copy.
    MachineBasicBlock::instr_iterator Terminator =
      MBB.getFirstInstrTerminator();
    MachineInstr *MI = BuildMI(MBB, Terminator, DebugLoc(),
                               TII->get(AMDGPU::COPY), BI.LiveMaskOut)
      .addReg(LiveMaskReg);
    LIS->InsertMachineInstrInMaps(*MI);
  }

  // Perform splitting after instruction scan to simplify iteration.
  if (!SplitPoints.empty()) {
    MachineBasicBlock *BB = &MBB;
    for (MachineInstr *MI : SplitPoints) {
      BB = splitBlock(BB, MI);
    }
  }
}

unsigned SIWholeQuadMode::findLiveMaskReg(MachineBasicBlock &MBB, BlockInfo &BI,
                                          MachineBasicBlock::iterator &Before) {
  assert(BI.LiveMaskIn);
  if (BI.LiveMaskIn == BI.LiveMaskOut)
    return BI.LiveMaskIn;

  // FIXME: make this more efficient than scanning all instructions in a block
  unsigned LiveMaskReg = BI.LiveMaskIn;
  auto II = MBB.getFirstNonPHI(), IE = MBB.end();

  while ((II != IE) && (II != Before)) {
    MachineInstr *I = &*II;
    if (LiveMaskRegs.count(I))
      LiveMaskReg = LiveMaskRegs[I];
    II++;
  }

  assert(LiveMaskReg);
  return LiveMaskReg;
}

void SIWholeQuadMode::processBlock(MachineBasicBlock &MBB, bool isEntry) {
  auto BII = Blocks.find(&MBB);
  if (BII == Blocks.end())
    return;

  BlockInfo &BI = BII->second;

  // This is a non-entry block that is WQM throughout, so no need to do
  // anything.
  if (!isEntry && BI.Needs == StateWQM && BI.OutNeeds != StateExact) {
    BI.InitialState = StateWQM;
    return;
  }

  LLVM_DEBUG(dbgs() << "\nProcessing block " << printMBBReference(MBB)
                    << ":\n");

  unsigned SavedWQMReg = 0;
  unsigned SavedNonWWMReg = 0;
  bool WQMFromExec = isEntry;
  char State = (isEntry || !(BI.InNeeds & StateWQM)) ? StateExact : StateWQM;
  char NonWWMState = 0;
  const TargetRegisterClass *BoolRC = TRI->getBoolRC();

  auto II = MBB.getFirstNonPHI(), IE = MBB.end();
  if (isEntry)
    ++II; // Skip the instruction that saves LiveMask

  // This stores the first instruction where it's safe to switch from WQM to
  // Exact or vice versa.
  MachineBasicBlock::iterator FirstWQM = IE;

  // This stores the first instruction where it's safe to switch from WWM to
  // Exact/WQM or to switch to WWM. It must always be the same as, or after,
  // FirstWQM since if it's safe to switch to/from WWM, it must be safe to
  // switch to/from WQM as well.
  MachineBasicBlock::iterator FirstWWM = IE;

  // Record initial state is block information.
  BI.InitialState = State;

  for (;;) {
    MachineBasicBlock::iterator Next = II;
    char Needs = StateExact | StateWQM; // WWM is disabled by default
    char OutNeeds = 0;

    if (FirstWQM == IE)
      FirstWQM = II;

    if (FirstWWM == IE)
      FirstWWM = II;

    // First, figure out the allowed states (Needs) based on the propagated
    // flags.
    if (II != IE) {
      MachineInstr &MI = *II;

      if (MI.isTerminator() || TII->mayReadEXEC(*MRI, MI)) {
        auto III = Instructions.find(&MI);
        if (III != Instructions.end()) {
          if (III->second.Needs & StateWWM)
            Needs = StateWWM;
          else if (III->second.Needs & StateWQM)
            Needs = StateWQM;
          else
            Needs &= ~III->second.Disabled;
          OutNeeds = III->second.OutNeeds;
        }
      } else {
        // If the instruction doesn't actually need a correct EXEC, then we can
        // safely leave WWM enabled.
        Needs = StateExact | StateWQM | StateWWM;
      }

      if (MI.isTerminator() && OutNeeds == StateExact)
        Needs = StateExact;

      ++Next;
    } else {
      // End of basic block
      if (BI.OutNeeds & StateWQM)
        Needs = StateWQM;
      else if (BI.OutNeeds == StateExact)
        Needs = StateExact;
      else
        Needs = StateWQM | StateExact;
    }

    // Now, transition if necessary.
    if (!(Needs & State)) {
      MachineBasicBlock::iterator First;
      if (State == StateWWM || Needs == StateWWM) {
        // We must switch to or from WWM
        First = FirstWWM;
      } else {
        // We only need to switch to/from WQM, so we can use FirstWQM
        First = FirstWQM;
      }

      MachineBasicBlock::iterator Before =
          prepareInsertion(MBB, First, II, Needs == StateWQM,
                           Needs == StateExact || WQMFromExec,
                           Needs == StateExact && isEntry);

      if (State == StateWWM) {
        assert(SavedNonWWMReg);
        fromWWM(MBB, Before, SavedNonWWMReg, NonWWMState);
        LIS->createAndComputeVirtRegInterval(SavedNonWWMReg);
        SavedNonWWMReg = 0;
        State = NonWWMState;
      }

      if (Needs == StateWWM) {
        NonWWMState = State;
        assert(!SavedNonWWMReg);
        SavedNonWWMReg = MRI->createVirtualRegister(BoolRC);
        toWWM(MBB, Before, SavedNonWWMReg);
        State = StateWWM;
      } else {
        if (State == StateWQM && (Needs & StateExact) && !(Needs & StateWQM)) {
          if (!WQMFromExec && (OutNeeds & StateWQM)) {
            assert(!SavedWQMReg);
            SavedWQMReg = MRI->createVirtualRegister(BoolRC);
          }

          toExact(MBB, Before, SavedWQMReg, findLiveMaskReg(MBB, BI, Before));
          State = StateExact;
        } else if (State == StateExact && (Needs & StateWQM) &&
                   !(Needs & StateExact)) {
          assert(WQMFromExec == (SavedWQMReg == 0));

          toWQM(MBB, Before, SavedWQMReg);

          if (SavedWQMReg) {
            LIS->createAndComputeVirtRegInterval(SavedWQMReg);
            SavedWQMReg = 0;
          }
          State = StateWQM;
        } else {
          // We can get here if we transitioned from WWM to a non-WWM state that
          // already matches our needs, but we shouldn't need to do anything.
          assert(Needs & State);
        }
      }
    }

    if (Needs != (StateExact | StateWQM | StateWWM)) {
      if (Needs != (StateExact | StateWQM))
        FirstWQM = IE;
      FirstWWM = IE;
    }

    if (II == IE)
      break;

    II = Next;
  }
  assert(!SavedWQMReg);
  assert(!SavedNonWWMReg);
}

bool SIWholeQuadMode::lowerLiveMaskQueries(unsigned LiveMaskReg) {
  bool Changed = false;
  for (MachineInstr *MI : LiveMaskQueries) {
    const DebugLoc &DL = MI->getDebugLoc();
    Register Dest = MI->getOperand(0).getReg();
    MachineInstr *Copy =
        BuildMI(*MI->getParent(), MI, DL, TII->get(AMDGPU::COPY), Dest)
            .addReg(LiveMaskReg);

    LIS->ReplaceMachineInstrInMaps(*MI, *Copy);
    MI->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

bool SIWholeQuadMode::lowerDemoteInstrs() {
  bool Changed = false;
  for (MachineInstr *MI : DemoteInstrs) {
    if (MI->getOpcode() != AMDGPU::SI_DEMOTE_I1)
      continue;
    MachineBasicBlock *MBB = MI->getParent();
    MI->setDesc(TII->get(AMDGPU::SI_KILL_I1_TERMINATOR));
    splitBlock(MBB, MI);
    Changed = true;
  }
  return Changed;
}

void SIWholeQuadMode::lowerCopyInstrs() {
  for (MachineInstr *MI : LowerToMovInstrs) {
    assert(MI->getNumExplicitOperands() == 2);

    const Register Reg = MI->getOperand(0).getReg();

    if (TRI->isVGPR(*MRI, Reg)) {
      const TargetRegisterClass *regClass =
          Reg.isVirtual() ? MRI->getRegClass(Reg) : TRI->getPhysRegClass(Reg);

      const unsigned MovOp = TII->getMovOpcode(regClass);
      MI->setDesc(TII->get(MovOp));

      // And make it implicitly depend on exec (like all VALU movs should do).
      MI->addOperand(MachineOperand::CreateReg(AMDGPU::EXEC, false, true));
    } else {
      MI->setDesc(TII->get(AMDGPU::COPY));
    }
  }
  for (MachineInstr *MI : LowerToCopyInstrs) {
    if (MI->getOpcode() == AMDGPU::V_SET_INACTIVE_B32 ||
        MI->getOpcode() == AMDGPU::V_SET_INACTIVE_B64) {
      assert(MI->getNumExplicitOperands() == 3);
      // the only reason we should be here is V_SET_INACTIVE has
      // an undef input so it is being replaced by a simple copy.
      // There should be a second undef source that we should remove.
      assert(MI->getOperand(2).isUndef());
      MI->RemoveOperand(2);
      MI->untieRegOperand(1);
    } else {
      assert(MI->getNumExplicitOperands() == 2);
    }

    MI->setDesc(TII->get(AMDGPU::COPY));
  }
}

bool SIWholeQuadMode::runOnMachineFunction(MachineFunction &MF) {
  Instructions.clear();
  Blocks.clear();
  LiveMaskQueries.clear();
  LowerToCopyInstrs.clear();
  LowerToMovInstrs.clear();
  DemoteInstrs.clear();
  LiveMaskRegs.clear();
  StateTransition.clear();
  HasWaterfalls = false;

  CallingConv = MF.getFunction().getCallingConv();

  ST = &MF.getSubtarget<GCNSubtarget>();

  TII = ST->getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();
  LIS = &getAnalysis<LiveIntervals>();
  MDT = &getAnalysis<MachineDominatorTree>();
  PDT = &getAnalysis<MachinePostDominatorTree>();

  const char GlobalFlags = analyzeFunction(MF);
  const bool NeedsLiveMask =
    !(DemoteInstrs.empty() && LiveMaskQueries.empty());
  const unsigned Exec = ST->isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
  unsigned LiveMaskReg = Exec;

  if (!(GlobalFlags & (StateWQM | StateWWM)) && LowerToCopyInstrs.empty() && LowerToMovInstrs.empty()) {
    // Shader only needs Exact mode
    const bool LoweredQueries = lowerLiveMaskQueries(LiveMaskReg);
    const bool LoweredDemotes = lowerDemoteInstrs();
    return LoweredQueries || LoweredDemotes;
  }

  MachineBasicBlock &Entry = MF.front();
  MachineBasicBlock::iterator EntryMI = Entry.getFirstNonPHI();

  // Store a copy of the original live mask when required
  if (NeedsLiveMask || (GlobalFlags & StateWQM)) {
    LiveMaskReg = MRI->createVirtualRegister(TRI->getBoolRC());
    MachineInstr *MI = BuildMI(Entry, EntryMI, DebugLoc(),
                               TII->get(AMDGPU::COPY), LiveMaskReg)
                           .addReg(Exec);
    LIS->InsertMachineInstrInMaps(*MI);
  }

  if ((GlobalFlags == StateWQM) && DemoteInstrs.empty()) {
    // Shader only needs WQM
    auto MI = BuildMI(Entry, EntryMI, DebugLoc(), TII->get(ST->isWave32() ?
              AMDGPU::S_WQM_B32 : AMDGPU::S_WQM_B64),
            Exec)
        .addReg(Exec);
    LIS->InsertMachineInstrInMaps(*MI);

    lowerLiveMaskQueries(LiveMaskReg);
    lowerCopyInstrs();
    return true;
  }

  if (NeedsLiveMask && (GlobalFlags & StateWQM)) {
    BlockInfo &BI = Blocks[&Entry];
    BI.LiveMaskIn = LiveMaskReg;
    analyzeLiveLanes(MF);
  } else {
    for (auto BII : Blocks) {
      BlockInfo &BI = Blocks[&*BII.first];
      BI.LiveMaskIn = LiveMaskReg;
      BI.LiveMaskOut = LiveMaskReg;
    }
  }

  LLVM_DEBUG(printInfo());

  lowerCopyInstrs();

  for (auto BII : Blocks) {
    processBlock(*BII.first, BII.first == &Entry);
  }

  if (NeedsLiveMask && (GlobalFlags & StateWQM)) {
    // Lowering blocks causes block splitting.
    // Hence live ranges and slot indexes cease to be valid here.
    for (auto BII : Blocks) {
      lowerBlock(*BII.first);
    }
  } else {
    lowerLiveMaskQueries(LiveMaskReg);
    lowerDemoteInstrs();
  }

  // Physical registers like SCC aren't tracked by default anyway, so just
  // removing the ranges we computed is the simplest option for maintaining
  // the analysis results.
  LIS->removeRegUnit(*MCRegUnitIterator(MCRegister::from(AMDGPU::SCC), TRI));

  return true;
}
