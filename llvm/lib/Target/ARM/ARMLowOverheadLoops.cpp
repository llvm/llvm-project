//===-- ARMLowOverheadLoops.cpp - CodeGen Low-overhead Loops ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Finalize v8.1-m low-overhead loops by converting the associated pseudo
/// instructions into machine operations.
/// The expectation is that the loop contains three pseudo instructions:
/// - t2*LoopStart - placed in the preheader or pre-preheader. The do-loop
///   form should be in the preheader, whereas the while form should be in the
///   preheaders only predecessor.
/// - t2LoopDec - placed within in the loop body.
/// - t2LoopEnd - the loop latch terminator.
///
/// In addition to this, we also look for the presence of the VCTP instruction,
/// which determines whether we can generated the tail-predicated low-overhead
/// loop form.
///
/// Assumptions and Dependencies:
/// Low-overhead loops are constructed and executed using a setup instruction:
/// DLS, WLS, DLSTP or WLSTP and an instruction that loops back: LE or LETP.
/// WLS(TP) and LE(TP) are branching instructions with a (large) limited range
/// but fixed polarity: WLS can only branch forwards and LE can only branch
/// backwards. These restrictions mean that this pass is dependent upon block
/// layout and block sizes, which is why it's the last pass to run. The same is
/// true for ConstantIslands, but this pass does not increase the size of the
/// basic blocks, nor does it change the CFG. Instructions are mainly removed
/// during the transform and pseudo instructions are replaced by real ones. In
/// some cases, when we have to revert to a 'normal' loop, we have to introduce
/// multiple instructions for a single pseudo (see RevertWhile and
/// RevertLoopEnd). To handle this situation, t2WhileLoopStart and t2LoopEnd
/// are defined to be as large as this maximum sequence of replacement
/// instructions.
///
/// A note on VPR.P0 (the lane mask):
/// VPT, VCMP, VPNOT and VCTP won't overwrite VPR.P0 when they update it in a
/// "VPT Active" context (which includes low-overhead loops and vpt blocks).
/// They will simply "and" the result of their calculation with the current
/// value of VPR.P0. You can think of it like this:
/// \verbatim
/// if VPT active:    ; Between a DLSTP/LETP, or for predicated instrs
///   VPR.P0 &= Value
/// else
///   VPR.P0 = Value
/// \endverbatim
/// When we're inside the low-overhead loop (between DLSTP and LETP), we always
/// fall in the "VPT active" case, so we can consider that all VPR writes by
/// one of those instruction is actually a "and".
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMBasicBlockInfo.h"
#include "ARMSubtarget.h"
#include "Thumb2InstrInfo.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineLoopUtils.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ReachingDefAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"

using namespace llvm;

#define DEBUG_TYPE "arm-low-overhead-loops"
#define ARM_LOW_OVERHEAD_LOOPS_NAME "ARM Low Overhead Loops pass"

static cl::opt<bool>
DisableTailPredication("arm-loloops-disable-tailpred", cl::Hidden,
    cl::desc("Disable tail-predication in the ARM LowOverheadLoop pass"),
    cl::init(false));

static bool isVectorPredicated(MachineInstr *MI) {
  int PIdx = llvm::findFirstVPTPredOperandIdx(*MI);
  return PIdx != -1 && MI->getOperand(PIdx + 1).getReg() == ARM::VPR;
}

static bool isVectorPredicate(MachineInstr *MI) {
  return MI->findRegisterDefOperandIdx(ARM::VPR) != -1;
}

static bool hasVPRUse(MachineInstr *MI) {
  return MI->findRegisterUseOperandIdx(ARM::VPR) != -1;
}

static bool isDomainMVE(MachineInstr *MI) {
  uint64_t Domain = MI->getDesc().TSFlags & ARMII::DomainMask;
  return Domain == ARMII::DomainMVE;
}

static bool shouldInspect(MachineInstr &MI) {
  return isDomainMVE(&MI) || isVectorPredicate(&MI) ||
    hasVPRUse(&MI);
}

namespace {

  using InstSet = SmallPtrSetImpl<MachineInstr *>;

  class PostOrderLoopTraversal {
    MachineLoop &ML;
    MachineLoopInfo &MLI;
    SmallPtrSet<MachineBasicBlock*, 4> Visited;
    SmallVector<MachineBasicBlock*, 4> Order;

  public:
    PostOrderLoopTraversal(MachineLoop &ML, MachineLoopInfo &MLI)
      : ML(ML), MLI(MLI) { }

    const SmallVectorImpl<MachineBasicBlock*> &getOrder() const {
      return Order;
    }

    // Visit all the blocks within the loop, as well as exit blocks and any
    // blocks properly dominating the header.
    void ProcessLoop() {
      std::function<void(MachineBasicBlock*)> Search = [this, &Search]
        (MachineBasicBlock *MBB) -> void {
        if (Visited.count(MBB))
          return;

        Visited.insert(MBB);
        for (auto *Succ : MBB->successors()) {
          if (!ML.contains(Succ))
            continue;
          Search(Succ);
        }
        Order.push_back(MBB);
      };

      // Insert exit blocks.
      SmallVector<MachineBasicBlock*, 2> ExitBlocks;
      ML.getExitBlocks(ExitBlocks);
      for (auto *MBB : ExitBlocks)
        Order.push_back(MBB);

      // Then add the loop body.
      Search(ML.getHeader());

      // Then try the preheader and its predecessors.
      std::function<void(MachineBasicBlock*)> GetPredecessor =
        [this, &GetPredecessor] (MachineBasicBlock *MBB) -> void {
        Order.push_back(MBB);
        if (MBB->pred_size() == 1)
          GetPredecessor(*MBB->pred_begin());
      };

      if (auto *Preheader = ML.getLoopPreheader())
        GetPredecessor(Preheader);
      else if (auto *Preheader = MLI.findLoopPreheader(&ML, true))
        GetPredecessor(Preheader);
    }
  };

  struct PredicatedMI {
    MachineInstr *MI = nullptr;
    SetVector<MachineInstr*> Predicates;

  public:
    PredicatedMI(MachineInstr *I, SetVector<MachineInstr *> &Preds) : MI(I) {
      assert(I && "Instruction must not be null!");
      Predicates.insert(Preds.begin(), Preds.end());
    }
  };

  // Represent the current state of the VPR and hold all instances which
  // represent a VPT block, which is a list of instructions that begins with a
  // VPT/VPST and has a maximum of four proceeding instructions. All
  // instructions within the block are predicated upon the vpr and we allow
  // instructions to define the vpr within in the block too.
  class VPTState {
    friend struct LowOverheadLoop;

    SmallVector<MachineInstr *, 4> Insts;

    static SmallVector<VPTState, 4> Blocks;
    static SetVector<MachineInstr *> CurrentPredicates;
    static std::map<MachineInstr *,
      std::unique_ptr<PredicatedMI>> PredicatedInsts;

    static void CreateVPTBlock(MachineInstr *MI) {
      assert(CurrentPredicates.size() && "Can't begin VPT without predicate");
      Blocks.emplace_back(MI);
      // The execution of MI is predicated upon the current set of instructions
      // that are AND'ed together to form the VPR predicate value. In the case
      // that MI is a VPT, CurrentPredicates will also just be MI.
      PredicatedInsts.emplace(
        MI, std::make_unique<PredicatedMI>(MI, CurrentPredicates));
    }

    static void reset() {
      Blocks.clear();
      PredicatedInsts.clear();
      CurrentPredicates.clear();
    }

    static void addInst(MachineInstr *MI) {
      Blocks.back().insert(MI);
      PredicatedInsts.emplace(
        MI, std::make_unique<PredicatedMI>(MI, CurrentPredicates));
    }

    static void addPredicate(MachineInstr *MI) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Adding VPT Predicate: " << *MI);
      CurrentPredicates.insert(MI);
    }

    static void resetPredicate(MachineInstr *MI) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Resetting VPT Predicate: " << *MI);
      CurrentPredicates.clear();
      CurrentPredicates.insert(MI);
    }

  public:
    // Have we found an instruction within the block which defines the vpr? If
    // so, not all the instructions in the block will have the same predicate.
    static bool hasUniformPredicate(VPTState &Block) {
      return getDivergent(Block) == nullptr;
    }

    // If it exists, return the first internal instruction which modifies the
    // VPR.
    static MachineInstr *getDivergent(VPTState &Block) {
      SmallVectorImpl<MachineInstr *> &Insts = Block.getInsts();
      for (unsigned i = 1; i < Insts.size(); ++i) {
        MachineInstr *Next = Insts[i];
        if (isVectorPredicate(Next))
          return Next; // Found an instruction altering the vpr.
      }
      return nullptr;
    }

    // Return whether the given instruction is predicated upon a VCTP.
    static bool isPredicatedOnVCTP(MachineInstr *MI, bool Exclusive = false) {
      SetVector<MachineInstr *> &Predicates = PredicatedInsts[MI]->Predicates;
      if (Exclusive && Predicates.size() != 1)
        return false;
      for (auto *PredMI : Predicates)
        if (isVCTP(PredMI))
          return true;
      return false;
    }

    // Is the VPST, controlling the block entry, predicated upon a VCTP.
    static bool isEntryPredicatedOnVCTP(VPTState &Block,
                                        bool Exclusive = false) {
      SmallVectorImpl<MachineInstr *> &Insts = Block.getInsts();
      return isPredicatedOnVCTP(Insts.front(), Exclusive);
    }

    // If this block begins with a VPT, we can check whether it's using
    // at least one predicated input(s), as well as possible loop invariant
    // which would result in it being implicitly predicated.
    static bool hasImplicitlyValidVPT(VPTState &Block,
                                      ReachingDefAnalysis &RDA) {
      SmallVectorImpl<MachineInstr *> &Insts = Block.getInsts();
      MachineInstr *VPT = Insts.front();
      assert(isVPTOpcode(VPT->getOpcode()) &&
             "Expected VPT block to begin with VPT/VPST");

      if (VPT->getOpcode() == ARM::MVE_VPST)
        return false;

      auto IsOperandPredicated = [&](MachineInstr *MI, unsigned Idx) {
        MachineInstr *Op = RDA.getMIOperand(MI, MI->getOperand(Idx));
        return Op && PredicatedInsts.count(Op) && isPredicatedOnVCTP(Op);
      };

      auto IsOperandInvariant = [&](MachineInstr *MI, unsigned Idx) {
        MachineOperand &MO = MI->getOperand(Idx);
        if (!MO.isReg() || !MO.getReg())
          return true;

        SmallPtrSet<MachineInstr *, 2> Defs;
        RDA.getGlobalReachingDefs(MI, MO.getReg(), Defs);
        if (Defs.empty())
          return true;

        for (auto *Def : Defs)
          if (Def->getParent() == VPT->getParent())
            return false;
        return true;
      };

      // Check that at least one of the operands is directly predicated on a
      // vctp and allow an invariant value too.
      return (IsOperandPredicated(VPT, 1) || IsOperandPredicated(VPT, 2)) &&
             (IsOperandPredicated(VPT, 1) || IsOperandInvariant(VPT, 1)) &&
             (IsOperandPredicated(VPT, 2) || IsOperandInvariant(VPT, 2));
    }

    static bool isValid(ReachingDefAnalysis &RDA) {
      // All predication within the loop should be based on vctp. If the block
      // isn't predicated on entry, check whether the vctp is within the block
      // and that all other instructions are then predicated on it.
      for (auto &Block : Blocks) {
        if (isEntryPredicatedOnVCTP(Block, false) ||
            hasImplicitlyValidVPT(Block, RDA))
          continue;

        SmallVectorImpl<MachineInstr *> &Insts = Block.getInsts();
        for (auto *MI : Insts) {
          // Check that any internal VCTPs are 'Then' predicated.
          if (isVCTP(MI) && getVPTInstrPredicate(*MI) != ARMVCC::Then)
            return false;
          // Skip other instructions that build up the predicate.
          if (MI->getOpcode() == ARM::MVE_VPST || isVectorPredicate(MI))
            continue;
          // Check that any other instructions are predicated upon a vctp.
          // TODO: We could infer when VPTs are implicitly predicated on the
          // vctp (when the operands are predicated).
          if (!isPredicatedOnVCTP(MI)) {
            LLVM_DEBUG(dbgs() << "ARM Loops: Can't convert: " << *MI);
            return false;
          }
        }
      }
      return true;
    }

    VPTState(MachineInstr *MI) { Insts.push_back(MI); }

    void insert(MachineInstr *MI) {
      Insts.push_back(MI);
      // VPT/VPST + 4 predicated instructions.
      assert(Insts.size() <= 5 && "Too many instructions in VPT block!");
    }

    bool containsVCTP() const {
      for (auto *MI : Insts)
        if (isVCTP(MI))
          return true;
      return false;
    }

    unsigned size() const { return Insts.size(); }
    SmallVectorImpl<MachineInstr *> &getInsts() { return Insts; }
  };

  struct LowOverheadLoop {

    MachineLoop &ML;
    MachineBasicBlock *Preheader = nullptr;
    MachineLoopInfo &MLI;
    ReachingDefAnalysis &RDA;
    const TargetRegisterInfo &TRI;
    const ARMBaseInstrInfo &TII;
    MachineFunction *MF = nullptr;
    MachineInstr *InsertPt = nullptr;
    MachineInstr *Start = nullptr;
    MachineInstr *Dec = nullptr;
    MachineInstr *End = nullptr;
    MachineOperand TPNumElements;
    SmallVector<MachineInstr*, 4> VCTPs;
    SmallPtrSet<MachineInstr*, 4> ToRemove;
    SmallPtrSet<MachineInstr*, 4> BlockMasksToRecompute;
    bool Revert = false;
    bool CannotTailPredicate = false;

    LowOverheadLoop(MachineLoop &ML, MachineLoopInfo &MLI,
                    ReachingDefAnalysis &RDA, const TargetRegisterInfo &TRI,
                    const ARMBaseInstrInfo &TII)
        : ML(ML), MLI(MLI), RDA(RDA), TRI(TRI), TII(TII),
          TPNumElements(MachineOperand::CreateImm(0)) {
      MF = ML.getHeader()->getParent();
      if (auto *MBB = ML.getLoopPreheader())
        Preheader = MBB;
      else if (auto *MBB = MLI.findLoopPreheader(&ML, true))
        Preheader = MBB;
      VPTState::reset();
    }

    // If this is an MVE instruction, check that we know how to use tail
    // predication with it. Record VPT blocks and return whether the
    // instruction is valid for tail predication.
    bool ValidateMVEInst(MachineInstr *MI);

    void AnalyseMVEInst(MachineInstr *MI) {
      CannotTailPredicate = !ValidateMVEInst(MI);
    }

    bool IsTailPredicationLegal() const {
      // For now, let's keep things really simple and only support a single
      // block for tail predication.
      return !Revert && FoundAllComponents() && !VCTPs.empty() &&
             !CannotTailPredicate && ML.getNumBlocks() == 1;
    }

    // Given that MI is a VCTP, check that is equivalent to any other VCTPs
    // found.
    bool AddVCTP(MachineInstr *MI);

    // Check that the predication in the loop will be equivalent once we
    // perform the conversion. Also ensure that we can provide the number
    // of elements to the loop start instruction.
    bool ValidateTailPredicate(MachineInstr *StartInsertPt);

    // Check that any values available outside of the loop will be the same
    // after tail predication conversion.
    bool ValidateLiveOuts();

    // Is it safe to define LR with DLS/WLS?
    // LR can be defined if it is the operand to start, because it's the same
    // value, or if it's going to be equivalent to the operand to Start.
    MachineInstr *isSafeToDefineLR();

    // Check the branch targets are within range and we satisfy our
    // restrictions.
    void Validate(ARMBasicBlockUtils *BBUtils);

    bool FoundAllComponents() const {
      return Start && Dec && End;
    }

    SmallVectorImpl<VPTState> &getVPTBlocks() {
      return VPTState::Blocks;
    }

    // Return the operand for the loop start instruction. This will be the loop
    // iteration count, or the number of elements if we're tail predicating.
    MachineOperand &getLoopStartOperand() {
      return IsTailPredicationLegal() ? TPNumElements : Start->getOperand(0);
    }

    unsigned getStartOpcode() const {
      bool IsDo = Start->getOpcode() == ARM::t2DoLoopStart;
      if (!IsTailPredicationLegal())
        return IsDo ? ARM::t2DLS : ARM::t2WLS;

      return VCTPOpcodeToLSTP(VCTPs.back()->getOpcode(), IsDo);
    }

    void dump() const {
      if (Start) dbgs() << "ARM Loops: Found Loop Start: " << *Start;
      if (Dec) dbgs() << "ARM Loops: Found Loop Dec: " << *Dec;
      if (End) dbgs() << "ARM Loops: Found Loop End: " << *End;
      if (!VCTPs.empty()) {
        dbgs() << "ARM Loops: Found VCTP(s):\n";
        for (auto *MI : VCTPs)
          dbgs() << " - " << *MI;
      }
      if (!FoundAllComponents())
        dbgs() << "ARM Loops: Not a low-overhead loop.\n";
      else if (!(Start && Dec && End))
        dbgs() << "ARM Loops: Failed to find all loop components.\n";
    }
  };

  class ARMLowOverheadLoops : public MachineFunctionPass {
    MachineFunction           *MF = nullptr;
    MachineLoopInfo           *MLI = nullptr;
    ReachingDefAnalysis       *RDA = nullptr;
    const ARMBaseInstrInfo    *TII = nullptr;
    MachineRegisterInfo       *MRI = nullptr;
    const TargetRegisterInfo  *TRI = nullptr;
    std::unique_ptr<ARMBasicBlockUtils> BBUtils = nullptr;

  public:
    static char ID;

    ARMLowOverheadLoops() : MachineFunctionPass(ID) { }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      AU.addRequired<MachineLoopInfo>();
      AU.addRequired<ReachingDefAnalysis>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoVRegs).set(
          MachineFunctionProperties::Property::TracksLiveness);
    }

    StringRef getPassName() const override {
      return ARM_LOW_OVERHEAD_LOOPS_NAME;
    }

  private:
    bool ProcessLoop(MachineLoop *ML);

    bool RevertNonLoops();

    void RevertWhile(MachineInstr *MI) const;

    bool RevertLoopDec(MachineInstr *MI) const;

    void RevertLoopEnd(MachineInstr *MI, bool SkipCmp = false) const;

    void ConvertVPTBlocks(LowOverheadLoop &LoLoop);

    MachineInstr *ExpandLoopStart(LowOverheadLoop &LoLoop);

    void Expand(LowOverheadLoop &LoLoop);

    void DCE(MachineInstr *MI, SmallPtrSetImpl<MachineInstr *> &ToRemove);

    void IterationCountDCE(LowOverheadLoop &LoLoop);
  };
}

char ARMLowOverheadLoops::ID = 0;

SmallVector<VPTState, 4> VPTState::Blocks;
SetVector<MachineInstr *> VPTState::CurrentPredicates;
std::map<MachineInstr *,
         std::unique_ptr<PredicatedMI>> VPTState::PredicatedInsts;

INITIALIZE_PASS(ARMLowOverheadLoops, DEBUG_TYPE, ARM_LOW_OVERHEAD_LOOPS_NAME,
                false, false)

bool LowOverheadLoop::ValidateTailPredicate(MachineInstr *StartInsertPt) {
  if (!StartInsertPt)
    return false;

  if (!IsTailPredicationLegal()) {
    LLVM_DEBUG(if (VCTPs.empty())
                 dbgs() << "ARM Loops: Didn't find a VCTP instruction.\n";
               dbgs() << "ARM Loops: Tail-predication is not valid.\n");
    return false;
  }

  assert(!VCTPs.empty() && "VCTP instruction expected but is not set");
  assert(ML.getBlocks().size() == 1 &&
         "Shouldn't be processing a loop with more than one block");

  if (DisableTailPredication) {
    LLVM_DEBUG(dbgs() << "ARM Loops: tail-predication is disabled\n");
    return false;
  }

  if (!VPTState::isValid(RDA))
    return false;

  if (!ValidateLiveOuts()) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Invalid live outs.\n");
    return false;
  }

  // For tail predication, we need to provide the number of elements, instead
  // of the iteration count, to the loop start instruction. The number of
  // elements is provided to the vctp instruction, so we need to check that
  // we can use this register at InsertPt.
  MachineInstr *VCTP = VCTPs.back();
  TPNumElements = VCTP->getOperand(1);
  Register NumElements = TPNumElements.getReg();

  // If the register is defined within loop, then we can't perform TP.
  // TODO: Check whether this is just a mov of a register that would be
  // available.
  if (RDA.hasLocalDefBefore(VCTP, NumElements)) {
    LLVM_DEBUG(dbgs() << "ARM Loops: VCTP operand is defined in the loop.\n");
    return false;
  }

  // The element count register maybe defined after InsertPt, in which case we
  // need to try to move either InsertPt or the def so that the [w|d]lstp can
  // use the value.
  MachineBasicBlock *InsertBB = StartInsertPt->getParent();

  if (!RDA.isReachingDefLiveOut(StartInsertPt, NumElements)) {
    if (auto *ElemDef = RDA.getLocalLiveOutMIDef(InsertBB, NumElements)) {
      if (RDA.isSafeToMoveForwards(ElemDef, StartInsertPt)) {
        ElemDef->removeFromParent();
        InsertBB->insert(MachineBasicBlock::iterator(StartInsertPt), ElemDef);
        LLVM_DEBUG(dbgs() << "ARM Loops: Moved element count def: "
                   << *ElemDef);
      } else if (RDA.isSafeToMoveBackwards(StartInsertPt, ElemDef)) {
        StartInsertPt->removeFromParent();
        InsertBB->insertAfter(MachineBasicBlock::iterator(ElemDef),
                              StartInsertPt);
        LLVM_DEBUG(dbgs() << "ARM Loops: Moved start past: " << *ElemDef);
      } else {
        // If we fail to move an instruction and the element count is provided
        // by a mov, use the mov operand if it will have the same value at the
        // insertion point
        MachineOperand Operand = ElemDef->getOperand(1);
        if (isMovRegOpcode(ElemDef->getOpcode()) &&
            RDA.getUniqueReachingMIDef(ElemDef, Operand.getReg()) ==
                RDA.getUniqueReachingMIDef(StartInsertPt, Operand.getReg())) {
          TPNumElements = Operand;
          NumElements = TPNumElements.getReg();
        } else {
          LLVM_DEBUG(dbgs()
                     << "ARM Loops: Unable to move element count to loop "
                     << "start instruction.\n");
          return false;
        }
      }
    }
  }

  // Could inserting the [W|D]LSTP cause some unintended affects? In a perfect
  // world the [w|d]lstp instruction would be last instruction in the preheader
  // and so it would only affect instructions within the loop body. But due to
  // scheduling, and/or the logic in this pass (above), the insertion point can
  // be moved earlier. So if the Loop Start isn't the last instruction in the
  // preheader, and if the initial element count is smaller than the vector
  // width, the Loop Start instruction will immediately generate one or more
  // false lane mask which can, incorrectly, affect the proceeding MVE
  // instructions in the preheader.
  auto CannotInsertWDLSTPBetween = [](MachineBasicBlock::iterator I,
                                      MachineBasicBlock::iterator E) {
    for (; I != E; ++I)
      if (shouldInspect(*I))
        return true;
    return false;
  };

  if (CannotInsertWDLSTPBetween(StartInsertPt, InsertBB->end()))
    return false;

  // Especially in the case of while loops, InsertBB may not be the
  // preheader, so we need to check that the register isn't redefined
  // before entering the loop.
  auto CannotProvideElements = [this](MachineBasicBlock *MBB,
                                      Register NumElements) {
    // NumElements is redefined in this block.
    if (RDA.hasLocalDefBefore(&MBB->back(), NumElements))
      return true;

    // Don't continue searching up through multiple predecessors.
    if (MBB->pred_size() > 1)
      return true;

    return false;
  };

  // Search backwards for a def, until we get to InsertBB.
  MachineBasicBlock *MBB = Preheader;
  while (MBB && MBB != InsertBB) {
    if (CannotProvideElements(MBB, NumElements)) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Unable to provide element count.\n");
      return false;
    }
    MBB = *MBB->pred_begin();
  }

  // Check that the value change of the element count is what we expect and
  // that the predication will be equivalent. For this we need:
  // NumElements = NumElements - VectorWidth. The sub will be a sub immediate
  // and we can also allow register copies within the chain too.
  auto IsValidSub = [](MachineInstr *MI, int ExpectedVecWidth) {
    return -getAddSubImmediate(*MI) == ExpectedVecWidth;
  };

  MBB = VCTP->getParent();
  // Remove modifications to the element count since they have no purpose in a
  // tail predicated loop. Explicitly refer to the vctp operand no matter which
  // register NumElements has been assigned to, since that is what the
  // modifications will be using
  if (auto *Def = RDA.getUniqueReachingMIDef(&MBB->back(),
                                             VCTP->getOperand(1).getReg())) {
    SmallPtrSet<MachineInstr*, 2> ElementChain;
    SmallPtrSet<MachineInstr*, 2> Ignore;
    unsigned ExpectedVectorWidth = getTailPredVectorWidth(VCTP->getOpcode());

    Ignore.insert(VCTPs.begin(), VCTPs.end());

    if (RDA.isSafeToRemove(Def, ElementChain, Ignore)) {
      bool FoundSub = false;

      for (auto *MI : ElementChain) {
        if (isMovRegOpcode(MI->getOpcode()))
          continue;

        if (isSubImmOpcode(MI->getOpcode())) {
          if (FoundSub || !IsValidSub(MI, ExpectedVectorWidth))
            return false;
          FoundSub = true;
        } else
          return false;
      }

      LLVM_DEBUG(dbgs() << "ARM Loops: Will remove element count chain:\n";
                 for (auto *MI : ElementChain)
                   dbgs() << " - " << *MI);
      ToRemove.insert(ElementChain.begin(), ElementChain.end());
    }
  }
  return true;
}

static bool isRegInClass(const MachineOperand &MO,
                         const TargetRegisterClass *Class) {
  return MO.isReg() && MO.getReg() && Class->contains(MO.getReg());
}

// MVE 'narrowing' operate on half a lane, reading from half and writing
// to half, which are referred to has the top and bottom half. The other
// half retains its previous value.
static bool retainsPreviousHalfElement(const MachineInstr &MI) {
  const MCInstrDesc &MCID = MI.getDesc();
  uint64_t Flags = MCID.TSFlags;
  return (Flags & ARMII::RetainsPreviousHalfElement) != 0;
}

// Some MVE instructions read from the top/bottom halves of their operand(s)
// and generate a vector result with result elements that are double the
// width of the input.
static bool producesDoubleWidthResult(const MachineInstr &MI) {
  const MCInstrDesc &MCID = MI.getDesc();
  uint64_t Flags = MCID.TSFlags;
  return (Flags & ARMII::DoubleWidthResult) != 0;
}

static bool isHorizontalReduction(const MachineInstr &MI) {
  const MCInstrDesc &MCID = MI.getDesc();
  uint64_t Flags = MCID.TSFlags;
  return (Flags & ARMII::HorizontalReduction) != 0;
}

// Can this instruction generate a non-zero result when given only zeroed
// operands? This allows us to know that, given operands with false bytes
// zeroed by masked loads, that the result will also contain zeros in those
// bytes.
static bool canGenerateNonZeros(const MachineInstr &MI) {

  // Check for instructions which can write into a larger element size,
  // possibly writing into a previous zero'd lane.
  if (producesDoubleWidthResult(MI))
    return true;

  switch (MI.getOpcode()) {
  default:
    break;
  // FIXME: VNEG FP and -0? I think we'll need to handle this once we allow
  // fp16 -> fp32 vector conversions.
  // Instructions that perform a NOT will generate 1s from 0s.
  case ARM::MVE_VMVN:
  case ARM::MVE_VORN:
  // Count leading zeros will do just that!
  case ARM::MVE_VCLZs8:
  case ARM::MVE_VCLZs16:
  case ARM::MVE_VCLZs32:
    return true;
  }
  return false;
}

// Look at its register uses to see if it only can only receive zeros
// into its false lanes which would then produce zeros. Also check that
// the output register is also defined by an FalseLanesZero instruction
// so that if tail-predication happens, the lanes that aren't updated will
// still be zeros.
static bool producesFalseLanesZero(MachineInstr &MI,
                                   const TargetRegisterClass *QPRs,
                                   const ReachingDefAnalysis &RDA,
                                   InstSet &FalseLanesZero) {
  if (canGenerateNonZeros(MI))
    return false;

  bool isPredicated = isVectorPredicated(&MI);
  // Predicated loads will write zeros to the falsely predicated bytes of the
  // destination register.
  if (MI.mayLoad())
    return isPredicated;

  auto IsZeroInit = [](MachineInstr *Def) {
    return !isVectorPredicated(Def) &&
           Def->getOpcode() == ARM::MVE_VMOVimmi32 &&
           Def->getOperand(1).getImm() == 0;
  };

  bool AllowScalars = isHorizontalReduction(MI);
  for (auto &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg())
      continue;
    if (!isRegInClass(MO, QPRs) && AllowScalars)
      continue;

    // Check that this instruction will produce zeros in its false lanes:
    // - If it only consumes false lanes zero or constant 0 (vmov #0)
    // - If it's predicated, it only matters that it's def register already has
    //   false lane zeros, so we can ignore the uses.
    SmallPtrSet<MachineInstr *, 2> Defs;
    RDA.getGlobalReachingDefs(&MI, MO.getReg(), Defs);
    for (auto *Def : Defs) {
      if (Def == &MI || FalseLanesZero.count(Def) || IsZeroInit(Def))
        continue;
      if (MO.isUse() && isPredicated)
        continue;
      return false;
    }
  }
  LLVM_DEBUG(dbgs() << "ARM Loops: Always False Zeros: " << MI);
  return true;
}

bool LowOverheadLoop::ValidateLiveOuts() {
  // We want to find out if the tail-predicated version of this loop will
  // produce the same values as the loop in its original form. For this to
  // be true, the newly inserted implicit predication must not change the
  // the (observable) results.
  // We're doing this because many instructions in the loop will not be
  // predicated and so the conversion from VPT predication to tail-predication
  // can result in different values being produced; due to the tail-predication
  // preventing many instructions from updating their falsely predicated
  // lanes. This analysis assumes that all the instructions perform lane-wise
  // operations and don't perform any exchanges.
  // A masked load, whether through VPT or tail predication, will write zeros
  // to any of the falsely predicated bytes. So, from the loads, we know that
  // the false lanes are zeroed and here we're trying to track that those false
  // lanes remain zero, or where they change, the differences are masked away
  // by their user(s).
  // All MVE stores have to be predicated, so we know that any predicate load
  // operands, or stored results are equivalent already. Other explicitly
  // predicated instructions will perform the same operation in the original
  // loop and the tail-predicated form too. Because of this, we can insert
  // loads, stores and other predicated instructions into our Predicated
  // set and build from there.
  const TargetRegisterClass *QPRs = TRI.getRegClass(ARM::MQPRRegClassID);
  SetVector<MachineInstr *> FalseLanesUnknown;
  SmallPtrSet<MachineInstr *, 4> FalseLanesZero;
  SmallPtrSet<MachineInstr *, 4> Predicated;
  MachineBasicBlock *Header = ML.getHeader();

  for (auto &MI : *Header) {
    if (!shouldInspect(MI))
      continue;

    if (isVCTP(&MI) || isVPTOpcode(MI.getOpcode()))
      continue;

    bool isPredicated = isVectorPredicated(&MI);
    bool retainsOrReduces =
      retainsPreviousHalfElement(MI) || isHorizontalReduction(MI);

    if (isPredicated)
      Predicated.insert(&MI);
    if (producesFalseLanesZero(MI, QPRs, RDA, FalseLanesZero))
      FalseLanesZero.insert(&MI);
    else if (MI.getNumDefs() == 0)
      continue;
    else if (!isPredicated && retainsOrReduces)
      return false;
    else if (!isPredicated)
      FalseLanesUnknown.insert(&MI);
  }

  auto HasPredicatedUsers = [this](MachineInstr *MI, const MachineOperand &MO,
                              SmallPtrSetImpl<MachineInstr *> &Predicated) {
    SmallPtrSet<MachineInstr *, 2> Uses;
    RDA.getGlobalUses(MI, MO.getReg(), Uses);
    for (auto *Use : Uses) {
      if (Use != MI && !Predicated.count(Use))
        return false;
    }
    return true;
  };

  // Visit the unknowns in reverse so that we can start at the values being
  // stored and then we can work towards the leaves, hopefully adding more
  // instructions to Predicated. Successfully terminating the loop means that
  // all the unknown values have to found to be masked by predicated user(s).
  // For any unpredicated values, we store them in NonPredicated so that we
  // can later check whether these form a reduction.
  SmallPtrSet<MachineInstr*, 2> NonPredicated;
  for (auto *MI : reverse(FalseLanesUnknown)) {
    for (auto &MO : MI->operands()) {
      if (!isRegInClass(MO, QPRs) || !MO.isDef())
        continue;
      if (!HasPredicatedUsers(MI, MO, Predicated)) {
        LLVM_DEBUG(dbgs() << "ARM Loops: Found an unknown def of : "
                          << TRI.getRegAsmName(MO.getReg()) << " at " << *MI);
        NonPredicated.insert(MI);
        break;
      }
    }
    // Any unknown false lanes have been masked away by the user(s).
    if (!NonPredicated.contains(MI))
      Predicated.insert(MI);
  }

  SmallPtrSet<MachineInstr *, 2> LiveOutMIs;
  SmallVector<MachineBasicBlock *, 2> ExitBlocks;
  ML.getExitBlocks(ExitBlocks);
  assert(ML.getNumBlocks() == 1 && "Expected single block loop!");
  assert(ExitBlocks.size() == 1 && "Expected a single exit block");
  MachineBasicBlock *ExitBB = ExitBlocks.front();
  for (const MachineBasicBlock::RegisterMaskPair &RegMask : ExitBB->liveins()) {
    // TODO: Instead of blocking predication, we could move the vctp to the exit
    // block and calculate it's operand there in or the preheader.
    if (RegMask.PhysReg == ARM::VPR)
      return false;
    // Check Q-regs that are live in the exit blocks. We don't collect scalars
    // because they won't be affected by lane predication.
    if (QPRs->contains(RegMask.PhysReg))
      if (auto *MI = RDA.getLocalLiveOutMIDef(Header, RegMask.PhysReg))
        LiveOutMIs.insert(MI);
  }

  // We've already validated that any VPT predication within the loop will be
  // equivalent when we perform the predication transformation; so we know that
  // any VPT predicated instruction is predicated upon VCTP. Any live-out
  // instruction needs to be predicated, so check this here. The instructions
  // in NonPredicated have been found to be a reduction that we can ensure its
  // legality.
  for (auto *MI : LiveOutMIs) {
    if (NonPredicated.count(MI) && FalseLanesUnknown.contains(MI)) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Unable to handle live out: " << *MI);
      return false;
    }
  }

  return true;
}

void LowOverheadLoop::Validate(ARMBasicBlockUtils *BBUtils) {
  if (Revert)
    return;

  auto ValidateRanges = [this, &BBUtils]() {
    if (!End->getOperand(1).isMBB())
      report_fatal_error("Expected LoopEnd to target basic block");

    // TODO Maybe there's cases where the target doesn't have to be the header,
    // but for now be safe and revert.
    if (End->getOperand(1).getMBB() != ML.getHeader()) {
      LLVM_DEBUG(dbgs() << "ARM Loops: LoopEnd is not targetting header.\n");
      return false;
    }

    // The WLS and LE instructions have 12-bits for the label offset. WLS
    // requires a positive offset, while LE uses negative.
    if (BBUtils->getOffsetOf(End) < BBUtils->getOffsetOf(ML.getHeader()) ||
        !BBUtils->isBBInRange(End, ML.getHeader(), 4094)) {
      LLVM_DEBUG(dbgs() << "ARM Loops: LE offset is out-of-range\n");
      return false;
    }

    if (Start->getOpcode() == ARM::t2WhileLoopStart &&
        (BBUtils->getOffsetOf(Start) >
         BBUtils->getOffsetOf(Start->getOperand(1).getMBB()) ||
         !BBUtils->isBBInRange(Start, Start->getOperand(1).getMBB(), 4094))) {
      LLVM_DEBUG(dbgs() << "ARM Loops: WLS offset is out-of-range!\n");
      return false;
    }
    return true;
  };

  auto FindStartInsertionPoint = [this]() -> MachineInstr* {
    // We can define LR because LR already contains the same value.
    if (Start->getOperand(0).getReg() == ARM::LR)
      return Start;

    unsigned CountReg = Start->getOperand(0).getReg();
    auto IsMoveLR = [&CountReg](MachineInstr *MI) {
      return MI->getOpcode() == ARM::tMOVr &&
             MI->getOperand(0).getReg() == ARM::LR &&
             MI->getOperand(1).getReg() == CountReg &&
             MI->getOperand(2).getImm() == ARMCC::AL;
    };

    MachineBasicBlock *MBB = Start->getParent();

    // Find an insertion point:
    // - Is there a (mov lr, Count) before Start? If so, and nothing else
    //   writes to Count before Start, we can insert at that mov.
    if (auto *LRDef = RDA.getUniqueReachingMIDef(Start, ARM::LR))
      if (IsMoveLR(LRDef) && RDA.hasSameReachingDef(Start, LRDef, CountReg))
        return LRDef;

    // - Is there a (mov lr, Count) after Start? If so, and nothing else writes
    //   to Count after Start, we can insert at that mov.
    if (auto *LRDef = RDA.getLocalLiveOutMIDef(MBB, ARM::LR))
      if (IsMoveLR(LRDef) && RDA.hasSameReachingDef(Start, LRDef, CountReg))
        return LRDef;

    // We've found no suitable LR def and Start doesn't use LR directly. Can we
    // just define LR anyway?
    return RDA.isSafeToDefRegAt(Start, ARM::LR) ? Start : nullptr;
  };

  InsertPt = FindStartInsertionPoint();
  Revert = !ValidateRanges() || !InsertPt;
  CannotTailPredicate = !ValidateTailPredicate(InsertPt);

  LLVM_DEBUG(if (!InsertPt)
               dbgs() << "ARM Loops: Unable to find safe insertion point.\n";
             else
                dbgs() << "ARM Loops: Start insertion point: " << *InsertPt;
             if (CannotTailPredicate)
                dbgs() << "ARM Loops: Couldn't validate tail predicate.\n"
            );
}

bool LowOverheadLoop::AddVCTP(MachineInstr *MI) {
  LLVM_DEBUG(dbgs() << "ARM Loops: Adding VCTP: " << *MI);
  if (VCTPs.empty()) {
    VCTPs.push_back(MI);
    return true;
  }

  // If we find another VCTP, check whether it uses the same value as the main VCTP.
  // If it does, store it in the VCTPs set, else refuse it.
  MachineInstr *Prev = VCTPs.back();
  if (!Prev->getOperand(1).isIdenticalTo(MI->getOperand(1)) ||
      !RDA.hasSameReachingDef(Prev, MI, MI->getOperand(1).getReg())) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Found VCTP with a different reaching "
                         "definition from the main VCTP");
    return false;
  }
  VCTPs.push_back(MI);
  return true;
}

bool LowOverheadLoop::ValidateMVEInst(MachineInstr* MI) {
  if (CannotTailPredicate)
    return false;

  if (!shouldInspect(*MI))
    return true;

  if (MI->getOpcode() == ARM::MVE_VPSEL ||
      MI->getOpcode() == ARM::MVE_VPNOT) {
    // TODO: Allow VPSEL and VPNOT, we currently cannot because:
    // 1) It will use the VPR as a predicate operand, but doesn't have to be
    //    instead a VPT block, which means we can assert while building up
    //    the VPT block because we don't find another VPT or VPST to being a new
    //    one.
    // 2) VPSEL still requires a VPR operand even after tail predicating,
    //    which means we can't remove it unless there is another
    //    instruction, such as vcmp, that can provide the VPR def.
    return false;
  }

  // Record all VCTPs and check that they're equivalent to one another.
  if (isVCTP(MI) && !AddVCTP(MI))
    return false;

  // Inspect uses first so that any instructions that alter the VPR don't
  // alter the predicate upon themselves.
  const MCInstrDesc &MCID = MI->getDesc();
  bool IsUse = false;
  unsigned LastOpIdx = MI->getNumOperands() - 1;
  for (auto &Op : enumerate(reverse(MCID.operands()))) {
    const MachineOperand &MO = MI->getOperand(LastOpIdx - Op.index());
    if (!MO.isReg() || !MO.isUse() || MO.getReg() != ARM::VPR)
      continue;

    if (ARM::isVpred(Op.value().OperandType)) {
      VPTState::addInst(MI);
      IsUse = true;
    } else if (MI->getOpcode() != ARM::MVE_VPST) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Found instruction using vpr: " << *MI);
      return false;
    }
  }

  // If we find an instruction that has been marked as not valid for tail
  // predication, only allow the instruction if it's contained within a valid
  // VPT block.
  bool RequiresExplicitPredication =
    (MCID.TSFlags & ARMII::ValidForTailPredication) == 0;
  if (isDomainMVE(MI) && RequiresExplicitPredication) {
    LLVM_DEBUG(if (!IsUse)
               dbgs() << "ARM Loops: Can't tail predicate: " << *MI);
    return IsUse;
  }

  // If the instruction is already explicitly predicated, then the conversion
  // will be fine, but ensure that all store operations are predicated.
  if (MI->mayStore())
    return IsUse;

  // If this instruction defines the VPR, update the predicate for the
  // proceeding instructions.
  if (isVectorPredicate(MI)) {
    // Clear the existing predicate when we're not in VPT Active state,
    // otherwise we add to it.
    if (!isVectorPredicated(MI))
      VPTState::resetPredicate(MI);
    else
      VPTState::addPredicate(MI);
  }

  // Finally once the predicate has been modified, we can start a new VPT
  // block if necessary.
  if (isVPTOpcode(MI->getOpcode()))
    VPTState::CreateVPTBlock(MI);

  return true;
}

bool ARMLowOverheadLoops::runOnMachineFunction(MachineFunction &mf) {
  const ARMSubtarget &ST = static_cast<const ARMSubtarget&>(mf.getSubtarget());
  if (!ST.hasLOB())
    return false;

  MF = &mf;
  LLVM_DEBUG(dbgs() << "ARM Loops on " << MF->getName() << " ------------- \n");

  MLI = &getAnalysis<MachineLoopInfo>();
  RDA = &getAnalysis<ReachingDefAnalysis>();
  MF->getProperties().set(MachineFunctionProperties::Property::TracksLiveness);
  MRI = &MF->getRegInfo();
  TII = static_cast<const ARMBaseInstrInfo*>(ST.getInstrInfo());
  TRI = ST.getRegisterInfo();
  BBUtils = std::unique_ptr<ARMBasicBlockUtils>(new ARMBasicBlockUtils(*MF));
  BBUtils->computeAllBlockSizes();
  BBUtils->adjustBBOffsetsAfter(&MF->front());

  bool Changed = false;
  for (auto ML : *MLI) {
    if (ML->isOutermost())
      Changed |= ProcessLoop(ML);
  }
  Changed |= RevertNonLoops();
  return Changed;
}

bool ARMLowOverheadLoops::ProcessLoop(MachineLoop *ML) {

  bool Changed = false;

  // Process inner loops first.
  for (auto I = ML->begin(), E = ML->end(); I != E; ++I)
    Changed |= ProcessLoop(*I);

  LLVM_DEBUG(dbgs() << "ARM Loops: Processing loop containing:\n";
             if (auto *Preheader = ML->getLoopPreheader())
               dbgs() << " - " << Preheader->getName() << "\n";
             else if (auto *Preheader = MLI->findLoopPreheader(ML))
               dbgs() << " - " << Preheader->getName() << "\n";
             else if (auto *Preheader = MLI->findLoopPreheader(ML, true))
               dbgs() << " - " << Preheader->getName() << "\n";
             for (auto *MBB : ML->getBlocks())
               dbgs() << " - " << MBB->getName() << "\n";
            );

  // Search the given block for a loop start instruction. If one isn't found,
  // and there's only one predecessor block, search that one too.
  std::function<MachineInstr*(MachineBasicBlock*)> SearchForStart =
    [&SearchForStart](MachineBasicBlock *MBB) -> MachineInstr* {
    for (auto &MI : *MBB) {
      if (isLoopStart(MI))
        return &MI;
    }
    if (MBB->pred_size() == 1)
      return SearchForStart(*MBB->pred_begin());
    return nullptr;
  };

  LowOverheadLoop LoLoop(*ML, *MLI, *RDA, *TRI, *TII);
  // Search the preheader for the start intrinsic.
  // FIXME: I don't see why we shouldn't be supporting multiple predecessors
  // with potentially multiple set.loop.iterations, so we need to enable this.
  if (LoLoop.Preheader)
    LoLoop.Start = SearchForStart(LoLoop.Preheader);
  else
    return false;

  // Find the low-overhead loop components and decide whether or not to fall
  // back to a normal loop. Also look for a vctp instructions and decide
  // whether we can convert that predicate using tail predication.
  for (auto *MBB : reverse(ML->getBlocks())) {
    for (auto &MI : *MBB) {
      if (MI.isDebugValue())
        continue;
      else if (MI.getOpcode() == ARM::t2LoopDec)
        LoLoop.Dec = &MI;
      else if (MI.getOpcode() == ARM::t2LoopEnd)
        LoLoop.End = &MI;
      else if (isLoopStart(MI))
        LoLoop.Start = &MI;
      else if (MI.getDesc().isCall()) {
        // TODO: Though the call will require LE to execute again, does this
        // mean we should revert? Always executing LE hopefully should be
        // faster than performing a sub,cmp,br or even subs,br.
        LoLoop.Revert = true;
        LLVM_DEBUG(dbgs() << "ARM Loops: Found call.\n");
      } else {
        // Record VPR defs and build up their corresponding vpt blocks.
        // Check we know how to tail predicate any mve instructions.
        LoLoop.AnalyseMVEInst(&MI);
      }
    }
  }

  LLVM_DEBUG(LoLoop.dump());
  if (!LoLoop.FoundAllComponents()) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Didn't find loop start, update, end\n");
    return false;
  }

  // Check that the only instruction using LoopDec is LoopEnd.
  // TODO: Check for copy chains that really have no effect.
  SmallPtrSet<MachineInstr*, 2> Uses;
  RDA->getReachingLocalUses(LoLoop.Dec, ARM::LR, Uses);
  if (Uses.size() > 1 || !Uses.count(LoLoop.End)) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Unable to remove LoopDec.\n");
    LoLoop.Revert = true;
  }
  LoLoop.Validate(BBUtils.get());
  Expand(LoLoop);
  return true;
}

// WhileLoopStart holds the exit block, so produce a cmp lr, 0 and then a
// beq that branches to the exit branch.
// TODO: We could also try to generate a cbz if the value in LR is also in
// another low register.
void ARMLowOverheadLoops::RevertWhile(MachineInstr *MI) const {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting to cmp: " << *MI);
  MachineBasicBlock *MBB = MI->getParent();
  MachineInstrBuilder MIB = BuildMI(*MBB, MI, MI->getDebugLoc(),
                                    TII->get(ARM::t2CMPri));
  MIB.add(MI->getOperand(0));
  MIB.addImm(0);
  MIB.addImm(ARMCC::AL);
  MIB.addReg(ARM::NoRegister);

  MachineBasicBlock *DestBB = MI->getOperand(1).getMBB();
  unsigned BrOpc = BBUtils->isBBInRange(MI, DestBB, 254) ?
    ARM::tBcc : ARM::t2Bcc;

  MIB = BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(BrOpc));
  MIB.add(MI->getOperand(1));   // branch target
  MIB.addImm(ARMCC::EQ);        // condition code
  MIB.addReg(ARM::CPSR);
  MI->eraseFromParent();
}

bool ARMLowOverheadLoops::RevertLoopDec(MachineInstr *MI) const {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting to sub: " << *MI);
  MachineBasicBlock *MBB = MI->getParent();
  SmallPtrSet<MachineInstr*, 1> Ignore;
  for (auto I = MachineBasicBlock::iterator(MI), E = MBB->end(); I != E; ++I) {
    if (I->getOpcode() == ARM::t2LoopEnd) {
      Ignore.insert(&*I);
      break;
    }
  }

  // If nothing defines CPSR between LoopDec and LoopEnd, use a t2SUBS.
  bool SetFlags = RDA->isSafeToDefRegAt(MI, ARM::CPSR, Ignore);

  MachineInstrBuilder MIB = BuildMI(*MBB, MI, MI->getDebugLoc(),
                                    TII->get(ARM::t2SUBri));
  MIB.addDef(ARM::LR);
  MIB.add(MI->getOperand(1));
  MIB.add(MI->getOperand(2));
  MIB.addImm(ARMCC::AL);
  MIB.addReg(0);

  if (SetFlags) {
    MIB.addReg(ARM::CPSR);
    MIB->getOperand(5).setIsDef(true);
  } else
    MIB.addReg(0);

  MI->eraseFromParent();
  return SetFlags;
}

// Generate a subs, or sub and cmp, and a branch instead of an LE.
void ARMLowOverheadLoops::RevertLoopEnd(MachineInstr *MI, bool SkipCmp) const {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting to cmp, br: " << *MI);

  MachineBasicBlock *MBB = MI->getParent();
  // Create cmp
  if (!SkipCmp) {
    MachineInstrBuilder MIB = BuildMI(*MBB, MI, MI->getDebugLoc(),
                                      TII->get(ARM::t2CMPri));
    MIB.addReg(ARM::LR);
    MIB.addImm(0);
    MIB.addImm(ARMCC::AL);
    MIB.addReg(ARM::NoRegister);
  }

  MachineBasicBlock *DestBB = MI->getOperand(1).getMBB();
  unsigned BrOpc = BBUtils->isBBInRange(MI, DestBB, 254) ?
    ARM::tBcc : ARM::t2Bcc;

  // Create bne
  MachineInstrBuilder MIB =
    BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(BrOpc));
  MIB.add(MI->getOperand(1));   // branch target
  MIB.addImm(ARMCC::NE);        // condition code
  MIB.addReg(ARM::CPSR);
  MI->eraseFromParent();
}

void ARMLowOverheadLoops::DCE(MachineInstr *MI,
                              SmallPtrSetImpl<MachineInstr *> &ToRemove) {
  // Collect the dead code and the MBBs in which they reside.
  SmallPtrSet<MachineInstr*, 4> Killed;
  RDA->collectKilledOperands(MI, Killed);
  SmallPtrSet<MachineBasicBlock*, 2> BasicBlocks;
  for (auto *Dead : Killed)
    BasicBlocks.insert(Dead->getParent());

  // Collect IT blocks in all affected basic blocks.
  std::map<MachineInstr *, SmallPtrSet<MachineInstr *, 2>> ITBlocks;
  for (auto *MBB : BasicBlocks) {
    for (auto &IT : *MBB) {
      if (IT.getOpcode() != ARM::t2IT)
        continue;
      RDA->getReachingLocalUses(&IT, ARM::ITSTATE, ITBlocks[&IT]);
    }
  }

  // If we're removing all of the instructions within an IT block, then
  // also remove the IT instruction.
  SmallPtrSet<MachineInstr*, 2> ModifiedITs;
  for (auto *Dead : Killed) {
    if (MachineOperand *MO = Dead->findRegisterUseOperand(ARM::ITSTATE)) {
      MachineInstr *IT = RDA->getMIOperand(Dead, *MO);
      auto &CurrentBlock = ITBlocks[IT];
      CurrentBlock.erase(Dead);
      if (CurrentBlock.empty())
        ModifiedITs.erase(IT);
      else
        ModifiedITs.insert(IT);
    }
  }

  // Delete the killed instructions only if we don't have any IT blocks that
  // need to be modified because we need to fixup the mask.
  // TODO: Handle cases where IT blocks are modified.
  if (ModifiedITs.empty()) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Will remove iteration count:\n";
               for (auto *MI : Killed)
                 dbgs() << " - " << *MI);
    ToRemove.insert(Killed.begin(), Killed.end());
  } else
    LLVM_DEBUG(dbgs() << "ARM Loops: Would need to modify IT block(s).\n");
}

// Perform dead code elimation on the loop iteration count setup expression.
// If we are tail-predicating, the number of elements to be processed is the
// operand of the VCTP instruction in the vector body, see getCount(), which is
// register $r3 in this example:
//
//   $lr = big-itercount-expression
//   ..
//   t2DoLoopStart renamable $lr
//   vector.body:
//     ..
//     $vpr = MVE_VCTP32 renamable $r3
//     renamable $lr = t2LoopDec killed renamable $lr, 1
//     t2LoopEnd renamable $lr, %vector.body
//     tB %end
//
// What we would like achieve here is to replace the do-loop start pseudo
// instruction t2DoLoopStart with:
//
//    $lr = MVE_DLSTP_32 killed renamable $r3
//
// Thus, $r3 which defines the number of elements, is written to $lr,
// and then we want to delete the whole chain that used to define $lr,
// see the comment below how this chain could look like.
//
void ARMLowOverheadLoops::IterationCountDCE(LowOverheadLoop &LoLoop) {
  if (!LoLoop.IsTailPredicationLegal())
    return;

  LLVM_DEBUG(dbgs() << "ARM Loops: Trying DCE on loop iteration count.\n");

  MachineInstr *Def = RDA->getMIOperand(LoLoop.Start, 0);
  if (!Def) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Couldn't find iteration count.\n");
    return;
  }

  // Collect and remove the users of iteration count.
  SmallPtrSet<MachineInstr*, 4> Killed  = { LoLoop.Start, LoLoop.Dec,
                                            LoLoop.End, LoLoop.InsertPt };
  SmallPtrSet<MachineInstr*, 2> Remove;
  if (RDA->isSafeToRemove(Def, Remove, Killed)) {
    LoLoop.ToRemove.insert(Remove.begin(), Remove.end());
    DCE(Def, LoLoop.ToRemove);
  } else
    LLVM_DEBUG(dbgs() << "ARM Loops: Unsafe to remove loop iteration count.\n");
}

MachineInstr* ARMLowOverheadLoops::ExpandLoopStart(LowOverheadLoop &LoLoop) {
  LLVM_DEBUG(dbgs() << "ARM Loops: Expanding LoopStart.\n");
  // When using tail-predication, try to delete the dead code that was used to
  // calculate the number of loop iterations.
  IterationCountDCE(LoLoop);

  MachineInstr *InsertPt = LoLoop.InsertPt;
  MachineInstr *Start = LoLoop.Start;
  MachineBasicBlock *MBB = InsertPt->getParent();
  bool IsDo = Start->getOpcode() == ARM::t2DoLoopStart;
  unsigned Opc = LoLoop.getStartOpcode();
  MachineOperand &Count = LoLoop.getLoopStartOperand();

  MachineInstrBuilder MIB =
    BuildMI(*MBB, InsertPt, InsertPt->getDebugLoc(), TII->get(Opc));

  MIB.addDef(ARM::LR);
  MIB.add(Count);
  if (!IsDo)
    MIB.add(Start->getOperand(1));

  // If we're inserting at a mov lr, then remove it as it's redundant.
  if (InsertPt != Start)
    LoLoop.ToRemove.insert(InsertPt);
  LoLoop.ToRemove.insert(Start);
  LLVM_DEBUG(dbgs() << "ARM Loops: Inserted start: " << *MIB);
  return &*MIB;
}

void ARMLowOverheadLoops::ConvertVPTBlocks(LowOverheadLoop &LoLoop) {
  auto RemovePredicate = [](MachineInstr *MI) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Removing predicate from: " << *MI);
    if (int PIdx = llvm::findFirstVPTPredOperandIdx(*MI)) {
      assert(MI->getOperand(PIdx).getImm() == ARMVCC::Then &&
             "Expected Then predicate!");
      MI->getOperand(PIdx).setImm(ARMVCC::None);
      MI->getOperand(PIdx+1).setReg(0);
    } else
      llvm_unreachable("trying to unpredicate a non-predicated instruction");
  };

  for (auto &Block : LoLoop.getVPTBlocks()) {
    SmallVectorImpl<MachineInstr *> &Insts = Block.getInsts();

    if (VPTState::isEntryPredicatedOnVCTP(Block, /*exclusive*/true)) {
      if (VPTState::hasUniformPredicate(Block)) {
        // A vpt block starting with VPST, is only predicated upon vctp and has no
        // internal vpr defs:
        // - Remove vpst.
        // - Unpredicate the remaining instructions.
        LLVM_DEBUG(dbgs() << "ARM Loops: Removing VPST: " << *Insts.front());
        LoLoop.ToRemove.insert(Insts.front());
        for (unsigned i = 1; i < Insts.size(); ++i)
          RemovePredicate(Insts[i]);
      } else {
        // The VPT block has a non-uniform predicate but it uses a vpst and its
        // entry is guarded only by a vctp, which means we:
        // - Need to remove the original vpst.
        // - Then need to unpredicate any following instructions, until
        //   we come across the divergent vpr def.
        // - Insert a new vpst to predicate the instruction(s) that following
        //   the divergent vpr def.
        // TODO: We could be producing more VPT blocks than necessary and could
        // fold the newly created one into a proceeding one.
        MachineInstr *Divergent = VPTState::getDivergent(Block);
        for (auto I = ++MachineBasicBlock::iterator(Insts.front()),
             E = ++MachineBasicBlock::iterator(Divergent); I != E; ++I)
          RemovePredicate(&*I);

        // Check if the instruction defining vpr is a vcmp so it can be combined
        // with the VPST This should be the divergent instruction
        MachineInstr *VCMP = VCMPOpcodeToVPT(Divergent->getOpcode()) != 0
          ? Divergent
          : nullptr;

        MachineInstrBuilder MIB;
        if (VCMP) {
          // Combine the VPST and VCMP into a VPT
          MIB = BuildMI(*Divergent->getParent(), Divergent,
                        Divergent->getDebugLoc(),
                        TII->get(VCMPOpcodeToVPT(VCMP->getOpcode())));
          MIB.addImm(ARMVCC::Then);
          // Register one
          MIB.add(VCMP->getOperand(1));
          // Register two
          MIB.add(VCMP->getOperand(2));
          // The comparison code, e.g. ge, eq, lt
          MIB.add(VCMP->getOperand(3));
          LLVM_DEBUG(dbgs()
                     << "ARM Loops: Combining with VCMP to VPT: " << *MIB);
          LoLoop.ToRemove.insert(VCMP);
        } else {
          // Create a VPST (with a null mask for now, we'll recompute it later)
          // or a VPT in case there was a VCMP right before it
          MIB = BuildMI(*Divergent->getParent(), Divergent,
                        Divergent->getDebugLoc(), TII->get(ARM::MVE_VPST));
          MIB.addImm(0);
          LLVM_DEBUG(dbgs() << "ARM Loops: Created VPST: " << *MIB);
        }
        LLVM_DEBUG(dbgs() << "ARM Loops: Removing VPST: " << *Insts.front());
        LoLoop.ToRemove.insert(Insts.front());
        LoLoop.BlockMasksToRecompute.insert(MIB.getInstr());
      }
    } else if (Block.containsVCTP()) {
      // The vctp will be removed, so the block mask of the vp(s)t will need
      // to be recomputed.
      LoLoop.BlockMasksToRecompute.insert(Insts.front());
    }
  }

  LoLoop.ToRemove.insert(LoLoop.VCTPs.begin(), LoLoop.VCTPs.end());
}

void ARMLowOverheadLoops::Expand(LowOverheadLoop &LoLoop) {

  // Combine the LoopDec and LoopEnd instructions into LE(TP).
  auto ExpandLoopEnd = [this](LowOverheadLoop &LoLoop) {
    MachineInstr *End = LoLoop.End;
    MachineBasicBlock *MBB = End->getParent();
    unsigned Opc = LoLoop.IsTailPredicationLegal() ?
      ARM::MVE_LETP : ARM::t2LEUpdate;
    MachineInstrBuilder MIB = BuildMI(*MBB, End, End->getDebugLoc(),
                                      TII->get(Opc));
    MIB.addDef(ARM::LR);
    MIB.add(End->getOperand(0));
    MIB.add(End->getOperand(1));
    LLVM_DEBUG(dbgs() << "ARM Loops: Inserted LE: " << *MIB);
    LoLoop.ToRemove.insert(LoLoop.Dec);
    LoLoop.ToRemove.insert(End);
    return &*MIB;
  };

  // TODO: We should be able to automatically remove these branches before we
  // get here - probably by teaching analyzeBranch about the pseudo
  // instructions.
  // If there is an unconditional branch, after I, that just branches to the
  // next block, remove it.
  auto RemoveDeadBranch = [](MachineInstr *I) {
    MachineBasicBlock *BB = I->getParent();
    MachineInstr *Terminator = &BB->instr_back();
    if (Terminator->isUnconditionalBranch() && I != Terminator) {
      MachineBasicBlock *Succ = Terminator->getOperand(0).getMBB();
      if (BB->isLayoutSuccessor(Succ)) {
        LLVM_DEBUG(dbgs() << "ARM Loops: Removing branch: " << *Terminator);
        Terminator->eraseFromParent();
      }
    }
  };

  if (LoLoop.Revert) {
    if (LoLoop.Start->getOpcode() == ARM::t2WhileLoopStart)
      RevertWhile(LoLoop.Start);
    else
      LoLoop.Start->eraseFromParent();
    bool FlagsAlreadySet = RevertLoopDec(LoLoop.Dec);
    RevertLoopEnd(LoLoop.End, FlagsAlreadySet);
  } else {
    LoLoop.Start = ExpandLoopStart(LoLoop);
    RemoveDeadBranch(LoLoop.Start);
    LoLoop.End = ExpandLoopEnd(LoLoop);
    RemoveDeadBranch(LoLoop.End);
    if (LoLoop.IsTailPredicationLegal())
      ConvertVPTBlocks(LoLoop);
    for (auto *I : LoLoop.ToRemove) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Erasing " << *I);
      I->eraseFromParent();
    }
    for (auto *I : LoLoop.BlockMasksToRecompute) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Recomputing VPT/VPST Block Mask: " << *I);
      recomputeVPTBlockMask(*I);
      LLVM_DEBUG(dbgs() << "           ... done: " << *I);
    }
  }

  PostOrderLoopTraversal DFS(LoLoop.ML, *MLI);
  DFS.ProcessLoop();
  const SmallVectorImpl<MachineBasicBlock*> &PostOrder = DFS.getOrder();
  for (auto *MBB : PostOrder) {
    recomputeLiveIns(*MBB);
    // FIXME: For some reason, the live-in print order is non-deterministic for
    // our tests and I can't out why... So just sort them.
    MBB->sortUniqueLiveIns();
  }

  for (auto *MBB : reverse(PostOrder))
    recomputeLivenessFlags(*MBB);

  // We've moved, removed and inserted new instructions, so update RDA.
  RDA->reset();
}

bool ARMLowOverheadLoops::RevertNonLoops() {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting any remaining pseudos...\n");
  bool Changed = false;

  for (auto &MBB : *MF) {
    SmallVector<MachineInstr*, 4> Starts;
    SmallVector<MachineInstr*, 4> Decs;
    SmallVector<MachineInstr*, 4> Ends;

    for (auto &I : MBB) {
      if (isLoopStart(I))
        Starts.push_back(&I);
      else if (I.getOpcode() == ARM::t2LoopDec)
        Decs.push_back(&I);
      else if (I.getOpcode() == ARM::t2LoopEnd)
        Ends.push_back(&I);
    }

    if (Starts.empty() && Decs.empty() && Ends.empty())
      continue;

    Changed = true;

    for (auto *Start : Starts) {
      if (Start->getOpcode() == ARM::t2WhileLoopStart)
        RevertWhile(Start);
      else
        Start->eraseFromParent();
    }
    for (auto *Dec : Decs)
      RevertLoopDec(Dec);

    for (auto *End : Ends)
      RevertLoopEnd(End);
  }
  return Changed;
}

FunctionPass *llvm::createARMLowOverheadLoopsPass() {
  return new ARMLowOverheadLoops();
}
