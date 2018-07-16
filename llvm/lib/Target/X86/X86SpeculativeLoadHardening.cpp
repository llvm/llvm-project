//====- X86SpeculativeLoadHardening.cpp - A Spectre v1 mitigation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Provide a pass which mitigates speculative execution attacks which operate
/// by speculating incorrectly past some predicate (a type check, bounds check,
/// or other condition) to reach a load with invalid inputs and leak the data
/// accessed by that load using a side channel out of the speculative domain.
///
/// For details on the attacks, see the first variant in both the Project Zero
/// writeup and the Spectre paper:
/// https://googleprojectzero.blogspot.com/2018/01/reading-privileged-memory-with-side.html
/// https://spectreattack.com/spectre.pdf
///
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>

using namespace llvm;

#define PASS_KEY "x86-speculative-load-hardening"
#define DEBUG_TYPE PASS_KEY

STATISTIC(NumCondBranchesTraced, "Number of conditional branches traced");
STATISTIC(NumBranchesUntraced, "Number of branches unable to trace");
STATISTIC(NumAddrRegsHardened,
          "Number of address mode used registers hardaned");
STATISTIC(NumPostLoadRegsHardened,
          "Number of post-load register values hardened");
STATISTIC(NumInstsInserted, "Number of instructions inserted");
STATISTIC(NumLFENCEsInserted, "Number of lfence instructions inserted");

static cl::opt<bool> HardenEdgesWithLFENCE(
    PASS_KEY "-lfence",
    cl::desc(
        "Use LFENCE along each conditional edge to harden against speculative "
        "loads rather than conditional movs and poisoned pointers."),
    cl::init(false), cl::Hidden);

static cl::opt<bool> EnablePostLoadHardening(
    PASS_KEY "-post-load",
    cl::desc("Harden the value loaded *after* it is loaded by "
             "flushing the loaded bits to 1. This is hard to do "
             "in general but can be done easily for GPRs."),
    cl::init(true), cl::Hidden);

static cl::opt<bool> FenceCallAndRet(
    PASS_KEY "-fence-call-and-ret",
    cl::desc("Use a full speculation fence to harden both call and ret edges "
             "rather than a lighter weight mitigation."),
    cl::init(false), cl::Hidden);

static cl::opt<bool> HardenInterprocedurally(
    PASS_KEY "-ip",
    cl::desc("Harden interprocedurally by passing our state in and out of "
             "functions in the high bits of the stack pointer."),
    cl::init(true), cl::Hidden);

static cl::opt<bool>
    HardenLoads(PASS_KEY "-loads",
                cl::desc("Sanitize loads from memory. When disable, no "
                         "significant security is provided."),
                cl::init(true), cl::Hidden);

namespace llvm {

void initializeX86SpeculativeLoadHardeningPassPass(PassRegistry &);

} // end namespace llvm

namespace {

class X86SpeculativeLoadHardeningPass : public MachineFunctionPass {
public:
  X86SpeculativeLoadHardeningPass() : MachineFunctionPass(ID) {
    initializeX86SpeculativeLoadHardeningPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "X86 speculative load hardening";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Pass identification, replacement for typeid.
  static char ID;

private:
  /// The information about a block's conditional terminators needed to trace
  /// our predicate state through the exiting edges.
  struct BlockCondInfo {
    MachineBasicBlock *MBB;

    // We mostly have one conditional branch, and in extremely rare cases have
    // two. Three and more are so rare as to be unimportant for compile time.
    SmallVector<MachineInstr *, 2> CondBrs;

    MachineInstr *UncondBr;
  };

  const X86Subtarget *Subtarget;
  MachineRegisterInfo *MRI;
  const X86InstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const TargetRegisterClass *PredStateRC;

  void hardenEdgesWithLFENCE(MachineFunction &MF);

  SmallVector<BlockCondInfo, 16> collectBlockCondInfo(MachineFunction &MF);

  void checkAllLoads(MachineFunction &MF, MachineSSAUpdater &PredStateSSA);

  unsigned saveEFLAGS(MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator InsertPt, DebugLoc Loc);
  void restoreEFLAGS(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, DebugLoc Loc,
                     unsigned OFReg);

  void mergePredStateIntoSP(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator InsertPt, DebugLoc Loc,
                            unsigned PredStateReg);
  unsigned extractPredStateFromSP(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator InsertPt,
                                  DebugLoc Loc);

  void
  hardenLoadAddr(MachineInstr &MI, MachineOperand &BaseMO,
                 MachineOperand &IndexMO, MachineSSAUpdater &PredStateSSA,
                 SmallDenseMap<unsigned, unsigned, 32> &AddrRegToHardenedReg);
  MachineInstr *
  sinkPostLoadHardenedInst(MachineInstr &MI,
                           SmallPtrSetImpl<MachineInstr *> &HardenedInstrs);
  bool canHardenRegister(unsigned Reg);
  void hardenPostLoad(MachineInstr &MI, MachineSSAUpdater &PredStateSSA);
  void checkReturnInstr(MachineInstr &MI, MachineSSAUpdater &PredStateSSA);
  void checkCallInstr(MachineInstr &MI, MachineSSAUpdater &PredStateSSA);
};

} // end anonymous namespace

char X86SpeculativeLoadHardeningPass::ID = 0;

void X86SpeculativeLoadHardeningPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
}

static MachineBasicBlock &splitEdge(MachineBasicBlock &MBB,
                                    MachineBasicBlock &Succ, int SuccCount,
                                    MachineInstr *Br, MachineInstr *&UncondBr,
                                    const X86InstrInfo &TII) {
  assert(!Succ.isEHPad() && "Shouldn't get edges to EH pads!");

  MachineFunction &MF = *MBB.getParent();

  MachineBasicBlock &NewMBB = *MF.CreateMachineBasicBlock();

  // We have to insert the new block immediately after the current one as we
  // don't know what layout-successor relationships the successor has and we
  // may not be able to (and generally don't want to) try to fix those up.
  MF.insert(std::next(MachineFunction::iterator(&MBB)), &NewMBB);

  // Update the branch instruction if necessary.
  if (Br) {
    assert(Br->getOperand(0).getMBB() == &Succ &&
           "Didn't start with the right target!");
    Br->getOperand(0).setMBB(&NewMBB);

    // If this successor was reached through a branch rather than fallthrough,
    // we might have *broken* fallthrough and so need to inject a new
    // unconditional branch.
    if (!UncondBr) {
      MachineBasicBlock &OldLayoutSucc =
          *std::next(MachineFunction::iterator(&NewMBB));
      assert(MBB.isSuccessor(&OldLayoutSucc) &&
             "Without an unconditional branch, the old layout successor should "
             "be an actual successor!");
      auto BrBuilder =
          BuildMI(&MBB, DebugLoc(), TII.get(X86::JMP_1)).addMBB(&OldLayoutSucc);
      // Update the unconditional branch now that we've added one.
      UncondBr = &*BrBuilder;
    }

    // Insert unconditional "jump Succ" instruction in the new block if
    // necessary.
    if (!NewMBB.isLayoutSuccessor(&Succ)) {
      SmallVector<MachineOperand, 4> Cond;
      TII.insertBranch(NewMBB, &Succ, nullptr, Cond, Br->getDebugLoc());
    }
  } else {
    assert(!UncondBr &&
           "Cannot have a branchless successor and an unconditional branch!");
    assert(NewMBB.isLayoutSuccessor(&Succ) &&
           "A non-branch successor must have been a layout successor before "
           "and now is a layout successor of the new block.");
  }

  // If this is the only edge to the successor, we can just replace it in the
  // CFG. Otherwise we need to add a new entry in the CFG for the new
  // successor.
  if (SuccCount == 1) {
    MBB.replaceSuccessor(&Succ, &NewMBB);
  } else {
    MBB.splitSuccessor(&Succ, &NewMBB);
  }

  // Hook up the edge from the new basic block to the old successor in the CFG.
  NewMBB.addSuccessor(&Succ);

  // Fix PHI nodes in Succ so they refer to NewMBB instead of MBB.
  for (MachineInstr &MI : Succ) {
    if (!MI.isPHI())
      break;
    for (int OpIdx = 1, NumOps = MI.getNumOperands(); OpIdx < NumOps;
         OpIdx += 2) {
      MachineOperand &OpV = MI.getOperand(OpIdx);
      MachineOperand &OpMBB = MI.getOperand(OpIdx + 1);
      assert(OpMBB.isMBB() && "Block operand to a PHI is not a block!");
      if (OpMBB.getMBB() != &MBB)
        continue;

      // If this is the last edge to the succesor, just replace MBB in the PHI
      if (SuccCount == 1) {
        OpMBB.setMBB(&NewMBB);
        break;
      }

      // Otherwise, append a new pair of operands for the new incoming edge.
      MI.addOperand(MF, OpV);
      MI.addOperand(MF, MachineOperand::CreateMBB(&NewMBB));
      break;
    }
  }

  // Inherit live-ins from the successor
  for (auto &LI : Succ.liveins())
    NewMBB.addLiveIn(LI);

  LLVM_DEBUG(dbgs() << "  Split edge from '" << MBB.getName() << "' to '"
                    << Succ.getName() << "'.\n");
  return NewMBB;
}

/// Removing duplicate PHI operands to leave the PHI in a canonical and
/// predictable form.
///
/// FIXME: It's really frustrating that we have to do this, but SSA-form in MIR
/// isn't what you might expect. We may have multiple entries in PHI nodes for
/// a single predecessor. This makes CFG-updating extremely complex, so here we
/// simplify all PHI nodes to a model even simpler than the IR's model: exactly
/// one entry per predecessor, regardless of how many edges there are.
static void canonicalizePHIOperands(MachineFunction &MF) {
  SmallPtrSet<MachineBasicBlock *, 4> Preds;
  SmallVector<int, 4> DupIndices;
  for (auto &MBB : MF)
    for (auto &MI : MBB) {
      if (!MI.isPHI())
        break;

      // First we scan the operands of the PHI looking for duplicate entries
      // a particular predecessor. We retain the operand index of each duplicate
      // entry found.
      for (int OpIdx = 1, NumOps = MI.getNumOperands(); OpIdx < NumOps;
           OpIdx += 2)
        if (!Preds.insert(MI.getOperand(OpIdx + 1).getMBB()).second)
          DupIndices.push_back(OpIdx);

      // Now walk the duplicate indices, removing both the block and value. Note
      // that these are stored as a vector making this element-wise removal
      // :w
      // potentially quadratic.
      //
      // FIXME: It is really frustrating that we have to use a quadratic
      // removal algorithm here. There should be a better way, but the use-def
      // updates required make that impossible using the public API.
      //
      // Note that we have to process these backwards so that we don't
      // invalidate other indices with each removal.
      while (!DupIndices.empty()) {
        int OpIdx = DupIndices.pop_back_val();
        // Remove both the block and value operand, again in reverse order to
        // preserve indices.
        MI.RemoveOperand(OpIdx + 1);
        MI.RemoveOperand(OpIdx);
      }

      Preds.clear();
    }
}

/// Helper to scan a function for loads vulnerable to misspeculation that we
/// want to harden.
///
/// We use this to avoid making changes to functions where there is nothing we
/// need to do to harden against misspeculation.
static bool hasVulnerableLoad(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Loads within this basic block after an LFENCE are not at risk of
      // speculatively executing with invalid predicates from prior control
      // flow. So break out of this block but continue scanning the function.
      if (MI.getOpcode() == X86::LFENCE)
        break;

      // Looking for loads only.
      if (!MI.mayLoad())
        continue;

      // An MFENCE is modeled as a load but isn't vulnerable to misspeculation.
      if (MI.getOpcode() == X86::MFENCE)
        continue;

      // We found a load.
      return true;
    }
  }

  // No loads found.
  return false;
}

bool X86SpeculativeLoadHardeningPass::runOnMachineFunction(
    MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** " << getPassName() << " : " << MF.getName()
                    << " **********\n");

  Subtarget = &MF.getSubtarget<X86Subtarget>();
  MRI = &MF.getRegInfo();
  TII = Subtarget->getInstrInfo();
  TRI = Subtarget->getRegisterInfo();
  // FIXME: Support for 32-bit.
  PredStateRC = &X86::GR64_NOSPRegClass;

  if (MF.begin() == MF.end())
    // Nothing to do for a degenerate empty function...
    return false;

  // We support an alternative hardening technique based on a debug flag.
  if (HardenEdgesWithLFENCE) {
    hardenEdgesWithLFENCE(MF);
    return true;
  }

  // Create a dummy debug loc to use for all the generated code here.
  DebugLoc Loc;

  MachineBasicBlock &Entry = *MF.begin();
  auto EntryInsertPt = Entry.SkipPHIsLabelsAndDebug(Entry.begin());

  // Do a quick scan to see if we have any checkable loads.
  bool HasVulnerableLoad = hasVulnerableLoad(MF);

  // See if we have any conditional branching blocks that we will need to trace
  // predicate state through.
  SmallVector<BlockCondInfo, 16> Infos = collectBlockCondInfo(MF);

  // If we have no interesting conditions or loads, nothing to do here.
  if (!HasVulnerableLoad && Infos.empty())
    return true;

  unsigned PredStateReg;
  unsigned PredStateSizeInBytes = TRI->getRegSizeInBits(*PredStateRC) / 8;

  // The poison value is required to be an all-ones value for many aspects of
  // this mitigation.
  const int PoisonVal = -1;
  unsigned PoisonReg = MRI->createVirtualRegister(PredStateRC);
  BuildMI(Entry, EntryInsertPt, Loc, TII->get(X86::MOV64ri32), PoisonReg)
      .addImm(PoisonVal);
  ++NumInstsInserted;

  // If we have loads being hardened and we've asked for call and ret edges to
  // get a full fence-based mitigation, inject that fence.
  if (HasVulnerableLoad && FenceCallAndRet) {
    // We need to insert an LFENCE at the start of the function to suspend any
    // incoming misspeculation from the caller. This helps two-fold: the caller
    // may not have been protected as this code has been, and this code gets to
    // not take any specific action to protect across calls.
    // FIXME: We could skip this for functions which unconditionally return
    // a constant.
    BuildMI(Entry, EntryInsertPt, Loc, TII->get(X86::LFENCE));
    ++NumInstsInserted;
    ++NumLFENCEsInserted;
  }

  // If we guarded the entry with an LFENCE and have no conditionals to protect
  // in blocks, then we're done.
  if (FenceCallAndRet && Infos.empty())
    // We may have changed the function's code at this point to insert fences.
    return true;

  // For every basic block in the function which can b
  if (HardenInterprocedurally && !FenceCallAndRet) {
    // Set up the predicate state by extracting it from the incoming stack
    // pointer so we pick up any misspeculation in our caller.
    PredStateReg = extractPredStateFromSP(Entry, EntryInsertPt, Loc);
  } else {
    // Otherwise, just build the predicate state itself by zeroing a register
    // as we don't need any initial state.
    PredStateReg = MRI->createVirtualRegister(PredStateRC);
    unsigned PredStateSubReg = MRI->createVirtualRegister(&X86::GR32RegClass);
    auto ZeroI = BuildMI(Entry, EntryInsertPt, Loc, TII->get(X86::MOV32r0),
                         PredStateSubReg);
    ++NumInstsInserted;
    MachineOperand *ZeroEFLAGSDefOp =
        ZeroI->findRegisterDefOperand(X86::EFLAGS);
    assert(ZeroEFLAGSDefOp && ZeroEFLAGSDefOp->isImplicit() &&
           "Must have an implicit def of EFLAGS!");
    ZeroEFLAGSDefOp->setIsDead(true);
    BuildMI(Entry, EntryInsertPt, Loc, TII->get(X86::SUBREG_TO_REG),
            PredStateReg)
        .addImm(0)
        .addReg(PredStateSubReg)
        .addImm(X86::sub_32bit);
  }

  // We're going to need to trace predicate state throughout the function's
  // CFG. Prepare for this by setting up our initial state of PHIs with unique
  // predecessor entries and all the initial predicate state.
  canonicalizePHIOperands(MF);

  // Track the updated values in an SSA updater to rewrite into SSA form at the
  // end.
  MachineSSAUpdater PredStateSSA(MF);
  PredStateSSA.Initialize(PredStateReg);
  PredStateSSA.AddAvailableValue(&Entry, PredStateReg);
  // Collect the inserted instructions so we can rewrite their uses of the
  // predicate state into SSA form.
  SmallVector<MachineInstr *, 16> CMovs;

  // Now walk all of the basic blocks looking for ones that end in conditional
  // jumps where we need to update this register along each edge.
  for (BlockCondInfo &Info : Infos) {
    MachineBasicBlock &MBB = *Info.MBB;
    SmallVectorImpl<MachineInstr *> &CondBrs = Info.CondBrs;
    MachineInstr *UncondBr = Info.UncondBr;

    LLVM_DEBUG(dbgs() << "Tracing predicate through block: " << MBB.getName()
                      << "\n");
    ++NumCondBranchesTraced;

    // Compute the non-conditional successor as either the target of any
    // unconditional branch or the layout successor.
    MachineBasicBlock *UncondSucc =
        UncondBr ? (UncondBr->getOpcode() == X86::JMP_1
                        ? UncondBr->getOperand(0).getMBB()
                        : nullptr)
                 : &*std::next(MachineFunction::iterator(&MBB));

    // Count how many edges there are to any given successor.
    SmallDenseMap<MachineBasicBlock *, int> SuccCounts;
    if (UncondSucc)
      ++SuccCounts[UncondSucc];
    for (auto *CondBr : CondBrs)
      ++SuccCounts[CondBr->getOperand(0).getMBB()];

    // A lambda to insert cmov instructions into a block checking all of the
    // condition codes in a sequence.
    auto BuildCheckingBlockForSuccAndConds =
        [&](MachineBasicBlock &MBB, MachineBasicBlock &Succ, int SuccCount,
            MachineInstr *Br, MachineInstr *&UncondBr,
            ArrayRef<X86::CondCode> Conds) {
          // First, we split the edge to insert the checking block into a safe
          // location.
          auto &CheckingMBB =
              (SuccCount == 1 && Succ.pred_size() == 1)
                  ? Succ
                  : splitEdge(MBB, Succ, SuccCount, Br, UncondBr, *TII);

          bool LiveEFLAGS = Succ.isLiveIn(X86::EFLAGS);
          if (!LiveEFLAGS)
            CheckingMBB.addLiveIn(X86::EFLAGS);

          // Now insert the cmovs to implement the checks.
          auto InsertPt = CheckingMBB.begin();
          assert((InsertPt == CheckingMBB.end() || !InsertPt->isPHI()) &&
                 "Should never have a PHI in the initial checking block as it "
                 "always has a single predecessor!");

          // We will wire each cmov to each other, but need to start with the
          // incoming pred state.
          unsigned CurStateReg = PredStateReg;

          for (X86::CondCode Cond : Conds) {
            auto CMovOp = X86::getCMovFromCond(Cond, PredStateSizeInBytes);

            unsigned UpdatedStateReg = MRI->createVirtualRegister(PredStateRC);
            auto CMovI = BuildMI(CheckingMBB, InsertPt, Loc, TII->get(CMovOp),
                                 UpdatedStateReg)
                             .addReg(CurStateReg)
                             .addReg(PoisonReg);
            // If this is the last cmov and the EFLAGS weren't originally
            // live-in, mark them as killed.
            if (!LiveEFLAGS && Cond == Conds.back())
              CMovI->findRegisterUseOperand(X86::EFLAGS)->setIsKill(true);

            ++NumInstsInserted;
            LLVM_DEBUG(dbgs() << "  Inserting cmov: "; CMovI->dump();
                       dbgs() << "\n");

            // The first one of the cmovs will be using the top level
            // `PredStateReg` and need to get rewritten into SSA form.
            if (CurStateReg == PredStateReg)
              CMovs.push_back(&*CMovI);

            // The next cmov should start from this one's def.
            CurStateReg = UpdatedStateReg;
          }

          // And put the last one into the available values for PredStateSSA.
          PredStateSSA.AddAvailableValue(&CheckingMBB, CurStateReg);
        };

    std::vector<X86::CondCode> UncondCodeSeq;
    for (auto *CondBr : CondBrs) {
      MachineBasicBlock &Succ = *CondBr->getOperand(0).getMBB();
      int &SuccCount = SuccCounts[&Succ];

      X86::CondCode Cond = X86::getCondFromBranchOpc(CondBr->getOpcode());
      X86::CondCode InvCond = X86::GetOppositeBranchCondition(Cond);
      UncondCodeSeq.push_back(Cond);

      BuildCheckingBlockForSuccAndConds(MBB, Succ, SuccCount, CondBr, UncondBr,
                                        {InvCond});

      // Decrement the successor count now that we've split one of the edges.
      // We need to keep the count of edges to the successor accurate in order
      // to know above when to *replace* the successor in the CFG vs. just
      // adding the new successor.
      --SuccCount;
    }

    // Since we may have split edges and changed the number of successors,
    // normalize the probabilities. This avoids doing it each time we split an
    // edge.
    MBB.normalizeSuccProbs();

    // Finally, we need to insert cmovs into the "fallthrough" edge. Here, we
    // need to intersect the other condition codes. We can do this by just
    // doing a cmov for each one.
    if (!UncondSucc)
      // If we have no fallthrough to protect (perhaps it is an indirect jump?)
      // just skip this and continue.
      continue;

    assert(SuccCounts[UncondSucc] == 1 &&
           "We should never have more than one edge to the unconditional "
           "successor at this point because every other edge must have been "
           "split above!");

    // Sort and unique the codes to minimize them.
    llvm::sort(UncondCodeSeq.begin(), UncondCodeSeq.end());
    UncondCodeSeq.erase(std::unique(UncondCodeSeq.begin(), UncondCodeSeq.end()),
                        UncondCodeSeq.end());

    // Build a checking version of the successor.
    BuildCheckingBlockForSuccAndConds(MBB, *UncondSucc, /*SuccCount*/ 1,
                                      UncondBr, UncondBr, UncondCodeSeq);
  }

  // We may also enter basic blocks in this function via exception handling
  // control flow. Here, if we are hardening interprocedurally, we need to
  // re-capture the predicate state from the throwing code. In the Itanium ABI,
  // the throw will always look like a call to __cxa_throw and will have the
  // predicate state in the stack pointer, so extract fresh predicate state from
  // the stack pointer and make it available in SSA.
  // FIXME: Handle non-itanium ABI EH models.
  if (HardenInterprocedurally) {
    for (MachineBasicBlock &MBB : MF) {
      assert(!MBB.isEHScopeEntry() && "Only Itanium ABI EH supported!");
      assert(!MBB.isEHFuncletEntry() && "Only Itanium ABI EH supported!");
      assert(!MBB.isCleanupFuncletEntry() && "Only Itanium ABI EH supported!");
      if (!MBB.isEHPad())
        continue;
      PredStateSSA.AddAvailableValue(
          &MBB,
          extractPredStateFromSP(MBB, MBB.SkipPHIsAndLabels(MBB.begin()), Loc));
    }
  }

  // Now check all of the loads using the predicate state.
  checkAllLoads(MF, PredStateSSA);

  // Now rewrite all the uses of the pred state using the SSA updater so that
  // we track updates through the CFG.
  for (MachineInstr *CMovI : CMovs)
    for (MachineOperand &Op : CMovI->operands()) {
      if (!Op.isReg() || Op.getReg() != PredStateReg)
        continue;

      PredStateSSA.RewriteUse(Op);
    }

  // If we are hardening interprocedurally, find each returning block and
  // protect the caller from being returned to through misspeculation.
  if (HardenInterprocedurally)
    for (MachineBasicBlock &MBB : MF) {
      if (MBB.empty())
        continue;

      MachineInstr &MI = MBB.back();
      if (!MI.isReturn())
        continue;

      checkReturnInstr(MI, PredStateSSA);
    }

  LLVM_DEBUG(dbgs() << "Final speculative load hardened function:\n"; MF.dump();
             dbgs() << "\n"; MF.verify(this));
  return true;
}

/// Implements the naive hardening approach of putting an LFENCE after every
/// potentially mis-predicted control flow construct.
///
/// We include this as an alternative mostly for the purpose of comparison. The
/// performance impact of this is expected to be extremely severe and not
/// practical for any real-world users.
void X86SpeculativeLoadHardeningPass::hardenEdgesWithLFENCE(
    MachineFunction &MF) {
  // First, we scan the function looking for blocks that are reached along edges
  // that we might want to harden.
  SmallSetVector<MachineBasicBlock *, 8> Blocks;
  for (MachineBasicBlock &MBB : MF) {
    // If there are no or only one successor, nothing to do here.
    if (MBB.succ_size() <= 1)
      continue;

    // Skip blocks unless their terminators start with a branch. Other
    // terminators don't seem interesting for guarding against misspeculation.
    auto TermIt = MBB.getFirstTerminator();
    if (TermIt == MBB.end() || !TermIt->isBranch())
      continue;

    // Add all the non-EH-pad succossors to the blocks we want to harden. We
    // skip EH pads because there isn't really a condition of interest on
    // entering.
    for (MachineBasicBlock *SuccMBB : MBB.successors())
      if (!SuccMBB->isEHPad())
        Blocks.insert(SuccMBB);
  }

  for (MachineBasicBlock *MBB : Blocks) {
    auto InsertPt = MBB->SkipPHIsAndLabels(MBB->begin());
    BuildMI(*MBB, InsertPt, DebugLoc(), TII->get(X86::LFENCE));
    ++NumInstsInserted;
    ++NumLFENCEsInserted;
  }
}

SmallVector<X86SpeculativeLoadHardeningPass::BlockCondInfo, 16>
X86SpeculativeLoadHardeningPass::collectBlockCondInfo(MachineFunction &MF) {
  SmallVector<BlockCondInfo, 16> Infos;

  // Walk the function and build up a summary for each block's conditions that
  // we need to trace through.
  for (MachineBasicBlock &MBB : MF) {
    // If there are no or only one successor, nothing to do here.
    if (MBB.succ_size() <= 1)
      continue;

    // We want to reliably handle any conditional branch terminators in the
    // MBB, so we manually analyze the branch. We can handle all of the
    // permutations here, including ones that analyze branch cannot.
    //
    // The approach is to walk backwards across the terminators, resetting at
    // any unconditional non-indirect branch, and track all conditional edges
    // to basic blocks as well as the fallthrough or unconditional successor
    // edge. For each conditional edge, we track the target and the opposite
    // condition code in order to inject a "no-op" cmov into that successor
    // that will harden the predicate. For the fallthrough/unconditional
    // edge, we inject a separate cmov for each conditional branch with
    // matching condition codes. This effectively implements an "and" of the
    // condition flags, even if there isn't a single condition flag that would
    // directly implement that. We don't bother trying to optimize either of
    // these cases because if such an optimization is possible, LLVM should
    // have optimized the conditional *branches* in that way already to reduce
    // instruction count. This late, we simply assume the minimal number of
    // branch instructions is being emitted and use that to guide our cmov
    // insertion.

    BlockCondInfo Info = {&MBB, {}, nullptr};

    // Now walk backwards through the terminators and build up successors they
    // reach and the conditions.
    for (MachineInstr &MI : llvm::reverse(MBB)) {
      // Once we've handled all the terminators, we're done.
      if (!MI.isTerminator())
        break;

      // If we see a non-branch terminator, we can't handle anything so bail.
      if (!MI.isBranch()) {
        Info.CondBrs.clear();
        break;
      }

      // If we see an unconditional branch, reset our state, clear any
      // fallthrough, and set this is the "else" successor.
      if (MI.getOpcode() == X86::JMP_1) {
        Info.CondBrs.clear();
        Info.UncondBr = &MI;
        continue;
      }

      // If we get an invalid condition, we have an indirect branch or some
      // other unanalyzable "fallthrough" case. We model this as a nullptr for
      // the destination so we can still guard any conditional successors.
      // Consider code sequences like:
      // ```
      //   jCC L1
      //   jmpq *%rax
      // ```
      // We still want to harden the edge to `L1`.
      if (X86::getCondFromBranchOpc(MI.getOpcode()) == X86::COND_INVALID) {
        Info.CondBrs.clear();
        Info.UncondBr = &MI;
        continue;
      }

      // We have a vanilla conditional branch, add it to our list.
      Info.CondBrs.push_back(&MI);
    }
    if (Info.CondBrs.empty()) {
      ++NumBranchesUntraced;
      LLVM_DEBUG(dbgs() << "WARNING: unable to secure successors of block:\n";
                 MBB.dump());
      continue;
    }

    Infos.push_back(Info);
  }

  return Infos;
}

/// Returns true if the instruction has no behavior (specified or otherwise)
/// that is based on the value of any of its register operands
///
/// A classical example of something that is inherently not data invariant is an
/// indirect jump -- the destination is loaded into icache based on the bits set
/// in the jump destination register.
///
/// FIXME: This should become part of our instruction tables.
static bool isDataInvariant(MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    // By default, assume that the instruction is not data invariant.
    return false;

    // Some target-independent operations that trivially lower to data-invariant
    // instructions.
  case TargetOpcode::COPY:
  case TargetOpcode::INSERT_SUBREG:
  case TargetOpcode::SUBREG_TO_REG:
    return true;

  // On x86 it is believed that imul is constant time w.r.t. the loaded data.
  // However, they set flags and are perhaps the most surprisingly constant
  // time operations so we call them out here separately.
  case X86::IMUL16rr:
  case X86::IMUL16rri8:
  case X86::IMUL16rri:
  case X86::IMUL32rr:
  case X86::IMUL32rri8:
  case X86::IMUL32rri:
  case X86::IMUL64rr:
  case X86::IMUL64rri32:
  case X86::IMUL64rri8:

  // Bit scanning and counting instructions that are somewhat surprisingly
  // constant time as they scan across bits and do other fairly complex
  // operations like popcnt, but are believed to be constant time on x86.
  // However, these set flags.
  case X86::BSF16rr:
  case X86::BSF32rr:
  case X86::BSF64rr:
  case X86::BSR16rr:
  case X86::BSR32rr:
  case X86::BSR64rr:
  case X86::LZCNT16rr:
  case X86::LZCNT32rr:
  case X86::LZCNT64rr:
  case X86::POPCNT16rr:
  case X86::POPCNT32rr:
  case X86::POPCNT64rr:
  case X86::TZCNT16rr:
  case X86::TZCNT32rr:
  case X86::TZCNT64rr:

  // Bit manipulation instructions are effectively combinations of basic
  // arithmetic ops, and should still execute in constant time. These also
  // set flags.
  case X86::BLCFILL32rr:
  case X86::BLCFILL64rr:
  case X86::BLCI32rr:
  case X86::BLCI64rr:
  case X86::BLCIC32rr:
  case X86::BLCIC64rr:
  case X86::BLCMSK32rr:
  case X86::BLCMSK64rr:
  case X86::BLCS32rr:
  case X86::BLCS64rr:
  case X86::BLSFILL32rr:
  case X86::BLSFILL64rr:
  case X86::BLSI32rr:
  case X86::BLSI64rr:
  case X86::BLSIC32rr:
  case X86::BLSIC64rr:
  case X86::BLSMSK32rr:
  case X86::BLSMSK64rr:
  case X86::BLSR32rr:
  case X86::BLSR64rr:
  case X86::TZMSK32rr:
  case X86::TZMSK64rr:

  // Bit extracting and clearing instructions should execute in constant time,
  // and set flags.
  case X86::BEXTR32rr:
  case X86::BEXTR64rr:
  case X86::BEXTRI32ri:
  case X86::BEXTRI64ri:
  case X86::BZHI32rr:
  case X86::BZHI64rr:

  // Basic arithmetic is constant time on the input but does set flags.
  case X86::ADC8rr:   case X86::ADC8ri:
  case X86::ADC16rr:  case X86::ADC16ri:   case X86::ADC16ri8:
  case X86::ADC32rr:  case X86::ADC32ri:   case X86::ADC32ri8:
  case X86::ADC64rr:  case X86::ADC64ri8:  case X86::ADC64ri32:
  case X86::ADD8rr:   case X86::ADD8ri:
  case X86::ADD16rr:  case X86::ADD16ri:   case X86::ADD16ri8:
  case X86::ADD32rr:  case X86::ADD32ri:   case X86::ADD32ri8:
  case X86::ADD64rr:  case X86::ADD64ri8:  case X86::ADD64ri32:
  case X86::AND8rr:   case X86::AND8ri:
  case X86::AND16rr:  case X86::AND16ri:   case X86::AND16ri8:
  case X86::AND32rr:  case X86::AND32ri:   case X86::AND32ri8:
  case X86::AND64rr:  case X86::AND64ri8:  case X86::AND64ri32:
  case X86::OR8rr:    case X86::OR8ri:
  case X86::OR16rr:   case X86::OR16ri:    case X86::OR16ri8:
  case X86::OR32rr:   case X86::OR32ri:    case X86::OR32ri8:
  case X86::OR64rr:   case X86::OR64ri8:   case X86::OR64ri32:
  case X86::SBB8rr:   case X86::SBB8ri:
  case X86::SBB16rr:  case X86::SBB16ri:   case X86::SBB16ri8:
  case X86::SBB32rr:  case X86::SBB32ri:   case X86::SBB32ri8:
  case X86::SBB64rr:  case X86::SBB64ri8:  case X86::SBB64ri32:
  case X86::SUB8rr:   case X86::SUB8ri:
  case X86::SUB16rr:  case X86::SUB16ri:   case X86::SUB16ri8:
  case X86::SUB32rr:  case X86::SUB32ri:   case X86::SUB32ri8:
  case X86::SUB64rr:  case X86::SUB64ri8:  case X86::SUB64ri32:
  case X86::XOR8rr:   case X86::XOR8ri:
  case X86::XOR16rr:  case X86::XOR16ri:   case X86::XOR16ri8:
  case X86::XOR32rr:  case X86::XOR32ri:   case X86::XOR32ri8:
  case X86::XOR64rr:  case X86::XOR64ri8:  case X86::XOR64ri32:
  // Arithmetic with just 32-bit and 64-bit variants and no immediates.
  case X86::ADCX32rr: case X86::ADCX64rr:
  case X86::ADOX32rr: case X86::ADOX64rr:
  case X86::ANDN32rr: case X86::ANDN64rr:
  // Just one operand for inc and dec.
  case X86::INC8r: case X86::INC16r: case X86::INC32r: case X86::INC64r:
  case X86::DEC8r: case X86::DEC16r: case X86::DEC32r: case X86::DEC64r:
    // Check whether the EFLAGS implicit-def is dead. We assume that this will
    // always find the implicit-def because this code should only be reached
    // for instructions that do in fact implicitly def this.
    if (!MI.findRegisterDefOperand(X86::EFLAGS)->isDead()) {
      // If we would clobber EFLAGS that are used, just bail for now.
      LLVM_DEBUG(dbgs() << "    Unable to harden post-load due to EFLAGS: ";
                 MI.dump(); dbgs() << "\n");
      return false;
    }

    // Otherwise, fallthrough to handle these the same as instructions that
    // don't set EFLAGS.
    LLVM_FALLTHROUGH;

  // Integer multiply w/o affecting flags is still believed to be constant
  // time on x86. Called out separately as this is among the most surprising
  // instructions to exhibit that behavior.
  case X86::MULX32rr:
  case X86::MULX64rr:

  // Arithmetic instructions that are both constant time and don't set flags.
  case X86::RORX32ri:
  case X86::RORX64ri:
  case X86::SARX32rr:
  case X86::SARX64rr:
  case X86::SHLX32rr:
  case X86::SHLX64rr:
  case X86::SHRX32rr:
  case X86::SHRX64rr:

  // LEA doesn't actually access memory, and its arithmetic is constant time.
  case X86::LEA16r:
  case X86::LEA32r:
  case X86::LEA64_32r:
  case X86::LEA64r:
    return true;
  }
}

/// Returns true if the instruction has no behavior (specified or otherwise)
/// that is based on the value loaded from memory or the value of any
/// non-address register operands.
///
/// For example, if the latency of the instruction is dependent on the
/// particular bits set in any of the registers *or* any of the bits loaded from
/// memory.
///
/// A classical example of something that is inherently not data invariant is an
/// indirect jump -- the destination is loaded into icache based on the bits set
/// in the jump destination register.
///
/// FIXME: This should become part of our instruction tables.
static bool isDataInvariantLoad(MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    // By default, assume that the load will immediately leak.
    return false;

  // On x86 it is believed that imul is constant time w.r.t. the loaded data.
  // However, they set flags and are perhaps the most surprisingly constant
  // time operations so we call them out here separately.
  case X86::IMUL16rm:
  case X86::IMUL16rmi8:
  case X86::IMUL16rmi:
  case X86::IMUL32rm:
  case X86::IMUL32rmi8:
  case X86::IMUL32rmi:
  case X86::IMUL64rm:
  case X86::IMUL64rmi32:
  case X86::IMUL64rmi8:

  // Bit scanning and counting instructions that are somewhat surprisingly
  // constant time as they scan across bits and do other fairly complex
  // operations like popcnt, but are believed to be constant time on x86.
  // However, these set flags.
  case X86::BSF16rm:
  case X86::BSF32rm:
  case X86::BSF64rm:
  case X86::BSR16rm:
  case X86::BSR32rm:
  case X86::BSR64rm:
  case X86::LZCNT16rm:
  case X86::LZCNT32rm:
  case X86::LZCNT64rm:
  case X86::POPCNT16rm:
  case X86::POPCNT32rm:
  case X86::POPCNT64rm:
  case X86::TZCNT16rm:
  case X86::TZCNT32rm:
  case X86::TZCNT64rm:

  // Bit manipulation instructions are effectively combinations of basic
  // arithmetic ops, and should still execute in constant time. These also
  // set flags.
  case X86::BLCFILL32rm:
  case X86::BLCFILL64rm:
  case X86::BLCI32rm:
  case X86::BLCI64rm:
  case X86::BLCIC32rm:
  case X86::BLCIC64rm:
  case X86::BLCMSK32rm:
  case X86::BLCMSK64rm:
  case X86::BLCS32rm:
  case X86::BLCS64rm:
  case X86::BLSFILL32rm:
  case X86::BLSFILL64rm:
  case X86::BLSI32rm:
  case X86::BLSI64rm:
  case X86::BLSIC32rm:
  case X86::BLSIC64rm:
  case X86::BLSMSK32rm:
  case X86::BLSMSK64rm:
  case X86::BLSR32rm:
  case X86::BLSR64rm:
  case X86::TZMSK32rm:
  case X86::TZMSK64rm:

  // Bit extracting and clearing instructions should execute in constant time,
  // and set flags.
  case X86::BEXTR32rm:
  case X86::BEXTR64rm:
  case X86::BEXTRI32mi:
  case X86::BEXTRI64mi:
  case X86::BZHI32rm:
  case X86::BZHI64rm:

  // Basic arithmetic is constant time on the input but does set flags.
  case X86::ADC8rm:
  case X86::ADC16rm:
  case X86::ADC32rm:
  case X86::ADC64rm:
  case X86::ADCX32rm:
  case X86::ADCX64rm:
  case X86::ADD8rm:
  case X86::ADD16rm:
  case X86::ADD32rm:
  case X86::ADD64rm:
  case X86::ADOX32rm:
  case X86::ADOX64rm:
  case X86::AND8rm:
  case X86::AND16rm:
  case X86::AND32rm:
  case X86::AND64rm:
  case X86::ANDN32rm:
  case X86::ANDN64rm:
  case X86::OR8rm:
  case X86::OR16rm:
  case X86::OR32rm:
  case X86::OR64rm:
  case X86::SBB8rm:
  case X86::SBB16rm:
  case X86::SBB32rm:
  case X86::SBB64rm:
  case X86::SUB8rm:
  case X86::SUB16rm:
  case X86::SUB32rm:
  case X86::SUB64rm:
  case X86::XOR8rm:
  case X86::XOR16rm:
  case X86::XOR32rm:
  case X86::XOR64rm:
    // Check whether the EFLAGS implicit-def is dead. We assume that this will
    // always find the implicit-def because this code should only be reached
    // for instructions that do in fact implicitly def this.
    if (!MI.findRegisterDefOperand(X86::EFLAGS)->isDead()) {
      // If we would clobber EFLAGS that are used, just bail for now.
      LLVM_DEBUG(dbgs() << "    Unable to harden post-load due to EFLAGS: ";
                 MI.dump(); dbgs() << "\n");
      return false;
    }

    // Otherwise, fallthrough to handle these the same as instructions that
    // don't set EFLAGS.
    LLVM_FALLTHROUGH;

  // Integer multiply w/o affecting flags is still believed to be constant
  // time on x86. Called out separately as this is among the most surprising
  // instructions to exhibit that behavior.
  case X86::MULX32rm:
  case X86::MULX64rm:

  // Arithmetic instructions that are both constant time and don't set flags.
  case X86::RORX32mi:
  case X86::RORX64mi:
  case X86::SARX32rm:
  case X86::SARX64rm:
  case X86::SHLX32rm:
  case X86::SHLX64rm:
  case X86::SHRX32rm:
  case X86::SHRX64rm:

  // Conversions are believed to be constant time and don't set flags.
  case X86::CVTTSD2SI64rm: case X86::VCVTTSD2SI64rm: case X86::VCVTTSD2SI64Zrm:
  case X86::CVTTSD2SIrm:   case X86::VCVTTSD2SIrm:   case X86::VCVTTSD2SIZrm:
  case X86::CVTTSS2SI64rm: case X86::VCVTTSS2SI64rm: case X86::VCVTTSS2SI64Zrm:
  case X86::CVTTSS2SIrm:   case X86::VCVTTSS2SIrm:   case X86::VCVTTSS2SIZrm:
  case X86::CVTSI2SDrm:    case X86::VCVTSI2SDrm:    case X86::VCVTSI2SDZrm:
  case X86::CVTSI2SSrm:    case X86::VCVTSI2SSrm:    case X86::VCVTSI2SSZrm:
  case X86::CVTSI642SDrm:  case X86::VCVTSI642SDrm:  case X86::VCVTSI642SDZrm:
  case X86::CVTSI642SSrm:  case X86::VCVTSI642SSrm:  case X86::VCVTSI642SSZrm:
  case X86::CVTSS2SDrm:    case X86::VCVTSS2SDrm:    case X86::VCVTSS2SDZrm:
  case X86::CVTSD2SSrm:    case X86::VCVTSD2SSrm:    case X86::VCVTSD2SSZrm:
  // AVX512 added unsigned integer conversions.
  case X86::VCVTTSD2USI64Zrm:
  case X86::VCVTTSD2USIZrm:
  case X86::VCVTTSS2USI64Zrm:
  case X86::VCVTTSS2USIZrm:
  case X86::VCVTUSI2SDZrm:
  case X86::VCVTUSI642SDZrm:
  case X86::VCVTUSI2SSZrm:
  case X86::VCVTUSI642SSZrm:

  // Loads to register don't set flags.
  case X86::MOV8rm:
  case X86::MOV8rm_NOREX:
  case X86::MOV16rm:
  case X86::MOV32rm:
  case X86::MOV64rm:
  case X86::MOVSX16rm8:
  case X86::MOVSX32rm16:
  case X86::MOVSX32rm8:
  case X86::MOVSX32rm8_NOREX:
  case X86::MOVSX64rm16:
  case X86::MOVSX64rm32:
  case X86::MOVSX64rm8:
  case X86::MOVZX16rm8:
  case X86::MOVZX32rm16:
  case X86::MOVZX32rm8:
  case X86::MOVZX32rm8_NOREX:
  case X86::MOVZX64rm16:
  case X86::MOVZX64rm8:
    return true;
  }
}

static bool isEFLAGSLive(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                         const TargetRegisterInfo &TRI) {
  // Check if EFLAGS are alive by seeing if there is a def of them or they
  // live-in, and then seeing if that def is in turn used.
  for (MachineInstr &MI : llvm::reverse(llvm::make_range(MBB.begin(), I))) {
    if (MachineOperand *DefOp = MI.findRegisterDefOperand(X86::EFLAGS)) {
      // If the def is dead, then EFLAGS is not live.
      if (DefOp->isDead())
        return false;

      // Otherwise we've def'ed it, and it is live.
      return true;
    }
    // While at this instruction, also check if we use and kill EFLAGS
    // which means it isn't live.
    if (MI.killsRegister(X86::EFLAGS, &TRI))
      return false;
  }

  // If we didn't find anything conclusive (neither definitely alive or
  // definitely dead) return whether it lives into the block.
  return MBB.isLiveIn(X86::EFLAGS);
}

void X86SpeculativeLoadHardeningPass::checkAllLoads(
    MachineFunction &MF, MachineSSAUpdater &PredStateSSA) {
  // If the actual checking of loads is disabled, skip doing anything here.
  if (!HardenLoads)
    return;

  SmallPtrSet<MachineInstr *, 16> HardenPostLoad;
  SmallPtrSet<MachineInstr *, 16> HardenLoadAddr;

  SmallSet<unsigned, 16> HardenedAddrRegs;

  SmallDenseMap<unsigned, unsigned, 32> AddrRegToHardenedReg;

  // Track the set of load-dependent registers through the basic block. Because
  // the values of these registers have an existing data dependency on a loaded
  // value which we would have checked, we can omit any checks on them.
  SparseBitVector<> LoadDepRegs;

  for (MachineBasicBlock &MBB : MF) {
    // We harden the loads of a basic block in several passes:
    //
    // 1) Collect all the loads which can have their loaded value hardened
    //    and all the loads that instead need their address hardened. During
    //    this walk we propagate load dependence for address hardened loads and
    //    also look for LFENCE to stop hardening wherever possible. When
    //    deciding whether or not to harden the loaded value or not, we check
    //    to see if any registers used in the address will have been hardened
    //    at this point and if so, harden any remaining address registers as
    //    that often successfully re-uses hardened addresses and minimizes
    //    instructions. FIXME: We should consider an aggressive mode where we
    //    continue to keep as many loads value hardened even when some address
    //    register hardening would be free (due to reuse).
    for (MachineInstr &MI : MBB) {
      // We naively assume that all def'ed registers of an instruction have
      // a data dependency on all of their operands.
      // FIXME: Do a more careful analysis of x86 to build a conservative model
      // here.
      if (llvm::any_of(MI.uses(), [&](MachineOperand &Op) {
            return Op.isReg() && LoadDepRegs.test(Op.getReg());
          }))
        for (MachineOperand &Def : MI.defs())
          if (Def.isReg())
            LoadDepRegs.set(Def.getReg());

      // Both Intel and AMD are guiding that they will change the semantics of
      // LFENCE to be a speculation barrier, so if we see an LFENCE, there is
      // no more need to guard things in this block.
      if (MI.getOpcode() == X86::LFENCE)
        break;

      // If this instruction cannot load, nothing to do.
      if (!MI.mayLoad())
        continue;

      // Some instructions which "load" are trivially safe or unimportant.
      if (MI.getOpcode() == X86::MFENCE)
        continue;

      // Extract the memory operand information about this instruction.
      // FIXME: This doesn't handle loading pseudo instructions which we often
      // could handle with similarly generic logic. We probably need to add an
      // MI-layer routine similar to the MC-layer one we use here which maps
      // pseudos much like this maps real instructions.
      const MCInstrDesc &Desc = MI.getDesc();
      int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);
      if (MemRefBeginIdx < 0) {
        LLVM_DEBUG(dbgs() << "WARNING: unable to harden loading instruction: ";
                   MI.dump());
        continue;
      }

      MemRefBeginIdx += X86II::getOperandBias(Desc);

      MachineOperand &BaseMO = MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg);
      MachineOperand &IndexMO =
          MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg);

      // If we have at least one (non-frame-index, non-RIP) register operand,
      // and neither operand is load-dependent, we need to check the load.
      unsigned BaseReg = 0, IndexReg = 0;
      if (!BaseMO.isFI() && BaseMO.getReg() != X86::RIP &&
          BaseMO.getReg() != X86::NoRegister)
        BaseReg = BaseMO.getReg();
      if (IndexMO.getReg() != X86::NoRegister)
        IndexReg = IndexMO.getReg();

      if (!BaseReg && !IndexReg)
        // No register operands!
        continue;

      // If any register operand is dependent, this load is dependent and we
      // needn't check it.
      // FIXME: Is this true in the case where we are hardening loads after
      // they complete? Unclear, need to investigate.
      if ((BaseReg && LoadDepRegs.test(BaseReg)) ||
          (IndexReg && LoadDepRegs.test(IndexReg)))
        continue;

      // If post-load hardening is enabled, this load is compatible with
      // post-load hardening, and we aren't already going to harden one of the
      // address registers, queue it up to be hardened post-load. Notably, even
      // once hardened this won't introduce a useful dependency that could prune
      // out subsequent loads.
      if (EnablePostLoadHardening && isDataInvariantLoad(MI) &&
          MI.getDesc().getNumDefs() == 1 && MI.getOperand(0).isReg() &&
          canHardenRegister(MI.getOperand(0).getReg()) &&
          !HardenedAddrRegs.count(BaseReg) &&
          !HardenedAddrRegs.count(IndexReg)) {
        HardenPostLoad.insert(&MI);
        HardenedAddrRegs.insert(MI.getOperand(0).getReg());
        continue;
      }

      // Record this instruction for address hardening and record its register
      // operands as being address-hardened.
      HardenLoadAddr.insert(&MI);
      if (BaseReg)
        HardenedAddrRegs.insert(BaseReg);
      if (IndexReg)
        HardenedAddrRegs.insert(IndexReg);

      for (MachineOperand &Def : MI.defs())
        if (Def.isReg())
          LoadDepRegs.set(Def.getReg());
    }

    // Now re-walk the instructions in the basic block, and apply whichever
    // hardening strategy we have elected. Note that we do this in a second
    // pass specifically so that we have the complete set of instructions for
    // which we will do post-load hardening and can defer it in certain
    // circumstances.
    //
    // FIXME: This could probably be made even more effective by doing it
    // across the entire function. Rather than just walking the flat list
    // backwards here, we could walk the function in PO and each block bottom
    // up, allowing us to in some cases sink hardening across block blocks. As
    // long as the in-block predicate state is used at the eventual hardening
    // site, this remains safe.
    for (MachineInstr &MI : MBB) {
      // We cannot both require hardening the def of a load and its address.
      assert(!(HardenLoadAddr.count(&MI) && HardenPostLoad.count(&MI)) &&
             "Requested to harden both the address and def of a load!");

      // Check if this is a load whose address needs to be hardened.
      if (HardenLoadAddr.erase(&MI)) {
        const MCInstrDesc &Desc = MI.getDesc();
        int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);
        assert(MemRefBeginIdx >= 0 && "Cannot have an invalid index here!");

        MemRefBeginIdx += X86II::getOperandBias(Desc);

        MachineOperand &BaseMO =
            MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg);
        MachineOperand &IndexMO =
            MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg);
        hardenLoadAddr(MI, BaseMO, IndexMO, PredStateSSA, AddrRegToHardenedReg);
        continue;
      }

      // Test if this instruction is one of our post load instructions (and
      // remove it from the set if so).
      if (HardenPostLoad.erase(&MI)) {
        assert(!MI.isCall() && "Must not try to post-load harden a call!");

        // If this is a data-invariant load, we want to try and sink any
        // hardening as far as possible.
        if (isDataInvariantLoad(MI)) {
          // Sink the instruction we'll need to harden as far as we can down the
          // graph.
          MachineInstr *SunkMI = sinkPostLoadHardenedInst(MI, HardenPostLoad);

          // If we managed to sink this instruction, update everything so we
          // harden that instruction when we reach it in the instruction
          // sequence.
          if (SunkMI != &MI) {
            // If in sinking there was no instruction needing to be hardened,
            // we're done.
            if (!SunkMI)
              continue;

            // Otherwise, add this to the set of defs we harden.
            HardenPostLoad.insert(SunkMI);
            continue;
          }
        }

        // The register def'ed by this instruction is trivially hardened so map
        // it to itself.
        AddrRegToHardenedReg[MI.getOperand(0).getReg()] =
            MI.getOperand(0).getReg();

        hardenPostLoad(MI, PredStateSSA);
        continue;
      }

      // After we finish processing the instruction and doing any hardening
      // necessary for it, we need to handle transferring the predicate state
      // into a call and recovering it after the call returns (if it returns).
      if (!MI.isCall())
        continue;

      // If we're not hardening interprocedurally, we can just skip calls.
      if (!HardenInterprocedurally)
        continue;

      auto InsertPt = MI.getIterator();
      DebugLoc Loc = MI.getDebugLoc();

      // First, we transfer the predicate state into the called function by
      // merging it into the stack pointer. This will kill the current def of
      // the state.
      unsigned StateReg = PredStateSSA.GetValueAtEndOfBlock(&MBB);
      mergePredStateIntoSP(MBB, InsertPt, Loc, StateReg);

      // If this call is also a return (because it is a tail call) we're done.
      if (MI.isReturn())
        continue;

      // Otherwise we need to step past the call and recover the predicate
      // state from SP after the return, and make this new state available.
      ++InsertPt;
      unsigned NewStateReg = extractPredStateFromSP(MBB, InsertPt, Loc);
      PredStateSSA.AddAvailableValue(&MBB, NewStateReg);
    }

    HardenPostLoad.clear();
    HardenLoadAddr.clear();
    HardenedAddrRegs.clear();
    AddrRegToHardenedReg.clear();

    // Currently, we only track data-dependent loads within a basic block.
    // FIXME: We should see if this is necessary or if we could be more
    // aggressive here without opening up attack avenues.
    LoadDepRegs.clear();
  }
}

/// Save EFLAGS into the returned GPR. This can in turn be restored with
/// `restoreEFLAGS`.
///
/// Note that LLVM can only lower very simple patterns of saved and restored
/// EFLAGS registers. The restore should always be within the same basic block
/// as the save so that no PHI nodes are inserted.
unsigned X86SpeculativeLoadHardeningPass::saveEFLAGS(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
    DebugLoc Loc) {
  // FIXME: Hard coding this to a 32-bit register class seems weird, but matches
  // what instruction selection does.
  unsigned Reg = MRI->createVirtualRegister(&X86::GR32RegClass);
  // We directly copy the FLAGS register and rely on later lowering to clean
  // this up into the appropriate setCC instructions.
  BuildMI(MBB, InsertPt, Loc, TII->get(X86::COPY), Reg).addReg(X86::EFLAGS);
  ++NumInstsInserted;
  return Reg;
}

/// Restore EFLAGS from the provided GPR. This should be produced by
/// `saveEFLAGS`.
///
/// This must be done within the same basic block as the save in order to
/// reliably lower.
void X86SpeculativeLoadHardeningPass::restoreEFLAGS(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt, DebugLoc Loc,
    unsigned Reg) {
  BuildMI(MBB, InsertPt, Loc, TII->get(X86::COPY), X86::EFLAGS).addReg(Reg);
  ++NumInstsInserted;
}

/// Takes the current predicate state (in a register) and merges it into the
/// stack pointer. The state is essentially a single bit, but we merge this in
/// a way that won't form non-canonical pointers and also will be preserved
/// across normal stack adjustments.
void X86SpeculativeLoadHardeningPass::mergePredStateIntoSP(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt, DebugLoc Loc,
    unsigned PredStateReg) {
  unsigned TmpReg = MRI->createVirtualRegister(PredStateRC);
  // FIXME: This hard codes a shift distance based on the number of bits needed
  // to stay canonical on 64-bit. We should compute this somehow and support
  // 32-bit as part of that.
  auto ShiftI = BuildMI(MBB, InsertPt, Loc, TII->get(X86::SHL64ri), TmpReg)
                    .addReg(PredStateReg, RegState::Kill)
                    .addImm(47);
  ShiftI->addRegisterDead(X86::EFLAGS, TRI);
  ++NumInstsInserted;
  auto OrI = BuildMI(MBB, InsertPt, Loc, TII->get(X86::OR64rr), X86::RSP)
                 .addReg(X86::RSP)
                 .addReg(TmpReg, RegState::Kill);
  OrI->addRegisterDead(X86::EFLAGS, TRI);
  ++NumInstsInserted;
}

/// Extracts the predicate state stored in the high bits of the stack pointer.
unsigned X86SpeculativeLoadHardeningPass::extractPredStateFromSP(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
    DebugLoc Loc) {
  unsigned PredStateReg = MRI->createVirtualRegister(PredStateRC);
  unsigned TmpReg = MRI->createVirtualRegister(PredStateRC);

  // We know that the stack pointer will have any preserved predicate state in
  // its high bit. We just want to smear this across the other bits. Turns out,
  // this is exactly what an arithmetic right shift does.
  BuildMI(MBB, InsertPt, Loc, TII->get(TargetOpcode::COPY), TmpReg)
      .addReg(X86::RSP);
  auto ShiftI =
      BuildMI(MBB, InsertPt, Loc, TII->get(X86::SAR64ri), PredStateReg)
          .addReg(TmpReg, RegState::Kill)
          .addImm(TRI->getRegSizeInBits(*PredStateRC) - 1);
  ShiftI->addRegisterDead(X86::EFLAGS, TRI);
  ++NumInstsInserted;

  return PredStateReg;
}

void X86SpeculativeLoadHardeningPass::hardenLoadAddr(
    MachineInstr &MI, MachineOperand &BaseMO, MachineOperand &IndexMO,
    MachineSSAUpdater &PredStateSSA,
    SmallDenseMap<unsigned, unsigned, 32> &AddrRegToHardenedReg) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc Loc = MI.getDebugLoc();

  // Check if EFLAGS are alive by seeing if there is a def of them or they
  // live-in, and then seeing if that def is in turn used.
  bool EFLAGSLive = isEFLAGSLive(MBB, MI.getIterator(), *TRI);

  SmallVector<MachineOperand *, 2> HardenOpRegs;

  if (BaseMO.isFI()) {
    // A frame index is never a dynamically controllable load, so only
    // harden it if we're covering fixed address loads as well.
    LLVM_DEBUG(
        dbgs() << "  Skipping hardening base of explicit stack frame load: ";
        MI.dump(); dbgs() << "\n");
  } else if (BaseMO.getReg() == X86::RIP ||
             BaseMO.getReg() == X86::NoRegister) {
    // For both RIP-relative addressed loads or absolute loads, we cannot
    // meaningfully harden them because the address being loaded has no
    // dynamic component.
    //
    // FIXME: When using a segment base (like TLS does) we end up with the
    // dynamic address being the base plus -1 because we can't mutate the
    // segment register here. This allows the signed 32-bit offset to point at
    // valid segment-relative addresses and load them successfully.
    LLVM_DEBUG(
        dbgs() << "  Cannot harden base of "
               << (BaseMO.getReg() == X86::RIP ? "RIP-relative" : "no-base")
               << " address in a load!");
  } else {
    assert(BaseMO.isReg() &&
           "Only allowed to have a frame index or register base.");
    HardenOpRegs.push_back(&BaseMO);
  }

  if (IndexMO.getReg() != X86::NoRegister &&
      (HardenOpRegs.empty() ||
       HardenOpRegs.front()->getReg() != IndexMO.getReg()))
    HardenOpRegs.push_back(&IndexMO);

  assert((HardenOpRegs.size() == 1 || HardenOpRegs.size() == 2) &&
         "Should have exactly one or two registers to harden!");
  assert((HardenOpRegs.size() == 1 ||
          HardenOpRegs[0]->getReg() != HardenOpRegs[1]->getReg()) &&
         "Should not have two of the same registers!");

  // Remove any registers that have alreaded been checked.
  llvm::erase_if(HardenOpRegs, [&](MachineOperand *Op) {
    // See if this operand's register has already been checked.
    auto It = AddrRegToHardenedReg.find(Op->getReg());
    if (It == AddrRegToHardenedReg.end())
      // Not checked, so retain this one.
      return false;

    // Otherwise, we can directly update this operand and remove it.
    Op->setReg(It->second);
    return true;
  });
  // If there are none left, we're done.
  if (HardenOpRegs.empty())
    return;

  // Compute the current predicate state.
  unsigned StateReg = PredStateSSA.GetValueAtEndOfBlock(&MBB);

  auto InsertPt = MI.getIterator();

  // If EFLAGS are live and we don't have access to instructions that avoid
  // clobbering EFLAGS we need to save and restore them. This in turn makes
  // the EFLAGS no longer live.
  unsigned FlagsReg = 0;
  if (EFLAGSLive && !Subtarget->hasBMI2()) {
    EFLAGSLive = false;
    FlagsReg = saveEFLAGS(MBB, InsertPt, Loc);
  }

  for (MachineOperand *Op : HardenOpRegs) {
    unsigned OpReg = Op->getReg();
    auto *OpRC = MRI->getRegClass(OpReg);
    unsigned TmpReg = MRI->createVirtualRegister(OpRC);

    // If this is a vector register, we'll need somewhat custom logic to handle
    // hardening it.
    if (!Subtarget->hasVLX() && (OpRC->hasSuperClassEq(&X86::VR128RegClass) ||
                                 OpRC->hasSuperClassEq(&X86::VR256RegClass))) {
      assert(Subtarget->hasAVX2() && "AVX2-specific register classes!");
      bool Is128Bit = OpRC->hasSuperClassEq(&X86::VR128RegClass);

      // Move our state into a vector register.
      // FIXME: We could skip this at the cost of longer encodings with AVX-512
      // but that doesn't seem likely worth it.
      unsigned VStateReg = MRI->createVirtualRegister(&X86::VR128RegClass);
      auto MovI =
          BuildMI(MBB, InsertPt, Loc, TII->get(X86::VMOV64toPQIrr), VStateReg)
              .addReg(StateReg);
      (void)MovI;
      ++NumInstsInserted;
      LLVM_DEBUG(dbgs() << "  Inserting mov: "; MovI->dump(); dbgs() << "\n");

      // Broadcast it across the vector register.
      unsigned VBStateReg = MRI->createVirtualRegister(OpRC);
      auto BroadcastI = BuildMI(MBB, InsertPt, Loc,
                                TII->get(Is128Bit ? X86::VPBROADCASTQrr
                                                  : X86::VPBROADCASTQYrr),
                                VBStateReg)
                            .addReg(VStateReg);
      (void)BroadcastI;
      ++NumInstsInserted;
      LLVM_DEBUG(dbgs() << "  Inserting broadcast: "; BroadcastI->dump();
                 dbgs() << "\n");

      // Merge our potential poison state into the value with a vector or.
      auto OrI =
          BuildMI(MBB, InsertPt, Loc,
                  TII->get(Is128Bit ? X86::VPORrr : X86::VPORYrr), TmpReg)
              .addReg(VBStateReg)
              .addReg(OpReg);
      (void)OrI;
      ++NumInstsInserted;
      LLVM_DEBUG(dbgs() << "  Inserting or: "; OrI->dump(); dbgs() << "\n");
    } else if (OpRC->hasSuperClassEq(&X86::VR128XRegClass) ||
               OpRC->hasSuperClassEq(&X86::VR256XRegClass) ||
               OpRC->hasSuperClassEq(&X86::VR512RegClass)) {
      assert(Subtarget->hasAVX512() && "AVX512-specific register classes!");
      bool Is128Bit = OpRC->hasSuperClassEq(&X86::VR128XRegClass);
      bool Is256Bit = OpRC->hasSuperClassEq(&X86::VR256XRegClass);
      if (Is128Bit || Is256Bit)
        assert(Subtarget->hasVLX() && "AVX512VL-specific register classes!");

      // Broadcast our state into a vector register.
      unsigned VStateReg = MRI->createVirtualRegister(OpRC);
      unsigned BroadcastOp =
          Is128Bit ? X86::VPBROADCASTQrZ128r
                   : Is256Bit ? X86::VPBROADCASTQrZ256r : X86::VPBROADCASTQrZr;
      auto BroadcastI =
          BuildMI(MBB, InsertPt, Loc, TII->get(BroadcastOp), VStateReg)
              .addReg(StateReg);
      (void)BroadcastI;
      ++NumInstsInserted;
      LLVM_DEBUG(dbgs() << "  Inserting broadcast: "; BroadcastI->dump();
                 dbgs() << "\n");

      // Merge our potential poison state into the value with a vector or.
      unsigned OrOp = Is128Bit ? X86::VPORQZ128rr
                               : Is256Bit ? X86::VPORQZ256rr : X86::VPORQZrr;
      auto OrI = BuildMI(MBB, InsertPt, Loc, TII->get(OrOp), TmpReg)
                     .addReg(VStateReg)
                     .addReg(OpReg);
      (void)OrI;
      ++NumInstsInserted;
      LLVM_DEBUG(dbgs() << "  Inserting or: "; OrI->dump(); dbgs() << "\n");
    } else {
      // FIXME: Need to support GR32 here for 32-bit code.
      assert(OpRC->hasSuperClassEq(&X86::GR64RegClass) &&
             "Not a supported register class for address hardening!");

      if (!EFLAGSLive) {
        // Merge our potential poison state into the value with an or.
        auto OrI = BuildMI(MBB, InsertPt, Loc, TII->get(X86::OR64rr), TmpReg)
                       .addReg(StateReg)
                       .addReg(OpReg);
        OrI->addRegisterDead(X86::EFLAGS, TRI);
        ++NumInstsInserted;
        LLVM_DEBUG(dbgs() << "  Inserting or: "; OrI->dump(); dbgs() << "\n");
      } else {
        // We need to avoid touching EFLAGS so shift out all but the least
        // significant bit using the instruction that doesn't update flags.
        auto ShiftI =
            BuildMI(MBB, InsertPt, Loc, TII->get(X86::SHRX64rr), TmpReg)
                .addReg(OpReg)
                .addReg(StateReg);
        (void)ShiftI;
        ++NumInstsInserted;
        LLVM_DEBUG(dbgs() << "  Inserting shrx: "; ShiftI->dump();
                   dbgs() << "\n");
      }
    }

    // Record this register as checked and update the operand.
    assert(!AddrRegToHardenedReg.count(Op->getReg()) &&
           "Should not have checked this register yet!");
    AddrRegToHardenedReg[Op->getReg()] = TmpReg;
    Op->setReg(TmpReg);
    ++NumAddrRegsHardened;
  }

  // And restore the flags if needed.
  if (FlagsReg)
    restoreEFLAGS(MBB, InsertPt, Loc, FlagsReg);
}

MachineInstr *X86SpeculativeLoadHardeningPass::sinkPostLoadHardenedInst(
    MachineInstr &InitialMI, SmallPtrSetImpl<MachineInstr *> &HardenedInstrs) {
  assert(isDataInvariantLoad(InitialMI) &&
         "Cannot get here with a non-invariant load!");

  // See if we can sink hardening the loaded value.
  auto SinkCheckToSingleUse =
      [&](MachineInstr &MI) -> Optional<MachineInstr *> {
    unsigned DefReg = MI.getOperand(0).getReg();

    // We need to find a single use which we can sink the check. We can
    // primarily do this because many uses may already end up checked on their
    // own.
    MachineInstr *SingleUseMI = nullptr;
    for (MachineInstr &UseMI : MRI->use_instructions(DefReg)) {
      // If we're already going to harden this use, it is data invariant and
      // within our block.
      if (HardenedInstrs.count(&UseMI)) {
        if (!isDataInvariantLoad(UseMI)) {
          // If we've already decided to harden a non-load, we must have sunk
          // some other post-load hardened instruction to it and it must itself
          // be data-invariant.
          assert(isDataInvariant(UseMI) &&
                 "Data variant instruction being hardened!");
          continue;
        }

        // Otherwise, this is a load and the load component can't be data
        // invariant so check how this register is being used.
        const MCInstrDesc &Desc = UseMI.getDesc();
        int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);
        assert(MemRefBeginIdx >= 0 &&
               "Should always have mem references here!");
        MemRefBeginIdx += X86II::getOperandBias(Desc);

        MachineOperand &BaseMO =
            UseMI.getOperand(MemRefBeginIdx + X86::AddrBaseReg);
        MachineOperand &IndexMO =
            UseMI.getOperand(MemRefBeginIdx + X86::AddrIndexReg);
        if ((BaseMO.isReg() && BaseMO.getReg() == DefReg) ||
            (IndexMO.isReg() && IndexMO.getReg() == DefReg))
          // The load uses the register as part of its address making it not
          // invariant.
          return {};

        continue;
      }

      if (SingleUseMI)
        // We already have a single use, this would make two. Bail.
        return {};

      // If this single use isn't data invariant, isn't in this block, or has
      // interfering EFLAGS, we can't sink the hardening to it.
      if (!isDataInvariant(UseMI) || UseMI.getParent() != MI.getParent())
        return {};

      // If this instruction defines multiple registers bail as we won't harden
      // all of them.
      if (UseMI.getDesc().getNumDefs() > 1)
        return {};

      // If this register isn't a virtual register we can't walk uses of sanely,
      // just bail. Also check that its register class is one of the ones we
      // can harden.
      unsigned UseDefReg = UseMI.getOperand(0).getReg();
      if (!TRI->isVirtualRegister(UseDefReg) ||
          !canHardenRegister(UseDefReg))
        return {};

      SingleUseMI = &UseMI;
    }

    // If SingleUseMI is still null, there is no use that needs its own
    // checking. Otherwise, it is the single use that needs checking.
    return {SingleUseMI};
  };

  MachineInstr *MI = &InitialMI;
  while (Optional<MachineInstr *> SingleUse = SinkCheckToSingleUse(*MI)) {
    // Update which MI we're checking now.
    MI = *SingleUse;
    if (!MI)
      break;
  }

  return MI;
}

bool X86SpeculativeLoadHardeningPass::canHardenRegister(unsigned Reg) {
  auto *RC = MRI->getRegClass(Reg);
  int RegBytes = TRI->getRegSizeInBits(*RC) / 8;
  if (RegBytes > 8)
    // We don't support post-load hardening of vectors.
    return false;

  // If this register class is explicitly constrained to a class that doesn't
  // require REX prefix, we may not be able to satisfy that constraint when
  // emitting the hardening instructions, so bail out here.
  // FIXME: This seems like a pretty lame hack. The way this comes up is when we
  // end up both with a NOREX and REX-only register as operands to the hardening
  // instructions. It would be better to fix that code to handle this situation
  // rather than hack around it in this way.
  const TargetRegisterClass *NOREXRegClasses[] = {
      &X86::GR8_NOREXRegClass, &X86::GR16_NOREXRegClass,
      &X86::GR32_NOREXRegClass, &X86::GR64_NOREXRegClass};
  if (RC == NOREXRegClasses[Log2_32(RegBytes)])
    return false;

  const TargetRegisterClass *GPRRegClasses[] = {
      &X86::GR8RegClass, &X86::GR16RegClass, &X86::GR32RegClass,
      &X86::GR64RegClass};
  return RC->hasSuperClassEq(GPRRegClasses[Log2_32(RegBytes)]);
}

// We can harden non-leaking loads into register without touching the address
// by just hiding all of the loaded bits. We use an `or` instruction to do
// this because having the poison value be all ones allows us to use the same
// value below. And the goal is just for the loaded bits to not be exposed to
// execution and coercing them to one is sufficient.
void X86SpeculativeLoadHardeningPass::hardenPostLoad(
    MachineInstr &MI, MachineSSAUpdater &PredStateSSA) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc Loc = MI.getDebugLoc();

  // For all of these, the def'ed register operand is operand zero.
  auto &DefOp = MI.getOperand(0);
  unsigned OldDefReg = DefOp.getReg();
  assert(canHardenRegister(OldDefReg) &&
         "Cannot harden this instruction's defined register!");

  auto *DefRC = MRI->getRegClass(OldDefReg);
  int DefRegBytes = TRI->getRegSizeInBits(*DefRC) / 8;

  unsigned OrOpCodes[] = {X86::OR8rr, X86::OR16rr, X86::OR32rr, X86::OR64rr};
  unsigned OrOpCode = OrOpCodes[Log2_32(DefRegBytes)];

  unsigned SubRegImms[] = {X86::sub_8bit, X86::sub_16bit, X86::sub_32bit};

  auto GetStateRegInRC = [&](const TargetRegisterClass &RC) {
    unsigned StateReg = PredStateSSA.GetValueAtEndOfBlock(&MBB);

    int Bytes = TRI->getRegSizeInBits(RC) / 8;
    // FIXME: Need to teach this about 32-bit mode.
    if (Bytes != 8) {
      unsigned SubRegImm = SubRegImms[Log2_32(Bytes)];
      unsigned NarrowStateReg = MRI->createVirtualRegister(&RC);
      BuildMI(MBB, MI.getIterator(), Loc, TII->get(TargetOpcode::COPY),
              NarrowStateReg)
          .addReg(StateReg, 0, SubRegImm);
      StateReg = NarrowStateReg;
    }
    return StateReg;
  };

  auto InsertPt = std::next(MI.getIterator());
  unsigned FlagsReg = 0;
  bool EFLAGSLive = isEFLAGSLive(MBB, InsertPt, *TRI);
  if (EFLAGSLive && !Subtarget->hasBMI2()) {
    FlagsReg = saveEFLAGS(MBB, InsertPt, Loc);
    EFLAGSLive = false;
  }

  if (!EFLAGSLive) {
    unsigned StateReg = GetStateRegInRC(*DefRC);
    unsigned NewDefReg = MRI->createVirtualRegister(DefRC);
    DefOp.setReg(NewDefReg);
    auto OrI = BuildMI(MBB, InsertPt, Loc, TII->get(OrOpCode), OldDefReg)
                   .addReg(StateReg)
                   .addReg(NewDefReg);
    OrI->addRegisterDead(X86::EFLAGS, TRI);
    ++NumInstsInserted;
    LLVM_DEBUG(dbgs() << "  Inserting or: "; OrI->dump(); dbgs() << "\n");
  } else {
    assert(Subtarget->hasBMI2() &&
           "Cannot harden loads and preserve EFLAGS without BMI2!");

    unsigned ShiftOpCode = DefRegBytes < 4 ? X86::SHRX32rr : X86::SHRX64rr;
    auto &ShiftRC =
        DefRegBytes < 4 ? X86::GR32_NOSPRegClass : X86::GR64_NOSPRegClass;
    int ShiftRegBytes = TRI->getRegSizeInBits(ShiftRC) / 8;
    unsigned DefSubRegImm = SubRegImms[Log2_32(DefRegBytes)];

    unsigned StateReg = GetStateRegInRC(ShiftRC);

    // First have the def instruction def a temporary register.
    unsigned TmpReg = MRI->createVirtualRegister(DefRC);
    DefOp.setReg(TmpReg);
    // Now copy it into a register of the shift RC.
    unsigned ShiftInputReg = TmpReg;
    if (DefRegBytes != ShiftRegBytes) {
      unsigned UndefReg = MRI->createVirtualRegister(&ShiftRC);
      BuildMI(MBB, InsertPt, Loc, TII->get(X86::IMPLICIT_DEF), UndefReg);
      ShiftInputReg = MRI->createVirtualRegister(&ShiftRC);
      BuildMI(MBB, InsertPt, Loc, TII->get(X86::INSERT_SUBREG), ShiftInputReg)
          .addReg(UndefReg)
          .addReg(TmpReg)
          .addImm(DefSubRegImm);
    }

    // We shift this once if the shift is wider than the def and thus we can
    // shift *all* of the def'ed bytes out. Otherwise we need to do two shifts.

    unsigned ShiftedReg = MRI->createVirtualRegister(&ShiftRC);
    auto Shift1I =
        BuildMI(MBB, InsertPt, Loc, TII->get(ShiftOpCode), ShiftedReg)
            .addReg(ShiftInputReg)
            .addReg(StateReg);
    (void)Shift1I;
    ++NumInstsInserted;
    LLVM_DEBUG(dbgs() << "  Inserting shrx: "; Shift1I->dump(); dbgs() << "\n");

    // The only way we have a bit left is if all 8 bytes were defined. Do an
    // extra shift to get the last bit in this case.
    if (DefRegBytes == ShiftRegBytes) {
      // We can just directly def the old def register as its the same size.
      ShiftInputReg = ShiftedReg;
      auto Shift2I =
          BuildMI(MBB, InsertPt, Loc, TII->get(ShiftOpCode), OldDefReg)
              .addReg(ShiftInputReg)
              .addReg(StateReg);
      (void)Shift2I;
      ++NumInstsInserted;
      LLVM_DEBUG(dbgs() << "  Inserting shrx: "; Shift2I->dump();
                 dbgs() << "\n");
    } else {
      // When we have different size shift register we need to fix up the
      // class. We can do that as we copy into the old def register.
      BuildMI(MBB, InsertPt, Loc, TII->get(TargetOpcode::COPY), OldDefReg)
          .addReg(ShiftedReg, 0, DefSubRegImm);
    }
  }

  if (FlagsReg)
    restoreEFLAGS(MBB, InsertPt, Loc, FlagsReg);

  ++NumPostLoadRegsHardened;
}

void X86SpeculativeLoadHardeningPass::checkReturnInstr(
    MachineInstr &MI, MachineSSAUpdater &PredStateSSA) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc Loc = MI.getDebugLoc();
  auto InsertPt = MI.getIterator();

  if (FenceCallAndRet) {
    // Simply forcibly block speculation of loads out of the function by using
    // an LFENCE. This is potentially a heavy-weight mitigation strategy, but
    // should be secure, is simple from an ABI perspective, and the cost can be
    // minimized through inlining.
    //
    // FIXME: We should investigate ways to establish a strong data-dependency
    // on the return. However, poisoning the stack pointer is unlikely to work
    // because the return is *predicted* rather than relying on the load of the
    // return address to actually resolve.
    BuildMI(MBB, InsertPt, Loc, TII->get(X86::LFENCE));
    ++NumInstsInserted;
    ++NumLFENCEsInserted;
    return;
  }

  // Take our predicate state, shift it to the high 17 bits (so that we keep
  // pointers canonical) and merge it into RSP. This will allow the caller to
  // extract it when we return (speculatively).
  mergePredStateIntoSP(MBB, InsertPt, Loc,
                       PredStateSSA.GetValueAtEndOfBlock(&MBB));
}

INITIALIZE_PASS_BEGIN(X86SpeculativeLoadHardeningPass, DEBUG_TYPE,
                      "X86 speculative load hardener", false, false)
INITIALIZE_PASS_END(X86SpeculativeLoadHardeningPass, DEBUG_TYPE,
                    "X86 speculative load hardener", false, false)

FunctionPass *llvm::createX86SpeculativeLoadHardeningPass() {
  return new X86SpeculativeLoadHardeningPass();
}
