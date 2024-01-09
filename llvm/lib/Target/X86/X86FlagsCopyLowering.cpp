//====- X86FlagsCopyLowering.cpp - Lowers COPY nodes of EFLAGS ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Lowers COPY nodes of EFLAGS by directly extracting and preserving individual
/// flag bits.
///
/// We have to do this by carefully analyzing and rewriting the usage of the
/// copied EFLAGS register because there is no general way to rematerialize the
/// entire EFLAGS register safely and efficiently. Using `popf` both forces
/// dynamic stack adjustment and can create correctness issues due to IF, TF,
/// and other non-status flags being overwritten. Using sequences involving
/// SAHF don't work on all x86 processors and are often quite slow compared to
/// directly testing a single status preserved in its own GPR.
///
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineDominators.h"
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

#define PASS_KEY "x86-flags-copy-lowering"
#define DEBUG_TYPE PASS_KEY

STATISTIC(NumCopiesEliminated, "Number of copies of EFLAGS eliminated");
STATISTIC(NumSetCCsInserted, "Number of setCC instructions inserted");
STATISTIC(NumTestsInserted, "Number of test instructions inserted");
STATISTIC(NumAddsInserted, "Number of adds instructions inserted");

namespace {

// Convenient array type for storing registers associated with each condition.
using CondRegArray = std::array<unsigned, X86::LAST_VALID_COND + 1>;

class X86FlagsCopyLoweringPass : public MachineFunctionPass {
public:
  X86FlagsCopyLoweringPass() : MachineFunctionPass(ID) { }

  StringRef getPassName() const override { return "X86 EFLAGS copy lowering"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Pass identification, replacement for typeid.
  static char ID;

private:
  MachineRegisterInfo *MRI = nullptr;
  const X86Subtarget *Subtarget = nullptr;
  const X86InstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const TargetRegisterClass *PromoteRC = nullptr;
  MachineDominatorTree *MDT = nullptr;

  CondRegArray collectCondsInRegs(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator CopyDefI);

  Register promoteCondToReg(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator TestPos,
                            const DebugLoc &TestLoc, X86::CondCode Cond);
  std::pair<unsigned, bool> getCondOrInverseInReg(
      MachineBasicBlock &TestMBB, MachineBasicBlock::iterator TestPos,
      const DebugLoc &TestLoc, X86::CondCode Cond, CondRegArray &CondRegs);
  void insertTest(MachineBasicBlock &MBB, MachineBasicBlock::iterator Pos,
                  const DebugLoc &Loc, unsigned Reg);

  void rewriteArithmetic(MachineBasicBlock &TestMBB,
                         MachineBasicBlock::iterator TestPos,
                         const DebugLoc &TestLoc, MachineInstr &MI,
                         MachineOperand &FlagUse, CondRegArray &CondRegs);
  void rewriteCMov(MachineBasicBlock &TestMBB,
                   MachineBasicBlock::iterator TestPos, const DebugLoc &TestLoc,
                   MachineInstr &CMovI, MachineOperand &FlagUse,
                   CondRegArray &CondRegs);
  void rewriteFCMov(MachineBasicBlock &TestMBB,
                    MachineBasicBlock::iterator TestPos,
                    const DebugLoc &TestLoc, MachineInstr &CMovI,
                    MachineOperand &FlagUse, CondRegArray &CondRegs);
  void rewriteCondJmp(MachineBasicBlock &TestMBB,
                      MachineBasicBlock::iterator TestPos,
                      const DebugLoc &TestLoc, MachineInstr &JmpI,
                      CondRegArray &CondRegs);
  void rewriteCopy(MachineInstr &MI, MachineOperand &FlagUse,
                   MachineInstr &CopyDefI);
  void rewriteSetCC(MachineBasicBlock &TestMBB,
                    MachineBasicBlock::iterator TestPos,
                    const DebugLoc &TestLoc, MachineInstr &SetCCI,
                    MachineOperand &FlagUse, CondRegArray &CondRegs);
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(X86FlagsCopyLoweringPass, DEBUG_TYPE,
                      "X86 EFLAGS copy lowering", false, false)
INITIALIZE_PASS_END(X86FlagsCopyLoweringPass, DEBUG_TYPE,
                    "X86 EFLAGS copy lowering", false, false)

FunctionPass *llvm::createX86FlagsCopyLoweringPass() {
  return new X86FlagsCopyLoweringPass();
}

char X86FlagsCopyLoweringPass::ID = 0;

void X86FlagsCopyLoweringPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineDominatorTree>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

namespace {
/// An enumeration of the arithmetic instruction mnemonics which have
/// interesting flag semantics.
///
/// We can map instruction opcodes into these mnemonics to make it easy to
/// dispatch with specific functionality.
enum class FlagArithMnemonic {
  ADC,
  RCL,
  RCR,
  SBB,
  SETB,
};
} // namespace

static FlagArithMnemonic getMnemonicFromOpcode(unsigned Opcode) {
  switch (Opcode) {
  default:
    report_fatal_error("No support for lowering a copy into EFLAGS when used "
                       "by this instruction!");

#define LLVM_EXPAND_INSTR_SIZES(MNEMONIC, SUFFIX)                              \
  case X86::MNEMONIC##8##SUFFIX:                                               \
  case X86::MNEMONIC##16##SUFFIX:                                              \
  case X86::MNEMONIC##32##SUFFIX:                                              \
  case X86::MNEMONIC##64##SUFFIX:

#define LLVM_EXPAND_ADC_SBB_INSTR(MNEMONIC)                                    \
  LLVM_EXPAND_INSTR_SIZES(MNEMONIC, rr)                                        \
  LLVM_EXPAND_INSTR_SIZES(MNEMONIC, rm)                                        \
  LLVM_EXPAND_INSTR_SIZES(MNEMONIC, mr)                                        \
  case X86::MNEMONIC##8ri:                                                     \
  case X86::MNEMONIC##16ri8:                                                   \
  case X86::MNEMONIC##32ri8:                                                   \
  case X86::MNEMONIC##64ri8:                                                   \
  case X86::MNEMONIC##16ri:                                                    \
  case X86::MNEMONIC##32ri:                                                    \
  case X86::MNEMONIC##64ri32:                                                  \
  case X86::MNEMONIC##8mi:                                                     \
  case X86::MNEMONIC##16mi8:                                                   \
  case X86::MNEMONIC##32mi8:                                                   \
  case X86::MNEMONIC##64mi8:                                                   \
  case X86::MNEMONIC##16mi:                                                    \
  case X86::MNEMONIC##32mi:                                                    \
  case X86::MNEMONIC##64mi32:                                                  \
  case X86::MNEMONIC##8i8:                                                     \
  case X86::MNEMONIC##16i16:                                                   \
  case X86::MNEMONIC##32i32:                                                   \
  case X86::MNEMONIC##64i32:

    LLVM_EXPAND_ADC_SBB_INSTR(ADC)
    return FlagArithMnemonic::ADC;

    LLVM_EXPAND_ADC_SBB_INSTR(SBB)
    return FlagArithMnemonic::SBB;

#undef LLVM_EXPAND_ADC_SBB_INSTR

    LLVM_EXPAND_INSTR_SIZES(RCL, rCL)
    LLVM_EXPAND_INSTR_SIZES(RCL, r1)
    LLVM_EXPAND_INSTR_SIZES(RCL, ri)
    return FlagArithMnemonic::RCL;

    LLVM_EXPAND_INSTR_SIZES(RCR, rCL)
    LLVM_EXPAND_INSTR_SIZES(RCR, r1)
    LLVM_EXPAND_INSTR_SIZES(RCR, ri)
    return FlagArithMnemonic::RCR;

#undef LLVM_EXPAND_INSTR_SIZES

  case X86::SETB_C32r:
  case X86::SETB_C64r:
    return FlagArithMnemonic::SETB;
  }
}

static MachineBasicBlock &splitBlock(MachineBasicBlock &MBB,
                                     MachineInstr &SplitI,
                                     const X86InstrInfo &TII) {
  MachineFunction &MF = *MBB.getParent();

  assert(SplitI.getParent() == &MBB &&
         "Split instruction must be in the split block!");
  assert(SplitI.isBranch() &&
         "Only designed to split a tail of branch instructions!");
  assert(X86::getCondFromBranch(SplitI) != X86::COND_INVALID &&
         "Must split on an actual jCC instruction!");

  // Dig out the previous instruction to the split point.
  MachineInstr &PrevI = *std::prev(SplitI.getIterator());
  assert(PrevI.isBranch() && "Must split after a branch!");
  assert(X86::getCondFromBranch(PrevI) != X86::COND_INVALID &&
         "Must split after an actual jCC instruction!");
  assert(!std::prev(PrevI.getIterator())->isTerminator() &&
         "Must only have this one terminator prior to the split!");

  // Grab the one successor edge that will stay in `MBB`.
  MachineBasicBlock &UnsplitSucc = *PrevI.getOperand(0).getMBB();

  // Analyze the original block to see if we are actually splitting an edge
  // into two edges. This can happen when we have multiple conditional jumps to
  // the same successor.
  bool IsEdgeSplit =
      std::any_of(SplitI.getIterator(), MBB.instr_end(),
                  [&](MachineInstr &MI) {
                    assert(MI.isTerminator() &&
                           "Should only have spliced terminators!");
                    return llvm::any_of(
                        MI.operands(), [&](MachineOperand &MOp) {
                          return MOp.isMBB() && MOp.getMBB() == &UnsplitSucc;
                        });
                  }) ||
      MBB.getFallThrough() == &UnsplitSucc;

  MachineBasicBlock &NewMBB = *MF.CreateMachineBasicBlock();

  // Insert the new block immediately after the current one. Any existing
  // fallthrough will be sunk into this new block anyways.
  MF.insert(std::next(MachineFunction::iterator(&MBB)), &NewMBB);

  // Splice the tail of instructions into the new block.
  NewMBB.splice(NewMBB.end(), &MBB, SplitI.getIterator(), MBB.end());

  // Copy the necessary succesors (and their probability info) into the new
  // block.
  for (auto SI = MBB.succ_begin(), SE = MBB.succ_end(); SI != SE; ++SI)
    if (IsEdgeSplit || *SI != &UnsplitSucc)
      NewMBB.copySuccessor(&MBB, SI);
  // Normalize the probabilities if we didn't end up splitting the edge.
  if (!IsEdgeSplit)
    NewMBB.normalizeSuccProbs();

  // Now replace all of the moved successors in the original block with the new
  // block. This will merge their probabilities.
  for (MachineBasicBlock *Succ : NewMBB.successors())
    if (Succ != &UnsplitSucc)
      MBB.replaceSuccessor(Succ, &NewMBB);

  // We should always end up replacing at least one successor.
  assert(MBB.isSuccessor(&NewMBB) &&
         "Failed to make the new block a successor!");

  // Now update all the PHIs.
  for (MachineBasicBlock *Succ : NewMBB.successors()) {
    for (MachineInstr &MI : *Succ) {
      if (!MI.isPHI())
        break;

      for (int OpIdx = 1, NumOps = MI.getNumOperands(); OpIdx < NumOps;
           OpIdx += 2) {
        MachineOperand &OpV = MI.getOperand(OpIdx);
        MachineOperand &OpMBB = MI.getOperand(OpIdx + 1);
        assert(OpMBB.isMBB() && "Block operand to a PHI is not a block!");
        if (OpMBB.getMBB() != &MBB)
          continue;

        // Replace the operand for unsplit successors
        if (!IsEdgeSplit || Succ != &UnsplitSucc) {
          OpMBB.setMBB(&NewMBB);

          // We have to continue scanning as there may be multiple entries in
          // the PHI.
          continue;
        }

        // When we have split the edge append a new successor.
        MI.addOperand(MF, OpV);
        MI.addOperand(MF, MachineOperand::CreateMBB(&NewMBB));
        break;
      }
    }
  }

  return NewMBB;
}

static X86::CondCode getCondFromFCMOV(unsigned Opcode) {
  switch (Opcode) {
  default: return X86::COND_INVALID;
  case X86::CMOVBE_Fp32:  case X86::CMOVBE_Fp64:  case X86::CMOVBE_Fp80:
    return X86::COND_BE;
  case X86::CMOVB_Fp32:   case X86::CMOVB_Fp64:   case X86::CMOVB_Fp80:
    return X86::COND_B;
  case X86::CMOVE_Fp32:   case X86::CMOVE_Fp64:   case X86::CMOVE_Fp80:
    return X86::COND_E;
  case X86::CMOVNBE_Fp32: case X86::CMOVNBE_Fp64: case X86::CMOVNBE_Fp80:
    return X86::COND_A;
  case X86::CMOVNB_Fp32:  case X86::CMOVNB_Fp64:  case X86::CMOVNB_Fp80:
    return X86::COND_AE;
  case X86::CMOVNE_Fp32:  case X86::CMOVNE_Fp64:  case X86::CMOVNE_Fp80:
    return X86::COND_NE;
  case X86::CMOVNP_Fp32:  case X86::CMOVNP_Fp64:  case X86::CMOVNP_Fp80:
    return X86::COND_NP;
  case X86::CMOVP_Fp32:   case X86::CMOVP_Fp64:   case X86::CMOVP_Fp80:
    return X86::COND_P;
  }
}

bool X86FlagsCopyLoweringPass::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** " << getPassName() << " : " << MF.getName()
                    << " **********\n");

  Subtarget = &MF.getSubtarget<X86Subtarget>();
  MRI = &MF.getRegInfo();
  TII = Subtarget->getInstrInfo();
  TRI = Subtarget->getRegisterInfo();
  MDT = &getAnalysis<MachineDominatorTree>();
  PromoteRC = &X86::GR8RegClass;

  if (MF.begin() == MF.end())
    // Nothing to do for a degenerate empty function...
    return false;

  // Collect the copies in RPO so that when there are chains where a copy is in
  // turn copied again we visit the first one first. This ensures we can find
  // viable locations for testing the original EFLAGS that dominate all the
  // uses across complex CFGs.
  SmallVector<MachineInstr *, 4> Copies;
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (MachineBasicBlock *MBB : RPOT)
    for (MachineInstr &MI : *MBB)
      if (MI.getOpcode() == TargetOpcode::COPY &&
          MI.getOperand(0).getReg() == X86::EFLAGS)
        Copies.push_back(&MI);

  for (MachineInstr *CopyI : Copies) {
    MachineBasicBlock &MBB = *CopyI->getParent();

    MachineOperand &VOp = CopyI->getOperand(1);
    assert(VOp.isReg() &&
           "The input to the copy for EFLAGS should always be a register!");
    MachineInstr &CopyDefI = *MRI->getVRegDef(VOp.getReg());
    if (CopyDefI.getOpcode() != TargetOpcode::COPY) {
      // FIXME: The big likely candidate here are PHI nodes. We could in theory
      // handle PHI nodes, but it gets really, really hard. Insanely hard. Hard
      // enough that it is probably better to change every other part of LLVM
      // to avoid creating them. The issue is that once we have PHIs we won't
      // know which original EFLAGS value we need to capture with our setCCs
      // below. The end result will be computing a complete set of setCCs that
      // we *might* want, computing them in every place where we copy *out* of
      // EFLAGS and then doing SSA formation on all of them to insert necessary
      // PHI nodes and consume those here. Then hoping that somehow we DCE the
      // unnecessary ones. This DCE seems very unlikely to be successful and so
      // we will almost certainly end up with a glut of dead setCC
      // instructions. Until we have a motivating test case and fail to avoid
      // it by changing other parts of LLVM's lowering, we refuse to handle
      // this complex case here.
      LLVM_DEBUG(
          dbgs() << "ERROR: Encountered unexpected def of an eflags copy: ";
          CopyDefI.dump());
      report_fatal_error(
          "Cannot lower EFLAGS copy unless it is defined in turn by a copy!");
    }

    auto Cleanup = make_scope_exit([&] {
      // All uses of the EFLAGS copy are now rewritten, kill the copy into
      // eflags and if dead the copy from.
      CopyI->eraseFromParent();
      if (MRI->use_empty(CopyDefI.getOperand(0).getReg()))
        CopyDefI.eraseFromParent();
      ++NumCopiesEliminated;
    });

    MachineOperand &DOp = CopyI->getOperand(0);
    assert(DOp.isDef() && "Expected register def!");
    assert(DOp.getReg() == X86::EFLAGS && "Unexpected copy def register!");
    if (DOp.isDead())
      continue;

    MachineBasicBlock *TestMBB = CopyDefI.getParent();
    auto TestPos = CopyDefI.getIterator();
    DebugLoc TestLoc = CopyDefI.getDebugLoc();

    LLVM_DEBUG(dbgs() << "Rewriting copy: "; CopyI->dump());

    // Walk up across live-in EFLAGS to find where they were actually def'ed.
    //
    // This copy's def may just be part of a region of blocks covered by
    // a single def of EFLAGS and we want to find the top of that region where
    // possible.
    //
    // This is essentially a search for a *candidate* reaching definition
    // location. We don't need to ever find the actual reaching definition here,
    // but we want to walk up the dominator tree to find the highest point which
    // would be viable for such a definition.
    auto HasEFLAGSClobber = [&](MachineBasicBlock::iterator Begin,
                                MachineBasicBlock::iterator End) {
      // Scan backwards as we expect these to be relatively short and often find
      // a clobber near the end.
      return llvm::any_of(
          llvm::reverse(llvm::make_range(Begin, End)), [&](MachineInstr &MI) {
            // Flag any instruction (other than the copy we are
            // currently rewriting) that defs EFLAGS.
            return &MI != CopyI && MI.findRegisterDefOperand(X86::EFLAGS);
          });
    };
    auto HasEFLAGSClobberPath = [&](MachineBasicBlock *BeginMBB,
                                    MachineBasicBlock *EndMBB) {
      assert(MDT->dominates(BeginMBB, EndMBB) &&
             "Only support paths down the dominator tree!");
      SmallPtrSet<MachineBasicBlock *, 4> Visited;
      SmallVector<MachineBasicBlock *, 4> Worklist;
      // We terminate at the beginning. No need to scan it.
      Visited.insert(BeginMBB);
      Worklist.push_back(EndMBB);
      do {
        auto *MBB = Worklist.pop_back_val();
        for (auto *PredMBB : MBB->predecessors()) {
          if (!Visited.insert(PredMBB).second)
            continue;
          if (HasEFLAGSClobber(PredMBB->begin(), PredMBB->end()))
            return true;
          // Enqueue this block to walk its predecessors.
          Worklist.push_back(PredMBB);
        }
      } while (!Worklist.empty());
      // No clobber found along a path from the begin to end.
      return false;
    };
    while (TestMBB->isLiveIn(X86::EFLAGS) && !TestMBB->pred_empty() &&
           !HasEFLAGSClobber(TestMBB->begin(), TestPos)) {
      // Find the nearest common dominator of the predecessors, as
      // that will be the best candidate to hoist into.
      MachineBasicBlock *HoistMBB =
          std::accumulate(std::next(TestMBB->pred_begin()), TestMBB->pred_end(),
                          *TestMBB->pred_begin(),
                          [&](MachineBasicBlock *LHS, MachineBasicBlock *RHS) {
                            return MDT->findNearestCommonDominator(LHS, RHS);
                          });

      // Now we need to scan all predecessors that may be reached along paths to
      // the hoist block. A clobber anywhere in any of these blocks the hoist.
      // Note that this even handles loops because we require *no* clobbers.
      if (HasEFLAGSClobberPath(HoistMBB, TestMBB))
        break;

      // We also need the terminators to not sneakily clobber flags.
      if (HasEFLAGSClobber(HoistMBB->getFirstTerminator()->getIterator(),
                           HoistMBB->instr_end()))
        break;

      // We found a viable location, hoist our test position to it.
      TestMBB = HoistMBB;
      TestPos = TestMBB->getFirstTerminator()->getIterator();
      // Clear the debug location as it would just be confusing after hoisting.
      TestLoc = DebugLoc();
    }
    LLVM_DEBUG({
      auto DefIt = llvm::find_if(
          llvm::reverse(llvm::make_range(TestMBB->instr_begin(), TestPos)),
          [&](MachineInstr &MI) {
            return MI.findRegisterDefOperand(X86::EFLAGS);
          });
      if (DefIt.base() != TestMBB->instr_begin()) {
        dbgs() << "  Using EFLAGS defined by: ";
        DefIt->dump();
      } else {
        dbgs() << "  Using live-in flags for BB:\n";
        TestMBB->dump();
      }
    });

    // While rewriting uses, we buffer jumps and rewrite them in a second pass
    // because doing so will perturb the CFG that we are walking to find the
    // uses in the first place.
    SmallVector<MachineInstr *, 4> JmpIs;

    // Gather the condition flags that have already been preserved in
    // registers. We do this from scratch each time as we expect there to be
    // very few of them and we expect to not revisit the same copy definition
    // many times. If either of those change sufficiently we could build a map
    // of these up front instead.
    CondRegArray CondRegs = collectCondsInRegs(*TestMBB, TestPos);

    // Collect the basic blocks we need to scan. Typically this will just be
    // a single basic block but we may have to scan multiple blocks if the
    // EFLAGS copy lives into successors.
    SmallVector<MachineBasicBlock *, 2> Blocks;
    SmallPtrSet<MachineBasicBlock *, 2> VisitedBlocks;
    Blocks.push_back(&MBB);

    do {
      MachineBasicBlock &UseMBB = *Blocks.pop_back_val();

      // Track when if/when we find a kill of the flags in this block.
      bool FlagsKilled = false;

      // In most cases, we walk from the beginning to the end of the block. But
      // when the block is the same block as the copy is from, we will visit it
      // twice. The first time we start from the copy and go to the end. The
      // second time we start from the beginning and go to the copy. This lets
      // us handle copies inside of cycles.
      // FIXME: This loop is *super* confusing. This is at least in part
      // a symptom of all of this routine needing to be refactored into
      // documentable components. Once done, there may be a better way to write
      // this loop.
      for (auto MII = (&UseMBB == &MBB && !VisitedBlocks.count(&UseMBB))
                          ? std::next(CopyI->getIterator())
                          : UseMBB.instr_begin(),
                MIE = UseMBB.instr_end();
           MII != MIE;) {
        MachineInstr &MI = *MII++;
        // If we are in the original copy block and encounter either the copy
        // def or the copy itself, break so that we don't re-process any part of
        // the block or process the instructions in the range that was copied
        // over.
        if (&MI == CopyI || &MI == &CopyDefI) {
          assert(&UseMBB == &MBB && VisitedBlocks.count(&MBB) &&
                 "Should only encounter these on the second pass over the "
                 "original block.");
          break;
        }

        MachineOperand *FlagUse = MI.findRegisterUseOperand(X86::EFLAGS);
        if (!FlagUse) {
          if (MI.findRegisterDefOperand(X86::EFLAGS)) {
            // If EFLAGS are defined, it's as-if they were killed. We can stop
            // scanning here.
            //
            // NB!!! Many instructions only modify some flags. LLVM currently
            // models this as clobbering all flags, but if that ever changes
            // this will need to be carefully updated to handle that more
            // complex logic.
            FlagsKilled = true;
            break;
          }
          continue;
        }

        LLVM_DEBUG(dbgs() << "  Rewriting use: "; MI.dump());

        // Check the kill flag before we rewrite as that may change it.
        if (FlagUse->isKill())
          FlagsKilled = true;

        // Once we encounter a branch, the rest of the instructions must also be
        // branches. We can't rewrite in place here, so we handle them below.
        //
        // Note that we don't have to handle tail calls here, even conditional
        // tail calls, as those are not introduced into the X86 MI until post-RA
        // branch folding or black placement. As a consequence, we get to deal
        // with the simpler formulation of conditional branches followed by tail
        // calls.
        if (X86::getCondFromBranch(MI) != X86::COND_INVALID) {
          auto JmpIt = MI.getIterator();
          do {
            JmpIs.push_back(&*JmpIt);
            ++JmpIt;
          } while (JmpIt != UseMBB.instr_end() &&
                   X86::getCondFromBranch(*JmpIt) !=
                       X86::COND_INVALID);
          break;
        }

        // Otherwise we can just rewrite in-place.
        if (X86::getCondFromCMov(MI) != X86::COND_INVALID) {
          rewriteCMov(*TestMBB, TestPos, TestLoc, MI, *FlagUse, CondRegs);
        } else if (getCondFromFCMOV(MI.getOpcode()) != X86::COND_INVALID) {
          rewriteFCMov(*TestMBB, TestPos, TestLoc, MI, *FlagUse, CondRegs);
        } else if (X86::getCondFromSETCC(MI) != X86::COND_INVALID) {
          rewriteSetCC(*TestMBB, TestPos, TestLoc, MI, *FlagUse, CondRegs);
        } else if (MI.getOpcode() == TargetOpcode::COPY) {
          rewriteCopy(MI, *FlagUse, CopyDefI);
        } else {
          // We assume all other instructions that use flags also def them.
          assert(MI.findRegisterDefOperand(X86::EFLAGS) &&
                 "Expected a def of EFLAGS for this instruction!");

          // NB!!! Several arithmetic instructions only *partially* update
          // flags. Theoretically, we could generate MI code sequences that
          // would rely on this fact and observe different flags independently.
          // But currently LLVM models all of these instructions as clobbering
          // all the flags in an undef way. We rely on that to simplify the
          // logic.
          FlagsKilled = true;

          // Generically handle remaining uses as arithmetic instructions.
          rewriteArithmetic(*TestMBB, TestPos, TestLoc, MI, *FlagUse,
                            CondRegs);
        }

        // If this was the last use of the flags, we're done.
        if (FlagsKilled)
          break;
      }

      // If the flags were killed, we're done with this block.
      if (FlagsKilled)
        continue;

      // Otherwise we need to scan successors for ones where the flags live-in
      // and queue those up for processing.
      for (MachineBasicBlock *SuccMBB : UseMBB.successors())
        if (SuccMBB->isLiveIn(X86::EFLAGS) &&
            VisitedBlocks.insert(SuccMBB).second) {
          // We currently don't do any PHI insertion and so we require that the
          // test basic block dominates all of the use basic blocks. Further, we
          // can't have a cycle from the test block back to itself as that would
          // create a cycle requiring a PHI to break it.
          //
          // We could in theory do PHI insertion here if it becomes useful by
          // just taking undef values in along every edge that we don't trace
          // this EFLAGS copy along. This isn't as bad as fully general PHI
          // insertion, but still seems like a great deal of complexity.
          //
          // Because it is theoretically possible that some earlier MI pass or
          // other lowering transformation could induce this to happen, we do
          // a hard check even in non-debug builds here.
          if (SuccMBB == TestMBB || !MDT->dominates(TestMBB, SuccMBB)) {
            LLVM_DEBUG({
              dbgs()
                  << "ERROR: Encountered use that is not dominated by our test "
                     "basic block! Rewriting this would require inserting PHI "
                     "nodes to track the flag state across the CFG.\n\nTest "
                     "block:\n";
              TestMBB->dump();
              dbgs() << "Use block:\n";
              SuccMBB->dump();
            });
            report_fatal_error(
                "Cannot lower EFLAGS copy when original copy def "
                "does not dominate all uses.");
          }

          Blocks.push_back(SuccMBB);

          // After this, EFLAGS will be recreated before each use.
          SuccMBB->removeLiveIn(X86::EFLAGS);
        }
    } while (!Blocks.empty());

    // Now rewrite the jumps that use the flags. These we handle specially
    // because if there are multiple jumps in a single basic block we'll have
    // to do surgery on the CFG.
    MachineBasicBlock *LastJmpMBB = nullptr;
    for (MachineInstr *JmpI : JmpIs) {
      // Past the first jump within a basic block we need to split the blocks
      // apart.
      if (JmpI->getParent() == LastJmpMBB)
        splitBlock(*JmpI->getParent(), *JmpI, *TII);
      else
        LastJmpMBB = JmpI->getParent();

      rewriteCondJmp(*TestMBB, TestPos, TestLoc, *JmpI, CondRegs);
    }

    // FIXME: Mark the last use of EFLAGS before the copy's def as a kill if
    // the copy's def operand is itself a kill.
  }

#ifndef NDEBUG
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.getOpcode() == TargetOpcode::COPY &&
          (MI.getOperand(0).getReg() == X86::EFLAGS ||
           MI.getOperand(1).getReg() == X86::EFLAGS)) {
        LLVM_DEBUG(dbgs() << "ERROR: Found a COPY involving EFLAGS: ";
                   MI.dump());
        llvm_unreachable("Unlowered EFLAGS copy!");
      }
#endif

  return true;
}

/// Collect any conditions that have already been set in registers so that we
/// can re-use them rather than adding duplicates.
CondRegArray X86FlagsCopyLoweringPass::collectCondsInRegs(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator TestPos) {
  CondRegArray CondRegs = {};

  // Scan backwards across the range of instructions with live EFLAGS.
  for (MachineInstr &MI :
       llvm::reverse(llvm::make_range(MBB.begin(), TestPos))) {
    X86::CondCode Cond = X86::getCondFromSETCC(MI);
    if (Cond != X86::COND_INVALID && !MI.mayStore() &&
        MI.getOperand(0).isReg() && MI.getOperand(0).getReg().isVirtual()) {
      assert(MI.getOperand(0).isDef() &&
             "A non-storing SETcc should always define a register!");
      CondRegs[Cond] = MI.getOperand(0).getReg();
    }

    // Stop scanning when we see the first definition of the EFLAGS as prior to
    // this we would potentially capture the wrong flag state.
    if (MI.findRegisterDefOperand(X86::EFLAGS))
      break;
  }
  return CondRegs;
}

Register X86FlagsCopyLoweringPass::promoteCondToReg(
    MachineBasicBlock &TestMBB, MachineBasicBlock::iterator TestPos,
    const DebugLoc &TestLoc, X86::CondCode Cond) {
  Register Reg = MRI->createVirtualRegister(PromoteRC);
  auto SetI = BuildMI(TestMBB, TestPos, TestLoc,
                      TII->get(X86::SETCCr), Reg).addImm(Cond);
  (void)SetI;
  LLVM_DEBUG(dbgs() << "    save cond: "; SetI->dump());
  ++NumSetCCsInserted;
  return Reg;
}

std::pair<unsigned, bool> X86FlagsCopyLoweringPass::getCondOrInverseInReg(
    MachineBasicBlock &TestMBB, MachineBasicBlock::iterator TestPos,
    const DebugLoc &TestLoc, X86::CondCode Cond, CondRegArray &CondRegs) {
  unsigned &CondReg = CondRegs[Cond];
  unsigned &InvCondReg = CondRegs[X86::GetOppositeBranchCondition(Cond)];
  if (!CondReg && !InvCondReg)
    CondReg = promoteCondToReg(TestMBB, TestPos, TestLoc, Cond);

  if (CondReg)
    return {CondReg, false};
  else
    return {InvCondReg, true};
}

void X86FlagsCopyLoweringPass::insertTest(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator Pos,
                                          const DebugLoc &Loc, unsigned Reg) {
  auto TestI =
      BuildMI(MBB, Pos, Loc, TII->get(X86::TEST8rr)).addReg(Reg).addReg(Reg);
  (void)TestI;
  LLVM_DEBUG(dbgs() << "    test cond: "; TestI->dump());
  ++NumTestsInserted;
}

void X86FlagsCopyLoweringPass::rewriteArithmetic(
    MachineBasicBlock &TestMBB, MachineBasicBlock::iterator TestPos,
    const DebugLoc &TestLoc, MachineInstr &MI, MachineOperand &FlagUse,
    CondRegArray &CondRegs) {
  // Arithmetic is either reading CF or OF. Figure out which condition we need
  // to preserve in a register.
  X86::CondCode Cond = X86::COND_INVALID;

  // The addend to use to reset CF or OF when added to the flag value.
  int Addend = 0;

  switch (getMnemonicFromOpcode(MI.getOpcode())) {
  case FlagArithMnemonic::ADC:
  case FlagArithMnemonic::RCL:
  case FlagArithMnemonic::RCR:
  case FlagArithMnemonic::SBB:
  case FlagArithMnemonic::SETB:
    Cond = X86::COND_B; // CF == 1
    // Set up an addend that when one is added will need a carry due to not
    // having a higher bit available.
    Addend = 255;
    break;
  }

  // Now get a register that contains the value of the flag input to the
  // arithmetic. We require exactly this flag to simplify the arithmetic
  // required to materialize it back into the flag.
  unsigned &CondReg = CondRegs[Cond];
  if (!CondReg)
    CondReg = promoteCondToReg(TestMBB, TestPos, TestLoc, Cond);

  MachineBasicBlock &MBB = *MI.getParent();

  // Insert an instruction that will set the flag back to the desired value.
  Register TmpReg = MRI->createVirtualRegister(PromoteRC);
  auto AddI =
      BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(), TII->get(X86::ADD8ri))
          .addDef(TmpReg, RegState::Dead)
          .addReg(CondReg)
          .addImm(Addend);
  (void)AddI;
  LLVM_DEBUG(dbgs() << "    add cond: "; AddI->dump());
  ++NumAddsInserted;
  FlagUse.setIsKill(true);
}

void X86FlagsCopyLoweringPass::rewriteCMov(MachineBasicBlock &TestMBB,
                                           MachineBasicBlock::iterator TestPos,
                                           const DebugLoc &TestLoc,
                                           MachineInstr &CMovI,
                                           MachineOperand &FlagUse,
                                           CondRegArray &CondRegs) {
  // First get the register containing this specific condition.
  X86::CondCode Cond = X86::getCondFromCMov(CMovI);
  unsigned CondReg;
  bool Inverted;
  std::tie(CondReg, Inverted) =
      getCondOrInverseInReg(TestMBB, TestPos, TestLoc, Cond, CondRegs);

  MachineBasicBlock &MBB = *CMovI.getParent();

  // Insert a direct test of the saved register.
  insertTest(MBB, CMovI.getIterator(), CMovI.getDebugLoc(), CondReg);

  // Rewrite the CMov to use the !ZF flag from the test, and then kill its use
  // of the flags afterward.
  CMovI.getOperand(CMovI.getDesc().getNumOperands() - 1)
      .setImm(Inverted ? X86::COND_E : X86::COND_NE);
  FlagUse.setIsKill(true);
  LLVM_DEBUG(dbgs() << "    fixed cmov: "; CMovI.dump());
}

void X86FlagsCopyLoweringPass::rewriteFCMov(MachineBasicBlock &TestMBB,
                                            MachineBasicBlock::iterator TestPos,
                                            const DebugLoc &TestLoc,
                                            MachineInstr &CMovI,
                                            MachineOperand &FlagUse,
                                            CondRegArray &CondRegs) {
  // First get the register containing this specific condition.
  X86::CondCode Cond = getCondFromFCMOV(CMovI.getOpcode());
  unsigned CondReg;
  bool Inverted;
  std::tie(CondReg, Inverted) =
      getCondOrInverseInReg(TestMBB, TestPos, TestLoc, Cond, CondRegs);

  MachineBasicBlock &MBB = *CMovI.getParent();

  // Insert a direct test of the saved register.
  insertTest(MBB, CMovI.getIterator(), CMovI.getDebugLoc(), CondReg);

  auto getFCMOVOpcode = [](unsigned Opcode, bool Inverted) {
    switch (Opcode) {
    default: llvm_unreachable("Unexpected opcode!");
    case X86::CMOVBE_Fp32: case X86::CMOVNBE_Fp32:
    case X86::CMOVB_Fp32:  case X86::CMOVNB_Fp32:
    case X86::CMOVE_Fp32:  case X86::CMOVNE_Fp32:
    case X86::CMOVP_Fp32:  case X86::CMOVNP_Fp32:
      return Inverted ? X86::CMOVE_Fp32 : X86::CMOVNE_Fp32;
    case X86::CMOVBE_Fp64: case X86::CMOVNBE_Fp64:
    case X86::CMOVB_Fp64:  case X86::CMOVNB_Fp64:
    case X86::CMOVE_Fp64:  case X86::CMOVNE_Fp64:
    case X86::CMOVP_Fp64:  case X86::CMOVNP_Fp64:
      return Inverted ? X86::CMOVE_Fp64 : X86::CMOVNE_Fp64;
    case X86::CMOVBE_Fp80: case X86::CMOVNBE_Fp80:
    case X86::CMOVB_Fp80:  case X86::CMOVNB_Fp80:
    case X86::CMOVE_Fp80:  case X86::CMOVNE_Fp80:
    case X86::CMOVP_Fp80:  case X86::CMOVNP_Fp80:
      return Inverted ? X86::CMOVE_Fp80 : X86::CMOVNE_Fp80;
    }
  };

  // Rewrite the CMov to use the !ZF flag from the test.
  CMovI.setDesc(TII->get(getFCMOVOpcode(CMovI.getOpcode(), Inverted)));
  FlagUse.setIsKill(true);
  LLVM_DEBUG(dbgs() << "    fixed fcmov: "; CMovI.dump());
}

void X86FlagsCopyLoweringPass::rewriteCondJmp(
    MachineBasicBlock &TestMBB, MachineBasicBlock::iterator TestPos,
    const DebugLoc &TestLoc, MachineInstr &JmpI, CondRegArray &CondRegs) {
  // First get the register containing this specific condition.
  X86::CondCode Cond = X86::getCondFromBranch(JmpI);
  unsigned CondReg;
  bool Inverted;
  std::tie(CondReg, Inverted) =
      getCondOrInverseInReg(TestMBB, TestPos, TestLoc, Cond, CondRegs);

  MachineBasicBlock &JmpMBB = *JmpI.getParent();

  // Insert a direct test of the saved register.
  insertTest(JmpMBB, JmpI.getIterator(), JmpI.getDebugLoc(), CondReg);

  // Rewrite the jump to use the !ZF flag from the test, and kill its use of
  // flags afterward.
  JmpI.getOperand(1).setImm(Inverted ? X86::COND_E : X86::COND_NE);
  JmpI.findRegisterUseOperand(X86::EFLAGS)->setIsKill(true);
  LLVM_DEBUG(dbgs() << "    fixed jCC: "; JmpI.dump());
}

void X86FlagsCopyLoweringPass::rewriteCopy(MachineInstr &MI,
                                           MachineOperand &FlagUse,
                                           MachineInstr &CopyDefI) {
  // Just replace this copy with the original copy def.
  MRI->replaceRegWith(MI.getOperand(0).getReg(),
                      CopyDefI.getOperand(0).getReg());
  MI.eraseFromParent();
}

void X86FlagsCopyLoweringPass::rewriteSetCC(MachineBasicBlock &TestMBB,
                                            MachineBasicBlock::iterator TestPos,
                                            const DebugLoc &TestLoc,
                                            MachineInstr &SetCCI,
                                            MachineOperand &FlagUse,
                                            CondRegArray &CondRegs) {
  X86::CondCode Cond = X86::getCondFromSETCC(SetCCI);
  // Note that we can't usefully rewrite this to the inverse without complex
  // analysis of the users of the setCC. Largely we rely on duplicates which
  // could have been avoided already being avoided here.
  unsigned &CondReg = CondRegs[Cond];
  if (!CondReg)
    CondReg = promoteCondToReg(TestMBB, TestPos, TestLoc, Cond);

  // Rewriting a register def is trivial: we just replace the register and
  // remove the setcc.
  if (!SetCCI.mayStore()) {
    assert(SetCCI.getOperand(0).isReg() &&
           "Cannot have a non-register defined operand to SETcc!");
    Register OldReg = SetCCI.getOperand(0).getReg();
    // Drop Kill flags on the old register before replacing. CondReg may have
    // a longer live range.
    MRI->clearKillFlags(OldReg);
    MRI->replaceRegWith(OldReg, CondReg);
    SetCCI.eraseFromParent();
    return;
  }

  // Otherwise, we need to emit a store.
  auto MIB = BuildMI(*SetCCI.getParent(), SetCCI.getIterator(),
                     SetCCI.getDebugLoc(), TII->get(X86::MOV8mr));
  // Copy the address operands.
  for (int i = 0; i < X86::AddrNumOperands; ++i)
    MIB.add(SetCCI.getOperand(i));

  MIB.addReg(CondReg);

  MIB.setMemRefs(SetCCI.memoperands());

  SetCCI.eraseFromParent();
}
