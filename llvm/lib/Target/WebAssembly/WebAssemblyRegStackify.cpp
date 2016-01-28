//===-- WebAssemblyRegStackify.cpp - Register Stackification --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a register stacking pass.
///
/// This pass reorders instructions to put register uses and defs in an order
/// such that they form single-use expression trees. Registers fitting this form
/// are then marked as "stackified", meaning references to them are replaced by
/// "push" and "pop" from the stack.
///
/// This is primarily a code size optimization, since temporary values on the
/// expression don't need to be named.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h" // for WebAssembly::ARGUMENT_*
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-reg-stackify"

namespace {
class WebAssemblyRegStackify final : public MachineFunctionPass {
  const char *getPassName() const override {
    return "WebAssembly Register Stackify";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<LiveIntervals>();
    AU.addPreserved<MachineBlockFrequencyInfo>();
    AU.addPreserved<SlotIndexes>();
    AU.addPreserved<LiveIntervals>();
    AU.addPreservedID(LiveVariablesID);
    AU.addPreserved<MachineDominatorTree>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyRegStackify() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyRegStackify::ID = 0;
FunctionPass *llvm::createWebAssemblyRegStackify() {
  return new WebAssemblyRegStackify();
}

// Decorate the given instruction with implicit operands that enforce the
// expression stack ordering constraints for an instruction which is on
// the expression stack.
static void ImposeStackOrdering(MachineInstr *MI) {
  // Write the opaque EXPR_STACK register.
  if (!MI->definesRegister(WebAssembly::EXPR_STACK))
    MI->addOperand(MachineOperand::CreateReg(WebAssembly::EXPR_STACK,
                                             /*isDef=*/true,
                                             /*isImp=*/true));

  // Also read the opaque EXPR_STACK register.
  if (!MI->readsRegister(WebAssembly::EXPR_STACK))
    MI->addOperand(MachineOperand::CreateReg(WebAssembly::EXPR_STACK,
                                             /*isDef=*/false,
                                             /*isImp=*/true));
}

// Test whether it's safe to move Def to just before Insert.
// TODO: Compute memory dependencies in a way that doesn't require always
// walking the block.
// TODO: Compute memory dependencies in a way that uses AliasAnalysis to be
// more precise.
static bool IsSafeToMove(const MachineInstr *Def, const MachineInstr *Insert,
                         AliasAnalysis &AA, const LiveIntervals &LIS,
                         const MachineRegisterInfo &MRI) {
  assert(Def->getParent() == Insert->getParent());
  bool SawStore = false, SawSideEffects = false;
  MachineBasicBlock::const_iterator D(Def), I(Insert);

  // Check for register dependencies.
  for (const MachineOperand &MO : Def->operands()) {
    if (!MO.isReg() || MO.isUndef())
      continue;
    unsigned Reg = MO.getReg();

    // If the register is dead here and at Insert, ignore it.
    if (MO.isDead() && Insert->definesRegister(Reg) &&
        !Insert->readsRegister(Reg))
      continue;

    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      // If the physical register is never modified, ignore it.
      if (!MRI.isPhysRegModified(Reg))
        continue;
      // Otherwise, it's a physical register with unknown liveness.
      return false;
    }

    // Ask LiveIntervals whether moving this virtual register use or def to
    // Insert will change value numbers are seen.
    const LiveInterval &LI = LIS.getInterval(Reg);
    VNInfo *DefVNI =
        MO.isDef() ? LI.getVNInfoAt(LIS.getInstructionIndex(Def).getRegSlot())
                   : LI.getVNInfoBefore(LIS.getInstructionIndex(Def));
    assert(DefVNI && "Instruction input missing value number");
    VNInfo *InsVNI = LI.getVNInfoBefore(LIS.getInstructionIndex(Insert));
    if (InsVNI && DefVNI != InsVNI)
      return false;
  }

  // Check for memory dependencies and side effects.
  for (--I; I != D; --I)
    SawSideEffects |= !I->isSafeToMove(&AA, SawStore);
  return !(SawStore && Def->mayLoad() && !Def->isInvariantLoad(&AA)) &&
         !(SawSideEffects && !Def->isSafeToMove(&AA, SawStore));
}

/// Test whether OneUse, a use of Reg, dominates all of Reg's other uses.
static bool OneUseDominatesOtherUses(unsigned Reg, const MachineOperand &OneUse,
                                     const MachineBasicBlock &MBB,
                                     const MachineRegisterInfo &MRI,
                                     const MachineDominatorTree &MDT) {
  for (const MachineOperand &Use : MRI.use_operands(Reg)) {
    if (&Use == &OneUse)
      continue;
    const MachineInstr *UseInst = Use.getParent();
    const MachineInstr *OneUseInst = OneUse.getParent();
    if (UseInst->getOpcode() == TargetOpcode::PHI) {
      // Test that the PHI use, which happens on the CFG edge rather than
      // within the PHI's own block, is dominated by the one selected use.
      const MachineBasicBlock *Pred =
          UseInst->getOperand(&Use - &UseInst->getOperand(0) + 1).getMBB();
      if (!MDT.dominates(&MBB, Pred))
        return false;
    } else if (UseInst == OneUseInst) {
      // Another use in the same instruction. We need to ensure that the one
      // selected use happens "before" it.
      if (&OneUse > &Use)
        return false;
    } else {
      // Test that the use is dominated by the one selected use.
      if (!MDT.dominates(OneUseInst, UseInst))
        return false;
    }
  }
  return true;
}

/// Get the appropriate tee_local opcode for the given register class.
static unsigned GetTeeLocalOpcode(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return WebAssembly::TEE_LOCAL_I32;
  if (RC == &WebAssembly::I64RegClass)
    return WebAssembly::TEE_LOCAL_I64;
  if (RC == &WebAssembly::F32RegClass)
    return WebAssembly::TEE_LOCAL_F32;
  if (RC == &WebAssembly::F64RegClass)
    return WebAssembly::TEE_LOCAL_F64;
  llvm_unreachable("Unexpected register class");
}

/// A single-use def in the same block with no intervening memory or register
/// dependencies; move the def down and nest it with the current instruction.
static MachineInstr *MoveForSingleUse(unsigned Reg, MachineInstr *Def,
                                      MachineBasicBlock &MBB,
                                      MachineInstr *Insert, LiveIntervals &LIS,
                                      WebAssemblyFunctionInfo &MFI) {
  MBB.splice(Insert, &MBB, Def);
  LIS.handleMove(Def);
  MFI.stackifyVReg(Reg);
  ImposeStackOrdering(Def);
  return Def;
}

/// A trivially cloneable instruction; clone it and nest the new copy with the
/// current instruction.
static MachineInstr *
RematerializeCheapDef(unsigned Reg, MachineOperand &Op, MachineInstr *Def,
                      MachineBasicBlock &MBB, MachineInstr *Insert,
                      LiveIntervals &LIS, WebAssemblyFunctionInfo &MFI,
                      MachineRegisterInfo &MRI, const WebAssemblyInstrInfo *TII,
                      const WebAssemblyRegisterInfo *TRI) {
  unsigned NewReg = MRI.createVirtualRegister(MRI.getRegClass(Reg));
  TII->reMaterialize(MBB, Insert, NewReg, 0, Def, *TRI);
  Op.setReg(NewReg);
  MachineInstr *Clone = &*std::prev(MachineBasicBlock::instr_iterator(Insert));
  LIS.InsertMachineInstrInMaps(Clone);
  LIS.createAndComputeVirtRegInterval(NewReg);
  MFI.stackifyVReg(NewReg);
  ImposeStackOrdering(Clone);

  // If that was the last use of the original, delete the original.
  // Otherwise shrink the LiveInterval.
  if (MRI.use_empty(Reg)) {
    SlotIndex Idx = LIS.getInstructionIndex(Def).getRegSlot();
    LIS.removePhysRegDefAt(WebAssembly::ARGUMENTS, Idx);
    LIS.removeVRegDefAt(LIS.getInterval(Reg), Idx);
    LIS.removeInterval(Reg);
    LIS.RemoveMachineInstrFromMaps(Def);
    Def->eraseFromParent();
  } else {
    LIS.shrinkToUses(&LIS.getInterval(Reg));
  }
  return Clone;
}

/// A multiple-use def in the same block with no intervening memory or register
/// dependencies; move the def down, nest it with the current instruction, and
/// insert a tee_local to satisfy the rest of the uses. As an illustration,
/// rewrite this:
///
///    Reg = INST ...        // Def
///    INST ..., Reg, ...    // Insert
///    INST ..., Reg, ...
///    INST ..., Reg, ...
///
/// to this:
///
///    Reg = INST ...        // Def (to become the new Insert)
///    TeeReg, NewReg = TEE_LOCAL_... Reg
///    INST ..., TeeReg, ... // Insert
///    INST ..., NewReg, ...
///    INST ..., NewReg, ...
///
/// with Reg and TeeReg stackified. This eliminates a get_local from the
/// resulting code.
static MachineInstr *MoveAndTeeForMultiUse(
    unsigned Reg, MachineOperand &Op, MachineInstr *Def, MachineBasicBlock &MBB,
    MachineInstr *Insert, LiveIntervals &LIS, WebAssemblyFunctionInfo &MFI,
    MachineRegisterInfo &MRI, const WebAssemblyInstrInfo *TII) {
  MBB.splice(Insert, &MBB, Def);
  LIS.handleMove(Def);
  const auto *RegClass = MRI.getRegClass(Reg);
  unsigned NewReg = MRI.createVirtualRegister(RegClass);
  unsigned TeeReg = MRI.createVirtualRegister(RegClass);
  MRI.replaceRegWith(Reg, NewReg);
  MachineInstr *Tee = BuildMI(MBB, Insert, Insert->getDebugLoc(),
                              TII->get(GetTeeLocalOpcode(RegClass)), TeeReg)
                          .addReg(NewReg, RegState::Define)
                          .addReg(Reg);
  Op.setReg(TeeReg);
  Def->getOperand(0).setReg(Reg);
  LIS.InsertMachineInstrInMaps(Tee);
  LIS.shrinkToUses(&LIS.getInterval(Reg));
  LIS.createAndComputeVirtRegInterval(NewReg);
  LIS.createAndComputeVirtRegInterval(TeeReg);
  MFI.stackifyVReg(Reg);
  MFI.stackifyVReg(TeeReg);
  ImposeStackOrdering(Def);
  ImposeStackOrdering(Tee);
  return Def;
}

namespace {
/// A stack for walking the tree of instructions being built, visiting the
/// MachineOperands in DFS order.
class TreeWalkerState {
  typedef MachineInstr::mop_iterator mop_iterator;
  typedef std::reverse_iterator<mop_iterator> mop_reverse_iterator;
  typedef iterator_range<mop_reverse_iterator> RangeTy;
  SmallVector<RangeTy, 4> Worklist;

public:
  explicit TreeWalkerState(MachineInstr *Insert) {
    const iterator_range<mop_iterator> &Range = Insert->explicit_uses();
    if (Range.begin() != Range.end())
      Worklist.push_back(reverse(Range));
  }

  bool Done() const { return Worklist.empty(); }

  MachineOperand &Pop() {
    RangeTy &Range = Worklist.back();
    MachineOperand &Op = *Range.begin();
    Range = drop_begin(Range, 1);
    if (Range.begin() == Range.end())
      Worklist.pop_back();
    assert((Worklist.empty() ||
            Worklist.back().begin() != Worklist.back().end()) &&
           "Empty ranges shouldn't remain in the worklist");
    return Op;
  }

  /// Push Instr's operands onto the stack to be visited.
  void PushOperands(MachineInstr *Instr) {
    const iterator_range<mop_iterator> &Range(Instr->explicit_uses());
    if (Range.begin() != Range.end())
      Worklist.push_back(reverse(Range));
  }

  /// Some of Instr's operands are on the top of the stack; remove them and
  /// re-insert them starting from the beginning (because we've commuted them).
  void ResetTopOperands(MachineInstr *Instr) {
    assert(HasRemainingOperands(Instr) &&
           "Reseting operands should only be done when the instruction has "
           "an operand still on the stack");
    Worklist.back() = reverse(Instr->explicit_uses());
  }

  /// Test whether Instr has operands remaining to be visited at the top of
  /// the stack.
  bool HasRemainingOperands(const MachineInstr *Instr) const {
    if (Worklist.empty())
      return false;
    const RangeTy &Range = Worklist.back();
    return Range.begin() != Range.end() && Range.begin()->getParent() == Instr;
  }

  /// Test whether the given register is present on the stack, indicating an
  /// operand in the tree that we haven't visited yet. Moving a definition of
  /// Reg to a point in the tree after that would change its value.
  bool IsOnStack(unsigned Reg) const {
    for (const RangeTy &Range : Worklist)
      for (const MachineOperand &MO : Range)
        if (MO.isReg() && MO.getReg() == Reg)
          return true;
    return false;
  }
};

/// State to keep track of whether commuting is in flight or whether it's been
/// tried for the current instruction and didn't work.
class CommutingState {
  /// There are effectively three states: the initial state where we haven't
  /// started commuting anything and we don't know anything yet, the tenative
  /// state where we've commuted the operands of the current instruction and are
  /// revisting it, and the declined state where we've reverted the operands
  /// back to their original order and will no longer commute it further.
  bool TentativelyCommuting;
  bool Declined;

  /// During the tentative state, these hold the operand indices of the commuted
  /// operands.
  unsigned Operand0, Operand1;

public:
  CommutingState() : TentativelyCommuting(false), Declined(false) {}

  /// Stackification for an operand was not successful due to ordering
  /// constraints. If possible, and if we haven't already tried it and declined
  /// it, commute Insert's operands and prepare to revisit it.
  void MaybeCommute(MachineInstr *Insert, TreeWalkerState &TreeWalker,
                    const WebAssemblyInstrInfo *TII) {
    if (TentativelyCommuting) {
      assert(!Declined &&
             "Don't decline commuting until you've finished trying it");
      // Commuting didn't help. Revert it.
      TII->commuteInstruction(Insert, /*NewMI=*/false, Operand0, Operand1);
      TentativelyCommuting = false;
      Declined = true;
    } else if (!Declined && TreeWalker.HasRemainingOperands(Insert)) {
      Operand0 = TargetInstrInfo::CommuteAnyOperandIndex;
      Operand1 = TargetInstrInfo::CommuteAnyOperandIndex;
      if (TII->findCommutedOpIndices(Insert, Operand0, Operand1)) {
        // Tentatively commute the operands and try again.
        TII->commuteInstruction(Insert, /*NewMI=*/false, Operand0, Operand1);
        TreeWalker.ResetTopOperands(Insert);
        TentativelyCommuting = true;
        Declined = false;
      }
    }
  }

  /// Stackification for some operand was successful. Reset to the default
  /// state.
  void Reset() {
    TentativelyCommuting = false;
    Declined = false;
  }
};
} // end anonymous namespace

bool WebAssemblyRegStackify::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "********** Register Stackifying **********\n"
                  "********** Function: "
               << MF.getName() << '\n');

  bool Changed = false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  WebAssemblyFunctionInfo &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  const auto *TII = MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  const auto *TRI = MF.getSubtarget<WebAssemblySubtarget>().getRegisterInfo();
  AliasAnalysis &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();
  LiveIntervals &LIS = getAnalysis<LiveIntervals>();

  // Walk the instructions from the bottom up. Currently we don't look past
  // block boundaries, and the blocks aren't ordered so the block visitation
  // order isn't significant, but we may want to change this in the future.
  for (MachineBasicBlock &MBB : MF) {
    // Don't use a range-based for loop, because we modify the list as we're
    // iterating over it and the end iterator may change.
    for (auto MII = MBB.rbegin(); MII != MBB.rend(); ++MII) {
      MachineInstr *Insert = &*MII;
      // Don't nest anything inside a phi.
      if (Insert->getOpcode() == TargetOpcode::PHI)
        break;

      // Don't nest anything inside an inline asm, because we don't have
      // constraints for $push inputs.
      if (Insert->getOpcode() == TargetOpcode::INLINEASM)
        break;

      // Iterate through the inputs in reverse order, since we'll be pulling
      // operands off the stack in LIFO order.
      CommutingState Commuting;
      TreeWalkerState TreeWalker(Insert);
      while (!TreeWalker.Done()) {
        MachineOperand &Op = TreeWalker.Pop();

        // We're only interested in explicit virtual register operands.
        if (!Op.isReg())
          continue;

        unsigned Reg = Op.getReg();
        assert(Op.isUse() && "explicit_uses() should only iterate over uses");
        assert(!Op.isImplicit() &&
               "explicit_uses() should only iterate over explicit operands");
        if (TargetRegisterInfo::isPhysicalRegister(Reg))
          continue;

        // Identify the definition for this register at this point. Most
        // registers are in SSA form here so we try a quick MRI query first.
        MachineInstr *Def = MRI.getUniqueVRegDef(Reg);
        if (!Def) {
          // MRI doesn't know what the Def is. Try asking LIS.
          const VNInfo *ValNo = LIS.getInterval(Reg).getVNInfoBefore(
              LIS.getInstructionIndex(Insert));
          if (!ValNo)
            continue;
          Def = LIS.getInstructionFromIndex(ValNo->def);
          if (!Def)
            continue;
        }

        // Don't nest an INLINE_ASM def into anything, because we don't have
        // constraints for $pop outputs.
        if (Def->getOpcode() == TargetOpcode::INLINEASM)
          continue;

        // Don't nest PHIs inside of anything.
        if (Def->getOpcode() == TargetOpcode::PHI)
          continue;

        // Argument instructions represent live-in registers and not real
        // instructions.
        if (Def->getOpcode() == WebAssembly::ARGUMENT_I32 ||
            Def->getOpcode() == WebAssembly::ARGUMENT_I64 ||
            Def->getOpcode() == WebAssembly::ARGUMENT_F32 ||
            Def->getOpcode() == WebAssembly::ARGUMENT_F64)
          continue;

        // Decide which strategy to take. Prefer to move a single-use value
        // over cloning it, and prefer cloning over introducing a tee_local.
        // For moving, we require the def to be in the same block as the use;
        // this makes things simpler (LiveIntervals' handleMove function only
        // supports intra-block moves) and it's MachineSink's job to catch all
        // the sinking opportunities anyway.
        bool SameBlock = Def->getParent() == &MBB;
        bool CanMove = SameBlock && IsSafeToMove(Def, Insert, AA, LIS, MRI) &&
                       !TreeWalker.IsOnStack(Reg);
        if (CanMove && MRI.hasOneUse(Reg)) {
          Insert = MoveForSingleUse(Reg, Def, MBB, Insert, LIS, MFI);
        } else if (Def->isAsCheapAsAMove() &&
                   TII->isTriviallyReMaterializable(Def, &AA)) {
          Insert = RematerializeCheapDef(Reg, Op, Def, MBB, Insert, LIS, MFI,
                                         MRI, TII, TRI);
        } else if (CanMove &&
                   OneUseDominatesOtherUses(Reg, Op, MBB, MRI, MDT)) {
          Insert = MoveAndTeeForMultiUse(Reg, Op, Def, MBB, Insert, LIS, MFI,
                                         MRI, TII);
        } else {
          // We failed to stackify the operand. If the problem was ordering
          // constraints, Commuting may be able to help.
          if (!CanMove && SameBlock)
            Commuting.MaybeCommute(Insert, TreeWalker, TII);
          // Proceed to the next operand.
          continue;
        }

        // We stackified an operand. Add the defining instruction's operands to
        // the worklist stack now to continue to build an ever deeper tree.
        Commuting.Reset();
        TreeWalker.PushOperands(Insert);
      }

      // If we stackified any operands, skip over the tree to start looking for
      // the next instruction we can build a tree on.
      if (Insert != &*MII) {
        ImposeStackOrdering(&*MII);
        MII = std::prev(
            make_reverse_iterator(MachineBasicBlock::iterator(Insert)));
        Changed = true;
      }
    }
  }

  // If we used EXPR_STACK anywhere, add it to the live-in sets everywhere so
  // that it never looks like a use-before-def.
  if (Changed) {
    MF.getRegInfo().addLiveIn(WebAssembly::EXPR_STACK);
    for (MachineBasicBlock &MBB : MF)
      MBB.addLiveIn(WebAssembly::EXPR_STACK);
  }

#ifndef NDEBUG
  // Verify that pushes and pops are performed in LIFO order.
  SmallVector<unsigned, 0> Stack;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : reverse(MI.explicit_operands())) {
        if (!MO.isReg())
          continue;
        unsigned Reg = MO.getReg();

        // Don't stackify physregs like SP or FP.
        if (!TargetRegisterInfo::isVirtualRegister(Reg))
          continue;

        if (MFI.isVRegStackified(Reg)) {
          if (MO.isDef())
            Stack.push_back(Reg);
          else
            assert(Stack.pop_back_val() == Reg &&
                   "Register stack pop should be paired with a push");
        }
      }
    }
    // TODO: Generalize this code to support keeping values on the stack across
    // basic block boundaries.
    assert(Stack.empty() &&
           "Register stack pushes and pops should be balanced");
  }
#endif

  return Changed;
}
