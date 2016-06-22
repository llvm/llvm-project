//===-- ImplicitNullChecks.cpp - Fold null checks into memory accesses ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass turns explicit null checks of the form
//
//   test %r10, %r10
//   je throw_npe
//   movl (%r10), %esi
//   ...
//
// to
//
//   faulting_load_op("movl (%r10), %esi", throw_npe)
//   ...
//
// With the help of a runtime that understands the .fault_maps section,
// faulting_load_op branches to throw_npe if executing movl (%r10), %esi incurs
// a page fault.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

static cl::opt<int> PageSize("imp-null-check-page-size",
                             cl::desc("The page size of the target in bytes"),
                             cl::init(4096));

#define DEBUG_TYPE "implicit-null-checks"

STATISTIC(NumImplicitNullChecks,
          "Number of explicit null checks made implicit");

namespace {

class ImplicitNullChecks : public MachineFunctionPass {
  /// Represents one null check that can be made implicit.
  class NullCheck {
    // The memory operation the null check can be folded into.
    MachineInstr *MemOperation;

    // The instruction actually doing the null check (Ptr != 0).
    MachineInstr *CheckOperation;

    // The block the check resides in.
    MachineBasicBlock *CheckBlock;

    // The block branched to if the pointer is non-null.
    MachineBasicBlock *NotNullSucc;

    // The block branched to if the pointer is null.
    MachineBasicBlock *NullSucc;

    // If this is non-null, then MemOperation has a dependency on on this
    // instruction; and it needs to be hoisted to execute before MemOperation.
    MachineInstr *OnlyDependency;

  public:
    explicit NullCheck(MachineInstr *memOperation, MachineInstr *checkOperation,
                       MachineBasicBlock *checkBlock,
                       MachineBasicBlock *notNullSucc,
                       MachineBasicBlock *nullSucc,
                       MachineInstr *onlyDependency)
        : MemOperation(memOperation), CheckOperation(checkOperation),
          CheckBlock(checkBlock), NotNullSucc(notNullSucc), NullSucc(nullSucc),
          OnlyDependency(onlyDependency) {}

    MachineInstr *getMemOperation() const { return MemOperation; }

    MachineInstr *getCheckOperation() const { return CheckOperation; }

    MachineBasicBlock *getCheckBlock() const { return CheckBlock; }

    MachineBasicBlock *getNotNullSucc() const { return NotNullSucc; }

    MachineBasicBlock *getNullSucc() const { return NullSucc; }

    MachineInstr *getOnlyDependency() const { return OnlyDependency; }
  };

  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  AliasAnalysis *AA = nullptr;
  MachineModuleInfo *MMI = nullptr;

  bool analyzeBlockForNullChecks(MachineBasicBlock &MBB,
                                 SmallVectorImpl<NullCheck> &NullCheckList);
  MachineInstr *insertFaultingLoad(MachineInstr *LoadMI, MachineBasicBlock *MBB,
                                   MachineBasicBlock *HandlerMBB);
  void rewriteNullChecks(ArrayRef<NullCheck> NullCheckList);

public:
  static char ID;

  ImplicitNullChecks() : MachineFunctionPass(ID) {
    initializeImplicitNullChecksPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::AllVRegsAllocated);
  }
};

/// \brief Detect re-ordering hazards and dependencies.
///
/// This class keeps track of defs and uses, and can be queried if a given
/// machine instruction can be re-ordered from after the machine instructions
/// seen so far to before them.
class HazardDetector {
  static MachineInstr *getUnknownMI() {
    return DenseMapInfo<MachineInstr *>::getTombstoneKey();
  }

  // Maps physical registers to the instruction defining them.  If there has
  // been more than one def of an specific register, that register is mapped to
  // getUnknownMI().
  DenseMap<unsigned, MachineInstr *> RegDefs;
  DenseSet<unsigned> RegUses;
  const TargetRegisterInfo &TRI;
  bool hasSeenClobber;
  AliasAnalysis &AA;

public:
  explicit HazardDetector(const TargetRegisterInfo &TRI, AliasAnalysis &AA)
      : TRI(TRI), hasSeenClobber(false), AA(AA) {}

  /// \brief Make a note of \p MI for later queries to isSafeToHoist.
  ///
  /// May clobber this HazardDetector instance.  \see isClobbered.
  void rememberInstruction(MachineInstr *MI);

  /// \brief Return true if it is safe to hoist \p MI from after all the
  /// instructions seen so far (via rememberInstruction) to before it.  If \p MI
  /// has one and only one transitive dependency, set \p Dependency to that
  /// instruction.  If there are more dependencies, return false.
  bool isSafeToHoist(MachineInstr *MI, MachineInstr *&Dependency);

  /// \brief Return true if this instance of HazardDetector has been clobbered
  /// (i.e. has no more useful information).
  ///
  /// A HazardDetecter is clobbered when it sees a construct it cannot
  /// understand, and it would have to return a conservative answer for all
  /// future queries.  Having a separate clobbered state lets the client code
  /// bail early, without making queries about all of the future instructions
  /// (which would have returned the most conservative answer anyway).
  ///
  /// Calling rememberInstruction or isSafeToHoist on a clobbered HazardDetector
  /// is an error.
  bool isClobbered() { return hasSeenClobber; }
};
}


void HazardDetector::rememberInstruction(MachineInstr *MI) {
  assert(!isClobbered() &&
         "Don't add instructions to a clobbered hazard detector");

  if (MI->mayStore() || MI->hasUnmodeledSideEffects()) {
    hasSeenClobber = true;
    return;
  }

  for (auto *MMO : MI->memoperands()) {
    // Right now we don't want to worry about LLVM's memory model.
    if (!MMO->isUnordered()) {
      hasSeenClobber = true;
      return;
    }
  }

  for (auto &MO : MI->operands()) {
    if (!MO.isReg() || !MO.getReg())
      continue;

    if (MO.isDef()) {
      auto It = RegDefs.find(MO.getReg());
      if (It == RegDefs.end())
        RegDefs.insert({MO.getReg(), MI});
      else {
        assert(It->second && "Found null MI?");
        It->second = getUnknownMI();
      }
    } else
      RegUses.insert(MO.getReg());
  }
}

bool HazardDetector::isSafeToHoist(MachineInstr *MI,
                                   MachineInstr *&Dependency) {
  assert(!isClobbered() && "isSafeToHoist cannot do anything useful!");
  Dependency = nullptr;

  // Right now we don't want to worry about LLVM's memory model.  This can be
  // made more precise later.
  for (auto *MMO : MI->memoperands())
    if (!MMO->isUnordered())
      return false;

  for (auto &MO : MI->operands()) {
    if (MO.isReg() && MO.getReg()) {
      for (auto &RegDef : RegDefs) {
        unsigned Reg = RegDef.first;
        MachineInstr *MI = RegDef.second;
        if (!TRI.regsOverlap(Reg, MO.getReg()))
          continue;

        // We found a write-after-write or read-after-write, see if the
        // instruction causing this dependency can be hoisted too.

        if (MI == getUnknownMI())
          // We don't have precise dependency information.
          return false;

        if (Dependency) {
          if (Dependency == MI)
            continue;
          // We already have one dependency, and we can track only one.
          return false;
        }

        // Now check if MI is actually a dependency that can be hoisted.

        // We don't want to track transitive dependencies.  We already know that
        // MI is the only instruction that defines Reg, but we need to be sure
        // that it does not use any registers that have been defined (trivially
        // checked below by ensuring that there are no register uses), and that
        // it is the only def for every register it defines (otherwise we could
        // violate a write after write hazard).
        auto IsMIOperandSafe = [&](MachineOperand &MO) {
          if (!MO.isReg() || !MO.getReg())
            return true;
          if (MO.isUse())
            return false;
          assert((!MO.isDef() || RegDefs.count(MO.getReg())) &&
                 "All defs must be tracked in RegDefs by now!");
          return !MO.isDef() || RegDefs.find(MO.getReg())->second == MI;
        };

        if (!all_of(MI->operands(), IsMIOperandSafe))
          return false;

        // Now check for speculation safety:
        bool SawStore = true;
        if (!MI->isSafeToMove(&AA, SawStore) || MI->mayLoad())
          return false;

        Dependency = MI;
      }

      if (MO.isDef())
        for (unsigned Reg : RegUses)
          if (TRI.regsOverlap(Reg, MO.getReg()))
            return false;  // We found a write-after-read
    }
  }

  return true;
}

bool ImplicitNullChecks::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getRegInfo().getTargetRegisterInfo();
  MMI = &MF.getMMI();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

  SmallVector<NullCheck, 16> NullCheckList;

  for (auto &MBB : MF)
    analyzeBlockForNullChecks(MBB, NullCheckList);

  if (!NullCheckList.empty())
    rewriteNullChecks(NullCheckList);

  return !NullCheckList.empty();
}

// Return true if any register aliasing \p Reg is live-in into \p MBB.
static bool AnyAliasLiveIn(const TargetRegisterInfo *TRI,
                           MachineBasicBlock *MBB, unsigned Reg) {
  for (MCRegAliasIterator AR(Reg, TRI, /*IncludeSelf*/ true); AR.isValid();
       ++AR)
    if (MBB->isLiveIn(*AR))
      return true;
  return false;
}

/// Analyze MBB to check if its terminating branch can be turned into an
/// implicit null check.  If yes, append a description of the said null check to
/// NullCheckList and return true, else return false.
bool ImplicitNullChecks::analyzeBlockForNullChecks(
    MachineBasicBlock &MBB, SmallVectorImpl<NullCheck> &NullCheckList) {
  typedef TargetInstrInfo::MachineBranchPredicate MachineBranchPredicate;

  MDNode *BranchMD = nullptr;
  if (auto *BB = MBB.getBasicBlock())
    BranchMD = BB->getTerminator()->getMetadata(LLVMContext::MD_make_implicit);

  if (!BranchMD)
    return false;

  MachineBranchPredicate MBP;

  if (TII->AnalyzeBranchPredicate(MBB, MBP, true))
    return false;

  // Is the predicate comparing an integer to zero?
  if (!(MBP.LHS.isReg() && MBP.RHS.isImm() && MBP.RHS.getImm() == 0 &&
        (MBP.Predicate == MachineBranchPredicate::PRED_NE ||
         MBP.Predicate == MachineBranchPredicate::PRED_EQ)))
    return false;

  // If we cannot erase the test instruction itself, then making the null check
  // implicit does not buy us much.
  if (!MBP.SingleUseCondition)
    return false;

  MachineBasicBlock *NotNullSucc, *NullSucc;

  if (MBP.Predicate == MachineBranchPredicate::PRED_NE) {
    NotNullSucc = MBP.TrueDest;
    NullSucc = MBP.FalseDest;
  } else {
    NotNullSucc = MBP.FalseDest;
    NullSucc = MBP.TrueDest;
  }

  // We handle the simplest case for now.  We can potentially do better by using
  // the machine dominator tree.
  if (NotNullSucc->pred_size() != 1)
    return false;

  // Starting with a code fragment like:
  //
  //   test %RAX, %RAX
  //   jne LblNotNull
  //
  //  LblNull:
  //   callq throw_NullPointerException
  //
  //  LblNotNull:
  //   Inst0
  //   Inst1
  //   ...
  //   Def = Load (%RAX + <offset>)
  //   ...
  //
  //
  // we want to end up with
  //
  //   Def = FaultingLoad (%RAX + <offset>), LblNull
  //   jmp LblNotNull ;; explicit or fallthrough
  //
  //  LblNotNull:
  //   Inst0
  //   Inst1
  //   ...
  //
  //  LblNull:
  //   callq throw_NullPointerException
  //
  //
  // To see why this is legal, consider the two possibilities:
  //
  //  1. %RAX is null: since we constrain <offset> to be less than PageSize, the
  //     load instruction dereferences the null page, causing a segmentation
  //     fault.
  //
  //  2. %RAX is not null: in this case we know that the load cannot fault, as
  //     otherwise the load would've faulted in the original program too and the
  //     original program would've been undefined.
  //
  // This reasoning cannot be extended to justify hoisting through arbitrary
  // control flow.  For instance, in the example below (in pseudo-C)
  //
  //    if (ptr == null) { throw_npe(); unreachable; }
  //    if (some_cond) { return 42; }
  //    v = ptr->field;  // LD
  //    ...
  //
  // we cannot (without code duplication) use the load marked "LD" to null check
  // ptr -- clause (2) above does not apply in this case.  In the above program
  // the safety of ptr->field can be dependent on some_cond; and, for instance,
  // ptr could be some non-null invalid reference that never gets loaded from
  // because some_cond is always true.

  unsigned PointerReg = MBP.LHS.getReg();

  HazardDetector HD(*TRI, *AA);

  for (auto MII = NotNullSucc->begin(), MIE = NotNullSucc->end(); MII != MIE;
       ++MII) {
    MachineInstr *MI = &*MII;
    unsigned BaseReg;
    int64_t Offset;
    MachineInstr *Dependency = nullptr;
    if (TII->getMemOpBaseRegImmOfs(MI, BaseReg, Offset, TRI))
      if (MI->mayLoad() && !MI->isPredicable() && BaseReg == PointerReg &&
          Offset < PageSize && MI->getDesc().getNumDefs() <= 1 &&
          HD.isSafeToHoist(MI, Dependency)) {

        auto DependencyOperandIsOk = [&](MachineOperand &MO) {
          assert(!(MO.isReg() && MO.isUse()) &&
                 "No transitive dependendencies please!");
          if (!MO.isReg() || !MO.getReg() || !MO.isDef())
            return true;

          // Make sure that we won't clobber any live ins to the sibling block
          // by hoisting Dependency.  For instance, we can't hoist INST to
          // before the null check (even if it safe, and does not violate any
          // dependencies in the non_null_block) if %rdx is live in to
          // _null_block.
          //
          //    test %rcx, %rcx
          //    je _null_block
          //  _non_null_block:
          //    %rdx<def> = INST
          //    ...
          if (AnyAliasLiveIn(TRI, NullSucc, MO.getReg()))
            return false;

          // Make sure Dependency isn't re-defining the base register.  Then we
          // won't get the memory operation on the address we want.
          if (TRI->regsOverlap(MO.getReg(), BaseReg))
            return false;

          return true;
        };

        bool DependencyOperandsAreOk =
            !Dependency ||
            all_of(Dependency->operands(), DependencyOperandIsOk);

        if (DependencyOperandsAreOk) {
          NullCheckList.emplace_back(MI, MBP.ConditionDef, &MBB, NotNullSucc,
                                     NullSucc, Dependency);
          return true;
        }
      }

    HD.rememberInstruction(MI);
    if (HD.isClobbered())
      return false;
  }

  return false;
}

/// Wrap a machine load instruction, LoadMI, into a FAULTING_LOAD_OP machine
/// instruction.  The FAULTING_LOAD_OP instruction does the same load as LoadMI
/// (defining the same register), and branches to HandlerMBB if the load
/// faults.  The FAULTING_LOAD_OP instruction is inserted at the end of MBB.
MachineInstr *
ImplicitNullChecks::insertFaultingLoad(MachineInstr *LoadMI,
                                       MachineBasicBlock *MBB,
                                       MachineBasicBlock *HandlerMBB) {
  const unsigned NoRegister = 0; // Guaranteed to be the NoRegister value for
                                 // all targets.

  DebugLoc DL;
  unsigned NumDefs = LoadMI->getDesc().getNumDefs();
  assert(NumDefs <= 1 && "other cases unhandled!");

  unsigned DefReg = NoRegister;
  if (NumDefs != 0) {
    DefReg = LoadMI->defs().begin()->getReg();
    assert(std::distance(LoadMI->defs().begin(), LoadMI->defs().end()) == 1 &&
           "expected exactly one def!");
  }

  auto MIB = BuildMI(MBB, DL, TII->get(TargetOpcode::FAULTING_LOAD_OP), DefReg)
                 .addMBB(HandlerMBB)
                 .addImm(LoadMI->getOpcode());

  for (auto &MO : LoadMI->uses())
    MIB.addOperand(MO);

  MIB.setMemRefs(LoadMI->memoperands_begin(), LoadMI->memoperands_end());

  return MIB;
}

/// Rewrite the null checks in NullCheckList into implicit null checks.
void ImplicitNullChecks::rewriteNullChecks(
    ArrayRef<ImplicitNullChecks::NullCheck> NullCheckList) {
  DebugLoc DL;

  for (auto &NC : NullCheckList) {
    // Remove the conditional branch dependent on the null check.
    unsigned BranchesRemoved = TII->RemoveBranch(*NC.getCheckBlock());
    (void)BranchesRemoved;
    assert(BranchesRemoved > 0 && "expected at least one branch!");

    if (auto *DepMI = NC.getOnlyDependency()) {
      DepMI->removeFromParent();
      NC.getCheckBlock()->insert(NC.getCheckBlock()->end(), DepMI);
    }

    // Insert a faulting load where the conditional branch was originally.  We
    // check earlier ensures that this bit of code motion is legal.  We do not
    // touch the successors list for any basic block since we haven't changed
    // control flow, we've just made it implicit.
    MachineInstr *FaultingLoad = insertFaultingLoad(
        NC.getMemOperation(), NC.getCheckBlock(), NC.getNullSucc());
    // Now the values defined by MemOperation, if any, are live-in of
    // the block of MemOperation.
    // The original load operation may define implicit-defs alongside
    // the loaded value.
    MachineBasicBlock *MBB = NC.getMemOperation()->getParent();
    for (const MachineOperand &MO : FaultingLoad->operands()) {
      if (!MO.isReg() || !MO.isDef())
        continue;
      unsigned Reg = MO.getReg();
      if (!Reg || MBB->isLiveIn(Reg))
        continue;
      MBB->addLiveIn(Reg);
    }

    if (auto *DepMI = NC.getOnlyDependency()) {
      for (auto &MO : DepMI->operands()) {
        if (!MO.isReg() || !MO.getReg() || !MO.isDef())
          continue;
        if (!NC.getNotNullSucc()->isLiveIn(MO.getReg()))
          NC.getNotNullSucc()->addLiveIn(MO.getReg());
      }
    }

    NC.getMemOperation()->eraseFromParent();
    NC.getCheckOperation()->eraseFromParent();

    // Insert an *unconditional* branch to not-null successor.
    TII->InsertBranch(*NC.getCheckBlock(), NC.getNotNullSucc(), nullptr,
                      /*Cond=*/None, DL);

    NumImplicitNullChecks++;
  }
}

char ImplicitNullChecks::ID = 0;
char &llvm::ImplicitNullChecksID = ImplicitNullChecks::ID;
INITIALIZE_PASS_BEGIN(ImplicitNullChecks, "implicit-null-checks",
                      "Implicit null checks", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(ImplicitNullChecks, "implicit-null-checks",
                    "Implicit null checks", false, false)
