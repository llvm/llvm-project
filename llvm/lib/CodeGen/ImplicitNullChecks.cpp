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

static cl::opt<unsigned> PageSize("imp-null-check-page-size",
                                  cl::desc("The page size of the target in "
                                           "bytes"),
                                  cl::init(4096));

#define DEBUG_TYPE "implicit-null-checks"

STATISTIC(NumImplicitNullChecks,
          "Number of explicit null checks made implicit");

namespace {

class ImplicitNullChecks : public MachineFunctionPass {
  /// Represents one null check that can be made implicit.
  struct NullCheck {
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

    NullCheck()
        : MemOperation(), CheckOperation(), CheckBlock(), NotNullSucc(),
          NullSucc() {}

    explicit NullCheck(MachineInstr *memOperation, MachineInstr *checkOperation,
                       MachineBasicBlock *checkBlock,
                       MachineBasicBlock *notNullSucc,
                       MachineBasicBlock *nullSucc)
        : MemOperation(memOperation), CheckOperation(checkOperation),
          CheckBlock(checkBlock), NotNullSucc(notNullSucc), NullSucc(nullSucc) {
    }
  };

  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  MachineModuleInfo *MMI = nullptr;

  bool analyzeBlockForNullChecks(MachineBasicBlock &MBB,
                                 SmallVectorImpl<NullCheck> &NullCheckList);
  MachineInstr *insertFaultingLoad(MachineInstr *LoadMI, MachineBasicBlock *MBB,
                                   MCSymbol *HandlerLabel);
  void rewriteNullChecks(ArrayRef<NullCheck> NullCheckList);

public:
  static char ID;

  ImplicitNullChecks() : MachineFunctionPass(ID) {
    initializeImplicitNullChecksPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

/// \brief Detect re-ordering hazards and dependencies.
///
/// This class keeps track of defs and uses, and can be queried if a given
/// machine instruction can be re-ordered from after the machine instructions
/// seen so far to before them.
class HazardDetector {
  DenseSet<unsigned> RegDefs;
  DenseSet<unsigned> RegUses;
  const TargetRegisterInfo &TRI;
  bool hasSeenClobber;

public:
  explicit HazardDetector(const TargetRegisterInfo &TRI) :
    TRI(TRI), hasSeenClobber(false) {}

  /// \brief Make a note of \p MI for later queries to isSafeToHoist.
  ///
  /// May clobber this HazardDetector instance.  \see isClobbered.
  void rememberInstruction(MachineInstr *MI);

  /// \brief Return true if it is safe to hoist \p MI from after all the
  /// instructions seen so far (via rememberInstruction) to before it.
  bool isSafeToHoist(MachineInstr *MI);

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

    if (MO.isDef())
      RegDefs.insert(MO.getReg());
    else
      RegUses.insert(MO.getReg());
  }
}

bool HazardDetector::isSafeToHoist(MachineInstr *MI) {
  assert(!isClobbered() && "isSafeToHoist cannot do anything useful!");

  // Right now we don't want to worry about LLVM's memory model.  This can be
  // made more precise later.
  for (auto *MMO : MI->memoperands())
    if (!MMO->isUnordered())
      return false;

  for (auto &MO : MI->operands()) {
    if (MO.isReg() && MO.getReg()) {
      for (unsigned Reg : RegDefs)
        if (TRI.regsOverlap(Reg, MO.getReg()))
          return false;  // We found a write-after-write or read-after-write

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

  SmallVector<NullCheck, 16> NullCheckList;

  for (auto &MBB : MF)
    analyzeBlockForNullChecks(MBB, NullCheckList);

  if (!NullCheckList.empty())
    rewriteNullChecks(NullCheckList);

  return !NullCheckList.empty();
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

  HazardDetector HD(*TRI);

  for (auto MII = NotNullSucc->begin(), MIE = NotNullSucc->end(); MII != MIE;
       ++MII) {
    MachineInstr *MI = &*MII;
    unsigned BaseReg, Offset;
    if (TII->getMemOpBaseRegImmOfs(MI, BaseReg, Offset, TRI))
      if (MI->mayLoad() && !MI->isPredicable() && BaseReg == PointerReg &&
          Offset < PageSize && MI->getDesc().getNumDefs() <= 1 &&
          HD.isSafeToHoist(MI)) {
        NullCheckList.emplace_back(MI, MBP.ConditionDef, &MBB, NotNullSucc,
                                   NullSucc);
        return true;
      }

    HD.rememberInstruction(MI);
    if (HD.isClobbered())
      return false;
  }

  return false;
}

/// Wrap a machine load instruction, LoadMI, into a FAULTING_LOAD_OP machine
/// instruction.  The FAULTING_LOAD_OP instruction does the same load as LoadMI
/// (defining the same register), and branches to HandlerLabel if the load
/// faults.  The FAULTING_LOAD_OP instruction is inserted at the end of MBB.
MachineInstr *ImplicitNullChecks::insertFaultingLoad(MachineInstr *LoadMI,
                                                     MachineBasicBlock *MBB,
                                                     MCSymbol *HandlerLabel) {
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
                 .addSym(HandlerLabel)
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
    MCSymbol *HandlerLabel = MMI->getContext().createTempSymbol();

    // Remove the conditional branch dependent on the null check.
    unsigned BranchesRemoved = TII->RemoveBranch(*NC.CheckBlock);
    (void)BranchesRemoved;
    assert(BranchesRemoved > 0 && "expected at least one branch!");

    // Insert a faulting load where the conditional branch was originally.  We
    // check earlier ensures that this bit of code motion is legal.  We do not
    // touch the successors list for any basic block since we haven't changed
    // control flow, we've just made it implicit.
    insertFaultingLoad(NC.MemOperation, NC.CheckBlock, HandlerLabel);
    NC.MemOperation->eraseFromParent();
    NC.CheckOperation->eraseFromParent();

    // Insert an *unconditional* branch to not-null successor.
    TII->InsertBranch(*NC.CheckBlock, NC.NotNullSucc, nullptr, /*Cond=*/None,
                      DL);

    // Emit the HandlerLabel as an EH_LABEL.
    BuildMI(*NC.NullSucc, NC.NullSucc->begin(), DL,
            TII->get(TargetOpcode::EH_LABEL)).addSym(HandlerLabel);

    NumImplicitNullChecks++;
  }
}

char ImplicitNullChecks::ID = 0;
char &llvm::ImplicitNullChecksID = ImplicitNullChecks::ID;
INITIALIZE_PASS_BEGIN(ImplicitNullChecks, "implicit-null-checks",
                      "Implicit null checks", false, false)
INITIALIZE_PASS_END(ImplicitNullChecks, "implicit-null-checks",
                    "Implicit null checks", false, false)
