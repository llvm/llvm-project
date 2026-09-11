//===-- EarlyIfConversion.cpp - If-conversion on SSA form machine code ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Early if-conversion is for out-of-order CPUs that don't have a lot of
// predicable instructions. The goal is to eliminate conditional branches that
// may mispredict.
//
// Instructions from both sides of the branch are executed specutatively, and a
// cmov instruction selects the result.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/EarlyIfConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineTraceMetrics.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "early-ifcvt"

// Absolute maximum number of instructions allowed per speculated block.
// This bypasses all other heuristics, so it should be set fairly high.
static cl::opt<unsigned>
BlockInstrLimit("early-ifcvt-limit", cl::init(30), cl::Hidden,
  cl::desc("Maximum number of instructions per speculated block."));

// Stress testing mode - disable heuristics.
static cl::opt<bool> Stress("stress-early-ifcvt", cl::Hidden,
  cl::desc("Turn all knobs to 11"));

// Enable analysis of data dependent branches (conditions derived from loads).
static cl::opt<bool> EnableDataDependentBranchAnalysis(
    "enable-early-ifcvt-data-dependent", cl::Hidden, cl::init(false),
    cl::desc("Enable hard-to-predict branch analysis for if-conversion"));

// Enable recognition of cascade if-else patterns (chains of triangles).
static cl::opt<bool> EnableCascadeIfConv(
    "enable-early-ifcvt-cascade", cl::Hidden, cl::init(false),
    cl::desc("Enable recognition of cascade if-else patterns"));

// Maximum depth for cascade patterns.
static cl::opt<unsigned>
    MaxCascadeDepth("early-ifcvt-max-cascade-depth", cl::Hidden, cl::init(8),
                    cl::desc("Maximum cascade depth for early if-conversion"));

// Limit the number steps we take when searching conditions that depend on
// values recently loaded from memory.
static cl::opt<unsigned>
    MaxNumSteps("early-ifcvt-max-steps", cl::Hidden, cl::init(16),
                cl::desc("Limit the number of steps taken when searching for a "
                         "recently loaded value"));

// Limit the work done when looking for calls between a load and the condition
// it feeds.
static cl::opt<unsigned> MaxRegionInstrs(
    "early-ifcvt-max-region-instrs", cl::Hidden, cl::init(64),
    cl::desc("Limit the number of blocks and instructions examined when "
             "searching for calls between a load and the condition it feeds"));

STATISTIC(NumDiamondsSeen,  "Number of diamonds");
STATISTIC(NumDiamondsConv,  "Number of diamonds converted");
STATISTIC(NumTrianglesSeen, "Number of triangles");
STATISTIC(NumTrianglesConv, "Number of triangles converted");
STATISTIC(NumDataDependant,
          "Number of data dependent conditional branches encountered");
STATISTIC(NumLikelyBiased, "Number of branches with a hot path encountered");
STATISTIC(NumCascadesSeen, "Number of cascade patterns detected");
STATISTIC(NumCascadesConv, "Number of cascade patterns converted");

//===----------------------------------------------------------------------===//
//                                 SSAIfConv
//===----------------------------------------------------------------------===//
//
// The SSAIfConv class performs if-conversion on SSA form machine code after
// determining if it is possible. The class contains no heuristics; external
// code should be used to determine when if-conversion is a good idea.
//
// SSAIfConv can convert both triangles and diamonds:
//
//   Triangle: Head              Diamond: Head
//              | \                       /  \_
//              |  \                     /    |
//              |  [TF]BB              FBB    TBB
//              |  /                     \    /
//              | /                       \  /
//             Tail                       Tail
//
// Instructions in the conditional blocks TBB and/or FBB are spliced into the
// Head block, and phis in the Tail block are converted to select instructions.
//
namespace {

/// What one cascade collapse step would do to one Tail phi.
struct CascadeSelectInfo {
  int CondCycles = 0;
  bool NeedsSelect = false;
};

struct CascadeResult {
  MachineBasicBlock *Head = nullptr;
  SmallVector<MachineBasicBlock *> Blocks;
  SmallVector<SmallVector<CascadeSelectInfo, 4>> Selects;
  explicit operator bool() const { return Head != nullptr; }
};

class SSAIfConv {
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  MachineRegisterInfo *MRI;

public:
  /// The block containing the conditional branch.
  MachineBasicBlock *Head;

  /// The block containing phis after the if-then-else.
  MachineBasicBlock *Tail;

  /// The 'true' conditional block as determined by analyzeBranch.
  MachineBasicBlock *TBB;

  /// The 'false' conditional block as determined by analyzeBranch.
  MachineBasicBlock *FBB;

  /// isTriangle - When there is no 'else' block, either TBB or FBB will be
  /// equal to Tail.
  bool isTriangle() const { return TBB == Tail || FBB == Tail; }

  /// Returns the Tail predecessor for the True side.
  MachineBasicBlock *getTPred() const { return TBB == Tail ? Head : TBB; }

  /// Returns the Tail predecessor for the  False side.
  MachineBasicBlock *getFPred() const { return FBB == Tail ? Head : FBB; }

  /// Information about each phi in the Tail block.
  struct PHIInfo {
    MachineInstr *PHI;
    Register TReg, FReg;
    // Latencies from Cond+Branch, TReg, and FReg to DstReg.
    int CondCycles = 0, TCycles = 0, FCycles = 0;

    PHIInfo(MachineInstr *phi) : PHI(phi) {}
  };

  SmallVector<PHIInfo, 8> PHIs;

  /// The branch condition determined by analyzeBranch.
  SmallVector<MachineOperand, 4> Cond;

private:
  /// Instructions in Head that define values used by the conditional blocks.
  /// The hoisted instructions must be inserted after these instructions.
  SmallPtrSet<MachineInstr*, 8> InsertAfter;

  /// Register units clobbered by the conditional blocks.
  BitVector ClobberedRegUnits;

  // Scratch pad for findInsertionPoint.
  SparseSet<MCRegUnit, MCRegUnit, MCRegUnitToIndex> LiveRegUnits;

  /// Insertion point in Head for speculatively executed instructions form TBB
  /// and FBB.
  MachineBasicBlock::iterator InsertionPoint;

  /// Return true if all non-terminator instructions in MBB can be safely
  /// speculated.
  bool canSpeculateInstrs(MachineBasicBlock *MBB);

  /// Return true if all non-terminator instructions in MBB can be safely
  /// predicated.
  bool canPredicateInstrs(MachineBasicBlock *MBB);

  /// Scan through instruction dependencies and update InsertAfter array.
  /// Return false if any dependency is incompatible with if conversion.
  bool InstrDependenciesAllowIfConv(MachineInstr *I);

  /// Predicate all instructions of the basic block with current condition
  /// except for terminators. Reverse the condition if ReversePredicate is set.
  void PredicateBlock(MachineBasicBlock *MBB, bool ReversePredicate);

  /// Find a valid insertion point in Head.
  bool findInsertionPoint();

  /// Replace PHI instructions in Tail with selects.
  void replacePHIInstrs();

  /// Insert selects and rewrite PHI operands to use them.
  void rewritePHIOperands();

  /// If virtual register has "killed" flag in TBB and FBB basic blocks, remove
  /// the flag in TBB instruction.
  void clearRepeatedKillFlagsFromTBB(MachineBasicBlock *TBB,
                                     MachineBasicBlock *FBB);

public:
  /// init - Initialize per-function data structures.
  void init(MachineFunction &MF) {
    TII = MF.getSubtarget().getInstrInfo();
    TRI = MF.getSubtarget().getRegisterInfo();
    MRI = &MF.getRegInfo();
    LiveRegUnits.clear();
    LiveRegUnits.setUniverse(TRI->getNumRegUnits());
    ClobberedRegUnits.clear();
    ClobberedRegUnits.resize(TRI->getNumRegUnits());
  }

  /// canConvertIf - If the sub-CFG headed by MBB can be if-converted,
  /// initialize the internal state, and return true.
  /// If predicate is set try to predicate the block otherwise try to
  /// speculatively execute it.
  bool canConvertIf(MachineBasicBlock *MBB, bool Predicate = false,
                    bool AllowMultiSuccTBB = false,
                    MachineBasicBlock *ExplicitTail = nullptr);

  /// convertIf - If-convert the last block passed to canConvertIf(), assuming
  /// it is possible. Add any blocks that are to be erased to RemoveBlocks.
  void convertIf(SmallVectorImpl<MachineBasicBlock *> &RemoveBlocks,
                 bool Predicate = false);

  /// matchCascade - match a cascade of conditional branches by walking up
  /// from MBB as a potential cascade end. A cascade is a chain of triangles
  /// where each block has 2 successors (next cascade block + Tail) and the
  /// last block has 1 successor (Tail). Returns the discovered Head and list
  /// of cascade blocks [BB1, ..., BBn] if valid, empty result otherwise.
  ///
  ///   Head --------+
  ///   |            |
  ///   v            |
  ///  BB1 ----------+
  ///   |            |
  ///   v            |
  ///  BB2 ----------+
  ///   |            |
  ///       ...
  ///       ...
  ///   |            |
  ///   v            v
  ///  BBn (MBB) -> Tail
  ///
  CascadeResult matchCascade(MachineBasicBlock *MBB);
};
} // end anonymous namespace

/// canSpeculateInstrs - Returns true if all the instructions in MBB can safely
/// be speculated. The terminators are not considered.
///
/// If instructions use any values that are defined in the head basic block,
/// the defining instructions are added to InsertAfter.
///
/// Any clobbered regunits are added to ClobberedRegUnits.
///
bool SSAIfConv::canSpeculateInstrs(MachineBasicBlock *MBB) {
  // Reject any live-in physregs. It's probably CPSR/EFLAGS, and very hard to
  // get right.
  if (!MBB->livein_empty()) {
    LLVM_DEBUG(dbgs() << printMBBReference(*MBB) << " has live-ins.\n");
    return false;
  }

  unsigned InstrCount = 0;

  // Check all instructions, except the terminators. It is assumed that
  // terminators never have side effects or define any used register values.
  for (MachineInstr &MI :
       llvm::make_range(MBB->begin(), MBB->getFirstTerminator())) {
    if (MI.isDebugInstr())
      continue;

    if (++InstrCount > BlockInstrLimit && !Stress) {
      LLVM_DEBUG(dbgs() << printMBBReference(*MBB) << " has more than "
                        << BlockInstrLimit << " instructions.\n");
      return false;
    }

    // There shouldn't normally be any phis in a single-predecessor block.
    if (MI.isPHI()) {
      LLVM_DEBUG(dbgs() << "Can't hoist: " << MI);
      return false;
    }

    // Don't speculate loads. Note that it may be possible and desirable to
    // speculate GOT or constant pool loads that are guaranteed not to trap,
    // but we don't support that for now.
    if (MI.mayLoad()) {
      LLVM_DEBUG(dbgs() << "Won't speculate load: " << MI);
      return false;
    }

    // We never speculate stores, so an AA pointer isn't necessary.
    bool DontMoveAcrossStore = true;
    if (!MI.isSafeToMove(DontMoveAcrossStore)) {
      LLVM_DEBUG(dbgs() << "Can't speculate: " << MI);
      return false;
    }

    // Check for any dependencies on Head instructions.
    if (!InstrDependenciesAllowIfConv(&MI))
      return false;
  }
  return true;
}

/// Check that there is no dependencies preventing if conversion.
///
/// If instruction uses any values that are defined in the head basic block,
/// the defining instructions are added to InsertAfter.
bool SSAIfConv::InstrDependenciesAllowIfConv(MachineInstr *I) {
  for (const MachineOperand &MO : I->operands()) {
    if (MO.isRegMask()) {
      LLVM_DEBUG(dbgs() << "Won't speculate regmask: " << *I);
      return false;
    }
    if (!MO.isReg())
      continue;
    Register Reg = MO.getReg();

    // Remember clobbered regunits.
    if (MO.isDef() && Reg.isPhysical())
      for (MCRegUnit Unit : TRI->regunits(Reg.asMCReg()))
        ClobberedRegUnits.set(static_cast<unsigned>(Unit));

    if (!MO.readsReg() || !Reg.isVirtual())
      continue;
    MachineInstr *DefMI = MRI->getVRegDef(Reg);
    if (!DefMI || DefMI->getParent() != Head)
      continue;
    if (InsertAfter.insert(DefMI).second)
      LLVM_DEBUG(dbgs() << printMBBReference(*I->getParent()) << " depends on "
                        << *DefMI);
    if (DefMI->isTerminator()) {
      LLVM_DEBUG(dbgs() << "Can't insert instructions below terminator.\n");
      return false;
    }
  }
  return true;
}

/// canPredicateInstrs - Returns true if all the instructions in MBB can safely
/// be predicates. The terminators are not considered.
///
/// If instructions use any values that are defined in the head basic block,
/// the defining instructions are added to InsertAfter.
///
/// Any clobbered regunits are added to ClobberedRegUnits.
///
bool SSAIfConv::canPredicateInstrs(MachineBasicBlock *MBB) {
  // Reject any live-in physregs. It's probably CPSR/EFLAGS, and very hard to
  // get right.
  if (!MBB->livein_empty()) {
    LLVM_DEBUG(dbgs() << printMBBReference(*MBB) << " has live-ins.\n");
    return false;
  }

  unsigned InstrCount = 0;

  // Check all instructions, except the terminators. It is assumed that
  // terminators never have side effects or define any used register values.
  for (MachineBasicBlock::iterator I = MBB->begin(),
                                   E = MBB->getFirstTerminator();
       I != E; ++I) {
    if (I->isDebugInstr())
      continue;

    if (++InstrCount > BlockInstrLimit && !Stress) {
      LLVM_DEBUG(dbgs() << printMBBReference(*MBB) << " has more than "
                        << BlockInstrLimit << " instructions.\n");
      return false;
    }

    // There shouldn't normally be any phis in a single-predecessor block.
    if (I->isPHI()) {
      LLVM_DEBUG(dbgs() << "Can't predicate: " << *I);
      return false;
    }

    // Check that instruction is predicable
    if (!TII->isPredicable(*I)) {
      LLVM_DEBUG(dbgs() << "Isn't predicable: " << *I);
      return false;
    }

    // Check that instruction is not already predicated.
    if (TII->isPredicated(*I) && !TII->canPredicatePredicatedInstr(*I)) {
      LLVM_DEBUG(dbgs() << "Is already predicated: " << *I);
      return false;
    }

    // Check for any dependencies on Head instructions.
    if (!InstrDependenciesAllowIfConv(&(*I)))
      return false;
  }
  return true;
}

// Apply predicate to all instructions in the machine block.
void SSAIfConv::PredicateBlock(MachineBasicBlock *MBB, bool ReversePredicate) {
  auto Condition = Cond;
  if (ReversePredicate) {
    bool CanRevCond = !TII->reverseBranchCondition(Condition);
    assert(CanRevCond && "Reversed predicate is not supported");
    (void)CanRevCond;
  }
  // Terminators don't need to be predicated as they will be removed.
  for (MachineBasicBlock::iterator I = MBB->begin(),
                                   E = MBB->getFirstTerminator();
       I != E; ++I) {
    if (I->isDebugInstr())
      continue;
    TII->PredicateInstruction(*I, Condition);
  }
}

/// Find an insertion point in Head for the speculated instructions. The
/// insertion point must be:
///
/// 1. Before any terminators.
/// 2. After any instructions in InsertAfter.
/// 3. Not have any clobbered regunits live.
///
/// This function sets InsertionPoint and returns true when successful, it
/// returns false if no valid insertion point could be found.
///
bool SSAIfConv::findInsertionPoint() {
  // Keep track of live regunits before the current position.
  // Only track RegUnits that are also in ClobberedRegUnits.
  LiveRegUnits.clear();
  SmallVector<MCRegister, 8> Reads;
  MachineBasicBlock::iterator FirstTerm = Head->getFirstTerminator();
  MachineBasicBlock::iterator I = Head->end();
  MachineBasicBlock::iterator B = Head->begin();
  while (I != B) {
    --I;
    // Some of the conditional code depends in I.
    if (InsertAfter.count(&*I)) {
      LLVM_DEBUG(dbgs() << "Can't insert code after " << *I);
      return false;
    }

    // Update live regunits.
    for (const MachineOperand &MO : I->operands()) {
      // We're ignoring regmask operands. That is conservatively correct.
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      if (!Reg.isPhysical())
        continue;
      // I clobbers Reg, so it isn't live before I.
      if (MO.isDef())
        for (MCRegUnit Unit : TRI->regunits(Reg.asMCReg()))
          LiveRegUnits.erase(Unit);
      // Unless I reads Reg.
      if (MO.readsReg())
        Reads.push_back(Reg.asMCReg());
    }
    // Anything read by I is live before I.
    while (!Reads.empty())
      for (MCRegUnit Unit : TRI->regunits(Reads.pop_back_val()))
        if (ClobberedRegUnits.test(static_cast<unsigned>(Unit)))
          LiveRegUnits.insert(Unit);

    // We can't insert before a terminator.
    if (I != FirstTerm && I->isTerminator())
      continue;

    // Some of the clobbered registers are live before I, not a valid insertion
    // point.
    if (!LiveRegUnits.empty()) {
      LLVM_DEBUG({
        dbgs() << "Would clobber";
        for (MCRegUnit LRU : LiveRegUnits)
          dbgs() << ' ' << printRegUnit(LRU, TRI);
        dbgs() << " live before " << *I;
      });
      continue;
    }

    // This is a valid insertion point.
    InsertionPoint = I;
    LLVM_DEBUG(dbgs() << "Can insert before " << *I);
    return true;
  }
  LLVM_DEBUG(dbgs() << "No legal insertion point found.\n");
  return false;
}



/// canConvertIf - analyze the sub-cfg rooted in MBB, and return true if it is
/// a potential candidate for if-conversion. Fill out the internal state.
///
bool SSAIfConv::canConvertIf(MachineBasicBlock *MBB, bool Predicate,
                             bool AllowMultiSuccTBB,
                             MachineBasicBlock *ExplicitTail) {
  Head = MBB;
  TBB = FBB = Tail = nullptr;

  if (Head->succ_size() != 2)
    return false;
  MachineBasicBlock *Succ0 = Head->succ_begin()[0];
  MachineBasicBlock *Succ1 = Head->succ_begin()[1];

  // Canonicalize so Succ0 has MBB as its single predecessor.
  if (Succ0->pred_size() != 1)
    std::swap(Succ0, Succ1);

  if (Succ0->pred_size() != 1 ||
      (!AllowMultiSuccTBB && Succ0->succ_size() != 1))
    return false;

  // Use explicit tail if provided (for cascade validation), otherwise compute.
  if (ExplicitTail) {
    Tail = ExplicitTail;
    LLVM_DEBUG(dbgs() << "\nCascade triangle: " << printMBBReference(*Head)
                      << " -> " << printMBBReference(*Succ0) << " -> "
                      << printMBBReference(*Tail) << '\n');
  } else {
    Tail = Succ0->succ_begin()[0];

    // This is not a triangle.
    if (Tail != Succ1) {
      // Check for a diamond. We won't deal with any critical edges.
      if (Succ1->pred_size() != 1 || Succ1->succ_size() != 1 ||
          Succ1->succ_begin()[0] != Tail)
        return false;
      LLVM_DEBUG(dbgs() << "\nDiamond: " << printMBBReference(*Head) << " -> "
                        << printMBBReference(*Succ0) << "/"
                        << printMBBReference(*Succ1) << " -> "
                        << printMBBReference(*Tail) << '\n');

      // Live-in physregs are tricky to get right when speculating code.
      if (!Tail->livein_empty()) {
        LLVM_DEBUG(dbgs() << "Tail has live-ins.\n");
        return false;
      }
    } else {
      LLVM_DEBUG(dbgs() << "\nTriangle: " << printMBBReference(*Head) << " -> "
                        << printMBBReference(*Succ0) << " -> "
                        << printMBBReference(*Tail) << '\n');
    }
  }

  // This is a triangle or a diamond.
  // Skip if we cannot predicate and there are no phis skip as there must be
  // side effects that can only be handled with predication.
  if (!Predicate && (Tail->empty() || !Tail->front().isPHI())) {
    LLVM_DEBUG(dbgs() << "No phis in tail.\n");
    return false;
  }

  // The branch we're looking to eliminate must be analyzable.
  Cond.clear();
  if (TII->analyzeBranch(*Head, TBB, FBB, Cond)) {
    LLVM_DEBUG(dbgs() << "Branch not analyzable.\n");
    return false;
  }

  // This is weird, probably some sort of degenerate CFG.
  if (!TBB) {
    LLVM_DEBUG(dbgs() << "analyzeBranch didn't find conditional branch.\n");
    return false;
  }

  // Make sure the analyzed branch is conditional; one of the successors
  // could be a landing pad. (Empty landing pads can be generated on Windows.)
  if (Cond.empty()) {
    LLVM_DEBUG(dbgs() << "analyzeBranch found an unconditional branch.\n");
    return false;
  }

  // analyzeBranch doesn't set FBB on a fall-through branch.
  // Make sure it is always set.
  FBB = TBB == Succ0 ? Succ1 : Succ0;

  // Any phis in the tail block must be convertible to selects.
  PHIs.clear();
  MachineBasicBlock *TPred = getTPred();
  MachineBasicBlock *FPred = getFPred();
  for (MachineBasicBlock::iterator I = Tail->begin(), E = Tail->end();
       I != E && I->isPHI(); ++I) {
    PHIs.push_back(&*I);
    PHIInfo &PI = PHIs.back();
    // Find PHI operands corresponding to TPred and FPred.
    for (unsigned i = 1; i != PI.PHI->getNumOperands(); i += 2) {
      if (PI.PHI->getOperand(i+1).getMBB() == TPred)
        PI.TReg = PI.PHI->getOperand(i).getReg();
      if (PI.PHI->getOperand(i+1).getMBB() == FPred)
        PI.FReg = PI.PHI->getOperand(i).getReg();
    }
    assert(PI.TReg.isVirtual() && "Bad PHI");
    assert(PI.FReg.isVirtual() && "Bad PHI");

    // Get target information.
    if (!TII->canInsertSelect(*Head, Cond, PI.PHI->getOperand(0).getReg(),
                              PI.TReg, PI.FReg, PI.CondCycles, PI.TCycles,
                              PI.FCycles)) {
      LLVM_DEBUG(dbgs() << "Can't convert: " << *PI.PHI);
      return false;
    }
  }

  // Check that the conditional instructions can be speculated.
  InsertAfter.clear();
  ClobberedRegUnits.reset();
  if (Predicate) {
    if (TBB != Tail && !canPredicateInstrs(TBB))
      return false;
    if (FBB != Tail && !canPredicateInstrs(FBB))
      return false;
  } else {
    if (TBB != Tail && !canSpeculateInstrs(TBB))
      return false;
    if (FBB != Tail && !canSpeculateInstrs(FBB))
      return false;
  }

  // Try to find a valid insertion point for the speculated instructions in the
  // head basic block.
  if (!findInsertionPoint())
    return false;

  // ExplicitTail means this is a validation call. Don't count towards the
  // triangle / diamonds seen stat - that should happen only when we consider
  // them for conversion seperately.
  if (!ExplicitTail) {
    if (isTriangle())
      ++NumTrianglesSeen;
    else
      ++NumDiamondsSeen;
  }
  return true;
}

/// \return true iff the two registers are known to have the same value.
static bool hasSameValue(const MachineRegisterInfo &MRI,
                         const TargetInstrInfo *TII, Register TReg,
                         Register FReg) {
  if (TReg == FReg)
    return true;

  if (!TReg.isVirtual() || !FReg.isVirtual())
    return false;

  const MachineInstr *TDef = MRI.getUniqueVRegDef(TReg);
  const MachineInstr *FDef = MRI.getUniqueVRegDef(FReg);
  if (!TDef || !FDef)
    return false;

  // If there are side-effects, all bets are off.
  if (TDef->hasUnmodeledSideEffects())
    return false;

  // If the instruction could modify memory, or there may be some intervening
  // store between the two, we can't consider them to be equal.
  if (TDef->mayLoadOrStore() && !TDef->isDereferenceableInvariantLoad())
    return false;

  // We also can't guarantee that they are the same if, for example, the
  // instructions are both a copy from a physical reg, because some other
  // instruction may have modified the value in that reg between the two
  // defining insts.
  if (any_of(TDef->uses(), [](const MachineOperand &MO) {
        return MO.isReg() && MO.getReg().isPhysical();
      }))
    return false;

  // Check whether the two defining instructions produce the same value(s).
  if (!TII->produceSameValue(*TDef, *FDef, &MRI))
    return false;

  // Further, check that the two defs come from corresponding operands.
  int TIdx = TDef->findRegisterDefOperandIdx(TReg, /*TRI=*/nullptr);
  int FIdx = FDef->findRegisterDefOperandIdx(FReg, /*TRI=*/nullptr);
  if (TIdx == -1 || FIdx == -1)
    return false;

  return TIdx == FIdx;
}

/// matchCascade - match a cascade pattern by walking up from MBB.
/// MBB is a potential cascade end (last block before Tail).
/// A cascade is a chain of triangles where:
/// - Head has 2 successors: one to first cascade block, one to Tail
/// - Each cascade block has 2 successors: next cascade + Tail (or 1 for last)
/// - Last cascade block (MBB) has 1 successor: Tail
/// Returns the discovered Head and cascade blocks if valid, empty otherwise.
///
///   Head --------+
///   |            |
///   v            |
///  BB1 ----------+
///   |            |
///   v            |
///  BB2 ----------+
///   |            |
///       ...
///       ...
///   |            |
///   v            v
///  BBn (MBB) -> Tail
///
CascadeResult SSAIfConv::matchCascade(MachineBasicBlock *MBB) {
  // Cascade end must have exactly 1 successor (to Tail) and 1 predecessor
  if (MBB->succ_size() != 1 || MBB->pred_size() != 1)
    return {};

  MachineBasicBlock *Tail = MBB->succ_begin()[0];

  // Collect cascade blocks from bottom to top: [BBn, BB(n-1), ..., BB1, Head]
  SmallVector<MachineBasicBlock *> Blocks;
  Blocks.push_back(MBB);
  MachineBasicBlock *Current = MBB;

  // Walk up the cascade chain
  while (true) {
    MachineBasicBlock *Pred = Current->pred_begin()[0];

    // Predecessor must have 2 successors: Current and Tail
    if (Pred->succ_size() != 2)
      break;

    MachineBasicBlock *S0 = Pred->succ_begin()[0];
    MachineBasicBlock *S1 = Pred->succ_begin()[1];

    // One successor must be Current, the other must be Tail
    if (!((S0 == Current && S1 == Tail) || (S1 == Current && S0 == Tail)))
      break;

    Blocks.push_back(Pred);

    // If Pred has multiple predecessors, it's the Head - stop here
    if (Pred->pred_size() != 1)
      break;

    // Pred has 1 predecessor, so the cascade continues. If we've reached
    // max depth, bail out entirely - we don't want to convert a partial
    // cascade.
    if (Blocks.size() > MaxCascadeDepth) {
      LLVM_DEBUG(dbgs() << "Cascade extends beyond max depth "
                        << MaxCascadeDepth << ", not converting.\n");
      return {};
    }

    Current = Pred;
  }

  // Need at least 3 blocks: [BBn, BB1, Head] for a minimal 2-block cascade
  if (Blocks.size() < 3)
    return {};

  // Reverse to get [HEAD, BB1, BB2, ..., BBn] order
  std::reverse(Blocks.begin(), Blocks.end());

  // Check that we can actually convert all the blocks in the cascade.
  // We skip the last cascade block (BBn) since it has a single successor
  // (to Tail), and is only used as a TBB (not a HEAD) - its speculatability
  // is validated when we check its predecessor.
  SmallVector<SmallVector<CascadeSelectInfo, 4>> CollectedSelects;
  size_t NumBlocksToValidate = Blocks.size() - 1;
  for (size_t I = 0; I < NumBlocksToValidate; ++I) {
    if (!canConvertIf(Blocks[I], /*Predicate=*/false,
                      /*AllowMultiSuccTBB=*/true,
                      /*ExplicitTail=*/Tail)) {
      LLVM_DEBUG(dbgs() << "Cannot convert cascade, block "
                        << printMBBReference(*Blocks[I])
                        << " is not if-convertible.\n");
      return {};
    }
    assert((CollectedSelects.empty() ||
            CollectedSelects.back().size() == PHIs.size()) &&
           "Tail phi list changed between cascade blocks");

    // Capture PHI info from canConvertIf's canInsertSelect call.
    auto &StepSelects = CollectedSelects.emplace_back();
    for (const PHIInfo &PI : PHIs)
      StepSelects.push_back(
          {PI.CondCycles, !hasSameValue(*MRI, TII, PI.TReg, PI.FReg)});
  }

  // The first block in Blocks is the cascade Head
  MachineBasicBlock *Head = Blocks.front();

  // Remove Head from the cascade blocks to convert
  Blocks.erase(Blocks.begin());

  LLVM_DEBUG({
    dbgs() << "\nCascade found: " << printMBBReference(*Head) << " -> [";
    for (auto *BB : Blocks)
      dbgs() << printMBBReference(*BB) << ", ";
    dbgs() << "] -> " << printMBBReference(*Tail) << "\n";
  });

  return {Head, std::move(Blocks), std::move(CollectedSelects)};
}

/// replacePHIInstrs - Completely replace PHI instructions with selects.
/// This is possible when the only Tail predecessors are the if-converted
/// blocks.
void SSAIfConv::replacePHIInstrs() {
  assert(Tail->pred_size() == 2 && "Cannot replace PHIs");
  MachineBasicBlock::iterator FirstTerm = Head->getFirstTerminator();
  assert(FirstTerm != Head->end() && "No terminators");
  DebugLoc HeadDL = FirstTerm->getDebugLoc();

  // Convert all PHIs to select instructions inserted before FirstTerm.
  for (PHIInfo &PI : PHIs) {
    LLVM_DEBUG(dbgs() << "If-converting " << *PI.PHI);
    Register DstReg = PI.PHI->getOperand(0).getReg();
    if (hasSameValue(*MRI, TII, PI.TReg, PI.FReg)) {
      // We do not need the select instruction if both incoming values are
      // equal, but we do need a COPY.
      BuildMI(*Head, FirstTerm, HeadDL, TII->get(TargetOpcode::COPY), DstReg)
          .addReg(PI.TReg);
    } else {
      TII->insertSelect(*Head, FirstTerm, HeadDL, DstReg, Cond, PI.TReg,
                        PI.FReg);
    }
    LLVM_DEBUG(dbgs() << "          --> " << *std::prev(FirstTerm));
    PI.PHI->eraseFromParent();
    PI.PHI = nullptr;
  }
}

/// rewritePHIOperands - When there are additional Tail predecessors, insert
/// select instructions in Head and rewrite PHI operands to use the selects.
/// Keep the PHI instructions in Tail to handle the other predecessors.
void SSAIfConv::rewritePHIOperands() {
  MachineBasicBlock::iterator FirstTerm = Head->getFirstTerminator();
  assert(FirstTerm != Head->end() && "No terminators");
  DebugLoc HeadDL = FirstTerm->getDebugLoc();

  // Convert all PHIs to select instructions inserted before FirstTerm.
  for (PHIInfo &PI : PHIs) {
    Register DstReg;

    LLVM_DEBUG(dbgs() << "If-converting " << *PI.PHI);
    if (hasSameValue(*MRI, TII, PI.TReg, PI.FReg)) {
      // We do not need the select instruction if both incoming values are
      // equal.
      DstReg = PI.TReg;
    } else {
      Register PHIDst = PI.PHI->getOperand(0).getReg();
      DstReg = MRI->createVirtualRegister(MRI->getRegClass(PHIDst));
      TII->insertSelect(*Head, FirstTerm, HeadDL,
                         DstReg, Cond, PI.TReg, PI.FReg);
      LLVM_DEBUG(dbgs() << "          --> " << *std::prev(FirstTerm));
    }

    // Rewrite PHI operands TPred -> (DstReg, Head), remove FPred.
    for (unsigned i = PI.PHI->getNumOperands(); i != 1; i -= 2) {
      MachineBasicBlock *MBB = PI.PHI->getOperand(i-1).getMBB();
      if (MBB == getTPred()) {
        PI.PHI->getOperand(i-1).setMBB(Head);
        PI.PHI->getOperand(i-2).setReg(DstReg);
      } else if (MBB == getFPred()) {
        PI.PHI->removeOperand(i-1);
        PI.PHI->removeOperand(i-2);
      }
    }
    LLVM_DEBUG(dbgs() << "          --> " << *PI.PHI);
  }
}

void SSAIfConv::clearRepeatedKillFlagsFromTBB(MachineBasicBlock *TBB,
                                              MachineBasicBlock *FBB) {
  assert(TBB != FBB);

  // Collect virtual registers killed in FBB.
  SmallDenseSet<Register> FBBKilledRegs;
  for (MachineInstr &MI : FBB->instrs()) {
    for (MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isKill() && MO.getReg().isVirtual())
        FBBKilledRegs.insert(MO.getReg());
    }
  }

  if (FBBKilledRegs.empty())
    return;

  // Find the same killed registers in TBB and clear kill flags for them.
  for (MachineInstr &MI : TBB->instrs()) {
    for (MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isKill() && FBBKilledRegs.contains(MO.getReg()))
        MO.setIsKill(false);
    }
  }
}

/// convertIf - Execute the if conversion after canConvertIf has determined the
/// feasibility.
///
/// Any basic blocks that need to be erased will be added to RemoveBlocks.
///
void SSAIfConv::convertIf(SmallVectorImpl<MachineBasicBlock *> &RemoveBlocks,
                          bool Predicate) {
  assert(Head && Tail && TBB && FBB && "Call canConvertIf first.");

  // Update statistics.
  if (isTriangle())
    ++NumTrianglesConv;
  else
    ++NumDiamondsConv;

  // If both blocks are going to be merged into Head, remove "killed" flag in
  // TBB for registers, which are killed in TBB and FBB. Otherwise, register
  // will be killed twice in Head after splice. Register killed twice is an
  // incorrect MIR.
  if (TBB != Tail && FBB != Tail)
    clearRepeatedKillFlagsFromTBB(TBB, FBB);

  // Move all instructions into Head, except for the terminators.
  if (TBB != Tail) {
    if (Predicate)
      PredicateBlock(TBB, /*ReversePredicate=*/false);
    Head->splice(InsertionPoint, TBB, TBB->begin(), TBB->getFirstTerminator());
  }
  if (FBB != Tail) {
    if (Predicate)
      PredicateBlock(FBB, /*ReversePredicate=*/true);
    Head->splice(InsertionPoint, FBB, FBB->begin(), FBB->getFirstTerminator());
  }
  // Are there extra Tail predecessors?
  bool ExtraPreds = Tail->pred_size() != 2;
  if (ExtraPreds)
    rewritePHIOperands();
  else
    replacePHIInstrs();

  // Fix up the CFG, temporarily leave Head without any successors.
  Head->removeSuccessor(TBB);
  Head->removeSuccessor(FBB, true);
  if (TBB != Tail)
    TBB->removeSuccessor(Tail, true);
  if (FBB != Tail)
    FBB->removeSuccessor(Tail, true);

  // Fix up Head's terminators.
  // It should become a single branch or a fallthrough.
  DebugLoc HeadDL = Head->getFirstTerminator()->getDebugLoc();
  TII->removeBranch(*Head);

  // Mark the now empty conditional blocks for removal and move them to the end.
  // It is likely that Head can fall
  // through to Tail, and we can join the two blocks.
  if (TBB != Tail) {
    RemoveBlocks.push_back(TBB);
    if (TBB != &TBB->getParent()->back())
      TBB->moveAfter(&TBB->getParent()->back());
  }
  if (FBB != Tail) {
    RemoveBlocks.push_back(FBB);
    if (FBB != &FBB->getParent()->back())
      FBB->moveAfter(&FBB->getParent()->back());
  }

  assert(Head->succ_empty() && "Additional head successors?");
  if (!ExtraPreds && Head->isLayoutSuccessor(Tail)) {
    // Splice Tail onto the end of Head.
    LLVM_DEBUG(dbgs() << "Joining tail " << printMBBReference(*Tail)
                      << " into head " << printMBBReference(*Head) << '\n');
    Head->splice(Head->end(), Tail,
                     Tail->begin(), Tail->end());
    Head->transferSuccessorsAndUpdatePHIs(Tail);
    RemoveBlocks.push_back(Tail);
    if (Tail != &Tail->getParent()->back())
      Tail->moveAfter(&Tail->getParent()->back());
  } else {
    // We need a branch to Tail, let code placement work it out later.
    LLVM_DEBUG(dbgs() << "Converting to unconditional branch.\n");
    SmallVector<MachineOperand, 0> EmptyCond;
    TII->insertBranch(*Head, Tail, nullptr, EmptyCond, HeadDL);
    Head->addSuccessor(Tail);
  }
  LLVM_DEBUG(dbgs() << *Head);
}

//===----------------------------------------------------------------------===//
//                           EarlyIfConverter Pass
//===----------------------------------------------------------------------===//

namespace {
class EarlyIfConverter {
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const TargetSubtargetInfo *STI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineDominatorTree *DomTree = nullptr;
  MachineLoopInfo *Loops = nullptr;
  MachineTraceMetrics *Traces = nullptr;
  MachineTraceMetrics::Ensemble *MinInstr = nullptr;
  MachineBranchProbabilityInfo *MBPI = nullptr;
  SSAIfConv IfConv;

  /// Cache of basic blocks verified to contain no call instructions, mapping
  /// each block to the number of instructions scanned in it.
  DenseMap<const MachineBasicBlock *, unsigned> NoCallBlocksCache;

  /// Set of blocks that must be converted (part of a cascade).
  /// These blocks bypass normal profitability checks in shouldConvertIf().
  SmallPtrSet<MachineBasicBlock *, 16> MustConvertBlocks;

public:
  EarlyIfConverter(MachineDominatorTree &DT, MachineLoopInfo &LI,
                   MachineTraceMetrics &MTM, MachineBranchProbabilityInfo *MBPI)
      : DomTree(&DT), Loops(&LI), Traces(&MTM), MBPI(MBPI) {}
  EarlyIfConverter() = delete;

  bool run(MachineFunction &MF);

private:
  bool tryConvertIf(MachineBasicBlock *);
  void detectCascades(MachineBasicBlock *);
  void convertIf();
  void invalidateTraces();
  bool shouldConvertIf();
  bool isConditionDataDependent(MachineBasicBlock *BB, bool RecordStats = true);
  bool isCascadeDataDependent(MachineBasicBlock *Head,
                              ArrayRef<MachineBasicBlock *> CascadeBlocks);
  bool doOperandsComeFromMemory(const MachineInstr *ConditionDef,
                                MachineBasicBlock *BB);
  bool hasCallOrLoopInRange(const MachineInstr *From, const MachineInstr *To);
  bool shouldConvertCascade(CascadeResult &Cascade, MachineBasicBlock *Tail);
  bool hasEnoughILP(MachineBasicBlock *TraceBlock,
                    SmallVectorImpl<MachineBasicBlock *> &ExtraBlocks,
                    unsigned CritLimit);
};

class EarlyIfConverterLegacy : public MachineFunctionPass {
public:
  static char ID;
  EarlyIfConverterLegacy() : MachineFunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override { return "Early If-Conversion"; }
};
} // end anonymous namespace

char EarlyIfConverterLegacy::ID = 0;
char &llvm::EarlyIfConverterLegacyID = EarlyIfConverterLegacy::ID;

INITIALIZE_PASS_BEGIN(EarlyIfConverterLegacy, DEBUG_TYPE, "Early If Converter",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineTraceMetricsWrapperPass)
INITIALIZE_PASS_END(EarlyIfConverterLegacy, DEBUG_TYPE, "Early If Converter",
                    false, false)

void EarlyIfConverterLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreserved<MachineRegisterClassInfoWrapperPass>();
  AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  AU.addRequired<MachineTraceMetricsWrapperPass>();
  AU.addPreserved<MachineTraceMetricsWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

namespace {
/// Update the dominator tree after if-conversion erased some blocks.
void updateDomTree(MachineDominatorTree *DomTree, const SSAIfConv &IfConv,
                   ArrayRef<MachineBasicBlock *> Removed) {
  // convertIf can remove TBB, FBB, and Tail can be merged into Head.
  // TBB and FBB should not dominate any blocks.
  // Tail children should be transferred to Head.
  MachineDomTreeNode *HeadNode = DomTree->getNode(IfConv.Head);
  for (auto *B : Removed) {
    MachineDomTreeNode *Node = DomTree->getNode(B);
    assert(Node != HeadNode && "Cannot erase the head node");
    while (!Node->isLeaf()) {
      assert(Node->getBlock() == IfConv.Tail && "Unexpected children");
      DomTree->changeImmediateDominator(*Node->begin(), HeadNode);
    }
    DomTree->eraseNode(B);
  }
}

/// Update LoopInfo after if-conversion.
void updateLoops(MachineLoopInfo *Loops,
                 ArrayRef<MachineBasicBlock *> Removed) {
  // If-conversion doesn't change loop structure, and it doesn't mess with back
  // edges, so updating LoopInfo is simply removing the dead blocks.
  for (auto *B : Removed)
    Loops->removeBlock(B);
}
} // namespace

/// Invalidate MachineTraceMetrics before if-conversion.
void EarlyIfConverter::invalidateTraces() {
  Traces->verifyAnalysis();
  Traces->invalidate(IfConv.Head);
  Traces->invalidate(IfConv.Tail);
  Traces->invalidate(IfConv.TBB);
  Traces->invalidate(IfConv.FBB);
  Traces->verifyAnalysis();
}

static bool isConstantPoolLoad(const MachineInstr *MI) {
  return MI->mayLoad() && any_of(MI->memoperands(), [](MachineMemOperand *MOp) {
           const PseudoSourceValue *PSV = MOp->getPseudoValue();
           return PSV && PSV->isConstantPool();
         });
}

/// Check whether the load in From and the condition in To are far apart, i.e.
/// whether a call or a loop can be executed between them. This is done by first
/// scanning the instructions within From and To MBBs. If no call is found, we
/// then scan all blocks which are dominated by From (the load) and can reach To
/// (the condition), looking for calls and for blocks belonging to a loop the
/// condition is not part of.
bool EarlyIfConverter::hasCallOrLoopInRange(const MachineInstr *From,
                                            const MachineInstr *To) {
  if (From == To)
    return false;

  LLVM_DEBUG(dbgs() << "  checking for a call or loop between " << *From
                    << "  and " << *To);
  assert(DomTree->dominates(From, To) && "From is expected to dominate To");

  const MachineBasicBlock *FromBB = From->getParent();
  const MachineBasicBlock *ToBB = To->getParent();

  unsigned NumScanned = 0;
  auto HitSearchLimit = [&](unsigned N) {
    NumScanned += N;
    if (NumScanned <= MaxRegionInstrs)
      return false;
    LLVM_DEBUG(dbgs() << "  hasCallOrLoopInRange scanned more than "
                      << MaxRegionInstrs << " instructions\n");
    return true;
  };
  auto FoundCall = [](const MachineInstr &MI) {
    LLVM_DEBUG(dbgs() << "  found a call before the condition: " << MI);
    return true;
  };
  auto IsCallOrHitSearchLimit = [&](const MachineInstr &MI) {
    if (HitSearchLimit(1))
      return true;
    if (!MI.isCall())
      return false;
    return FoundCall(MI);
  };

  // If From and To are in the same block, just check (From, To).
  if (FromBB == ToBB) {
    for (const MachineInstr &MI :
         make_range(std::next(From->getIterator()), To->getIterator()))
      if (IsCallOrHitSearchLimit(MI))
        return true;
    return false;
  }

  // Check (From, end of From's block] and [start of To's block, To).
  for (const MachineInstr &MI :
       make_range(std::next(From->getIterator()), FromBB->instr_end()))
    if (IsCallOrHitSearchLimit(MI))
      return true;
  for (const MachineInstr &MI :
       make_range(ToBB->instr_begin(), To->getIterator()))
    if (IsCallOrHitSearchLimit(MI))
      return true;

  // Enqueued guards the traversal: the endpoint blocks are traversed through
  // but their instructions were already handled above.
  SmallPtrSet<const MachineBasicBlock *, 16> Enqueued = {FromBB, ToBB};
  SmallVector<const MachineBasicBlock *, 16> Worklist;
  auto Enqueue = [&](const MachineBasicBlock *BB) {
    if (DomTree->dominates(FromBB, BB) && Enqueued.insert(BB).second)
      Worklist.push_back(BB);
  };

  for (const MachineBasicBlock *Pred : ToBB->predecessors())
    Enqueue(Pred);

  while (!Worklist.empty()) {
    const MachineBasicBlock *BB = Worklist.pop_back_val();

    // If the block belongs to a loop containing neither the load nor the
    // condition, that loop is executed entirely between the two, so consider
    // them far apart.
    if (const MachineLoop *BBLoop = Loops->getLoopFor(BB)) {
      if (!BBLoop->contains(ToBB) && !BBLoop->contains(FromBB)) {
        LLVM_DEBUG(dbgs() << "  found a loop before the condition in "
                          << printMBBReference(*BB) << '\n');
        return true;
      }
    }

    // Next check for calls in the block.
    auto CacheIt = NoCallBlocksCache.find(BB);
    if (CacheIt != NoCallBlocksCache.end()) {
      if (HitSearchLimit(CacheIt->second))
        return true;
    } else {
      for (const MachineInstr &MI : *BB)
        if (IsCallOrHitSearchLimit(MI))
          return true;
      NoCallBlocksCache[BB] = BB->size();
    }

    for (const MachineBasicBlock *Pred : BB->predecessors())
      Enqueue(Pred);
  }

  return false;
}

/// Check if a register's value comes from a memory load by walking the
/// def-use chain. We want to prioritize converting branches which
/// depend on values loaded from memory (unless they are loop invariant,
/// or come from a constant pool). The walk starts from the definition of
/// ConditionDef's first operand, which is not ConditionDef itself for
/// instructions such as FCMPSrr, where that operand is a use. BB is the block
/// whose terminator consumes the condition.
bool EarlyIfConverter::doOperandsComeFromMemory(
    const MachineInstr *ConditionDef, MachineBasicBlock *BB) {
  Register Reg = ConditionDef->getOperand(0).getReg();
  if (!Reg.isVirtual())
    return false;

  LLVM_DEBUG(dbgs() << "  doOperandsComeFromMemory starting from reg "
                    << printReg(Reg) << "\n");

  // The condition is consumed by the branch terminating BB, so this is the
  // end of the interval a load has to survive without a call in between.
  const MachineInstr *Br = &*BB->getFirstTerminator();
  MachineLoop *IfConvLoop = Loops->getLoopFor(BB);

  // Walk the def-use chain.
  SmallPtrSet<const MachineInstr *, 8> VisitedInstrs;
  SmallVector<const MachineInstr *> Worklist;

  MachineInstr *DefMI = MRI->getVRegDef(Reg);
  // The operand is defined outside of the function - it does not
  // come from memory access.
  if (!DefMI)
    return false;

  Worklist.push_back(DefMI);

  while (!Worklist.empty() && VisitedInstrs.size() < MaxNumSteps) {
    const MachineInstr *MI = Worklist.pop_back_val();
    if (!VisitedInstrs.insert(MI).second)
      continue;

    // Don't walk through PHIs: a value arriving on a back edge is loaded in a
    // previous iteration, so the interval between the load and the branch is
    // not the one hasCallOrLoopInRange measures.
    if (MI->isPHI())
      continue;

    const MachineBasicBlock *Parent = MI->getParent();
    MachineLoop *ParentLoop = Loops->getLoopFor(Parent);

    // If the instruction is outside the loop, skip it (loop-invariant).
    if (IfConvLoop && ParentLoop != IfConvLoop)
      continue;

    // Check if this instruction is a load, and there are no calls or loops
    // between the load and the condition (which would break the "close in
    // time" assumption).
    if (MI->mayLoad() && !isConstantPoolLoad(MI) &&
        !MI->isDereferenceableInvariantLoad()) {
      // If the load doesn't dominate the branch (e.g., comes after it in
      // the same block via a loop back-edge), it can't affect this iteration.
      // If not - check if there is a call or a loop between the load
      // instruction and the branch.
      if (!DomTree->dominates(MI, Br) || hasCallOrLoopInRange(MI, Br))
        continue;

      return true;
    }

    // Walk through all register use operands and find their definitions.
    for (const MachineOperand &MO : MI->operands()) {
      if (!MO.isReg() || !MO.isUse())
        continue;
      Register UseReg = MO.getReg();
      if (!UseReg.isVirtual())
        continue;

      if (MachineInstr *UseDef = MRI->getVRegDef(UseReg)) {
        if (!VisitedInstrs.count(UseDef)) {
          Worklist.push_back(UseDef);
        }
      }
    }
  }

  return false;
}

/// Check if the branch condition is data-dependent (comes from memory loads).
/// RecordStats should be false when the same branch may be examined again
/// later, so that it is only counted once.
bool EarlyIfConverter::isConditionDataDependent(MachineBasicBlock *BB,
                                                bool RecordStats) {
  TargetInstrInfo::MachineBranchPredicate MBP;
  if (TII->analyzeBranchPredicate(*BB, MBP, /*AllowModify=*/false))
    return false;

  if (!MBP.ConditionDef)
    return false;

  // If the branch is biased (not 50/50), don't consider it data dependent.
  // This is to prevent converting unprofitable checks such as
  // `x[i] != 0;`
  if (MBP.TrueDest && MBP.FalseDest && MBPI) {
    auto TBBProb = MBPI->getEdgeProbability(BB, MBP.TrueDest);
    auto FBBProb = MBPI->getEdgeProbability(BB, MBP.FalseDest);
    if (TBBProb != FBBProb) {
      if (RecordStats)
        ++NumLikelyBiased;
      return false;
    }
  }

  // Check if operands used to compute the branch condition were loaded recently
  // from memory, starting by the ConditionDef itself and walking up the use-def
  // chain.
  if (doOperandsComeFromMemory(MBP.ConditionDef, BB)) {
    if (RecordStats)
      ++NumDataDependant;
    return true;
  }

  return false;
}

/// Check if ALL branches in a cascade are data-dependent (come from loads).
/// Cascade conversion always requires data-dependent branches, so this check
/// is enabled whenever cascade conversion is enabled.
bool EarlyIfConverter::isCascadeDataDependent(
    MachineBasicBlock *Head, ArrayRef<MachineBasicBlock *> CascadeBlocks) {

  // Check Head block
  if (!isConditionDataDependent(Head, /*RecordStats=*/false)) {
    LLVM_DEBUG(dbgs() << "Cascade: Head not data-dependent\n");
    return false;
  }

  // Check all cascade blocks except the last one (which has only 1 successor).
  // Don't record stats because we are not converting anything yet.
  for (auto *CascadeBlock : CascadeBlocks.drop_back()) {
    if (!isConditionDataDependent(CascadeBlock, /*RecordStats=*/false)) {
      LLVM_DEBUG(dbgs() << "Cascade: " << printMBBReference(*CascadeBlock)
                        << " not data-dependent\n");
      return false;
    }
  }

  LLVM_DEBUG(dbgs() << "Cascade: all branches are data-dependent\n");
  return true;
}

namespace {
/// Helper class to simplify emission of cycle counts into optimization remarks.
struct Cycles {
  const char *Key;
  unsigned Value;
  Cycles(const char *K, unsigned V) : Key(K), Value(V) {}
};
template <typename Remark> Remark &operator<<(Remark &R, Cycles C) {
  return R << ore::NV(C.Key, C.Value) << (C.Value == 1 ? " cycle" : " cycles");
}
} // anonymous namespace

// Adjust cycles with downward saturation.
static unsigned adjCycles(unsigned Cyc, int Delta) {
  if (Delta < 0 && Cyc + Delta > Cyc)
    return 0;
  return Cyc + Delta;
}

/// Count the instructions of MBB that a predecessor would have to speculate to
/// absorb it. This matches what canSpeculateInstrs() checks against
/// BlockInstrLimit: the non-debug instructions ahead of the terminators.
/// If we don't perform this check we could bail out of a cascade conversion
/// midway by exceeding `BlockInstrLimit`.
static unsigned countSpeculatedInstrs(const MachineBasicBlock &MBB) {
  return count_if(make_range(MBB.begin(), MBB.getFirstTerminator()),
                  [](const MachineInstr &MI) { return !MI.isDebugInstr(); });
}

/// Apply profitability check for cascade conversion.
bool EarlyIfConverter::shouldConvertCascade(CascadeResult &Cascade,
                                            MachineBasicBlock *Tail) {
  MachineBasicBlock *Head = Cascade.Head;
  auto &CascadeBlocks = Cascade.Blocks;

  if (!isCascadeDataDependent(Head, CascadeBlocks)) {
    LLVM_DEBUG(dbgs() << "Cascade: not all branches are data-dependent\n");
    return false;
  }

  // Calculate CritLimit using cascade formula.
  unsigned CascadeSize = CascadeBlocks.size();
  unsigned MispredictPenalty = STI->getMispredictionPenalty();
  unsigned CritLimit = std::min(MispredictPenalty * 25 * CascadeSize / 100,
                                2 * MispredictPenalty);

  LLVM_DEBUG(dbgs() << "Cascade: size=" << CascadeSize
                    << ", CritLimit=" << CritLimit
                    << " (MispredictPenalty=" << MispredictPenalty << ")\n");

  if (CritLimit == 0) {
    LLVM_DEBUG(dbgs() << "Cascade: CritLimit is 0, skipping\n");
    return false;
  }

  MachineOptimizationRemarkEmitter MORE(*Head->getParent(), nullptr);
  SmallVector<MachineBasicBlock *, 8> CondBlocks;
  CondBlocks.push_back(Head);
  append_range(CondBlocks, ArrayRef(CascadeBlocks).drop_back());
  assert(CondBlocks.size() == Cascade.Selects.size() &&
         "Mismatched cascade condition info");

  unsigned NumPHIs = Cascade.Selects.front().size();
  assert(NumPHIs == range_size(Tail->phis()) &&
         "Tail phi list changed since matchCascade()");

  // A step only needs a select if the phi's two incoming values differ. But
  // once some deeper step has produced a select, every shallower step takes
  // that select's result as an incoming value and needs one too.
  SmallVector<int, 4> DeepestSelectStep(NumPHIs, -1);
  for (auto [I, Step] : enumerate(Cascade.Selects))
    for (auto [J, Sel] : enumerate(Step))
      if (Sel.NeedsSelect)
        DeepestSelectStep[J] = I;

  if (!Stress) {
    unsigned NumSelects = 0;
    for (int Deepest : DeepestSelectStep)
      NumSelects += Deepest + 1;
    unsigned TotalInstrs = NumSelects;
    for (MachineBasicBlock *BB : CascadeBlocks)
      TotalInstrs += countSpeculatedInstrs(*BB);

    LLVM_DEBUG(dbgs() << "Cascade: collapsed cascade holds " << TotalInstrs
                      << " instructions (" << NumSelects
                      << " of them selects), limit is " << BlockInstrLimit
                      << '\n');

    if (TotalInstrs > BlockInstrLimit) {
      LLVM_DEBUG(dbgs() << "Cascade: collapsed cascade exceeds instruction "
                           "limit, skipping\n");
      MORE.emit([&]() {
        return MachineOptimizationRemarkMissed(DEBUG_TYPE,
                                               "CascadeIfConversion",
                                               Head->back().getDebugLoc(), Head)
               << "did not if-convert cascade with "
               << ore::NV("CascadeSize", CascadeSize)
               << " blocks: the speculated blocks would hold "
               << ore::NV("TotalInstrs", TotalInstrs)
               << " instructions, exceeding the limit of "
               << ore::NV("BlockInstrLimit", BlockInstrLimit) << ".";
      });
      return false;
    }
  }

  // Check if there's enough ILP to hide the speculated cascade blocks.
  if (!hasEnoughILP(Tail, CascadeBlocks, CritLimit)) {
    LLVM_DEBUG(dbgs() << "Cascade: not enough ILP, skipping\n");
    MORE.emit([&]() {
      return MachineOptimizationRemarkMissed(DEBUG_TYPE, "CascadeIfConversion",
                                             Head->back().getDebugLoc(), Head)
             << "did not if-convert cascade with "
             << ore::NV("CascadeSize", CascadeSize)
             << " blocks: not enough ILP to hide speculated instructions.";
    });
    return false;
  }

  // Compute CSEL chain critical path depth.
  // After conversion, we get a chain of CSELs where each CSEL depends on:
  //   1. The condition from that block's branch
  //   2. The previous CSEL's output (chain dependency)
  // The chain is rooted at the deepest cascade block and ends at Head, which is
  // the order convertIf() creates the selects in.
  if (!MinInstr)
    MinInstr = Traces->getEnsemble(MachineTraceStrategy::TS_MinInstrCount);

  MachineTraceMetrics::Trace TailTrace = MinInstr->getTrace(Tail);

  SmallVector<unsigned, 8> BranchDepths;
  for (MachineBasicBlock *CondBlock : CondBlocks) {
    MachineTraceMetrics::Trace BlockTrace = MinInstr->getTrace(CondBlock);
    BranchDepths.push_back(
        BlockTrace.getInstrCycles(*CondBlock->getFirstTerminator()).Depth);
  }

  // Each phi gets its own budget and its own chain: select latency depends on
  // the phi's register class, and a phi with a lot of slack must not raise the
  // bar for a phi with none. All of them have to be profitable.
  unsigned MaxExtension = 0;
  bool ExceedsLimit = false;
  for (auto [J, PHI] : enumerate(Tail->phis())) {
    unsigned Slack = TailTrace.getInstrSlack(PHI);
    unsigned MaxDepth = Slack + TailTrace.getInstrCycles(PHI).Depth;

    // convertIf() collapses the cascade bottom-up, so the deepest block's
    // condition roots the CSEL chain and Head's condition feeds the last
    // select.
    LLVM_DEBUG(dbgs() << "CSEL chain depth computation for " << PHI);
    unsigned CSELChainDepth = 0;
    for (int I = DeepestSelectStep[J]; I >= 0; --I) {
      int CondCycles = Cascade.Selects[I][J].CondCycles;
      CSELChainDepth =
          adjCycles(std::max(BranchDepths[I], CSELChainDepth), CondCycles);

      LLVM_DEBUG(dbgs() << "  " << printMBBReference(*CondBlocks[I])
                        << ": branch depth=" << BranchDepths[I]
                        << ", CondCycles=" << CondCycles
                        << ", new CSEL depth=" << CSELChainDepth << "\n");
    }

    unsigned Extension =
        CSELChainDepth > MaxDepth ? CSELChainDepth - MaxDepth : 0;
    MaxExtension = std::max(MaxExtension, Extension);

    LLVM_DEBUG(dbgs() << "Final CSEL chain depth: " << CSELChainDepth
                      << ", MaxDepth: " << MaxDepth << ", Extension: "
                      << Extension << ", CritLimit: " << CritLimit << '\n');

    if (Extension > CritLimit) {
      LLVM_DEBUG(dbgs() << "Cascade: critical path extension exceeds "
                           "limit, skipping\n");
      ExceedsLimit = true;
    }
  }

  if (ExceedsLimit) {
    MORE.emit([&]() {
      MachineOptimizationRemarkMissed R(DEBUG_TYPE, "CascadeIfConversion",
                                        Head->back().getDebugLoc(), Head);
      R << "did not if-convert cascade with "
        << ore::NV("CascadeSize", CascadeSize)
        << " blocks: critical path extension of "
        << Cycles("CritPathExtension", MaxExtension)
        << " exceeds the threshold of " << Cycles("CritLimit", CritLimit)
        << ".";
      return R;
    });
    return false;
  }

  MORE.emit([&]() {
    MachineOptimizationRemark R(DEBUG_TYPE, "CascadeIfConversion",
                                Head->back().getDebugLoc(), Head);
    R << "performing cascade if-conversion on "
      << ore::NV("CascadeSize", CascadeSize)
      << " blocks: critical path extension is "
      << Cycles("CritPathExtension", MaxExtension)
      << ", staying under the threshold of " << Cycles("CritLimit", CritLimit)
      << ".";
    return R;
  });

  return true;
}

/// Check if there is enough ILP to hide the latency of speculatively executing
/// the extra blocks
bool EarlyIfConverter::hasEnoughILP(
    MachineBasicBlock *TraceBlock,
    SmallVectorImpl<MachineBasicBlock *> &ExtraBlocks, unsigned CritLimit) {
  if (!MinInstr)
    MinInstr = Traces->getEnsemble(MachineTraceStrategy::TS_MinInstrCount);

  MachineTraceMetrics::Trace Trace = MinInstr->getTrace(TraceBlock);
  unsigned CritPath = Trace.getCriticalPath();
  SmallVector<const MachineBasicBlock *, 8> ConstExtraBlocks(
      ExtraBlocks.begin(), ExtraBlocks.end());
  unsigned ResLength = Trace.getResourceLength(ConstExtraBlocks);

  LLVM_DEBUG(dbgs() << "ILP check: CriticalPath=" << CritPath
                    << ", ResourceLength=" << ResLength
                    << ", CritLimit=" << CritLimit << "\n");

  if (ResLength > CritPath + CritLimit) {
    LLVM_DEBUG(dbgs() << "Not enough ILP: resource length " << ResLength
                      << " exceeds critical path " << CritPath << " + limit "
                      << CritLimit << "\n");
    return false;
  }
  return true;
}

/// Apply cost model and heuristics to the if-conversion in IfConv.
/// Return true if the conversion is a good idea.
///
bool EarlyIfConverter::shouldConvertIf() {
  bool InCascade = MustConvertBlocks.erase(IfConv.Head);

  // Stress testing mode disables all cost considerations.
  if (Stress)
    return true;

  if (InCascade) {
    LLVM_DEBUG(dbgs() << "Block is part of cascade, skipping profitability\n");
    return true;
  }

  // Do not try to if-convert if the condition has a high chance of being
  // predictable.
  MachineLoop *CurrentLoop = Loops->getLoopFor(IfConv.Head);
  // If the condition is in a loop, consider it predictable if the condition
  // itself or all its operands are loop-invariant. E.g. this considers a load
  // from a loop-invariant address predictable; we were unable to prove that it
  // doesn't alias any of the memory-writes in the loop, but it is likely to
  // read to same value multiple times.
  if (CurrentLoop && any_of(IfConv.Cond, [&](MachineOperand &MO) {
        if (!MO.isReg() || !MO.isUse())
          return false;
        Register Reg = MO.getReg();
        if (Reg.isPhysical())
          return false;

        MachineInstr *Def = MRI->getVRegDef(Reg);
        return CurrentLoop->isLoopInvariant(*Def) ||
               all_of(Def->operands(), [&](MachineOperand &Op) {
                 if (Op.isImm())
                   return true;
                 if (!Op.isReg() || !Op.isUse())
                   return true;
                 Register Reg = Op.getReg();
                 if (Reg.isPhysical())
                   return false;

                 MachineInstr *Def = MRI->getVRegDef(Reg);
                 return CurrentLoop->isLoopInvariant(*Def);
               });
      }))
    return false;

  if (!MinInstr)
    MinInstr = Traces->getEnsemble(MachineTraceStrategy::TS_MinInstrCount);

  MachineTraceMetrics::Trace TBBTrace = MinInstr->getTrace(IfConv.getTPred());
  MachineTraceMetrics::Trace FBBTrace = MinInstr->getTrace(IfConv.getFPred());
  LLVM_DEBUG(dbgs() << "TBB: " << TBBTrace << "FBB: " << FBBTrace);
  unsigned MinCrit = std::min(TBBTrace.getCriticalPath(),
                              FBBTrace.getCriticalPath());

  // Set a somewhat arbitrary limit on the critical path extension we accept.
  // When hard-to-predict analysis is enabled, use full MispredictPenalty for
  // hard-to-predict branches, half for others. Otherwise use half for all.
  bool DataDependent = false;
  if (EnableDataDependentBranchAnalysis)
    DataDependent = isConditionDataDependent(IfConv.Head);

  unsigned CritLimit = DataDependent ? STI->getMispredictionPenalty()
                                     : STI->getMispredictionPenalty() / 2;

  MachineBasicBlock &MBB = *IfConv.Head;
  MachineOptimizationRemarkEmitter MORE(*MBB.getParent(), nullptr);

  // Emit analysis remark about data-dependent condition.
  if (DataDependent) {
    MORE.emit([&]() {
      return MachineOptimizationRemarkAnalysis(DEBUG_TYPE,
                                               "DataDependentCondition",
                                               MBB.back().getDebugLoc(), &MBB)
             << "branch condition is data-dependent (from memory load), "
             << "using higher CritLimit of " << ore::NV("CritLimit", CritLimit)
             << " cycles";
    });
  }

  // If-conversion only makes sense when there is unexploited ILP. Compute the
  // maximum-ILP resource length of the trace after if-conversion. Compare it
  // to the shortest critical path.
  SmallVector<const MachineBasicBlock*, 1> ExtraBlocks;
  if (IfConv.TBB != IfConv.Tail)
    ExtraBlocks.push_back(IfConv.TBB);
  unsigned ResLength = FBBTrace.getResourceLength(ExtraBlocks);
  LLVM_DEBUG(dbgs() << "Resource length " << ResLength
                    << ", minimal critical path " << MinCrit << '\n');
  if (ResLength > MinCrit + CritLimit) {
    LLVM_DEBUG(dbgs() << "Not enough available ILP.\n");
    MORE.emit([&]() {
      MachineOptimizationRemarkMissed R(DEBUG_TYPE, "IfConversion",
                                        MBB.findDebugLoc(MBB.back()), &MBB);
      R << "did not if-convert branch: the resulting critical path ("
        << Cycles("ResLength", ResLength)
        << ") would extend the shorter leg's critical path ("
        << Cycles("MinCrit", MinCrit) << ") by more than the threshold of "
        << Cycles("CritLimit", CritLimit)
        << ", which cannot be hidden by available ILP.";
      return R;
    });
    return false;
  }

  // Assume that the depth of the first head terminator will also be the depth
  // of the select instruction inserted, as determined by the flag dependency.
  // TBB / FBB data dependencies may delay the select even more.
  MachineTraceMetrics::Trace HeadTrace = MinInstr->getTrace(IfConv.Head);
  unsigned BranchDepth =
      HeadTrace.getInstrCycles(*IfConv.Head->getFirstTerminator()).Depth;
  LLVM_DEBUG(dbgs() << "Branch depth: " << BranchDepth << '\n');

  // Look at all the tail phis, and compute the critical path extension caused
  // by inserting select instructions.
  MachineTraceMetrics::Trace TailTrace = MinInstr->getTrace(IfConv.Tail);
  struct CriticalPathInfo {
    unsigned Extra; // Count of extra cycles that the component adds.
    unsigned Depth; // Absolute depth of the component in cycles.
  };
  CriticalPathInfo Cond{};
  CriticalPathInfo TBlock{};
  CriticalPathInfo FBlock{};
  bool ShouldConvert = true;
  for (SSAIfConv::PHIInfo &PI : IfConv.PHIs) {
    unsigned Slack = TailTrace.getInstrSlack(*PI.PHI);
    unsigned MaxDepth = Slack + TailTrace.getInstrCycles(*PI.PHI).Depth;
    LLVM_DEBUG(dbgs() << "Slack " << Slack << ":\t" << *PI.PHI);

    // The condition is pulled into the critical path.
    unsigned CondDepth = adjCycles(BranchDepth, PI.CondCycles);
    if (CondDepth > MaxDepth) {
      unsigned Extra = CondDepth - MaxDepth;
      LLVM_DEBUG(dbgs() << "Condition adds " << Extra << " cycles.\n");
      if (Extra > Cond.Extra)
        Cond = {Extra, CondDepth};
      if (Extra > CritLimit) {
        LLVM_DEBUG(dbgs() << "Exceeds limit of " << CritLimit << '\n');
        ShouldConvert = false;
      }
    }

    // The TBB value is pulled into the critical path.
    unsigned TDepth = adjCycles(TBBTrace.getPHIDepth(*PI.PHI), PI.TCycles);
    if (TDepth > MaxDepth) {
      unsigned Extra = TDepth - MaxDepth;
      LLVM_DEBUG(dbgs() << "TBB data adds " << Extra << " cycles.\n");
      if (Extra > TBlock.Extra)
        TBlock = {Extra, TDepth};
      if (Extra > CritLimit) {
        LLVM_DEBUG(dbgs() << "Exceeds limit of " << CritLimit << '\n');
        ShouldConvert = false;
      }
    }

    // The FBB value is pulled into the critical path.
    unsigned FDepth = adjCycles(FBBTrace.getPHIDepth(*PI.PHI), PI.FCycles);
    if (FDepth > MaxDepth) {
      unsigned Extra = FDepth - MaxDepth;
      LLVM_DEBUG(dbgs() << "FBB data adds " << Extra << " cycles.\n");
      if (Extra > FBlock.Extra)
        FBlock = {Extra, FDepth};
      if (Extra > CritLimit) {
        LLVM_DEBUG(dbgs() << "Exceeds limit of " << CritLimit << '\n');
        ShouldConvert = false;
      }
    }
  }

  // Organize by "short" and "long" legs, since the diagnostics get confusing
  // when referring to the "true" and "false" sides of the branch, given that
  // those don't always correlate with what the user wrote in source-terms.
  const CriticalPathInfo Short = TBlock.Extra > FBlock.Extra ? FBlock : TBlock;
  const CriticalPathInfo Long = TBlock.Extra > FBlock.Extra ? TBlock : FBlock;

  if (ShouldConvert) {
    MORE.emit([&]() {
      MachineOptimizationRemark R(DEBUG_TYPE, "IfConversion",
                                  MBB.back().getDebugLoc(), &MBB);
      R << "performing if-conversion on branch: the condition adds "
        << Cycles("CondCycles", Cond.Extra) << " to the critical path";
      if (Short.Extra > 0)
        R << ", and the short leg adds another "
          << Cycles("ShortCycles", Short.Extra);
      if (Long.Extra > 0)
        R << ", and the long leg adds another "
          << Cycles("LongCycles", Long.Extra);
      R << ", each staying under the threshold of "
        << Cycles("CritLimit", CritLimit) << ".";
      return R;
    });
  } else {
    MORE.emit([&]() {
      MachineOptimizationRemarkMissed R(DEBUG_TYPE, "IfConversion",
                                        MBB.back().getDebugLoc(), &MBB);
      R << "did not if-convert branch: the condition would add "
        << Cycles("CondCycles", Cond.Extra) << " to the critical path";
      if (Cond.Extra > CritLimit)
        R << " exceeding the limit of " << Cycles("CritLimit", CritLimit);
      if (Short.Extra > 0) {
        R << ", and the short leg would add another "
          << Cycles("ShortCycles", Short.Extra);
        if (Short.Extra > CritLimit)
          R << " exceeding the limit of " << Cycles("CritLimit", CritLimit);
      }
      if (Long.Extra > 0) {
        R << ", and the long leg would add another "
          << Cycles("LongCycles", Long.Extra);
        if (Long.Extra > CritLimit)
          R << " exceeding the limit of " << Cycles("CritLimit", CritLimit);
      }
      R << ".";
      return R;
    });
  }

  return ShouldConvert;
}

/// Perform the actual if-conversion and update analyses.
void EarlyIfConverter::convertIf() {
  SmallVector<MachineBasicBlock *, 4> RemoveBlocks;
  invalidateTraces();
  IfConv.convertIf(RemoveBlocks);
  updateDomTree(DomTree, IfConv, RemoveBlocks);
  updateLoops(Loops, RemoveBlocks);
  // Head absorbs the instructions of the removed blocks, including any calls,
  // so a Head cached as call-free may no longer be.
  NoCallBlocksCache.erase(IfConv.Head);
  for (MachineBasicBlock *MBB : RemoveBlocks) {
    NoCallBlocksCache.erase(MBB);
    MustConvertBlocks.erase(MBB);
    MBB->eraseFromParent();
  }
}

/// Try to detect a cascade ending at MBB and mark blocks for conversion.
/// Since we visit blocks in postorder, the cascade end is visited before
/// its head, allowing us to mark cascade blocks before they are visited.
void EarlyIfConverter::detectCascades(MachineBasicBlock *MBB) {
  if (!EnableCascadeIfConv || !EnableDataDependentBranchAnalysis ||
      MustConvertBlocks.count(MBB))
    return;

  auto Cascade = IfConv.matchCascade(MBB);
  if (!Cascade)
    return;

  ++NumCascadesSeen;
  MachineBasicBlock *Tail = MBB->succ_begin()[0];
  LLVM_DEBUG(dbgs() << "Found cascade with " << Cascade.Blocks.size()
                    << " blocks, Head=" << printMBBReference(*Cascade.Head)
                    << ", Tail=" << printMBBReference(*Tail) << "\n");

  if (!shouldConvertCascade(Cascade, Tail))
    return;

  ++NumCascadesConv;
  NumDataDependant += Cascade.Blocks.size();

  // Mark Head and all cascade blocks except the last one.
  // The last cascade block (BBn) is never a head - it's always the TBB.
  MustConvertBlocks.insert(Cascade.Head);
  for (size_t I = 0; I + 1 < Cascade.Blocks.size(); ++I)
    MustConvertBlocks.insert(Cascade.Blocks[I]);
  LLVM_DEBUG(dbgs() << "Marked " << (1 + Cascade.Blocks.size() - 1)
                    << " blocks for cascade conversion\n");
}

/// Attempt repeated if-conversion on MBB, return true if successful.
///
bool EarlyIfConverter::tryConvertIf(MachineBasicBlock *MBB) {
  detectCascades(MBB);

  bool Changed = false;
  while (IfConv.canConvertIf(MBB) && shouldConvertIf()) {
    convertIf();
    Changed = true;
  }

  // A mark that survives means a cascade block was not converted. We only
  // convert cascades if the entire cascade conversion is profitable, so
  // something went wrong.
  assert(!MustConvertBlocks.contains(MBB) &&
         "cascade block failed to if-convert");

  return Changed;
}

bool EarlyIfConverter::run(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** EARLY IF-CONVERSION **********\n"
                    << "********** Function: " << MF.getName() << '\n');

  STI = &MF.getSubtarget();
  // Only run if conversion if the target wants it.
  if (!STI->enableEarlyIfConversion())
    return false;

  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  MRI = &MF.getRegInfo();
  MinInstr = nullptr;

  bool Changed = false;
  IfConv.init(MF);

  MustConvertBlocks.clear();

  // Visit blocks in dominator tree post-order. The post-order enables nested
  // if-conversion in a single pass. The tryConvertIf() function may erase
  // blocks, but only blocks dominated by the head block. This makes it safe to
  // update the dominator tree while the post-order iterator is still active.
  for (auto *DomNode : post_order(DomTree))
    if (tryConvertIf(DomNode->getBlock()))
      Changed = true;

  assert(MustConvertBlocks.empty() && "cascade block failed to if-convert");

  return Changed;
}

PreservedAnalyses
EarlyIfConverterPass::run(MachineFunction &MF,
                          MachineFunctionAnalysisManager &MFAM) {
  MachineDominatorTree &MDT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  MachineLoopInfo &LI = MFAM.getResult<MachineLoopAnalysis>(MF);
  MachineTraceMetrics &MTM = MFAM.getResult<MachineTraceMetricsAnalysis>(MF);
  MachineBranchProbabilityInfo *MBPI = nullptr;
  if (EnableDataDependentBranchAnalysis)
    MBPI = &MFAM.getResult<MachineBranchProbabilityAnalysis>(MF);

  EarlyIfConverter Impl(MDT, LI, MTM, MBPI);
  bool Changed = Impl.run(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserve<MachineDominatorTreeAnalysis>();
  PA.preserve<MachineLoopAnalysis>();
  PA.preserve<MachineTraceMetricsAnalysis>();
  return PA;
}

bool EarlyIfConverterLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MachineDominatorTree &MDT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MachineLoopInfo &LI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MachineTraceMetrics &MTM =
      getAnalysis<MachineTraceMetricsWrapperPass>().getMTM();
  MachineBranchProbabilityInfo *MBPI = nullptr;
  if (EnableDataDependentBranchAnalysis)
    MBPI = &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();

  return EarlyIfConverter(MDT, LI, MTM, MBPI).run(MF);
}

//===----------------------------------------------------------------------===//
//                           EarlyIfPredicator Pass
//===----------------------------------------------------------------------===//

namespace {
class EarlyIfPredicator : public MachineFunctionPass {
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  TargetSchedModel SchedModel;
  MachineRegisterInfo *MRI = nullptr;
  MachineDominatorTree *DomTree = nullptr;
  MachineBranchProbabilityInfo *MBPI = nullptr;
  MachineLoopInfo *Loops = nullptr;
  SSAIfConv IfConv;

public:
  static char ID;
  EarlyIfPredicator() : MachineFunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override { return "Early If-predicator"; }

protected:
  bool tryConvertIf(MachineBasicBlock *);
  bool shouldConvertIf();
};
} // end anonymous namespace

#undef DEBUG_TYPE
#define DEBUG_TYPE "early-if-predicator"

char EarlyIfPredicator::ID = 0;
char &llvm::EarlyIfPredicatorID = EarlyIfPredicator::ID;

INITIALIZE_PASS_BEGIN(EarlyIfPredicator, DEBUG_TYPE, "Early If Predicator",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_END(EarlyIfPredicator, DEBUG_TYPE, "Early If Predicator", false,
                    false)

void EarlyIfPredicator::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

/// Apply the target heuristic to decide if the transformation is profitable.
bool EarlyIfPredicator::shouldConvertIf() {
  auto TrueProbability = MBPI->getEdgeProbability(IfConv.Head, IfConv.TBB);
  if (IfConv.isTriangle()) {
    MachineBasicBlock &IfBlock =
        (IfConv.TBB == IfConv.Tail) ? *IfConv.FBB : *IfConv.TBB;

    unsigned ExtraPredCost = 0;
    unsigned Cycles = 0;
    for (MachineInstr &I : IfBlock) {
      unsigned NumCycles = SchedModel.computeInstrLatency(&I, false);
      if (NumCycles > 1)
        Cycles += NumCycles - 1;
      ExtraPredCost += TII->getPredicationCost(I);
    }

    return TII->isProfitableToIfCvt(IfBlock, Cycles, ExtraPredCost,
                                    TrueProbability);
  }
  unsigned TExtra = 0;
  unsigned FExtra = 0;
  unsigned TCycle = 0;
  unsigned FCycle = 0;
  for (MachineInstr &I : *IfConv.TBB) {
    unsigned NumCycles = SchedModel.computeInstrLatency(&I, false);
    if (NumCycles > 1)
      TCycle += NumCycles - 1;
    TExtra += TII->getPredicationCost(I);
  }
  for (MachineInstr &I : *IfConv.FBB) {
    unsigned NumCycles = SchedModel.computeInstrLatency(&I, false);
    if (NumCycles > 1)
      FCycle += NumCycles - 1;
    FExtra += TII->getPredicationCost(I);
  }
  return TII->isProfitableToIfCvt(*IfConv.TBB, TCycle, TExtra, *IfConv.FBB,
                                  FCycle, FExtra, TrueProbability);
}

/// Attempt repeated if-conversion on MBB, return true if successful.
///
bool EarlyIfPredicator::tryConvertIf(MachineBasicBlock *MBB) {
  bool Changed = false;
  while (IfConv.canConvertIf(MBB, /*Predicate*/ true) && shouldConvertIf()) {
    // If-convert MBB and update analyses.
    SmallVector<MachineBasicBlock *, 4> RemoveBlocks;
    IfConv.convertIf(RemoveBlocks, /*Predicate*/ true);
    Changed = true;
    updateDomTree(DomTree, IfConv, RemoveBlocks);
    updateLoops(Loops, RemoveBlocks);
    for (MachineBasicBlock *MBB : RemoveBlocks)
      MBB->eraseFromParent();
  }
  return Changed;
}

bool EarlyIfPredicator::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** EARLY IF-PREDICATOR **********\n"
                    << "********** Function: " << MF.getName() << '\n');
  if (skipFunction(MF.getFunction()))
    return false;

  const TargetSubtargetInfo &STI = MF.getSubtarget();
  TII = STI.getInstrInfo();
  TRI = STI.getRegisterInfo();
  MRI = &MF.getRegInfo();
  SchedModel.init(&STI);
  DomTree = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  Loops = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MBPI = &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();

  bool Changed = false;
  IfConv.init(MF);

  // Visit blocks in dominator tree post-order. The post-order enables nested
  // if-conversion in a single pass. The tryConvertIf() function may erase
  // blocks, but only blocks dominated by the head block. This makes it safe to
  // update the dominator tree while the post-order iterator is still active.
  for (auto *DomNode : post_order(DomTree))
    if (tryConvertIf(DomNode->getBlock()))
      Changed = true;

  return Changed;
}
