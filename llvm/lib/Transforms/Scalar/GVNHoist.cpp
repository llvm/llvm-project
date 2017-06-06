//===- GVNHoist.cpp - Hoist scalar and load expressions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass hoists expressions from branches to a common dominator. It uses
// GVN (global value numbering) to discover expressions computing the same
// values. The primary goals of code-hoisting are:
// 1. To reduce the code size.
// 2. In some cases reduce critical path (by exposing more ILP).
//
// Hoisting may affect the performance in some cases. To mitigate that, hoisting
// is disabled in the following cases.
// 1. Scalars across calls.
// 2. geps when corresponding load/store cannot be hoisted.
//
// TODO: Hoist from >2 successors. Currently GVNHoist will not hoist stores
// in this case because it works on two instructions at a time.
// entry:
//   switch i32 %c1, label %exit1 [
//     i32 0, label %sw0
//     i32 1, label %sw1
//   ]
//
// sw0:
//   store i32 1, i32* @G
//   br label %exit
//
// sw1:
//   store i32 1, i32* @G
//   br label %exit
//
// exit1:
//   store i32 1, i32* @G
//   ret void
// exit:
//   ret void
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "gvn-hoist"

STATISTIC(NumHoisted, "Number of instructions hoisted");
STATISTIC(NumRemoved, "Number of instructions removed");
STATISTIC(NumLoadsHoisted, "Number of loads hoisted");
STATISTIC(NumLoadsRemoved, "Number of loads removed");
STATISTIC(NumStoresHoisted, "Number of stores hoisted");
STATISTIC(NumStoresRemoved, "Number of stores removed");
STATISTIC(NumCallsHoisted, "Number of calls hoisted");
STATISTIC(NumCallsRemoved, "Number of calls removed");

static cl::opt<int>
    MaxHoistedThreshold("gvn-max-hoisted", cl::Hidden, cl::init(-1),
                        cl::desc("Max number of instructions to hoist "
                                 "(default unlimited = -1)"));
static cl::opt<int> MaxNumberOfBBSInPath(
    "gvn-hoist-max-bbs", cl::Hidden, cl::init(4),
    cl::desc("Max number of basic blocks on the path between "
             "hoisting locations (default = 4, unlimited = -1)"));

static cl::opt<int> MaxDepthInBB(
    "gvn-hoist-max-depth", cl::Hidden, cl::init(100),
    cl::desc("Hoist instructions from the beginning of the BB up to the "
             "maximum specified depth (default = 100, unlimited = -1)"));

static cl::opt<int>
    MaxChainLength("gvn-hoist-max-chain-length", cl::Hidden, cl::init(10),
                   cl::desc("Maximum length of dependent chains to hoist "
                            "(default = 10, unlimited = -1)"));

namespace llvm {

// Provides a sorting function based on the execution order of two instructions.
struct SortByDFSIn {
private:
  DenseMap<const Value *, unsigned> &DFSNumber;

public:
  SortByDFSIn(DenseMap<const Value *, unsigned> &D) : DFSNumber(D) {}

  // Returns true when A executes before B.
  bool operator()(const Instruction *A, const Instruction *B) const {
    const BasicBlock *BA = A->getParent();
    const BasicBlock *BB = B->getParent();
    unsigned ADFS, BDFS;
    if (BA == BB) {
      ADFS = DFSNumber.lookup(A);
      BDFS = DFSNumber.lookup(B);
    } else {
      ADFS = DFSNumber.lookup(BA);
      BDFS = DFSNumber.lookup(BB);
    }
    assert(ADFS && BDFS);
    return ADFS < BDFS;
  }
};

// A map from a pair of VNs to all the instructions with those VNs.
typedef DenseMap<std::pair<unsigned, unsigned>, SmallVector<Instruction *, 4>>
    VNtoInsns;
// An invalid value number Used when inserting a single value number into
// VNtoInsns.
enum : unsigned { InvalidVN = ~2U };

// Records all scalar instructions candidate for code hoisting.
class InsnInfo {
  VNtoInsns VNtoScalars;

public:
  // Inserts I and its value number in VNtoScalars.
  void insert(Instruction *I, GVN::ValueTable &VN) {
    // Scalar instruction.
    unsigned V = VN.lookupOrAdd(I);
    VNtoScalars[{V, InvalidVN}].push_back(I);
  }

  const VNtoInsns &getVNTable() const { return VNtoScalars; }
};

// Records all load instructions candidate for code hoisting.
class LoadInfo {
  VNtoInsns VNtoLoads;

public:
  // Insert Load and the value number of its memory address in VNtoLoads.
  void insert(LoadInst *Load, GVN::ValueTable &VN) {
    if (Load->isSimple()) {
      unsigned V = VN.lookupOrAdd(Load->getPointerOperand());
      VNtoLoads[{V, InvalidVN}].push_back(Load);
    }
  }

  const VNtoInsns &getVNTable() const { return VNtoLoads; }
};

// Records all store instructions candidate for code hoisting.
class StoreInfo {
  VNtoInsns VNtoStores;

public:
  // Insert the Store and a hash number of the store address and the stored
  // value in VNtoStores.
  void insert(StoreInst *Store, GVN::ValueTable &VN) {
    if (!Store->isSimple())
      return;
    // Hash the store address and the stored value.
    Value *Ptr = Store->getPointerOperand();
    Value *Val = Store->getValueOperand();
    VNtoStores[{VN.lookupOrAdd(Ptr), VN.lookupOrAdd(Val)}].push_back(Store);
  }

  const VNtoInsns &getVNTable() const { return VNtoStores; }
};

// Records all call instructions candidate for code hoisting.
class CallInfo {
  VNtoInsns VNtoCallsScalars;
  VNtoInsns VNtoCallsLoads;
  VNtoInsns VNtoCallsStores;

public:
  // Insert Call and its value numbering in one of the VNtoCalls* containers.
  void insert(CallInst *Call, GVN::ValueTable &VN) {
    // A call that doesNotAccessMemory is handled as a Scalar,
    // onlyReadsMemory will be handled as a Load instruction,
    // all other calls will be handled as stores.
    unsigned V = VN.lookupOrAdd(Call);
    auto Entry = std::make_pair(V, InvalidVN);

    if (Call->doesNotAccessMemory())
      VNtoCallsScalars[Entry].push_back(Call);
    else if (Call->onlyReadsMemory())
      VNtoCallsLoads[Entry].push_back(Call);
    else
      VNtoCallsStores[Entry].push_back(Call);
  }

  const VNtoInsns &getScalarVNTable() const { return VNtoCallsScalars; }

  const VNtoInsns &getLoadVNTable() const { return VNtoCallsLoads; }

  const VNtoInsns &getStoreVNTable() const { return VNtoCallsStores; }
};

typedef DenseMap<const BasicBlock *, bool> BBSideEffectsSet;
typedef SmallVector<Instruction *, 4> SmallVecInsn;
typedef SmallVectorImpl<Instruction *> SmallVecImplInsn;

static void combineKnownMetadata(Instruction *ReplInst, Instruction *I) {
  static const unsigned KnownIDs[] = {
      LLVMContext::MD_tbaa,           LLVMContext::MD_alias_scope,
      LLVMContext::MD_noalias,        LLVMContext::MD_range,
      LLVMContext::MD_fpmath,         LLVMContext::MD_invariant_load,
      LLVMContext::MD_invariant_group};
  combineMetadata(ReplInst, I, KnownIDs);
}

// This pass hoists common computations across branches sharing common
// dominator. The primary goal is to reduce the code size, and in some
// cases reduce critical path (by exposing more ILP).
class GVNHoist {
public:
  GVNHoist(DominatorTree *DT, AliasAnalysis *AA, MemoryDependenceResults *MD,
           MemorySSA *MSSA)
      : DT(DT), AA(AA), MD(MD), MSSA(MSSA),
        MSSAUpdater(make_unique<MemorySSAUpdater>(MSSA)),
        HoistingGeps(false),
        HoistedCtr(0)
  { }

  bool run(Function &F) {
    VN.setDomTree(DT);
    VN.setAliasAnalysis(AA);
    VN.setMemDep(MD);
    bool Res = false;
    // Perform DFS Numbering of instructions.
    unsigned BBI = 0;
    for (const BasicBlock *BB : depth_first(&F.getEntryBlock())) {
      DFSNumber[BB] = ++BBI;
      unsigned I = 0;
      for (auto &Inst : *BB)
        DFSNumber[&Inst] = ++I;
    }

    int ChainLength = 0;

    // FIXME: use lazy evaluation of VN to avoid the fix-point computation.
    while (1) {
      if (MaxChainLength != -1 && ++ChainLength >= MaxChainLength)
        return Res;

      auto HoistStat = hoistExpressions(F);
      if (HoistStat.first + HoistStat.second == 0)
        return Res;

      if (HoistStat.second > 0)
        // To address a limitation of the current GVN, we need to rerun the
        // hoisting after we hoisted loads or stores in order to be able to
        // hoist all scalars dependent on the hoisted ld/st.
        VN.clear();

      Res = true;
    }

    return Res;
  }

private:
  GVN::ValueTable VN;
  DominatorTree *DT;
  AliasAnalysis *AA;
  MemoryDependenceResults *MD;
  MemorySSA *MSSA;
  std::unique_ptr<MemorySSAUpdater> MSSAUpdater;
  const bool HoistingGeps;
  DenseMap<const Value *, unsigned> DFSNumber;
  BBSideEffectsSet BBSideEffects;
  DenseSet<const BasicBlock*> HoistBarrier;
  int HoistedCtr;

  enum InsKind { Unknown, Scalar, Load, Store };

  // Return true when there are exception handling in BB.
  bool hasEH(const BasicBlock *BB) {
    auto It = BBSideEffects.find(BB);
    if (It != BBSideEffects.end())
      return It->second;

    if (BB->isEHPad() || BB->hasAddressTaken()) {
      BBSideEffects[BB] = true;
      return true;
    }

    if (BB->getTerminator()->mayThrow()) {
      BBSideEffects[BB] = true;
      return true;
    }

    BBSideEffects[BB] = false;
    return false;
  }

  // Return true when a successor of BB dominates A.
  bool successorDominate(const BasicBlock *BB, const BasicBlock *A) {
    for (const BasicBlock *Succ : BB->getTerminator()->successors())
      if (DT->dominates(Succ, A))
        return true;

    return false;
  }

  // Return true when all paths from HoistBB to the end of the function pass
  // through one of the blocks in WL.
  bool hoistingFromAllPaths(const BasicBlock *HoistBB,
                            SmallPtrSetImpl<const BasicBlock *> &WL) {

    // Copy WL as the loop will remove elements from it.
    SmallPtrSet<const BasicBlock *, 2> WorkList(WL.begin(), WL.end());

    for (auto It = df_begin(HoistBB), E = df_end(HoistBB); It != E;) {
      // There exists a path from HoistBB to the exit of the function if we are
      // still iterating in DF traversal and we removed all instructions from
      // the work list.
      if (WorkList.empty())
        return false;

      const BasicBlock *BB = *It;
      if (WorkList.erase(BB)) {
        // Stop DFS traversal when BB is in the work list.
        It.skipChildren();
        continue;
      }

      // We reached the leaf Basic Block => not all paths have this instruction.
      if (!BB->getTerminator()->getNumSuccessors())
        return false;

      // When reaching the back-edge of a loop, there may be a path through the
      // loop that does not pass through B or C before exiting the loop.
      if (successorDominate(BB, HoistBB))
        return false;

      // Increment DFS traversal when not skipping children.
      ++It;
    }

    return true;
  }

  /* Return true when I1 appears before I2 in the instructions of BB.  */
  bool firstInBB(const Instruction *I1, const Instruction *I2) {
    assert(I1->getParent() == I2->getParent());
    unsigned I1DFS = DFSNumber.lookup(I1);
    unsigned I2DFS = DFSNumber.lookup(I2);
    assert(I1DFS && I2DFS);
    return I1DFS < I2DFS;
  }

  // Return true when there are memory uses of Def in BB.
  bool hasMemoryUse(const Instruction *NewPt, MemoryDef *Def,
                    const BasicBlock *BB) {
    const MemorySSA::AccessList *Acc = MSSA->getBlockAccesses(BB);
    if (!Acc)
      return false;

    Instruction *OldPt = Def->getMemoryInst();
    const BasicBlock *OldBB = OldPt->getParent();
    const BasicBlock *NewBB = NewPt->getParent();
    bool ReachedNewPt = false;

    for (const MemoryAccess &MA : *Acc)
      if (const MemoryUse *MU = dyn_cast<MemoryUse>(&MA)) {
        Instruction *Insn = MU->getMemoryInst();

        // Do not check whether MU aliases Def when MU occurs after OldPt.
        if (BB == OldBB && firstInBB(OldPt, Insn))
          break;

        // Do not check whether MU aliases Def when MU occurs before NewPt.
        if (BB == NewBB) {
          if (!ReachedNewPt) {
            if (firstInBB(Insn, NewPt))
              continue;
            ReachedNewPt = true;
          }
        }
        if (MemorySSAUtil::defClobbersUseOrDef(Def, MU, *AA))
          return true;
      }

    return false;
  }

  // Return true when there are exception handling or loads of memory Def
  // between Def and NewPt.  This function is only called for stores: Def is
  // the MemoryDef of the store to be hoisted.

  // Decrement by 1 NBBsOnAllPaths for each block between HoistPt and BB, and
  // return true when the counter NBBsOnAllPaths reaces 0, except when it is
  // initialized to -1 which is unlimited.
  bool hasEHOrLoadsOnPath(const Instruction *NewPt, MemoryDef *Def,
                          int &NBBsOnAllPaths) {
    const BasicBlock *NewBB = NewPt->getParent();
    const BasicBlock *OldBB = Def->getBlock();
    assert(DT->dominates(NewBB, OldBB) && "invalid path");
    assert(DT->dominates(Def->getDefiningAccess()->getBlock(), NewBB) &&
           "def does not dominate new hoisting point");

    // Walk all basic blocks reachable in depth-first iteration on the inverse
    // CFG from OldBB to NewBB. These blocks are all the blocks that may be
    // executed between the execution of NewBB and OldBB. Hoisting an expression
    // from OldBB into NewBB has to be safe on all execution paths.
    for (auto I = idf_begin(OldBB), E = idf_end(OldBB); I != E;) {
      const BasicBlock *BB = *I;
      if (BB == NewBB) {
        // Stop traversal when reaching HoistPt.
        I.skipChildren();
        continue;
      }

      // Stop walk once the limit is reached.
      if (NBBsOnAllPaths == 0)
        return true;

      // Impossible to hoist with exceptions on the path.
      if (hasEH(BB))
        return true;

      // No such instruction after HoistBarrier in a basic block was
      // selected for hoisting so instructions selected within basic block with
      // a hoist barrier can be hoisted.
      if ((BB != OldBB) && HoistBarrier.count(BB))
        return true;

      // Check that we do not move a store past loads.
      if (hasMemoryUse(NewPt, Def, BB))
        return true;

      // -1 is unlimited number of blocks on all paths.
      if (NBBsOnAllPaths != -1)
        --NBBsOnAllPaths;

      ++I;
    }

    return false;
  }

  // Return true when there are exception handling between HoistPt and BB.
  // Decrement by 1 NBBsOnAllPaths for each block between HoistPt and BB, and
  // return true when the counter NBBsOnAllPaths reaches 0, except when it is
  // initialized to -1 which is unlimited.
  bool hasEHOnPath(const BasicBlock *HoistPt, const BasicBlock *SrcBB,
                   int &NBBsOnAllPaths) {
    assert(DT->dominates(HoistPt, SrcBB) && "Invalid path");

    // Walk all basic blocks reachable in depth-first iteration on
    // the inverse CFG from BBInsn to NewHoistPt. These blocks are all the
    // blocks that may be executed between the execution of NewHoistPt and
    // BBInsn. Hoisting an expression from BBInsn into NewHoistPt has to be safe
    // on all execution paths.
    for (auto I = idf_begin(SrcBB), E = idf_end(SrcBB); I != E;) {
      const BasicBlock *BB = *I;
      if (BB == HoistPt) {
        // Stop traversal when reaching NewHoistPt.
        I.skipChildren();
        continue;
      }

      // Stop walk once the limit is reached.
      if (NBBsOnAllPaths == 0)
        return true;

      // Impossible to hoist with exceptions on the path.
      if (hasEH(BB))
        return true;

      // No such instruction after HoistBarrier in a basic block was
      // selected for hoisting so instructions selected within basic block with
      // a hoist barrier can be hoisted.
      if ((BB != SrcBB) && HoistBarrier.count(BB))
        return true;

      // -1 is unlimited number of blocks on all paths.
      if (NBBsOnAllPaths != -1)
        --NBBsOnAllPaths;

      ++I;
    }

    return false;
  }

  // Return true when it is safe to hoist a memory load or store U from OldPt
  // to NewPt.
  bool safeToHoistLdSt(const Instruction *NewPt, const Instruction *OldPt,
                       MemoryUseOrDef *U, InsKind K, int &NBBsOnAllPaths) {

    // In place hoisting is safe.
    if (NewPt == OldPt)
      return true;

    const BasicBlock *NewBB = NewPt->getParent();
    const BasicBlock *OldBB = OldPt->getParent();
    const BasicBlock *UBB = U->getBlock();

    // Check for dependences on the Memory SSA.
    MemoryAccess *D = U->getDefiningAccess();
    BasicBlock *DBB = D->getBlock();
    if (DT->properlyDominates(NewBB, DBB))
      // Cannot move the load or store to NewBB above its definition in DBB.
      return false;

    if (NewBB == DBB && !MSSA->isLiveOnEntryDef(D))
      if (auto *UD = dyn_cast<MemoryUseOrDef>(D))
        if (firstInBB(NewPt, UD->getMemoryInst()))
          // Cannot move the load or store to NewPt above its definition in D.
          return false;

    // Check for unsafe hoistings due to side effects.
    if (K == InsKind::Store) {
      if (hasEHOrLoadsOnPath(NewPt, dyn_cast<MemoryDef>(U), NBBsOnAllPaths))
        return false;
    } else if (hasEHOnPath(NewBB, OldBB, NBBsOnAllPaths))
      return false;

    if (UBB == NewBB) {
      if (DT->properlyDominates(DBB, NewBB))
        return true;
      assert(UBB == DBB);
      assert(MSSA->locallyDominates(D, U));
    }

    // No side effects: it is safe to hoist.
    return true;
  }

  // Return true when it is safe to hoist scalar instructions from all blocks in
  // WL to HoistBB.
  bool safeToHoistScalar(const BasicBlock *HoistBB,
                         SmallPtrSetImpl<const BasicBlock *> &WL,
                         int &NBBsOnAllPaths) {
    // Check that the hoisted expression is needed on all paths.
    if (!hoistingFromAllPaths(HoistBB, WL))
      return false;

    for (const BasicBlock *BB : WL)
      if (hasEHOnPath(HoistBB, BB, NBBsOnAllPaths))
        return false;

    return true;
  }

  // Each element of a hoisting list contains the basic block where to hoist and
  // a list of instructions to be hoisted.
  typedef std::pair<BasicBlock *, SmallVecInsn> HoistingPointInfo;
  typedef SmallVector<HoistingPointInfo, 4> HoistingPointList;

  // Partition InstructionsToHoist into a set of candidates which can share a
  // common hoisting point. The partitions are collected in HPL. IsScalar is
  // true when the instructions in InstructionsToHoist are scalars. IsLoad is
  // true when the InstructionsToHoist are loads, false when they are stores.
  void partitionCandidates(SmallVecImplInsn &InstructionsToHoist,
                           HoistingPointList &HPL, InsKind K) {
    // No need to sort for two instructions.
    if (InstructionsToHoist.size() > 2) {
      SortByDFSIn Pred(DFSNumber);
      std::sort(InstructionsToHoist.begin(), InstructionsToHoist.end(), Pred);
    }

    int NumBBsOnAllPaths = MaxNumberOfBBSInPath;

    SmallVecImplInsn::iterator II = InstructionsToHoist.begin();
    SmallVecImplInsn::iterator Start = II;
    Instruction *HoistPt = *II;
    BasicBlock *HoistBB = HoistPt->getParent();
    MemoryUseOrDef *UD;
    if (K != InsKind::Scalar)
      UD = MSSA->getMemoryAccess(HoistPt);

    for (++II; II != InstructionsToHoist.end(); ++II) {
      Instruction *Insn = *II;
      BasicBlock *BB = Insn->getParent();
      BasicBlock *NewHoistBB;
      Instruction *NewHoistPt;

      if (BB == HoistBB) { // Both are in the same Basic Block.
        NewHoistBB = HoistBB;
        NewHoistPt = firstInBB(Insn, HoistPt) ? Insn : HoistPt;
      } else {
        // If the hoisting point contains one of the instructions,
        // then hoist there, otherwise hoist before the terminator.
        NewHoistBB = DT->findNearestCommonDominator(HoistBB, BB);
        if (NewHoistBB == BB)
          NewHoistPt = Insn;
        else if (NewHoistBB == HoistBB)
          NewHoistPt = HoistPt;
        else
          NewHoistPt = NewHoistBB->getTerminator();
      }

      SmallPtrSet<const BasicBlock *, 2> WL;
      WL.insert(HoistBB);
      WL.insert(BB);

      if (K == InsKind::Scalar) {
        if (safeToHoistScalar(NewHoistBB, WL, NumBBsOnAllPaths)) {
          // Extend HoistPt to NewHoistPt.
          HoistPt = NewHoistPt;
          HoistBB = NewHoistBB;
          continue;
        }
      } else {
        // When NewBB already contains an instruction to be hoisted, the
        // expression is needed on all paths.
        // Check that the hoisted expression is needed on all paths: it is
        // unsafe to hoist loads to a place where there may be a path not
        // loading from the same address: for instance there may be a branch on
        // which the address of the load may not be initialized.
        if ((HoistBB == NewHoistBB || BB == NewHoistBB ||
             hoistingFromAllPaths(NewHoistBB, WL)) &&
            // Also check that it is safe to move the load or store from HoistPt
            // to NewHoistPt, and from Insn to NewHoistPt.
            safeToHoistLdSt(NewHoistPt, HoistPt, UD, K, NumBBsOnAllPaths) &&
            safeToHoistLdSt(NewHoistPt, Insn, MSSA->getMemoryAccess(Insn),
                            K, NumBBsOnAllPaths)) {
          // Extend HoistPt to NewHoistPt.
          HoistPt = NewHoistPt;
          HoistBB = NewHoistBB;
          continue;
        }
      }

      // At this point it is not safe to extend the current hoisting to
      // NewHoistPt: save the hoisting list so far.
      if (std::distance(Start, II) > 1)
        HPL.push_back({HoistBB, SmallVecInsn(Start, II)});

      // Start over from BB.
      Start = II;
      if (K != InsKind::Scalar)
        UD = MSSA->getMemoryAccess(*Start);
      HoistPt = Insn;
      HoistBB = BB;
      NumBBsOnAllPaths = MaxNumberOfBBSInPath;
    }

    // Save the last partition.
    if (std::distance(Start, II) > 1)
      HPL.push_back({HoistBB, SmallVecInsn(Start, II)});
  }

  // Initialize HPL from Map.
  void computeInsertionPoints(const VNtoInsns &Map, HoistingPointList &HPL,
                              InsKind K) {
    for (const auto &Entry : Map) {
      if (MaxHoistedThreshold != -1 && ++HoistedCtr > MaxHoistedThreshold)
        return;

      const SmallVecInsn &V = Entry.second;
      if (V.size() < 2)
        continue;

      // Compute the insertion point and the list of expressions to be hoisted.
      SmallVecInsn InstructionsToHoist;
      for (auto I : V)
        // We don't need to check for hoist-barriers here because if
        // I->getParent() is a barrier then I precedes the barrier.
        if (!hasEH(I->getParent()))
          InstructionsToHoist.push_back(I);

      if (!InstructionsToHoist.empty())
        partitionCandidates(InstructionsToHoist, HPL, K);
    }
  }

  // Return true when all operands of Instr are available at insertion point
  // HoistPt. When limiting the number of hoisted expressions, one could hoist
  // a load without hoisting its access function. So before hoisting any
  // expression, make sure that all its operands are available at insert point.
  bool allOperandsAvailable(const Instruction *I,
                            const BasicBlock *HoistPt) const {
    for (const Use &Op : I->operands())
      if (const auto *Inst = dyn_cast<Instruction>(&Op))
        if (!DT->dominates(Inst->getParent(), HoistPt))
          return false;

    return true;
  }

  // Same as allOperandsAvailable with recursive check for GEP operands.
  bool allGepOperandsAvailable(const Instruction *I,
                               const BasicBlock *HoistPt) const {
    for (const Use &Op : I->operands())
      if (const auto *Inst = dyn_cast<Instruction>(&Op))
        if (!DT->dominates(Inst->getParent(), HoistPt)) {
          if (const GetElementPtrInst *GepOp =
                  dyn_cast<GetElementPtrInst>(Inst)) {
            if (!allGepOperandsAvailable(GepOp, HoistPt))
              return false;
            // Gep is available if all operands of GepOp are available.
          } else {
            // Gep is not available if it has operands other than GEPs that are
            // defined in blocks not dominating HoistPt.
            return false;
          }
        }
    return true;
  }

  // Make all operands of the GEP available.
  void makeGepsAvailable(Instruction *Repl, BasicBlock *HoistPt,
                         const SmallVecInsn &InstructionsToHoist,
                         Instruction *Gep) const {
    assert(allGepOperandsAvailable(Gep, HoistPt) &&
           "GEP operands not available");

    Instruction *ClonedGep = Gep->clone();
    for (unsigned i = 0, e = Gep->getNumOperands(); i != e; ++i)
      if (Instruction *Op = dyn_cast<Instruction>(Gep->getOperand(i))) {

        // Check whether the operand is already available.
        if (DT->dominates(Op->getParent(), HoistPt))
          continue;

        // As a GEP can refer to other GEPs, recursively make all the operands
        // of this GEP available at HoistPt.
        if (GetElementPtrInst *GepOp = dyn_cast<GetElementPtrInst>(Op))
          makeGepsAvailable(ClonedGep, HoistPt, InstructionsToHoist, GepOp);
      }

    // Copy Gep and replace its uses in Repl with ClonedGep.
    ClonedGep->insertBefore(HoistPt->getTerminator());

    // Conservatively discard any optimization hints, they may differ on the
    // other paths.
    ClonedGep->dropUnknownNonDebugMetadata();

    // If we have optimization hints which agree with each other along different
    // paths, preserve them.
    for (const Instruction *OtherInst : InstructionsToHoist) {
      const GetElementPtrInst *OtherGep;
      if (auto *OtherLd = dyn_cast<LoadInst>(OtherInst))
        OtherGep = cast<GetElementPtrInst>(OtherLd->getPointerOperand());
      else
        OtherGep = cast<GetElementPtrInst>(
            cast<StoreInst>(OtherInst)->getPointerOperand());
      ClonedGep->andIRFlags(OtherGep);
    }

    // Replace uses of Gep with ClonedGep in Repl.
    Repl->replaceUsesOfWith(Gep, ClonedGep);
  }

  // In the case Repl is a load or a store, we make all their GEPs
  // available: GEPs are not hoisted by default to avoid the address
  // computations to be hoisted without the associated load or store.
  bool makeGepOperandsAvailable(Instruction *Repl, BasicBlock *HoistPt,
                                const SmallVecInsn &InstructionsToHoist) const {
    // Check whether the GEP of a ld/st can be synthesized at HoistPt.
    GetElementPtrInst *Gep = nullptr;
    Instruction *Val = nullptr;
    if (auto *Ld = dyn_cast<LoadInst>(Repl)) {
      Gep = dyn_cast<GetElementPtrInst>(Ld->getPointerOperand());
    } else if (auto *St = dyn_cast<StoreInst>(Repl)) {
      Gep = dyn_cast<GetElementPtrInst>(St->getPointerOperand());
      Val = dyn_cast<Instruction>(St->getValueOperand());
      // Check that the stored value is available.
      if (Val) {
        if (isa<GetElementPtrInst>(Val)) {
          // Check whether we can compute the GEP at HoistPt.
          if (!allGepOperandsAvailable(Val, HoistPt))
            return false;
        } else if (!DT->dominates(Val->getParent(), HoistPt))
          return false;
      }
    }

    // Check whether we can compute the Gep at HoistPt.
    if (!Gep || !allGepOperandsAvailable(Gep, HoistPt))
      return false;

    makeGepsAvailable(Repl, HoistPt, InstructionsToHoist, Gep);

    if (Val && isa<GetElementPtrInst>(Val))
      makeGepsAvailable(Repl, HoistPt, InstructionsToHoist, Val);

    return true;
  }

  std::pair<unsigned, unsigned> hoist(HoistingPointList &HPL) {
    unsigned NI = 0, NL = 0, NS = 0, NC = 0, NR = 0;
    for (const HoistingPointInfo &HP : HPL) {
      // Find out whether we already have one of the instructions in HoistPt,
      // in which case we do not have to move it.
      BasicBlock *HoistPt = HP.first;
      const SmallVecInsn &InstructionsToHoist = HP.second;
      Instruction *Repl = nullptr;
      for (Instruction *I : InstructionsToHoist)
        if (I->getParent() == HoistPt)
          // If there are two instructions in HoistPt to be hoisted in place:
          // update Repl to be the first one, such that we can rename the uses
          // of the second based on the first.
          if (!Repl || firstInBB(I, Repl))
            Repl = I;

      // Keep track of whether we moved the instruction so we know whether we
      // should move the MemoryAccess.
      bool MoveAccess = true;
      if (Repl) {
        // Repl is already in HoistPt: it remains in place.
        assert(allOperandsAvailable(Repl, HoistPt) &&
               "instruction depends on operands that are not available");
        MoveAccess = false;
      } else {
        // When we do not find Repl in HoistPt, select the first in the list
        // and move it to HoistPt.
        Repl = InstructionsToHoist.front();

        // We can move Repl in HoistPt only when all operands are available.
        // The order in which hoistings are done may influence the availability
        // of operands.
        if (!allOperandsAvailable(Repl, HoistPt)) {

          // When HoistingGeps there is nothing more we can do to make the
          // operands available: just continue.
          if (HoistingGeps)
            continue;

          // When not HoistingGeps we need to copy the GEPs.
          if (!makeGepOperandsAvailable(Repl, HoistPt, InstructionsToHoist))
            continue;
        }

        // Move the instruction at the end of HoistPt.
        Instruction *Last = HoistPt->getTerminator();
        MD->removeInstruction(Repl);
        Repl->moveBefore(Last);

        DFSNumber[Repl] = DFSNumber[Last]++;
      }

      MemoryAccess *NewMemAcc = MSSA->getMemoryAccess(Repl);

      if (MoveAccess) {
        if (MemoryUseOrDef *OldMemAcc =
                dyn_cast_or_null<MemoryUseOrDef>(NewMemAcc)) {
          // The definition of this ld/st will not change: ld/st hoisting is
          // legal when the ld/st is not moved past its current definition.
          MemoryAccess *Def = OldMemAcc->getDefiningAccess();
          NewMemAcc =
            MSSAUpdater->createMemoryAccessInBB(Repl, Def, HoistPt, MemorySSA::End);
          OldMemAcc->replaceAllUsesWith(NewMemAcc);
          MSSAUpdater->removeMemoryAccess(OldMemAcc);
        }
      }

      if (isa<LoadInst>(Repl))
        ++NL;
      else if (isa<StoreInst>(Repl))
        ++NS;
      else if (isa<CallInst>(Repl))
        ++NC;
      else // Scalar
        ++NI;

      // Remove and rename all other instructions.
      for (Instruction *I : InstructionsToHoist)
        if (I != Repl) {
          ++NR;
          if (auto *ReplacementLoad = dyn_cast<LoadInst>(Repl)) {
            ReplacementLoad->setAlignment(
                std::min(ReplacementLoad->getAlignment(),
                         cast<LoadInst>(I)->getAlignment()));
            ++NumLoadsRemoved;
          } else if (auto *ReplacementStore = dyn_cast<StoreInst>(Repl)) {
            ReplacementStore->setAlignment(
                std::min(ReplacementStore->getAlignment(),
                         cast<StoreInst>(I)->getAlignment()));
            ++NumStoresRemoved;
          } else if (auto *ReplacementAlloca = dyn_cast<AllocaInst>(Repl)) {
            ReplacementAlloca->setAlignment(
                std::max(ReplacementAlloca->getAlignment(),
                         cast<AllocaInst>(I)->getAlignment()));
          } else if (isa<CallInst>(Repl)) {
            ++NumCallsRemoved;
          }

          if (NewMemAcc) {
            // Update the uses of the old MSSA access with NewMemAcc.
            MemoryAccess *OldMA = MSSA->getMemoryAccess(I);
            OldMA->replaceAllUsesWith(NewMemAcc);
            MSSAUpdater->removeMemoryAccess(OldMA);
          }

          Repl->andIRFlags(I);
          combineKnownMetadata(Repl, I);
          I->replaceAllUsesWith(Repl);
          // Also invalidate the Alias Analysis cache.
          MD->removeInstruction(I);
          I->eraseFromParent();
        }

      // Remove MemorySSA phi nodes with the same arguments.
      if (NewMemAcc) {
        SmallPtrSet<MemoryPhi *, 4> UsePhis;
        for (User *U : NewMemAcc->users())
          if (MemoryPhi *Phi = dyn_cast<MemoryPhi>(U))
            UsePhis.insert(Phi);

        for (auto *Phi : UsePhis) {
          auto In = Phi->incoming_values();
          if (all_of(In, [&](Use &U) { return U == NewMemAcc; })) {
            Phi->replaceAllUsesWith(NewMemAcc);
            MSSAUpdater->removeMemoryAccess(Phi);
          }
        }
      }
    }

    NumHoisted += NL + NS + NC + NI;
    NumRemoved += NR;
    NumLoadsHoisted += NL;
    NumStoresHoisted += NS;
    NumCallsHoisted += NC;
    return {NI, NL + NC + NS};
  }

  // Hoist all expressions. Returns Number of scalars hoisted
  // and number of non-scalars hoisted.
  std::pair<unsigned, unsigned> hoistExpressions(Function &F) {
    InsnInfo II;
    LoadInfo LI;
    StoreInfo SI;
    CallInfo CI;
    for (BasicBlock *BB : depth_first(&F.getEntryBlock())) {
      int InstructionNb = 0;
      for (Instruction &I1 : *BB) {
        // If I1 cannot guarantee progress, subsequent instructions
        // in BB cannot be hoisted anyways.
        if (!isGuaranteedToTransferExecutionToSuccessor(&I1)) {
           HoistBarrier.insert(BB);
           break;
        }
        // Only hoist the first instructions in BB up to MaxDepthInBB. Hoisting
        // deeper may increase the register pressure and compilation time.
        if (MaxDepthInBB != -1 && InstructionNb++ >= MaxDepthInBB)
          break;

        // Do not value number terminator instructions.
        if (isa<TerminatorInst>(&I1))
          break;

        if (auto *Load = dyn_cast<LoadInst>(&I1))
          LI.insert(Load, VN);
        else if (auto *Store = dyn_cast<StoreInst>(&I1))
          SI.insert(Store, VN);
        else if (auto *Call = dyn_cast<CallInst>(&I1)) {
          if (auto *Intr = dyn_cast<IntrinsicInst>(Call)) {
            if (isa<DbgInfoIntrinsic>(Intr) ||
                Intr->getIntrinsicID() == Intrinsic::assume)
              continue;
          }
          if (Call->mayHaveSideEffects())
            break;

          if (Call->isConvergent())
            break;

          CI.insert(Call, VN);
        } else if (HoistingGeps || !isa<GetElementPtrInst>(&I1))
          // Do not hoist scalars past calls that may write to memory because
          // that could result in spills later. geps are handled separately.
          // TODO: We can relax this for targets like AArch64 as they have more
          // registers than X86.
          II.insert(&I1, VN);
      }
    }

    HoistingPointList HPL;
    computeInsertionPoints(II.getVNTable(), HPL, InsKind::Scalar);
    computeInsertionPoints(LI.getVNTable(), HPL, InsKind::Load);
    computeInsertionPoints(SI.getVNTable(), HPL, InsKind::Store);
    computeInsertionPoints(CI.getScalarVNTable(), HPL, InsKind::Scalar);
    computeInsertionPoints(CI.getLoadVNTable(), HPL, InsKind::Load);
    computeInsertionPoints(CI.getStoreVNTable(), HPL, InsKind::Store);
    return hoist(HPL);
  }
};

class GVNHoistLegacyPass : public FunctionPass {
public:
  static char ID;

  GVNHoistLegacyPass() : FunctionPass(ID) {
    initializeGVNHoistLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &MD = getAnalysis<MemoryDependenceWrapperPass>().getMemDep();
    auto &MSSA = getAnalysis<MemorySSAWrapperPass>().getMSSA();

    GVNHoist G(&DT, &AA, &MD, &MSSA);
    return G.run(F);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<MemoryDependenceWrapperPass>();
    AU.addRequired<MemorySSAWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<MemorySSAWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
} // namespace

PreservedAnalyses GVNHoistPass::run(Function &F, FunctionAnalysisManager &AM) {
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  AliasAnalysis &AA = AM.getResult<AAManager>(F);
  MemoryDependenceResults &MD = AM.getResult<MemoryDependenceAnalysis>(F);
  MemorySSA &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();
  GVNHoist G(&DT, &AA, &MD, &MSSA);
  if (!G.run(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<MemorySSAAnalysis>();
  PA.preserve<GlobalsAA>();
  return PA;
}

char GVNHoistLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(GVNHoistLegacyPass, "gvn-hoist",
                      "Early GVN Hoisting of Expressions", false, false)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(GVNHoistLegacyPass, "gvn-hoist",
                    "Early GVN Hoisting of Expressions", false, false)

FunctionPass *llvm::createGVNHoistPass() { return new GVNHoistLegacyPass(); }
