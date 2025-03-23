//===- MergeICmps.cpp - Optimize chains of integer comparisons ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass turns chains of integer comparisons into memcmp (the memcmp is
// later typically inlined as a chain of efficient hardware comparisons). This
// typically benefits c++ member or nonmember operator==().
//
// The basic idea is to replace a longer chain of integer comparisons loaded
// from contiguous memory locations into a shorter chain of larger integer
// comparisons. Benefits are double:
//  - There are less jumps, and therefore less opportunities for mispredictions
//    and I-cache misses.
//  - Code size is smaller, both because jumps are removed and because the
//    encoding of a 2*n byte compare is smaller than that of two n-byte
//    compares.
//
// Example:
//
//  struct S {
//    int a;
//    char b;
//    char c;
//    uint16_t d;
//    bool operator==(const S& o) const {
//      return a == o.a && b == o.b && c == o.c && d == o.d;
//    }
//  };
//
//  Is optimized as :
//
//    bool S::operator==(const S& o) const {
//      return memcmp(this, &o, 8) == 0;
//    }
//
//  Which will later be expanded (ExpandMemCmp) as a single 8-bytes icmp.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/MergeICmps.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

#define DEBUG_TYPE "mergeicmps"

// A BCE atom "Binary Compare Expression Atom" represents an integer load
// that is a constant offset from a base value, e.g. `a` or `o.c` in the example
// at the top.
struct BCEAtom {
  BCEAtom() = default;
  BCEAtom(GetElementPtrInst *GEP, LoadInst *LoadI, int BaseId, APInt Offset)
      : GEP(GEP), LoadI(LoadI), BaseId(BaseId), Offset(std::move(Offset)) {}

  BCEAtom(const BCEAtom &) = delete;
  BCEAtom &operator=(const BCEAtom &) = delete;

  BCEAtom(BCEAtom &&that) = default;
  BCEAtom &operator=(BCEAtom &&that) {
    if (this == &that)
      return *this;
    GEP = that.GEP;
    LoadI = that.LoadI;
    BaseId = that.BaseId;
    Offset = std::move(that.Offset);
    return *this;
  }

  // We want to order BCEAtoms by (Base, Offset). However we cannot use
  // the pointer values for Base because these are non-deterministic.
  // To make sure that the sort order is stable, we first assign to each atom
  // base value an index based on its order of appearance in the chain of
  // comparisons. We call this index `BaseOrdering`. For example, for:
  //    b[3] == c[2] && a[1] == d[1] && b[4] == c[3]
  //    |  block 1 |    |  block 2 |    |  block 3 |
  // b gets assigned index 0 and a index 1, because b appears as LHS in block 1,
  // which is before block 2.
  // We then sort by (BaseOrdering[LHS.Base()], LHS.Offset), which is stable.
  bool operator<(const BCEAtom &O) const {
    return BaseId != O.BaseId ? BaseId < O.BaseId : Offset.slt(O.Offset);
  }

  GetElementPtrInst *GEP = nullptr;
  LoadInst *LoadI = nullptr;
  unsigned BaseId = 0;
  APInt Offset;
};

// A class that assigns increasing ids to values in the order in which they are
// seen. See comment in `BCEAtom::operator<()``.
class BaseIdentifier {
public:
  // Returns the id for value `Base`, after assigning one if `Base` has not been
  // seen before.
  int getBaseId(const Value *Base) {
    assert(Base && "invalid base");
    const auto Insertion = BaseToIndex.try_emplace(Base, Order);
    if (Insertion.second)
      ++Order;
    return Insertion.first->second;
  }

private:
  unsigned Order = 1;
  DenseMap<const Value*, int> BaseToIndex;
};


// All Instructions related to a comparison.
typedef SmallDenseSet<const Instruction *, 8> InstructionSet;

// If this value is a load from a constant offset w.r.t. a base address, and
// there are no other users of the load or address, returns the base address and
// the offset.
BCEAtom visitICmpLoadOperand(Value *const Val, BaseIdentifier &BaseId, InstructionSet* BlockInsts) {
  auto *const LoadI = dyn_cast<LoadInst>(Val);
  if (!LoadI)
    return {};
  LLVM_DEBUG(dbgs() << "load\n");
  if (LoadI->isUsedOutsideOfBlock(LoadI->getParent())) {
    LLVM_DEBUG(dbgs() << "used outside of block\n");
    return {};
  }
  // Do not optimize atomic loads to non-atomic memcmp
  if (!LoadI->isSimple()) {
    LLVM_DEBUG(dbgs() << "volatile or atomic\n");
    return {};
  }
  Value *Addr = LoadI->getOperand(0);
  if (Addr->getType()->getPointerAddressSpace() != 0) {
    LLVM_DEBUG(dbgs() << "from non-zero AddressSpace\n");
    return {};
  }
  const auto &DL = LoadI->getDataLayout();
  if (!isDereferenceablePointer(Addr, LoadI->getType(), DL)) {
    LLVM_DEBUG(dbgs() << "not dereferenceable\n");
    // We need to make sure that we can do comparison in any order, so we
    // require memory to be unconditionally dereferenceable.
    return {};
  }

  APInt Offset = APInt(DL.getIndexTypeSizeInBits(Addr->getType()), 0);
  Value *Base = Addr;
  auto *GEP = dyn_cast<GetElementPtrInst>(Addr);
  if (GEP) {
    LLVM_DEBUG(dbgs() << "GEP\n");
    if (GEP->isUsedOutsideOfBlock(LoadI->getParent())) {
      LLVM_DEBUG(dbgs() << "used outside of block\n");
      return {};
    }
    if (!GEP->accumulateConstantOffset(DL, Offset))
      return {};
    Base = GEP->getPointerOperand();
    BlockInsts->insert(GEP);
  }
  BlockInsts->insert(LoadI);
  return BCEAtom(GEP, LoadI, BaseId.getBaseId(Base), Offset);
}


// An abstract parent class that can either be a comparison of
// two BCEAtoms with the same offsets to a base pointer (BCECmp)
// or a comparison of a single BCEAtom with a constant (BCEConstCmp).
struct Comparison {
public:
  enum CompKind {
    CK_ConstCmp,
    CK_BceCmp,
  };
private:
  const CompKind Kind;
public:
  int SizeBits;
  const ICmpInst *CmpI;

  Comparison(CompKind K, int SizeBits, const ICmpInst *CmpI)
        : Kind(K), SizeBits(SizeBits), CmpI(CmpI) {}
  CompKind getKind() const { return Kind; }

  virtual ~Comparison() = default;
  bool areContiguous(const Comparison& Other) const;
  bool operator<(const Comparison &Other) const;
};

// A comparison between a BCE atom and an integer constant.
// If these BCE atoms are chained and access adjacent memory then they too can be merged, e.g.
// ```
// int *p = ...;
// int a = p[0];
// int b = p[1];
// return a == 100 && b == 2;
// ```
struct BCEConstCmp : public Comparison {
  BCEAtom Lhs;
  Constant* Const;

  BCEConstCmp(BCEAtom L, Constant* Const, int SizeBits, const ICmpInst *CmpI)
      : Comparison(CK_ConstCmp, SizeBits,CmpI), Lhs(std::move(L)), Const(Const) {}
  static bool classof(const Comparison* C) {
    return C->getKind() == CK_ConstCmp;
  }
};

// A comparison between two BCE atoms, e.g. `a == o.a` in the example at the
// top.
// Note: the terminology is misleading: the comparison is symmetric, so there
// is no real {l/r}hs. What we want though is to have the same base on the
// left (resp. right), so that we can detect consecutive loads. To ensure this
// we put the smallest atom on the left.
struct BCECmp : public Comparison {
  BCEAtom Lhs;
  BCEAtom Rhs;

  BCECmp(BCEAtom L, BCEAtom R, int SizeBits, const ICmpInst *CmpI)
      : Comparison(CK_BceCmp, SizeBits,CmpI), Lhs(std::move(L)), Rhs(std::move(R))  {
    if (Rhs < Lhs) std::swap(Rhs, Lhs);
  }
  static bool classof(const Comparison* C) {
    return C->getKind() == CK_BceCmp;
  }
};

// TODO: this can be improved to take alignment into account.
bool Comparison::areContiguous(const Comparison& Other) const {
  assert(isa<BCEConstCmp>(this) == isa<BCEConstCmp>(Other) && "Comparisons are of same kind");
  if (isa<BCEConstCmp>(this)) {
    const auto& First = cast<BCEConstCmp>(this);
    const auto& Second = cast<BCEConstCmp>(Other);

    return First->Lhs.BaseId == Second.Lhs.BaseId &&
           First->Lhs.Offset + First->SizeBits / 8 == Second.Lhs.Offset;
  }
  const auto& First = cast<BCECmp>(this);
  const auto& Second = cast<BCECmp>(Other);

  return First->Lhs.BaseId == Second.Lhs.BaseId &&
         First->Rhs.BaseId == Second.Rhs.BaseId &&
         First->Lhs.Offset + First->SizeBits / 8 == Second.Lhs.Offset &&
         First->Rhs.Offset + First->SizeBits / 8 == Second.Rhs.Offset;
}
bool Comparison::operator<(const Comparison& Other) const {
  assert(isa<BCEConstCmp>(this) == isa<BCEConstCmp>(Other) && "Comparisons are of same kind");
  if (isa<BCEConstCmp>(this)) {
    const auto& First = cast<BCEConstCmp>(this);
    const auto& Second = cast<BCEConstCmp>(Other);
    return First->Lhs < Second.Lhs;
  }
  const auto& First = cast<BCECmp>(this);
  const auto& Second = cast<BCECmp>(Other);
  return std::tie(First->Lhs,First->Rhs) < std::tie(Second.Lhs,Second.Rhs);
}

// Represents multiple comparisons inside of a single basic block.
// This happens if multiple basic blocks have previously been merged into a single block using a select node.
class IntraCmpChain {
  // TODO: this could probably be a unique-ptr but current impl relies on some copies
  std::vector<std::shared_ptr<Comparison>> CmpChain;

public:
  IntraCmpChain(std::shared_ptr<Comparison> C) : CmpChain{C} {}
  IntraCmpChain combine(const IntraCmpChain OtherChain) {
    CmpChain.insert(CmpChain.end(), OtherChain.CmpChain.begin(), OtherChain.CmpChain.end());
    return *this;
  }
  std::vector<std::shared_ptr<Comparison>> getCmpChain() const {
    return CmpChain;
  }
};

// A basic block that contains one or more comparisons.
class MultBCECmpBlock {
 public:
  MultBCECmpBlock(std::vector<std::shared_ptr<Comparison>> Cmps, BasicBlock *BB, InstructionSet BlockInsts)
      : BB(BB), BlockInsts(std::move(BlockInsts)), Cmps(std::move(Cmps)) {}

  std::vector<std::shared_ptr<Comparison>> getCmps() {
    return Cmps;
  }

  // Returns true if the block does other works besides comparison.
  bool doesOtherWork() const;

  // Returns true if the non-BCE-cmp instructions can be separated from BCE-cmp
  // instructions in the block.
  bool canSplit(AliasAnalysis &AA) const;

  // Return true if all the relevant instructions in the BCE-cmp-block can
  // be sunk below this instruction. By doing this, we know we can separate the
  // BCE-cmp-block instructions from the non-BCE-cmp-block instructions in the
  // block.
  bool canSinkBCECmpInst(const Instruction *, AliasAnalysis &AA) const;

  // Returns all instructions that should be split off of the comparison chain.
  llvm::SmallVector<Instruction *, 4> getAllSplitInsts(AliasAnalysis &AA) const;

  // The basic block where this comparison happens.
  BasicBlock *BB;
  // Instructions relating to the BCECmp and branch.
  InstructionSet BlockInsts;

private:
  std::vector<std::shared_ptr<Comparison>> Cmps;
};

// A basic block with single a comparison between two BCE atoms.
// The block might do extra work besides the atom comparison, in which case
// doesOtherWork() returns true. Under some conditions, the block can be
// split into the atom comparison part and the "other work" part
// (see canSplit()).
class SingleBCECmpBlock {
 public:
  SingleBCECmpBlock(std::shared_ptr<Comparison> Cmp, BasicBlock* BB, unsigned OrigOrder)
      : BB(BB), OrigOrder(OrigOrder), Cmp(std::move(Cmp)) {}

  SingleBCECmpBlock(std::shared_ptr<Comparison> Cmp, BasicBlock* BB, unsigned OrigOrder,
                    llvm::SmallVector<Instruction *, 4> SplitInsts)
      : BB(BB), OrigOrder(OrigOrder), RequireSplit(true), Cmp(std::move(Cmp)), SplitInsts(SplitInsts) {}

  const BCEAtom* Lhs() const {
    if (auto *const BceConstCmp = dyn_cast<BCEConstCmp>(Cmp.get()))
      return &BceConstCmp->Lhs;
    auto *const BceCmp = cast<BCECmp>(Cmp.get());
    return &BceCmp->Lhs;
  }
  const Comparison* getCmp() const { return Cmp.get(); }
  bool operator<(const SingleBCECmpBlock &O) const {
    return *Cmp < *O.Cmp;
  }

  // We can separate the BCE-cmp-block instructions and the non-BCE-cmp-block
  // instructions. Split the old block and move all non-BCE-cmp-insts into the
  // new parent block.
  void split(BasicBlock *NewParent, AliasAnalysis &AA) const;

  // The basic block where this comparison happens.
  BasicBlock *BB;
  // Original order of this block in the chain.
  unsigned OrigOrder = 0;
  // The block requires splitting.
  bool RequireSplit = false;

private:
  std::shared_ptr<Comparison> Cmp;
  llvm::SmallVector<Instruction *, 4> SplitInsts;
};

bool MultBCECmpBlock::canSinkBCECmpInst(const Instruction *Inst,
                                    AliasAnalysis &AA) const {
  // If this instruction may clobber the loads and is in middle of the BCE cmp
  // block instructions, then bail for now.
  if (Inst->mayWriteToMemory()) {
    auto MayClobber = [&](LoadInst *LI) {
      // If a potentially clobbering instruction comes before the load,
      // we can still safely sink the load.
      return (Inst->getParent() != LI->getParent() || !Inst->comesBefore(LI)) &&
             isModSet(AA.getModRefInfo(Inst, MemoryLocation::get(LI)));
    };
    auto CmpLoadsAreClobbered = [&](const auto& Cmp) {
      if (auto *const BceConstCmp = dyn_cast<BCEConstCmp>(Cmp.get()))
        return MayClobber(BceConstCmp->Lhs.LoadI);
      auto *const BceCmp = cast<BCECmp>(Cmp.get());
      return MayClobber(BceCmp->Lhs.LoadI) || MayClobber(BceCmp->Rhs.LoadI);
    };
    if (llvm::any_of(Cmps, CmpLoadsAreClobbered))
      return false;
  }
  // Make sure this instruction does not use any of the BCE cmp block
  // instructions as operand.
  return llvm::none_of(Inst->operands(), [&](const Value *Op) {
    const Instruction *OpI = dyn_cast<Instruction>(Op);
    return OpI && BlockInsts.contains(OpI);
  });
}

void SingleBCECmpBlock::split(BasicBlock *NewParent, AliasAnalysis &AA) const {
  // Do the actual splitting.
  for (Instruction *Inst : reverse(SplitInsts))
    Inst->moveBeforePreserving(*NewParent, NewParent->begin());
}

bool MultBCECmpBlock::canSplit(AliasAnalysis &AA) const {
  for (Instruction &Inst : *BB) {
    if (!BlockInsts.count(&Inst)) {
      if (!canSinkBCECmpInst(&Inst, AA))
        return false;
    }
  }
  return true;
}

bool MultBCECmpBlock::doesOtherWork() const {
  // TODO(courbet): Can we allow some other things ? This is very conservative.
  // We might be able to get away with anything does not have any side
  // effects outside of the basic block.
  // Note: The GEPs and/or loads are not necessarily in the same block.
  for (const Instruction &Inst : *BB) {
    if (!BlockInsts.count(&Inst))
      return true;
  }
  return false;
}

llvm::SmallVector<Instruction *, 4> MultBCECmpBlock::getAllSplitInsts(AliasAnalysis &AA) const {
  llvm::SmallVector<Instruction *, 4> SplitInsts;
  for (Instruction& Inst : *BB) {
    if (BlockInsts.count(&Inst))
      continue;
    assert(canSinkBCECmpInst(&Inst, AA) && "Split unsplittable block");
    // This is a non-BCE-cmp-block instruction. And it can be separated
    // from the BCE-cmp-block instructions.
    SplitInsts.push_back(&Inst);
  }
  return SplitInsts;
}


// Visit the given comparison. If this is a comparison between two valid
// BCE atoms, or between a BCE atom and a constant, returns the comparison.
std::optional<std::shared_ptr<Comparison>> visitICmp(const ICmpInst *const CmpI,
                                const ICmpInst::Predicate ExpectedPredicate,
                                BaseIdentifier &BaseId, InstructionSet *BlockInsts) {
  // The comparison can only be used once:
  //  - For intermediate blocks, as a branch condition.
  //  - For the final block, as an incoming value for the Phi.
  // If there are any other uses of the comparison, we cannot merge it with
  // other comparisons as we would create an orphan use of the value.
  if (!CmpI->hasOneUse()) {
    LLVM_DEBUG(dbgs() << "cmp has several uses\n");
    return std::nullopt;
  }
  if (CmpI->getPredicate() != ExpectedPredicate)
    return std::nullopt;
  LLVM_DEBUG(dbgs() << "cmp "
                    << (ExpectedPredicate == ICmpInst::ICMP_EQ ? "eq" : "ne")
                    << "\n");
  // First operand is always a load
  auto Lhs = visitICmpLoadOperand(CmpI->getOperand(0), BaseId, BlockInsts);
  if (!Lhs.BaseId)
    return std::nullopt;

  // Second operand can either be load if doing compare between two BCE atoms or 
  // can be constant if comparing adjacent memory to constant
  auto* RhsOperand = CmpI->getOperand(1);
  const auto &DL = CmpI->getDataLayout();
  int SizeBits = DL.getTypeSizeInBits(CmpI->getOperand(0)->getType());

  BlockInsts->insert(CmpI);
  if (auto const& Const = dyn_cast<Constant>(RhsOperand))
    return std::make_shared<BCEConstCmp>(BCEConstCmp(std::move(Lhs), Const, SizeBits, CmpI));

  auto Rhs = visitICmpLoadOperand(RhsOperand, BaseId, BlockInsts);
  if (!Rhs.BaseId)
    return std::nullopt;
  return std::make_shared<BCECmp>(BCECmp(std::move(Lhs), std::move(Rhs), SizeBits, CmpI));
}

// Chain of comparisons inside a single basic block connected using `select` nodes.
std::optional<IntraCmpChain> visitComparison(Value*, ICmpInst::Predicate, BaseIdentifier&, InstructionSet*);

std::optional<IntraCmpChain> visitSelect(const SelectInst *const SelectI,
                                  ICmpInst::Predicate ExpectedPredicate, BaseIdentifier& BaseId, InstructionSet *BlockInsts) {
  if (!SelectI->hasOneUse()) {
    LLVM_DEBUG(dbgs() << "select has several uses\n");
    return std::nullopt;
  }
  auto* Cmp1 = dyn_cast<ICmpInst>(SelectI->getOperand(0));
  auto* Sel1 = dyn_cast<SelectInst>(SelectI->getOperand(0));
  auto const& Cmp2 = dyn_cast<ICmpInst>(SelectI->getOperand(1));
  auto const& ConstantI = dyn_cast<Constant>(SelectI->getOperand(2));

  if (!(Cmp1 || Sel1) || !Cmp2 || !ConstantI || !ConstantI->isZeroValue())
    return std::nullopt;

  auto Lhs = visitComparison(SelectI->getOperand(0),ExpectedPredicate,BaseId,BlockInsts);
  if (!Lhs)
    return std::nullopt;
  auto Rhs = visitComparison(Cmp2,ExpectedPredicate,BaseId,BlockInsts);
  if (!Rhs)
    return std::nullopt;

  BlockInsts->insert(SelectI);
  return Lhs->combine(std::move(*Rhs));
}

std::optional<IntraCmpChain> visitComparison(Value *Cond,
            ICmpInst::Predicate ExpectedPredicate,BaseIdentifier &BaseId, InstructionSet *BlockInsts) {
  if (auto *CmpI = dyn_cast<ICmpInst>(Cond)) {
    auto CmpVisit = visitICmp(CmpI, ExpectedPredicate, BaseId, BlockInsts);
    if (!CmpVisit)
      return std::nullopt;
    return IntraCmpChain(*CmpVisit);
  }
  if (auto *SelectI = dyn_cast<SelectInst>(Cond))
    return visitSelect(SelectI, ExpectedPredicate, BaseId, BlockInsts);

  return std::nullopt;
}

// Visit the given comparison block. If this is a comparison between two valid
// BCE atoms, returns the comparison.
std::optional<MultBCECmpBlock> visitCmpBlock(Value *const Val,
                                         BasicBlock *const Block,
                                         const BasicBlock *const PhiBlock,
                                         BaseIdentifier &BaseId) {
  if (Block->empty())
    return std::nullopt;
  auto *const BranchI = dyn_cast<BranchInst>(Block->getTerminator());
  if (!BranchI)
    return std::nullopt;
  LLVM_DEBUG(dbgs() << "branch\n");
  Value *Cond;
  ICmpInst::Predicate ExpectedPredicate;
  if (BranchI->isUnconditional()) {
    // In this case, we expect an incoming value which is the result of the
    // comparison. This is the last link in the chain of comparisons (note
    // that this does not mean that this is the last incoming value, blocks
    // can be reordered).
    Cond = Val;
    ExpectedPredicate = ICmpInst::ICMP_EQ;
  } else {
    // In this case, we expect a constant incoming value (the comparison is
    // chained).
    const auto *const Const = cast<ConstantInt>(Val);
    LLVM_DEBUG(dbgs() << "const\n");
    if (!Const->isZero())
      return std::nullopt;
    LLVM_DEBUG(dbgs() << "false\n");
    assert(BranchI->getNumSuccessors() == 2 && "expecting a cond branch");
    BasicBlock *const FalseBlock = BranchI->getSuccessor(1);
    Cond = BranchI->getCondition();
    ExpectedPredicate =
        FalseBlock == PhiBlock ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_NE;
  }

  InstructionSet BlockInsts;
  std::optional<IntraCmpChain> Result = visitComparison(Cond, ExpectedPredicate, BaseId, &BlockInsts);
  if (!Result)
    return std::nullopt;

  BlockInsts.insert(BranchI);
  return MultBCECmpBlock(Result->getCmpChain(), Block, BlockInsts);
}

void emitDebugInfo(std::shared_ptr<Comparison> Cmp, BasicBlock* BB) {
  LLVM_DEBUG(dbgs() << "Block '" << BB->getName());
  if (auto* ConstCmp = dyn_cast<BCEConstCmp>(Cmp.get())) {
    LLVM_DEBUG(dbgs() << "': Found constant-cmp of " << Cmp->SizeBits
    << " bits including " << ConstCmp->Lhs.BaseId << " + "
    << ConstCmp->Lhs.Offset << "\n");
    return;
  }
  auto* BceCmp = cast<BCECmp>(Cmp.get());
  LLVM_DEBUG(dbgs() << "': Found cmp of " << BceCmp->SizeBits
  << " bits between " << BceCmp->Lhs.BaseId << " + "
  << BceCmp->Lhs.Offset << " and "
  << BceCmp->Rhs.BaseId << " + "
  << BceCmp->Rhs.Offset << "\n");
}

// Enqueues all comparisons of a mult-block.
// If the block requires splitting then adds `OtherInsts` to the block too.
static inline void enqueueSingleCmps(std::vector<SingleBCECmpBlock> &Comparisons,
                                MultBCECmpBlock &&CmpBlock, AliasAnalysis &AA, bool RequireSplit) {
  bool hasAlreadySplit = false;
  for (auto& Cmp : CmpBlock.getCmps()) {
    emitDebugInfo(Cmp, CmpBlock.BB);
    unsigned OrigOrder = Comparisons.size();
    if (RequireSplit && !hasAlreadySplit) {
      hasAlreadySplit = true;
      auto SplitInsts = CmpBlock.getAllSplitInsts(AA);
      Comparisons.push_back(SingleBCECmpBlock(Cmp, CmpBlock.BB, OrigOrder, SplitInsts));
      continue;
    }
    Comparisons.push_back(SingleBCECmpBlock(Cmp, CmpBlock.BB, OrigOrder));
  }
}

// A chain of comparisons.
class BCECmpChain {
public:
  using ContiguousBlocks = std::vector<SingleBCECmpBlock>;

  BCECmpChain(const std::vector<BasicBlock *> &Blocks, PHINode &Phi,
              AliasAnalysis &AA);

  bool simplify(const TargetLibraryInfo &TLI, AliasAnalysis &AA,
                DomTreeUpdater &DTU);

  bool multBlockOnlyPartiallyMerged();

  bool atLeastOneMerged() const {
    return any_of(MergedBlocks_,
                  [](const auto &Blocks) { return Blocks.size() > 1; });
  };

private:
  PHINode &Phi_;
  // The list of all blocks in the chain, grouped by contiguity.
  // First all BCE comparisons followed by all BCE-Const comparisons.
  std::vector<ContiguousBlocks> MergedBlocks_;
  // The original entry block (before sorting);
  BasicBlock *EntryBlock_;
};


// Returns true if a merge in the chain depends on a basic block where not every comparison is merged.
// NOTE: This is pretty restrictive and could potentially be handled using an improved tradeoff heuristic.
bool BCECmpChain::multBlockOnlyPartiallyMerged() {
  llvm::SmallDenseSet<const BasicBlock*, 8> UnmergedBlocks, MergedBB;

  for (auto& Merged : MergedBlocks_) {
    if (Merged.size() == 1) {
      UnmergedBlocks.insert(Merged[0].BB);
      continue;
    }
    for (auto& C : Merged)
      MergedBB.insert(C.BB);
  }
  return llvm::any_of(MergedBB, [&](const BasicBlock* BB){
    return UnmergedBlocks.contains(BB);
  });
}

static unsigned getMinOrigOrder(const BCECmpChain::ContiguousBlocks &Blocks) {
  unsigned MinOrigOrder = std::numeric_limits<unsigned>::max();
  for (const SingleBCECmpBlock &Block : Blocks)
    MinOrigOrder = std::min(MinOrigOrder, Block.OrigOrder);
  return MinOrigOrder;
}

/// Given a chain of comparison blocks (of the same kind), groups the blocks into contiguous
/// ranges that can be merged together into a single comparison.
template<class RandomIt>
static void mergeBlocks(RandomIt First, RandomIt Last,
                        std::vector<BCECmpChain::ContiguousBlocks>* MergedBlocks) {
  // Sort to detect continuous offsets.
  llvm::sort(First, Last,
             [](const SingleBCECmpBlock &LhsBlock, const SingleBCECmpBlock &RhsBlock) {
              return LhsBlock < RhsBlock;
             });

  BCECmpChain::ContiguousBlocks *LastMergedBlock = nullptr;
  int Offset = MergedBlocks->size();
  for (auto& BlockIt = First; BlockIt != Last; ++BlockIt) {
    if (!LastMergedBlock || !LastMergedBlock->back().getCmp()->areContiguous(*BlockIt->getCmp())) {
      MergedBlocks->emplace_back();
      LastMergedBlock = &MergedBlocks->back();
    } else {
      LLVM_DEBUG(dbgs() << "Merging block " << BlockIt->BB->getName() << " into "
                        << LastMergedBlock->back().BB->getName() << "\n");
    }
    LastMergedBlock->push_back(std::move(*BlockIt));
  }

  // While we allow reordering for merging, do not reorder unmerged comparisons.
  // Doing so may introduce branch on poison.
  llvm::sort(MergedBlocks->begin() + Offset, MergedBlocks->end(), [](const BCECmpChain::ContiguousBlocks &LhsBlocks,
                              const BCECmpChain::ContiguousBlocks &RhsBlocks) {
    return getMinOrigOrder(LhsBlocks) < getMinOrigOrder(RhsBlocks);
  });
}


BCECmpChain::BCECmpChain(const std::vector<BasicBlock *> &Blocks, PHINode &Phi,
                         AliasAnalysis &AA)
    : Phi_(Phi) {
  assert(!Blocks.empty() && "a chain should have at least one block");
  // Now look inside blocks to check for BCE comparisons.
  std::vector<SingleBCECmpBlock> Comparisons;
  BaseIdentifier BaseId;
  for (BasicBlock *const Block : Blocks) {
    assert(Block && "invalid block");
    std::optional<MultBCECmpBlock> CmpBlock = visitCmpBlock(
        Phi.getIncomingValueForBlock(Block), Block, Phi.getParent(), BaseId);
    if (!CmpBlock) {
      LLVM_DEBUG(dbgs() << "chain with invalid BCECmpBlock, no merge.\n");
      return;
    }
    if (CmpBlock->doesOtherWork()) {
      LLVM_DEBUG(dbgs() << "block '" << CmpBlock->BB->getName()
                        << "' does extra work besides compare\n");
      if (Comparisons.empty()) {
        // This is the initial block in the chain, in case this block does other
        // work, we can try to split the block and move the irrelevant
        // instructions to the predecessor.
        //
        // If this is not the initial block in the chain, splitting it wont
        // work.
        //
        // As once split, there will still be instructions before the BCE cmp
        // instructions that do other work in program order, i.e. within the
        // chain before sorting. Unless we can abort the chain at this point
        // and start anew.
        //
        // NOTE: we only handle blocks a with single predecessor for now.
        if (CmpBlock->canSplit(AA)) {
          LLVM_DEBUG(dbgs()
                     << "Split initial block '" << CmpBlock->BB->getName()
                     << "' that does extra work besides compare\n");
          enqueueSingleCmps(Comparisons, std::move(*CmpBlock), AA, true);
        } else {
          LLVM_DEBUG(dbgs()
                     << "ignoring initial block '" << CmpBlock->BB->getName()
                     << "' that does extra work besides compare\n");
        }
        continue;
      }
      // TODO(courbet): Right now we abort the whole chain. We could be
      // merging only the blocks that don't do other work and resume the
      // chain from there. For example:
      //  if (a[0] == b[0]) {  // bb1
      //    if (a[1] == b[1]) {  // bb2
      //      some_value = 3; //bb3
      //      if (a[2] == b[2]) { //bb3
      //        do a ton of stuff  //bb4
      //      }
      //    }
      //  }
      //
      // This is:
      //
      // bb1 --eq--> bb2 --eq--> bb3* -eq--> bb4 --+
      //  \            \           \               \
      //   ne           ne          ne              \
      //    \            \           \               v
      //     +------------+-----------+----------> bb_phi
      //
      // We can only merge the first two comparisons, because bb3* does
      // "other work" (setting some_value to 3).
      // We could still merge bb1 and bb2 though.
      return;
    }
    enqueueSingleCmps(Comparisons, std::move(*CmpBlock), AA, false);
  }
  
  // It is possible we have no suitable comparison to merge.
  if (Comparisons.empty()) {
    LLVM_DEBUG(dbgs() << "chain with no BCE basic blocks, no merge\n");
    return;
  }

  EntryBlock_ = Comparisons[0].BB;

  auto isConstCmp = [](SingleBCECmpBlock& C) { return isa<BCEConstCmp>(C.getCmp()); };
  auto BceIt = std::partition(Comparisons.begin(), Comparisons.end(), isConstCmp);

  // The chain that requires splitting should always be first.
  // If no chain requires splitting then defaults to BCE-comparisons coming first.
  if (std::any_of(Comparisons.begin(), BceIt,
                   [](const SingleBCECmpBlock &B) { return B.RequireSplit; })) {
    mergeBlocks(Comparisons.begin(), BceIt, &MergedBlocks_);
    mergeBlocks(BceIt, Comparisons.end(), &MergedBlocks_);
  } else {
    mergeBlocks(BceIt, Comparisons.end(), &MergedBlocks_);
    mergeBlocks(Comparisons.begin(), BceIt, &MergedBlocks_);
  }
}

namespace {

// A class to compute the name of a set of merged basic blocks.
// This is optimized for the common case of no block names.
class MergedBlockName {
  // Storage for the uncommon case of several named blocks.
  SmallString<16> Scratch;

public:
  explicit MergedBlockName(ArrayRef<SingleBCECmpBlock> Comparisons)
      : Name(makeName(Comparisons)) {}
  const StringRef Name;

private:
  StringRef makeName(ArrayRef<SingleBCECmpBlock> Comparisons) {
    assert(!Comparisons.empty() && "no basic block");
    // Fast path: only one block, or no names at all.
    if (Comparisons.size() == 1)
      return Comparisons[0].BB->getName();
    // Since multiple comparisons can come from the same basic block
    // (when using select inst) don't want to repeat same name twice
    UniqueVector<StringRef> UniqueNames;
    for (const auto& B : Comparisons)
      UniqueNames.insert(B.BB->getName());
    const int size = std::accumulate(UniqueNames.begin(), UniqueNames.end(), 0,
                                     [](int i, const StringRef &Name) {
                                       return i + Name.size();
                                     });
    if (size == 0)
      return StringRef("", 0);

    // Slow path: at least two blocks, at least one block with a name.
    Scratch.clear();
    // We'll have `size` bytes for name and `Comparisons.size() - 1` bytes for
    // separators.
    Scratch.reserve(size + UniqueNames.size() - 1);
    const auto append = [this](StringRef str) {
      Scratch.append(str.begin(), str.end());
    };
    // UniqueVector's index starts at 1
    append(UniqueNames[1]);
    for (int I = 2, E = UniqueNames.size(); I <= E; ++I) {
      StringRef BBName = UniqueNames[I];
      if (!BBName.empty()) {
        append("+");
        append(BBName);
      }
    }
    return Scratch.str();
  }
};
} // namespace


// Add a branch to the next basic block in the chain.
void updateBranching(Value* CondResult,
                     IRBuilder<>& Builder,
                     BasicBlock *BB,
                     BasicBlock *const NextCmpBlock,
                     PHINode &Phi,
                     LLVMContext &Context,
                     const TargetLibraryInfo &TLI,
                     AliasAnalysis &AA, DomTreeUpdater &DTU) {
  BasicBlock *const PhiBB = Phi.getParent();
  if (NextCmpBlock == PhiBB) {
    // Continue to phi, passing it the comparison result.
    Builder.CreateBr(PhiBB);
    Phi.addIncoming(CondResult, BB);
    DTU.applyUpdates({{DominatorTree::Insert, BB, PhiBB}});
  } else {
    // Continue to next block if equal, exit to phi else.
    Builder.CreateCondBr(CondResult, NextCmpBlock, PhiBB);
    Phi.addIncoming(ConstantInt::getFalse(Context), BB);
    DTU.applyUpdates({{DominatorTree::Insert, BB, NextCmpBlock},
                      {DominatorTree::Insert, BB, PhiBB}});
  }
}

// Builds constant-struct to compare pointer to during memcmp(). Has to be a chain of const-comparisons.
AllocaInst* buildStruct(ArrayRef<SingleBCECmpBlock>& Comparisons, IRBuilder<>& Builder, LLVMContext &Context) {
  std::vector<Constant*> Constants;
  std::vector<Type*> Types;

  for (const auto& BceBlock : Comparisons) {
    assert(isa<BCEConstCmp>(BceBlock.getCmp()) && "Const-cmp-chain can only contain const comparisons");
    auto* ConstCmp = cast<BCEConstCmp>(BceBlock.getCmp());
    Constants.emplace_back(ConstCmp->Const);
    Types.emplace_back(ConstCmp->Lhs.LoadI->getType());
  }
  // NOTE: Could check if all elements are of the same type and then use an array instead, if that is more performat.
  auto* StructType = StructType::get(Context, Types, /* currently only matches packed offsets */ true);
  auto* StructAlloca = Builder.CreateAlloca(StructType,nullptr);
  auto *StructConstant = ConstantStruct::get(StructType, Constants);
  Builder.CreateStore(StructConstant, StructAlloca);

  return StructAlloca;
}

// Merges the given contiguous comparison blocks into one memcmp block.
static BasicBlock *mergeComparisons(ArrayRef<SingleBCECmpBlock> Comparisons,
                                    BasicBlock *const InsertBefore,
                                    BasicBlock *const NextCmpBlock,
                                    PHINode &Phi,
                                    LLVMContext &Context,
                                    const TargetLibraryInfo &TLI,
                                    AliasAnalysis &AA, DomTreeUpdater &DTU) {
  assert(Comparisons.size() > 1 && "merging multiple comparisons");
  const SingleBCECmpBlock &FirstCmp = Comparisons[0];

  // Create a new cmp block before next cmp block.
  BasicBlock *const BB =
      BasicBlock::Create(Context, MergedBlockName(Comparisons).Name,
                         NextCmpBlock->getParent(), InsertBefore);
  IRBuilder<> Builder(BB);
  // Add the GEPs from the first BCECmpBlock.
  Value *Lhs, *Rhs;
  if (FirstCmp.Lhs()->GEP)
    Lhs = Builder.Insert(FirstCmp.Lhs()->GEP->clone());
  else
    Lhs = FirstCmp.Lhs()->LoadI->getPointerOperand();

  if (isa<BCEConstCmp>(FirstCmp.getCmp())) {
    Rhs = buildStruct(Comparisons, Builder, Context);
  } else {
    auto* FirstBceCmp = cast<BCECmp>(FirstCmp.getCmp());
    if (FirstBceCmp->Rhs.GEP)
      Rhs = Builder.Insert(FirstBceCmp->Rhs.GEP->clone());
    else
      Rhs = FirstBceCmp->Rhs.LoadI->getPointerOperand();
  }
  LLVM_DEBUG(dbgs() << "Merging " << Comparisons.size() << " comparisons -> "
                    << BB->getName() << "\n");

  // If there is one block that requires splitting, we do it now, i.e.
  // just before we know we will collapse the chain. The instructions
  // can be executed before any of the instructions in the chain.
  const auto* ToSplit = llvm::find_if(
      Comparisons, [](const SingleBCECmpBlock &B) { return B.RequireSplit; });
  if (ToSplit != Comparisons.end()) {
    LLVM_DEBUG(dbgs() << "Splitting non_BCE work to header\n");
    ToSplit->split(BB, AA);
  }

  // memcmp expects a 'size_t' argument and returns 'int'.
  unsigned SizeTBits = TLI.getSizeTSize(*Phi.getModule());
  unsigned IntBits = TLI.getIntSize();
  const unsigned TotalSizeBits = std::accumulate(
      Comparisons.begin(), Comparisons.end(), 0u,
      [](int Size, const SingleBCECmpBlock &C) { return Size + C.getCmp()->SizeBits; });

  // Create memcmp() == 0.
  const auto &DL = Phi.getDataLayout();
  Value *const MemCmpCall = emitMemCmp(
      Lhs, Rhs,
      ConstantInt::get(Builder.getIntNTy(SizeTBits), TotalSizeBits / 8),
      Builder, DL, &TLI);
  Value* IsEqual = Builder.CreateICmpEQ(
      MemCmpCall, ConstantInt::get(Builder.getIntNTy(IntBits), 0));

  updateBranching(IsEqual, Builder, BB, NextCmpBlock, Phi, Context, TLI, AA, DTU);
  return BB;
}

// Keep existing block if it isn't merged. Only change the branches.
// Also handles not splitting mult-blocks that use select instructions.
static BasicBlock *updateOriginalBlock(BasicBlock *const BB,
                                    BasicBlock *const InsertBefore,
                                    BasicBlock *const NextCmpBlock,
                                    PHINode &Phi,
                                    LLVMContext &Context,
                                    const TargetLibraryInfo &TLI,
                                    AliasAnalysis &AA, DomTreeUpdater &DTU) {
  BasicBlock *MultBB = BasicBlock::Create(Context, BB->getName(),
                         NextCmpBlock->getParent(), InsertBefore);
  auto *const BranchI = cast<BranchInst>(BB->getTerminator());
  Value* CondResult = nullptr;
  if (BranchI->isUnconditional())
    CondResult = Phi.getIncomingValueForBlock(BB);
  else
    CondResult = cast<Value>(BranchI->getCondition());
  // Transfer all instructions except the branching terminator to the new block.
  MultBB->splice(MultBB->end(), BB, BB->begin(), std::prev(BB->end()));
  IRBuilder<> Builder(MultBB);
  updateBranching(CondResult, Builder, MultBB, NextCmpBlock, Phi, Context, TLI, AA, DTU);

  return MultBB;
}

bool BCECmpChain::simplify(const TargetLibraryInfo &TLI, AliasAnalysis &AA,
                           DomTreeUpdater &DTU) {
  assert(atLeastOneMerged() && "simplifying trivial BCECmpChain");
  LLVM_DEBUG(dbgs() << "Simplifying comparison chain starting at block "
                    << EntryBlock_->getName() << "\n");

  // Effectively merge blocks. We go in the reverse direction from the phi block
  // so that the next block is always available to branch to.
  BasicBlock *InsertBefore = EntryBlock_;
  BasicBlock *NextCmpBlock = Phi_.getParent();
  SmallDenseSet<const BasicBlock*, 8> ExistingBlocksToKeep;
  LLVMContext &Context = NextCmpBlock->getContext();
  for (const auto &Cmps : reverse(MergedBlocks_)) {
    // If there is only a single comparison then nothing should
    // be merged and can use original block.
    if (Cmps.size() == 1) {
      // If a comparison from a mult-block is already handled
      // then don't emit same block again.
      BasicBlock *const BB = Cmps[0].BB;
      if (ExistingBlocksToKeep.contains(BB))
        continue;
      ExistingBlocksToKeep.insert(BB);
      InsertBefore = NextCmpBlock = updateOriginalBlock(
        BB, InsertBefore, NextCmpBlock, Phi_, Context, TLI, AA, DTU);
    } else {
      InsertBefore = NextCmpBlock = mergeComparisons(
          Cmps, InsertBefore, NextCmpBlock, Phi_, Context, TLI, AA, DTU);
    }
  }

  // Replace the original cmp chain with the new cmp chain by pointing all
  // predecessors of EntryBlock_ to NextCmpBlock instead. This makes all cmp
  // blocks in the old chain unreachable.
  while (!pred_empty(EntryBlock_)) {
    BasicBlock* const Pred = *pred_begin(EntryBlock_);
    LLVM_DEBUG(dbgs() << "Updating jump into old chain from " << Pred->getName()
                      << "\n");
    Pred->getTerminator()->replaceUsesOfWith(EntryBlock_, NextCmpBlock);
    DTU.applyUpdates({{DominatorTree::Delete, Pred, EntryBlock_},
                      {DominatorTree::Insert, Pred, NextCmpBlock}});
  }

  // If the old cmp chain was the function entry, we need to update the function
  // entry.
  const bool ChainEntryIsFnEntry = EntryBlock_->isEntryBlock();
  if (ChainEntryIsFnEntry && DTU.hasDomTree()) {
    LLVM_DEBUG(dbgs() << "Changing function entry from "
                      << EntryBlock_->getName() << " to "
                      << NextCmpBlock->getName() << "\n");
    DTU.getDomTree().setNewRoot(NextCmpBlock);
    DTU.applyUpdates({{DominatorTree::Delete, NextCmpBlock, EntryBlock_}});
  }
  EntryBlock_ = nullptr;

  // Delete merged blocks. This also removes incoming values in phi.
  SmallVector<BasicBlock *, 16> DeadBlocks;
  for (const auto &Blocks : MergedBlocks_) {
    for (const SingleBCECmpBlock &Block : Blocks) {
      // Many single blocks can refer to the same multblock coming from an select instruction.
      // TODO: preferrably use a set instead
      if (llvm::is_contained(DeadBlocks, Block.BB))
        continue;
      LLVM_DEBUG(dbgs() << "Deleting merged block " << Block.BB->getName()
                        << "\n");
      DeadBlocks.push_back(Block.BB);
    }
  }
  DeleteDeadBlocks(DeadBlocks, &DTU);

  MergedBlocks_.clear();
  return true;
}

std::vector<BasicBlock *> getOrderedBlocks(PHINode &Phi,
                                           BasicBlock *const LastBlock,
                                           int NumBlocks) {
  // Walk up from the last block to find other blocks.
  std::vector<BasicBlock *> Blocks(NumBlocks);
  assert(LastBlock && "invalid last block");
  BasicBlock *CurBlock = LastBlock;
  for (int BlockIndex = NumBlocks - 1; BlockIndex > 0; --BlockIndex) {
    if (CurBlock->hasAddressTaken()) {
      // Somebody is jumping to the block through an address, all bets are
      // off.
      LLVM_DEBUG(dbgs() << "skip: block " << BlockIndex
                        << " has its address taken\n");
      return {};
    }
    Blocks[BlockIndex] = CurBlock;
    auto *SinglePredecessor = CurBlock->getSinglePredecessor();
    if (!SinglePredecessor) {
      // The block has two or more predecessors.
      LLVM_DEBUG(dbgs() << "skip: block " << BlockIndex
                        << " has two or more predecessors\n");
      return {};
    }
    if (Phi.getBasicBlockIndex(SinglePredecessor) < 0) {
      // The block does not link back to the phi.
      LLVM_DEBUG(dbgs() << "skip: block " << BlockIndex
                        << " does not link back to the phi\n");
      return {};
    }
    CurBlock = SinglePredecessor;
  }
  Blocks[0] = CurBlock;
  return Blocks;
}

template<typename T>
bool isInvalidPrevBlock(PHINode &Phi, unsigned I) {
  auto* IncomingValue = Phi.getIncomingValue(I);
  return !isa<T>(IncomingValue) ||
    cast<T>(IncomingValue)->getParent() != Phi.getIncomingBlock(I);
}

bool processPhi(PHINode &Phi, const TargetLibraryInfo &TLI, AliasAnalysis &AA,
                DomTreeUpdater &DTU) {
  LLVM_DEBUG(dbgs() << "processPhi()\n");
  if (Phi.getNumIncomingValues() <= 1) {
    LLVM_DEBUG(dbgs() << "skip: only one incoming value in phi\n");
    return false;
  }
  // We are looking for something that has the following structure:
  //   bb1 --eq--> bb2 --eq--> bb3 --eq--> bb4 --+
  //     \            \           \               \
  //      ne           ne          ne              \
  //       \            \           \               v
  //        +------------+-----------+----------> bb_phi
  //
  //  - The last basic block (bb4 here) must branch unconditionally to bb_phi.
  //    It's the only block that contributes a non-constant value to the Phi.
  //  - All other blocks (b1, b2, b3) must have exactly two successors, one of
  //    them being the phi block.
  //  - All intermediate blocks (bb2, bb3) must have only one predecessor.
  //  - Blocks cannot do other work besides the comparison, see doesOtherWork()

  // The blocks are not necessarily ordered in the phi, so we start from the
  // last block and reconstruct the order.
  BasicBlock *LastBlock = nullptr;
  for (unsigned I = 0; I < Phi.getNumIncomingValues(); ++I) {
    if (isa<ConstantInt>(Phi.getIncomingValue(I))) continue;
    if (LastBlock) {
      // There are several non-constant values.
      LLVM_DEBUG(dbgs() << "skip: several non-constant values\n");
      return false;
    }
    if (isInvalidPrevBlock<ICmpInst>(Phi,I) && isInvalidPrevBlock<SelectInst>(Phi,I)) {
      // Non-constant incoming value is not from a cmp instruction or not
      // produced by the last block. We could end up processing the value
      // producing block more than once.
      //
      // This is an uncommon case, so we bail.
      LLVM_DEBUG(
          dbgs()
          << "skip: non-constant value not from cmp or not from last block.\n");
      return false;
    }
    LastBlock = Phi.getIncomingBlock(I);
  }
  if (!LastBlock) {
    // There is no non-constant block.
    LLVM_DEBUG(dbgs() << "skip: no non-constant block\n");
    return false;
  }
  if (LastBlock->getSingleSuccessor() != Phi.getParent()) {
    LLVM_DEBUG(dbgs() << "skip: last block non-phi successor\n");
    return false;
  }

  const auto Blocks =
      getOrderedBlocks(Phi, LastBlock, Phi.getNumIncomingValues());

  if (Blocks.empty()) return false;
  BCECmpChain CmpChain(Blocks, Phi, AA);

  if (!CmpChain.atLeastOneMerged()) {
    LLVM_DEBUG(dbgs() << "skip: nothing merged\n");
    return false;
  }

  if (CmpChain.multBlockOnlyPartiallyMerged()) {
    LLVM_DEBUG(dbgs() << "chain uses not fully merged basic block, no merge\n");
    return false;
  }

  return CmpChain.simplify(TLI, AA, DTU);
}

static bool runImpl(Function &F, const TargetLibraryInfo &TLI,
                    const TargetTransformInfo &TTI, AliasAnalysis &AA,
                    DominatorTree *DT) {
  LLVM_DEBUG(dbgs() << "MergeICmpsLegacyPass: " << F.getName() << "\n");

  // We only try merging comparisons if the target wants to expand memcmp later.
  // The rationale is to avoid turning small chains into memcmp calls.
  if (!TTI.enableMemCmpExpansion(F.hasOptSize(), true))
    return false;

  // If we don't have memcmp avaiable we can't emit calls to it.
  if (!TLI.has(LibFunc_memcmp))
    return false;

  DomTreeUpdater DTU(DT, /*PostDominatorTree*/ nullptr,
                     DomTreeUpdater::UpdateStrategy::Eager);

  bool MadeChange = false;

  for (BasicBlock &BB : llvm::drop_begin(F)) {
    // A Phi operation is always first in a basic block.
    if (auto *const Phi = dyn_cast<PHINode>(&*BB.begin()))
      MadeChange |= processPhi(*Phi, TLI, AA, DTU);
  }

  return MadeChange;
}

class MergeICmpsLegacyPass : public FunctionPass {
public:
  static char ID;

  MergeICmpsLegacyPass() : FunctionPass(ID) {
    initializeMergeICmpsLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F)) return false;
    const auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    const auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    // MergeICmps does not need the DominatorTree, but we update it if it's
    // already available.
    auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    return runImpl(F, TLI, TTI, AA, DTWP ? &DTWP->getDomTree() : nullptr);
  }

 private:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
};

} // namespace

char MergeICmpsLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(MergeICmpsLegacyPass, "mergeicmps",
                      "Merge contiguous icmps into a memcmp", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(MergeICmpsLegacyPass, "mergeicmps",
                    "Merge contiguous icmps into a memcmp", false, false)

Pass *llvm::createMergeICmpsLegacyPass() { return new MergeICmpsLegacyPass(); }

PreservedAnalyses MergeICmpsPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto &AA = AM.getResult<AAManager>(F);
  auto *DT = AM.getCachedResult<DominatorTreeAnalysis>(F);
  const bool MadeChanges = runImpl(F, TLI, TTI, AA, DT);
  if (!MadeChanges)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
