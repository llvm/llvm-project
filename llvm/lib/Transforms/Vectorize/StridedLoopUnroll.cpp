//===-------- StridedLoopUnroll.cpp - Loop idiom vectorization
//-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a transformation pass that identifies and optimizes
// a specific class of nested loops operating over 2D data (e.g., image or
// matrix).
//
// The pass looks for loops with the following characteristics:
//
//   - An outer loop with canonical induction over rows (y-dimension).
//   - An inner loop with canonical induction over columns (x-dimension).
//   - Inner loop performs unit-stride loads/stores via pointer induction.
//   - Outer loop increments the base pointers with constant strides.
//   - Loops have predictable trip counts (starting at zero, step = 1).
//
// When such a structure is recognized, the pass performs loop versioning:
//
//   1. The first version is intended to be consumed by the default
//      LoopVectorize pass.
//
//   2. The second version assumes regular strided memory access and is marked
//      for further transformation (e.g., unrolling or custom vectorization).
//
// This enables aggressive optimization of memory-bound loop nests, particularly
// for architectures where strided memory patterns can be handled efficiently.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/StridedLoopUnroll.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "strided-loop-unroll"

namespace {
struct LoadInfo {
  LoadInst *Instr;
  Value *Stride;
};

struct StoreInfo {
  StoreInst *Instr;
  Value *Stride;
};

class StridedLoopUnroll {
  Loop *CurLoop = nullptr;
  TargetTransformInfo *TTI;
  const DataLayout *DL;
  ScalarEvolution *SE;
  LoopAccessInfoManager LAIs;

public:
  StridedLoopUnroll(DominatorTree *DT, LoopInfo *LI, TargetTransformInfo *TTI,
                    const DataLayout *DL, ScalarEvolution *SE,
                    AliasAnalysis *AA, AssumptionCache *AC)
      : TTI(TTI), DL(DL), SE(SE), LAIs(*SE, *AA, *DT, *LI, TTI, nullptr, AC) {}

  bool run(Loop *L);

private:
  /// \name Stride Loop Idiom Handling
  /// @{

  Value *widenVectorizedInstruction(Instruction *I, SmallVectorImpl<Value *> &,
                                    FixedVectorType *VecTy, unsigned int VF);
  bool recognizeStridedSpecialCases();

  void transformStridedSpecialCases(BasicBlock *Header, BasicBlock *Latch,
                                    BasicBlock *Preheader, Loop *SubLoop,
                                    SmallVectorImpl<LoadInst *> &Loads,
                                    StoreInst *Store,
                                    SmallVectorImpl<Value *> &PostOrder,
                                    SmallVectorImpl<Value *> &PreOrder);
  void changeInductionVarIncrement(Value *IncomingValue, unsigned VF);
  std::optional<Value *> getDynamicStrideFromMemOp(Value *Value,
                                                   Instruction *InsertionPt);
  std::optional<Value *> getStrideFromAddRecExpr(const SCEVAddRecExpr *AR,
                                                 Instruction *InsertionPt);

  /// @}
};

static cl::opt<bool>
    SkipPass("strided-loop-unroll-disable", cl::init(false), cl::Hidden,
             cl::desc("Skip running strided loop unroll optimization."));

class StridedLoopUnrollVersioning {
  Loop *CurLoop = nullptr;
  DominatorTree *DT;
  LoopInfo *LI;
  TargetTransformInfo *TTI;
  const DataLayout *DL;
  ScalarEvolution *SE;
  LoopAccessInfoManager LAIs;
  AssumptionCache *AC;
  OptimizationRemarkEmitter *ORE;

  const LoopAccessInfo *LAI = nullptr;

public:
  StridedLoopUnrollVersioning(DominatorTree *DT, LoopInfo *LI,
                              TargetTransformInfo *TTI, const DataLayout *DL,
                              ScalarEvolution *SE, AliasAnalysis *AA,
                              AssumptionCache *AC,
                              OptimizationRemarkEmitter *ORE, Function *F)
      : DT(DT), LI(LI), TTI(TTI), DL(DL), SE(SE),
        LAIs(*SE, *AA, *DT, *LI, TTI, nullptr, AC), AC(AC), ORE(ORE) {}

  bool run(Loop *L);

private:
  /// \name Countable Loop Idiom Handling
  /// @{

  void setNoAliasToLoop(Loop *VerLoop);
  bool recognizeStridedSpecialCases();
  void transformStridedSpecialCases(
      PHINode *OuterInductionVar, PHINode *InnerInductionVar, StoreInst *Stores,
      BasicBlock *PreheaderBB, BasicBlock *BodyBB, BasicBlock *HeaderBB,
      BasicBlock *LatchBB, SmallVectorImpl<const SCEV *> &AlignmentInfo,
      unsigned UnrollSize);
  void eliminateRedundantLoads(BasicBlock *BB) {
    // Map from load pointer to the first load instruction
    DenseMap<Value *, LoadInst *> LoadMap;
    SmallVector<LoadInst *, 16> ToDelete;

    // First pass: collect all loads and find duplicates
    for (Instruction &I : *BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        Value *Ptr = LI->getPointerOperand();

        // Check if we've seen a load from this address
        auto It = LoadMap.find(Ptr);
        if (It != LoadMap.end() && !LI->isVolatile()) {
          // Found duplicate - check if they're compatible
          LoadInst *FirstLoad = It->second;
          if (FirstLoad->getType() == LI->getType() &&
              FirstLoad->getAlign() == LI->getAlign() &&
              !FirstLoad->isVolatile()) {
            // Replace this load with the first one
            LI->replaceAllUsesWith(FirstLoad);
            ToDelete.push_back(LI);
          }
        } else {
          // First load from this address
          LoadMap[Ptr] = LI;
        }
      }
    }

    // Delete redundant loads
    for (LoadInst *LI : ToDelete) {
      LI->eraseFromParent();
    }
  }

  void hoistInvariantLoadsToPreheader(Loop *L);
  /// @}
};

} // anonymous namespace

PreservedAnalyses StridedLoopUnrollPass::run(Loop &L, LoopAnalysisManager &AM,
                                             LoopStandardAnalysisResults &AR,
                                             LPMUpdater &) {
  const auto *DL = &L.getHeader()->getDataLayout();

  StridedLoopUnroll LIV(&AR.DT, &AR.LI, &AR.TTI, DL, &AR.SE, &AR.AA, &AR.AC);
  if (!LIV.run(&L))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

//===----------------------------------------------------------------------===//
//
//          Implementation of StridedLoopUnroll
//
//===----------------------------------------------------------------------===//
bool StridedLoopUnroll::run(Loop *L) {
  CurLoop = L;

  Function &F = *L->getHeader()->getParent();

  // If the loop could not be converted to canonical form, it must have an
  // indirectbr in it, just give up.
  if (!L->getLoopPreheader())
    return false;

  LLVM_DEBUG(dbgs() << DEBUG_TYPE " Scanning: F[" << F.getName() << "] Loop %"
                    << CurLoop->getHeader()->getName() << "\n");

  if (recognizeStridedSpecialCases()) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE " Will transform: F[" << F.getName()
                      << "] Loop %" << CurLoop->getHeader()->getName() << "\n");
    return true;
  }

  LLVM_DEBUG(dbgs() << DEBUG_TYPE " Will not transform: F[" << F.getName()
                    << "] Loop %" << CurLoop->getHeader()->getName() << "\n");
  return false;
}

bool StridedLoopUnrollVersioning::run(Loop *L) {
  CurLoop = L;

  if (!TTI->getVScaleForTuning())
    return false;

  Function &F = *L->getHeader()->getParent();
  if (F.hasOptSize())
    return false;

  // If the loop could not be converted to canonical form, it must have an
  // indirectbr in it, just give up.
  if (!L->getLoopPreheader())
    return false;

  LLVM_DEBUG(dbgs() << DEBUG_TYPE " Scanning: F[" << F.getName() << "] Loop %"
                    << CurLoop->getHeader()->getName() << "\n");

  if (recognizeStridedSpecialCases()) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE " Will transform: F[" << F.getName()
                      << "] Loop %" << CurLoop->getHeader()->getName() << "\n");

    return true;
  }
  LLVM_DEBUG(dbgs() << DEBUG_TYPE " Will not transform: F[" << F.getName()
                    << "] Loop %" << CurLoop->getHeader()->getName() << "\n");
  return false;
}

/// \returns the number of elements for Ty.
static unsigned getNumElements(Type *Ty) {
  assert(!isa<ScalableVectorType>(Ty) &&
         "ScalableVectorType is not supported.");
  if (auto *VecTy = dyn_cast<FixedVectorType>(Ty))
    return VecTy->getNumElements();
  return 1;
}

/// \returns the vector type of ScalarTy based on vectorization factor.
static FixedVectorType *getGroupedWidenedType(Type *OriginalVecTy, unsigned VF,
                                              const DataLayout &DL) {
  auto ScalarTy = OriginalVecTy->getScalarType();
  auto GroupScalarTy = Type::getIntNTy(
      ScalarTy->getContext(), DL.getTypeSizeInBits(ScalarTy).getFixedValue() *
                                  getNumElements(OriginalVecTy));
  return FixedVectorType::get(GroupScalarTy, VF);
}

/// \returns the vector type of ScalarTy based on vectorization factor.
static FixedVectorType *getWidenedType(Type *VecTy, unsigned VF) {
  return FixedVectorType::get(VecTy->getScalarType(),
                              VF * getNumElements(VecTy));
}

static void findUnconnectedToLoad(Instruction *start,
                                  SmallPtrSetImpl<Value *> &NotConnected,
                                  SmallPtrSetImpl<Value *> &Connected) {
  SmallPtrSet<Value *, 32> outerVisited;
  SmallVector<Value *> Worklist;

  Worklist.push_back(start);

  while (!Worklist.empty()) {
    SmallVector<Value *> innerWorklist;
    SmallPtrSet<Value *, 32> innerVisited;
    bool connected = false;
    Value *OuterVal = Worklist.back();
    Worklist.pop_back();

    if (isa<LoadInst>(OuterVal))
      continue;

    innerWorklist.push_back(OuterVal);
    if (!outerVisited.insert(OuterVal).second)
      continue;

    while (!innerWorklist.empty()) {
      Value *val = innerWorklist.back();
      innerWorklist.pop_back();

      // ignore phinodes
      if (dyn_cast<PHINode>(val)) {
        continue;
      }

      // Only process instructions (skip constants, arguments, etc.)
      auto *inst = dyn_cast<Instruction>(val);
      if (!inst) {
        continue;
      }

      // Already innerVisited?
      if (!innerVisited.insert(val).second)
        continue;

      bool shouldBreak = isa<LoadInst>(inst);
      // If this is a load, do not proceed from here!
      connected = isa<LoadInst>(inst) &&
                  start->getParent()->getName() == inst->getParent()->getName();
      if (shouldBreak)
        break;

      // Add operands to the worklist
      for (auto &op : inst->operands()) {
        if (auto I = dyn_cast<Instruction>(op.get())) {
          if (I->getParent() == start->getParent())
            innerWorklist.push_back(op.get());
        } else
          innerWorklist.push_back(op.get());
      }
    }
    if (!connected)
      NotConnected.insert(OuterVal);
    else
      Connected.insert(OuterVal);
    if (auto I = dyn_cast<Instruction>(OuterVal)) {
      for (auto &op : I->operands()) {
        Worklist.push_back(op.get());
      }
    }
  }
}

void StridedLoopUnroll::changeInductionVarIncrement(Value *IncomingValue,
                                                    unsigned VF) {
  if (auto I = dyn_cast<Instruction>(IncomingValue)) {
    switch (I->getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::GetElementPtr: {
      IRBuilder<> Builder(I);
      I->setOperand(1, Builder.CreateMul(
                           I->getOperand(1),
                           ConstantInt::get(I->getOperand(1)->getType(), VF)));
      break;
    }
    default:
      llvm_unreachable("Can't change increment in this InductionVar");
    }
  }
}

Value *StridedLoopUnroll::widenVectorizedInstruction(
    Instruction *I, SmallVectorImpl<Value *> &Ops, FixedVectorType *VecTy,
    unsigned int VF) {
  IRBuilder<> Builder(I);
  auto Opcode = I->getOpcode();
  switch (Opcode) {
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem:
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::FNeg:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    Value *V = Builder.CreateNAryOp(Opcode, Ops);
    return V;
  }
  case Instruction::Select: {
    Value *V = Builder.CreateSelect(Ops[0], Ops[1], Ops[2]);
    return V;
  }
  case Instruction::ICmp: {
    auto Cmp = dyn_cast<ICmpInst>(I);
    Value *V = Builder.CreateICmp(Cmp->getPredicate(), Ops[0], Ops[1]);
    return V;
  }
  case Instruction::ShuffleVector: {
    auto SV = dyn_cast<ShuffleVectorInst>(I);
    ArrayRef<int> Mask = SV->getShuffleMask();
    std::vector<int> RepeatedMask;
    RepeatedMask.reserve(Mask.size() * VF);

    for (unsigned int i = 0; i < VF; ++i) {
      llvm::append_range(RepeatedMask, Mask);
    }

    ArrayRef<int> NewMask(RepeatedMask);
    Value *Shuffle =
        Builder.CreateShuffleVector(Ops[0], Ops[1], NewMask, I->getName());
    return Shuffle;
  }
  case Instruction::InsertElement: {
    Value *A =
        Builder.CreateInsertElement(Ops[0], Ops[1], Ops[2], I->getName());
    return A;
  }
  case Instruction::Load: {
    Value *L = Builder.CreateLoad(VecTy, Ops[0], I->getName());
    return L;
  }
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::Trunc:
  case Instruction::FPTrunc:
  case Instruction::FPExt: {
    Value *V = Builder.CreateCast(static_cast<Instruction::CastOps>(Opcode),
                                  Ops[0], VecTy, "");
    return V;
  }
  default:
    llvm_unreachable("Can't handle widening this Opcode");
  }
}
bool isInstructionDepends(Instruction *Dependant, Instruction *Target) {
  SmallPtrSet<Instruction *, 16> Visited;
  SmallVector<Instruction *, 16> Worklist;

  // Start from the terminator
  Instruction *Terminator = Dependant->getParent()->getTerminator();
  Worklist.push_back(Terminator);

  while (!Worklist.empty()) {
    Instruction *Current = Worklist.pop_back_val();

    // Skip if already visited
    if (!Visited.insert(Current).second)
      continue;

    // Found our target
    if (Current == Target) {
      return true;
    }

    // Add operands that are instructions in the same BB
    for (Use &U : Current->operands()) {
      if (Instruction *OpInst = dyn_cast<Instruction>(U.get())) {
        if (OpInst->getParent() == Dependant->getParent() &&
            Visited.find(OpInst) == Visited.end()) {
          Worklist.push_back(OpInst);
        }
      }
    }
  }

  return false;
}

// InnerInductionVar will be transformed to static
void StridedLoopUnroll::transformStridedSpecialCases(
    BasicBlock *Header, BasicBlock *Latch, BasicBlock *Preheader, Loop *SubLoop,
    SmallVectorImpl<LoadInst *> &Loads, StoreInst *Store,
    SmallVectorImpl<Value *> &PostOrder, SmallVectorImpl<Value *> &PreOrder) {

  // auto InnerPreheader = SubLoop->getLoopPreheader();

  auto Stride = getDynamicStrideFromMemOp(Store->getPointerOperand(),
                                          Preheader->getTerminator());

  SmallPtrSet<llvm::Value *, 32> Connected;
  SmallPtrSet<llvm::Value *, 32> NotConnected;
  SmallDenseMap<llvm::Value *, llvm::Value *, 32> Replacements;

  auto StoredInstruction = dyn_cast<Instruction>(Store->getValueOperand());
  findUnconnectedToLoad(StoredInstruction, NotConnected, Connected);

  auto convertConstant = [&](auto val) {
    auto constVal = cast<Constant>(val);
    unsigned numElements =
        cast<FixedVectorType>(val->getType())->getNumElements();
    SmallVector<Constant *, 16> elements;

    // Extract original elements
    for (unsigned i = 0; i < numElements; ++i)
      elements.push_back(constVal->getAggregateElement(i));

    auto originalElements = elements;
    for (unsigned int copy = 0; copy != (*TTI->getVScaleForTuning()) - 1;
         ++copy)
      elements.append(originalElements);
    Constant *newConst = ConstantVector::get(elements);
    return newConst;
  };

  // Process in post-order (leafs to root)
  for (Value *val : PostOrder) {
    if (Connected.contains(val)) {
      if (auto *I = dyn_cast<Instruction>(val)) {
        SmallVector<Value *, 4> Operands(I->operands());
        for (auto op_it = Operands.begin(); op_it != Operands.end(); ++op_it) {
          if (Replacements.contains(*op_it))
            *op_it = Replacements[*op_it];
          else if (auto OrigVecTy =
                       llvm::dyn_cast<llvm::VectorType>((*op_it)->getType())) {
            if (auto Iop = dyn_cast<Instruction>(*op_it)) {
              if (Iop->getParent() != Store->getParent()) {
                assert(!Connected.contains(*op_it));

                IRBuilder<> Builder(I);

                std::vector<llvm::Constant *> Consts;
                for (unsigned int i = 0; i != *TTI->getVScaleForTuning(); i++) {
                  for (size_t j = 0;
                       j != OrigVecTy->getElementCount().getFixedValue(); j++) {
                    Consts.push_back(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(Builder.getContext()), j));
                  }
                }

                llvm::Constant *maskConst = llvm::ConstantVector::get(Consts);
                assert(maskConst != nullptr);

                llvm::Value *splat = Builder.CreateShuffleVector(
                    Iop, llvm::PoisonValue::get(Iop->getType()), maskConst);
                assert(splat != nullptr);
                Replacements.insert({*op_it, splat});
                *op_it = splat;
              }
            } else if (isa<Constant>(*op_it)) { // not instruction
              auto replacement = convertConstant(*op_it);
              assert(!!replacement);
              Replacements.insert({*op_it, replacement});
              *op_it = replacement;
            }
          }
        }

        auto NewVecTy =
            getWidenedType(I->getType(), *TTI->getVScaleForTuning());
        Value *NI = widenVectorizedInstruction(I, Operands, NewVecTy,
                                               *TTI->getVScaleForTuning());

        assert(NI != nullptr);
        Replacements.insert({I, NI});
      }
    } else if (NotConnected.contains(val)) {
      if (val->getType()->isVectorTy() && isa<Constant>(val)) {
        auto replacement = convertConstant(val);
        Replacements.insert({val, replacement});
      }
    } else if (auto Load = dyn_cast<LoadInst>(val)) {
      auto It =
          std::find_if(Loads.begin(), Loads.end(),
                       [Load](auto &&LoadInstr) { return LoadInstr == Load; });
      if (It != Loads.end()) {
        auto Stride = getDynamicStrideFromMemOp((*It)->getPointerOperand(),
                                                Preheader->getTerminator());

        auto GroupedVecTy = getGroupedWidenedType(
            Load->getType(), *TTI->getVScaleForTuning(), *DL);
        auto VecTy =
            getWidenedType(Load->getType(), *TTI->getVScaleForTuning());
        ElementCount NewElementCount = GroupedVecTy->getElementCount();

        IRBuilder<> Builder(Load);
        auto *NewInst = Builder.CreateIntrinsic(
            Intrinsic::experimental_vp_strided_load,
            {GroupedVecTy, Load->getPointerOperand()->getType(),
             (*Stride)->getType()},
            {Load->getPointerOperand(), *Stride,
             Builder.getAllOnesMask(NewElementCount),
             Builder.getInt32(NewElementCount.getKnownMinValue())});
        auto Cast = Builder.CreateBitCast(NewInst, VecTy);
        Replacements.insert({Load, Cast});
      }
    }
  }

  IRBuilder<> Builder(Store);
  auto VecTy = getGroupedWidenedType(Store->getValueOperand()->getType(),
                                     *TTI->getVScaleForTuning(), *DL);
  ElementCount NewElementCount = VecTy->getElementCount();

  assert(Replacements.find(Store->getValueOperand()) != Replacements.end());
  auto Cast =
      Builder.CreateBitCast(Replacements[Store->getValueOperand()], VecTy);

  Builder.CreateIntrinsic(
      Intrinsic::experimental_vp_strided_store,
      {VecTy, Store->getPointerOperand()->getType(), (*Stride)->getType()},
      {Cast, Store->getPointerOperand(), *Stride,
       Builder.getAllOnesMask(NewElementCount),
       Builder.getInt32(NewElementCount.getKnownMinValue())});

  for (auto &&PN : CurLoop->getHeader()->phis()) {
    InductionDescriptor IndDesc;

    if (InductionDescriptor::isInductionPHI(&PN, CurLoop, SE, IndDesc)) {
      if (IndDesc.getKind() == InductionDescriptor::IK_PtrInduction)
        changeInductionVarIncrement(
            PN.getIncomingValueForBlock(CurLoop->getLoopLatch()),
            *TTI->getVScaleForTuning());
      else if (IndDesc.getKind() == InductionDescriptor::IK_IntInduction)
        changeInductionVarIncrement(IndDesc.getInductionBinOp(),
                                    *TTI->getVScaleForTuning());
    }
  }

  if (Store->use_empty())
    Store->eraseFromParent();

  for (auto OldOp : PreOrder)
    if (OldOp->use_empty())
      if (auto I = dyn_cast<Instruction>(OldOp))
        I->eraseFromParent();
}

std::optional<Value *>
StridedLoopUnroll::getStrideFromAddRecExpr(const SCEVAddRecExpr *AR,
                                           Instruction *InsertionPt) {
  auto Step = AR->getStepRecurrence(*SE);
  if (isa<SCEVConstant>(Step))
    return std::nullopt;
  SCEVExpander Expander(*SE, *DL, "stride");
  Value *StrideValue =
      Expander.expandCodeFor(Step, Step->getType(), InsertionPt);
  return StrideValue;
}

std::optional<Value *>
StridedLoopUnroll::getDynamicStrideFromMemOp(Value *V,
                                             Instruction *InsertionPt) {
  const SCEV *S = SE->getSCEV(V);
  if (const SCEVAddRecExpr *InnerLoopAR = dyn_cast<SCEVAddRecExpr>(S)) {
    if (auto *constant =
            dyn_cast<SCEVConstant>(InnerLoopAR->getStepRecurrence(*SE))) {
      // We need to form 64-bit groups
      if (constant->getAPInt() != 8) {
        return std::nullopt;
      }

      const auto *Add = dyn_cast<SCEVAddExpr>(InnerLoopAR->getStart());
      if (Add) {
        for (const SCEV *Op : Add->operands()) {
          // Look for the outer recurrence: { %dst, +, sext(%i_dst_stride) }
          // <outer loop>
          const auto *AR = dyn_cast<SCEVAddRecExpr>(Op);
          if (!AR)
            continue;

          return getStrideFromAddRecExpr(AR, InsertionPt);
        }
      } else if (const SCEVAddRecExpr *AR =
                     dyn_cast<SCEVAddRecExpr>(InnerLoopAR->getStart())) {
        return getStrideFromAddRecExpr(AR, InsertionPt);
      }
    }
  }
  return std::nullopt;
}

bool StridedLoopUnroll::recognizeStridedSpecialCases() {
  auto Stride = getOptionalIntLoopAttribute(CurLoop, "llvm.stride.loop_idiom");
  if (!Stride)
    return false;

  auto SubLoops = CurLoop->getSubLoops();

  if (SubLoops.size() > 2 || SubLoops.empty())
    return false;

  auto SubLoop = SubLoops.size() == 2 ? SubLoops[1] : SubLoops[0];

  auto Preheader = SubLoop->getLoopPreheader();
  auto Header = SubLoop->getHeader();
  auto Latch = SubLoop->getLoopLatch();

  if (Header != Latch)
    return false;

  SmallVector<LoadInst *> Loads;
  SmallVector<StoreInst *> Stores;

  llvm::SmallPtrSet<llvm::Instruction *, 32> NotVisited;
  llvm::SmallVector<llvm::Instruction *, 8> WorkList;

  for (auto &&I : *Header) {
    if (auto &&Store = dyn_cast<StoreInst>(&I)) {
      WorkList.push_back(Store);
    } else {
      NotVisited.insert(&I);
    }
  }

  if (WorkList.size() != 1)
    return false;

  while (!WorkList.empty()) {
    llvm::Instruction *I = WorkList.back();
    WorkList.pop_back();

    if (auto *Load = dyn_cast<LoadInst>(I)) {
      NotVisited.erase(I);
      Loads.push_back(Load);
    } else if (auto *Store = dyn_cast<StoreInst>(I)) {
      if (auto *ValueInst = dyn_cast<Instruction>(Store->getValueOperand()))
        WorkList.push_back(ValueInst);
      NotVisited.erase(I);
      Stores.push_back(Store);
    } else {
      // Add operand instructions to the worklist
      for (llvm::Value *Op : I->operands()) {
        if (auto *DepInst = llvm::dyn_cast<llvm::Instruction>(Op))
          if (DepInst->getParent() == Header)
            WorkList.push_back(DepInst);
      }
      NotVisited.erase(I);
    }
  }

  if (Stores.size() != 1)
    return false;

  SmallPtrSet<llvm::Value *, 32> Connected;
  SmallPtrSet<llvm::Value *, 32> NotConnected;

  auto StoredInstruction = dyn_cast<Instruction>(Stores[0]->getValueOperand());
  findUnconnectedToLoad(StoredInstruction, NotConnected, Connected);

  llvm::SmallVector<Value *, 16> PostOrder;
  llvm::SmallVector<Value *, 16> PreOrder;
  llvm::SmallPtrSet<Value *, 16> Visited;
  llvm::SmallPtrSet<Value *, 16> InStack;
  SmallVector<llvm::Value *, 32> Worklist;

  Worklist.push_back(StoredInstruction);

  auto shouldVisit = [Header](auto *Val) {
    return !isa<PHINode>(Val) &&
           (!isa<Instruction>(Val) ||
            dyn_cast<Instruction>(Val)->getParent() == Header);
  };
  auto shouldVisitOperands = [](auto *Val) {
    return !isa<PHINode>(Val) && !isa<LoadInst>(Val);
  };

  while (!Worklist.empty()) {
    Value *val = Worklist.back();
    assert(!isa<Instruction>(val) ||
           dyn_cast<Instruction>(val)->getParent() == Header ||
           dyn_cast<Instruction>(val)->getParent() == Preheader);

    if (InStack.contains(val)) {
      // We've finished processing children, add to post-order
      Worklist.pop_back();
      InStack.erase(val);
      PostOrder.push_back(val);
    } else if (!Visited.contains(val)) {
      // First time seeing this node
      Visited.insert(val);
      InStack.insert(val);
      PreOrder.push_back(val);

      // Add children to worklist
      if (auto I = dyn_cast<Instruction>(val))
        if (shouldVisitOperands(I))
          for (auto &Op : I->operands())
            if (shouldVisit(Op.get()) && !Visited.contains(Op.get())) {
              Worklist.push_back(Op.get());
            }
    } else {
      // Already visited, skip
      Worklist.pop_back();
    }
  }

  // Process in post-order (leafs to root)
  for (Value *val : PostOrder) {
    if (Connected.contains(val)) {
      if (auto *I = dyn_cast<Instruction>(val)) {
        SmallVector<Value *, 4> Operands(I->operands());
        for (auto op_it = Operands.begin(); op_it != Operands.end(); ++op_it) {
          if (isa<llvm::VectorType>((*op_it)->getType())) {
            if (!isa<Instruction>(*op_it) && !isa<Constant>(*op_it)) {
              return false;
            }
          }
        }
      } else { // We don't handle Non-instructions connected to Load
        return false;
      }
    } else if (NotConnected.contains(val) &&
               (!val->getType()->isVectorTy() || !isa<Constant>(val))) {
      return false;
    } else if (auto Load = dyn_cast<LoadInst>(val)) {
      if (std::find(Loads.begin(), Loads.end(), Load) == Loads.end())
        return false;
    }
  }

  transformStridedSpecialCases(Header, Latch, Preheader, SubLoop, Loads,
                               Stores[0], PostOrder, PreOrder);

  return true;
}

namespace {

bool canHandleInstruction(Instruction *I) {
  auto Opcode = I->getOpcode();
  switch (Opcode) {
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem:
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::FNeg:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::Trunc:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::ShuffleVector:
  case Instruction::Br:
  case Instruction::PHI:
  case Instruction::GetElementPtr:
  case Instruction::ICmp:
    return true;
  default:
    return false;
  }
}

} // anonymous namespace

PreservedAnalyses
StridedLoopUnrollVersioningPass::run(Function &F,
                                     FunctionAnalysisManager &FAM) {
  bool Changed = false;

  if (SkipPass)
    return PreservedAnalyses::all();

  auto &LI = FAM.getResult<LoopAnalysis>(F);
  if (LI.empty())
    return PreservedAnalyses::all();
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
  auto &AA = FAM.getResult<AAManager>(F);
  auto &AC = FAM.getResult<AssumptionAnalysis>(F);
  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  // Iterate over all loops in the function
  std::vector<Loop *> Loops(LI.begin(), LI.end());
  for (Loop *L : Loops) {
    // L may be deleted, so check it's still valid!
    if (std::find(LI.begin(), LI.end(), L) == LI.end())
      continue;

    const auto *DL = &L->getHeader()->getDataLayout();

    StridedLoopUnrollVersioning LIV(&DT, &LI, &TTI, DL, &SE, &AA, &AC, &ORE,
                                    &F);
    bool ThisChanged = LIV.run(L);
    Changed |= ThisChanged;
  }

  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

void StridedLoopUnrollVersioning::setNoAliasToLoop(Loop *VerLoop) {
  // Get latch terminator instruction.
  Instruction *I = VerLoop->getLoopLatch()->getTerminator();
  // Create alias scope domain.
  MDBuilder MDB(I->getContext());
  MDNode *NewDomain = MDB.createAnonymousAliasScopeDomain("LIVVDomain");
  StringRef Name = "LVAliasScope";
  MDNode *NewScope = MDB.createAnonymousAliasScope(NewDomain, Name);
  SmallVector<Metadata *, 4> Scopes{NewScope}, NoAliases{NewScope};
  // Iterate over each instruction of loop.
  // set no-alias for all load & store instructions.
  for (auto *Block : CurLoop->getBlocks()) {
    for (auto &Inst : *Block) {
      // Only interested in instruction that may modify or read memory.
      if (!Inst.mayReadFromMemory() && !Inst.mayWriteToMemory())
        continue;
      // Set no-alias for current instruction.
      Inst.setMetadata(
          LLVMContext::MD_noalias,
          MDNode::concatenate(Inst.getMetadata(LLVMContext::MD_noalias),
                              MDNode::get(Inst.getContext(), NoAliases)));
      // set alias-scope for current instruction.
      Inst.setMetadata(
          LLVMContext::MD_alias_scope,
          MDNode::concatenate(Inst.getMetadata(LLVMContext::MD_alias_scope),
                              MDNode::get(Inst.getContext(), Scopes)));
    }
  }
}

void StridedLoopUnrollVersioning::hoistInvariantLoadsToPreheader(Loop *L) {
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    // If no preheader, try the header
    Preheader = L->getHeader();
  }

  // Find all invariant loads in the loop
  SmallVector<LoadInst *, 8> InvariantLoads;

  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        Value *Ptr = LI->getPointerOperand();

        if (L->isLoopInvariant(Ptr)) {
          InvariantLoads.push_back(LI);
        }
      }
    }
  }

  // Move loads to preheader and eliminate duplicates
  DenseMap<Value *, LoadInst *> HoistedLoads;

  for (LoadInst *LI : InvariantLoads) {
    Value *Ptr = LI->getPointerOperand();

    if (HoistedLoads.count(Ptr)) {
      // Already hoisted this load, replace uses
      LI->replaceAllUsesWith(HoistedLoads[Ptr]);
      LI->eraseFromParent();
    } else {
      // Move to preheader
      LI->moveBefore(*Preheader, Preheader->getTerminator()->getIterator());
      HoistedLoads[Ptr] = LI;
    }
  }
}

// InnerInductionVar will be transformed to static
void StridedLoopUnrollVersioning::transformStridedSpecialCases(
    PHINode *OuterInductionVar, PHINode *InnerInductionVar, StoreInst *Store,
    BasicBlock *PreheaderBB, BasicBlock *BodyBB, BasicBlock *HeaderBB,
    BasicBlock *LatchBB, SmallVectorImpl<const SCEV *> &AlignmentInfo,
    unsigned UnrollSize) {

  PredicatedScalarEvolution PSE(*SE, *CurLoop);

  auto VLAI = &LAIs.getInfo(*CurLoop);
  LoopVersioning LVer2(*VLAI, VLAI->getRuntimePointerChecking()->getChecks(),
                       CurLoop, LI, DT, SE, true);
  LVer2.versionLoop();

#ifdef EXPENSIVE_CHECKS
  assert(DT->verify(DominatorTree::VerificationLevel::Fast));
#endif

  addStringMetadataToLoop(LVer2.getNonVersionedLoop(),
                          "llvm.mem.string_loop_idiom");
  setNoAliasToLoop(LVer2.getVersionedLoop());

  auto VersionedLoop = LVer2.getVersionedLoop();
  auto NewInnerLoop = VersionedLoop->getSubLoops()[0];
  auto InnerLoopBounds = NewInnerLoop->getBounds(*SE);
  auto OuterLoopBounds = VersionedLoop->getBounds(*SE);

  for (BasicBlock *BB : VersionedLoop->blocks()) {
    BB->setName(BB->getName() + ".strided.vectorized");
  }

  UnrollLoopOptions ULO;
  ULO.Count = UnrollSize;
  ULO.Force = true;
  ULO.Runtime = true;
  ULO.AllowExpensiveTripCount = false;
  ULO.UnrollRemainder = false;
  ULO.SCEVExpansionBudget = -1;

  UnrollLoop(NewInnerLoop, ULO, LI, SE, DT, AC, TTI, ORE, false);

  hoistInvariantLoadsToPreheader(VersionedLoop);

  for (BasicBlock *BB : VersionedLoop->blocks()) {
    eliminateRedundantLoads(BB);
  }

  for (BasicBlock *BB : VersionedLoop->blocks()) {
    DenseMap<Value *, Value *> LoadCSE;
    SmallVector<Instruction *, 16> DeadInsts;

    for (Instruction &I : *BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (!LI->isVolatile()) {
          Value *Ptr = LI->getPointerOperand();
          if (LoadCSE.count(Ptr)) {
            // Reuse previous load
            LI->replaceAllUsesWith(LoadCSE[Ptr]);
            DeadInsts.push_back(LI);
          } else {
            LoadCSE[Ptr] = LI;
          }
        }
      }
    }

    for (auto *I : DeadInsts) {
      I->eraseFromParent();
    }
  }

  for (BasicBlock *BB : VersionedLoop->blocks()) {
    eliminateRedundantLoads(BB);
  }

  if (InnerLoopBounds) {
    setNoAliasToLoop(VersionedLoop);
    setNoAliasToLoop(VersionedLoop->getSubLoops()[0]);
    addStringMetadataToLoop(VersionedLoop, "llvm.stride.loop_idiom");
    VersionedLoop->setLoopAlreadyUnrolled();

    assert(std::distance(pred_begin(VersionedLoop->getLoopPreheader()),
                         pred_end(VersionedLoop->getLoopPreheader())) == 1);
    for (auto *Pred : predecessors(VersionedLoop->getLoopPreheader())) {
      BranchInst *PHBranch = cast<BranchInst>(Pred->getTerminator());
      IRBuilder<> Builder(PHBranch);

      Value *innerZero =
          Constant::getNullValue(InnerLoopBounds->getFinalIVValue().getType());
      Value *outerZero =
          Constant::getNullValue(OuterLoopBounds->getFinalIVValue().getType());

      Value *innerMask = Builder.getIntN(
          InnerLoopBounds->getFinalIVValue().getType()->getIntegerBitWidth(),
          UnrollSize - 1);
      Value *innerAndResult = Builder.CreateAnd(
          &InnerLoopBounds->getFinalIVValue(), innerMask, "inner_mod_unroll");
      Value *innerIsNotDivisible =
          Builder.CreateICmpNE(innerAndResult, innerZero, "innerIsDivUnroll");

      Value *const32 = Builder.getIntN(
          InnerLoopBounds->getFinalIVValue().getType()->getIntegerBitWidth(),
          32);
      Value *innerNotSmallerThan = Builder.CreateICmpUGE(
          &InnerLoopBounds->getFinalIVValue(), const32, "inner_not_less_32");

      auto o = TTI->getVScaleForTuning();
      assert(!!o);

      Value *mask = Builder.getIntN(
          OuterLoopBounds->getFinalIVValue().getType()->getIntegerBitWidth(),
          *o - 1);
      Value *andResult = Builder.CreateAnd(&OuterLoopBounds->getFinalIVValue(),
                                           mask, "div_unroll");
      Value *isNotDivisible =
          Builder.CreateICmpNE(andResult, outerZero, "is_div_unroll");
      Value *Check1 = Builder.CreateOr(innerIsNotDivisible, isNotDivisible);
      Value *Check2 = Builder.CreateOr(Check1, innerNotSmallerThan);

      Value *AlignmentCheck = Builder.getFalse();

      for (auto &&PtrSCEV : AlignmentInfo) {
        const unsigned Alignment = 8;
        // Expand SCEV to get runtime value
        SCEVExpander Expander(*SE, *DL, "align.check");
        Value *PtrValue =
            Expander.expandCodeFor(PtrSCEV, Builder.getPtrTy(), PHBranch);

        Type *I64 = Type::getInt64Ty(PtrValue->getContext());
        bool AllowsMisaligned = TTI->isLegalStridedLoadStore(
            VectorType::get(I64, ElementCount::getFixed(8)), Align(1));

        if (!AllowsMisaligned) {
          // Create alignment check: (ptr & (alignment-1)) == 0
          Value *PtrInt =
              Builder.CreatePtrToInt(PtrValue, Builder.getInt64Ty());
          Value *Mask = Builder.getInt64(Alignment - 1);
          Value *Masked = Builder.CreateAnd(PtrInt, Mask);
          Value *IsAligned = Builder.CreateICmpNE(Masked, Builder.getInt64(0));

          AlignmentCheck = Builder.CreateOr(AlignmentCheck, IsAligned);
        }
      }
      Value *Check3 = Builder.CreateOr(Check2, AlignmentCheck);

      PHBranch->setCondition(Check3);

#ifdef EXPENSIVE_CHECKS
      assert(DT->verify(DominatorTree::VerificationLevel::Fast));
#endif
    }
  }
}

bool StridedLoopUnrollVersioning::recognizeStridedSpecialCases() {
  if (!TTI->supportsScalableVectors() || !TTI->getMinPageSize().has_value())
    return false;

  auto LoopBlocks = CurLoop->getBlocks();

  auto SubLoops = CurLoop->getSubLoops();

  if (SubLoops.size() != 1)
    return false;

  auto InnerLoop = SubLoops[0];

  auto OuterLoopBounds = CurLoop->getBounds(*SE);
  auto InnerLoopBounds = InnerLoop->getBounds(*SE);

  if (!OuterLoopBounds || !InnerLoopBounds)
    return false;

  // We want both loops to be straightforward loops
  if (!OuterLoopBounds->getStepValue() || !InnerLoopBounds->getStepValue())
    return false;

  // We want for-loops that start in zero and end in a variable that
  // is immutable inside the loop
  if (!isa<Constant>(&OuterLoopBounds->getInitialIVValue()) ||
      !isa<Constant>(&InnerLoopBounds->getInitialIVValue()) ||
      isa<Constant>(&OuterLoopBounds->getFinalIVValue()) ||
      isa<Constant>(&InnerLoopBounds->getFinalIVValue()) ||
      !dyn_cast<Constant>(&OuterLoopBounds->getInitialIVValue())
           ->isZeroValue() ||
      !dyn_cast<Constant>(&OuterLoopBounds->getInitialIVValue())->isZeroValue())
    return false;

  // We want to loop by one step and the condition must be to end
  // at the specified final value
  if (!isa<Constant>(OuterLoopBounds->getStepValue()) ||
      !isa<Constant>(InnerLoopBounds->getStepValue()) ||
      !dyn_cast<Constant>(OuterLoopBounds->getStepValue())->isOneValue() ||
      !dyn_cast<Constant>(OuterLoopBounds->getStepValue())->isOneValue() ||
      OuterLoopBounds->getCanonicalPredicate() != ICmpInst::ICMP_NE ||
      InnerLoopBounds->getCanonicalPredicate() != ICmpInst::ICMP_NE)
    return false;

  BasicBlock *OuterLoopHeader = CurLoop->getHeader();
  BasicBlock *OuterLoopLatch = CurLoop->getLoopLatch();

  // In StridedLoopUnrollVersioning::run we have already checked that the loop
  // has a preheader so we can assume it's in a canonical form.
  if (CurLoop->getNumBackEdges() != 1 || CurLoop->getNumBlocks() != 3 ||
      !OuterLoopHeader || !OuterLoopLatch)
    return false;

  BasicBlock *ForLoop =
      OuterLoopHeader != LoopBlocks[0] && OuterLoopLatch != LoopBlocks[0]
          ? LoopBlocks[0]
      : OuterLoopHeader != LoopBlocks[1] && OuterLoopLatch != LoopBlocks[1]
          ? LoopBlocks[1]
          : LoopBlocks[2];

  // We must have two canonical induction variables
  auto OuterInductionVariable = CurLoop->getCanonicalInductionVariable();
  auto InnerInductionVariable = InnerLoop->getCanonicalInductionVariable();

  SmallVector<LoadInst *> Loads;
  SmallVector<StoreInst *> Stores;

  if (!OuterInductionVariable || !InnerInductionVariable)
    return false;

  for (auto &&PN : OuterLoopHeader->phis()) {
    if (PN.getNumIncomingValues() != 2)
      return false;

    InductionDescriptor IndDesc;

    // Check if PN is a simple induction PHI:
    // - For pointer IVs: require exactly one increment (feeds back into PN)
    //   and one mem-op address (feeding a single load/store).
    // - For integer IVs: only accept the designated outer IV.
    // Reject if shape is more complex (multiple users, non-load/store ops).
    if (InductionDescriptor::isInductionPHI(&PN, CurLoop, SE, IndDesc)) {
      if (IndDesc.getKind() == InductionDescriptor::IK_PtrInduction) {
        Value *IncrementGEP = nullptr, *MemOpGEP = nullptr;
        for (auto &&User : PN.uses()) {
          if (std::distance(User.getUser()->use_begin(),
                            User.getUser()->use_end()) != 1)
            return false;
          if (User.getUser()->use_begin()->getUser() == &PN)
            IncrementGEP = User.getUser();
          else if (!MemOpGEP)
            MemOpGEP = User.getUser();
          else
            return false;
        }

        if (!MemOpGEP || !IncrementGEP)
          return false;

        auto MemOp = MemOpGEP->use_begin()->getUser();
        if (!isa<LoadInst>(MemOp) && !isa<StoreInst>(MemOp))
          return false;
      } else if (IndDesc.getKind() == InductionDescriptor::IK_IntInduction)
        if (&PN != OuterInductionVariable)
          return false;
    } else
      return false;
  }

  llvm::SmallPtrSet<llvm::Instruction *, 32> NotVisited;
  llvm::SmallVector<llvm::Instruction *, 8> WorkList;

  for (auto &&BB : CurLoop->getBlocks())
    for (auto &&V : *BB)
      if (BB != ForLoop)
        if (!canHandleInstruction(&V))
          return false;
  for (auto &&Loop : CurLoop->getSubLoops())
    for (auto &&BB : Loop->getBlocks())
      for (auto &&V : *BB)
        if (BB != ForLoop)
          if (!canHandleInstruction(&V))
            return false;

  // Collect pointers needing alignment
  SmallVector<const SCEV *, 8> AlignmentInfo;
  unsigned UnrollSize = 8;

  for (BasicBlock *BB : CurLoop->blocks()) {
    for (Instruction &I : *BB) {
      Value *Ptr = nullptr;
      uint64_t size = 0;

      if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
        Ptr = LI->getPointerOperand();
        TypeSize typeSize = DL->getTypeAllocSize(I.getType());
        if (size == 0)
          size = typeSize;
        else if (size != typeSize)
          return false;
      } else if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
        Ptr = SI->getPointerOperand();
        TypeSize typeSize =
            DL->getTypeAllocSize(SI->getValueOperand()->getType());
        if (size == 0)
          size = typeSize;
        else if (size != typeSize)
          return false;
        UnrollSize = 8 / size;
      } else
        continue;

      const SCEV *S = SE->getSCEV(Ptr);

      if (const SCEVAddRecExpr *InnerLoopAR = dyn_cast<SCEVAddRecExpr>(S)) {
        if (auto *constant =
                dyn_cast<SCEVConstant>(InnerLoopAR->getStepRecurrence(*SE))) {
          if (constant->getAPInt() != size)
            return false; // must be contiguous

          if (const SCEVAddRecExpr *AR =
                  dyn_cast<SCEVAddRecExpr>(InnerLoopAR->getStart())) {
            auto Step = AR->getStepRecurrence(*SE);
            if (isa<SCEVConstant>(Step))
              return false;
            else {
              const SCEVUnknown *Unknown = nullptr;

              if (size > 1) {
                if (auto mul = dyn_cast<SCEVMulExpr>(Step)) {
                  if (mul->getNumOperands() == 2) {
                    if (auto constant =
                            dyn_cast<SCEVConstant>(mul->getOperand(0))) {
                      if (constant->getAPInt() != size)
                        return false;
                    } else
                      return false;
                    Unknown = dyn_cast<SCEVUnknown>(mul->getOperand(1));
                    if (auto CastExtend =
                            dyn_cast<SCEVCastExpr>(mul->getOperand(1)))
                      Unknown = dyn_cast<SCEVUnknown>(CastExtend->getOperand());
                  } else
                    return false;
                } else
                  return false;
              }
              if (!Unknown) {
                Unknown = dyn_cast<SCEVUnknown>(Step);
                if (auto CastExtend = dyn_cast<SCEVCastExpr>(Step))
                  Unknown = dyn_cast<SCEVUnknown>(CastExtend->getOperand());
              }
              if (Unknown) { // stride should be fixed but not constant
                if (isa<Instruction>(Unknown->getValue()))
                  return false;
              } else
                return false;
            }

            AlignmentInfo.push_back(AR->getStart());
          } else
            return false;
        } else
          return false;
      } else if (!CurLoop->isLoopInvariant(Ptr))
        return false;
    }
  }

  // Initialize NotVisited and WorkList
  // Check that we can handle all instructions during the Strided Loop Unroll
  // pass We will ignore the exit condition and the increment from the induction
  // variable
  for (auto &&I : *ForLoop) {
    if (auto &&Store = dyn_cast<StoreInst>(&I)) {
      WorkList.push_back(Store);
      Stores.push_back(Store);
    } else if (&I != OuterInductionVariable && &I != InnerInductionVariable) {
      if (I.getParent() != InnerLoop->getHeader() &&
          &I != InnerLoop->getHeader()->getTerminator() &&
          &I != dyn_cast<BranchInst>(InnerLoop->getHeader()->getTerminator())
                    ->getCondition())
        NotVisited.insert(&I);
    }
  }

  if (WorkList.size() != 1 || Stores.size() != 1)
    return false;

  // Check dependencies between instructions that the outer loop
  // arithmetic is self-contained
  while (!WorkList.empty()) {
    llvm::Instruction *I = WorkList.back();
    WorkList.pop_back();

    /* Should check for loops, possibly with NotVisited */
    if (auto *Load = dyn_cast<LoadInst>(I)) {
      // We stop at load
      NotVisited.erase(I);

      auto Pointer = Load->getPointerOperand();
      if (auto *GEP = dyn_cast<GetElementPtrInst>(Pointer)) {
        NotVisited.erase(GEP);
        Loads.push_back(Load);
      } else
        return false;
    } else if (auto *Store = dyn_cast<StoreInst>(I)) {
      if (auto *ValueInst = dyn_cast<Instruction>(Store->getValueOperand()))
        WorkList.push_back(ValueInst);
      NotVisited.erase(I);
      auto Pointer = Store->getPointerOperand();
      if (auto *GEP = dyn_cast<GetElementPtrInst>(Pointer)) {
        NotVisited.erase(GEP);
      } else
        return false;
    } else {
      // Add operand instructions to the worklist
      for (llvm::Value *Op : I->operands())
        if (auto *DepInst = llvm::dyn_cast<llvm::Instruction>(Op))
          WorkList.push_back(DepInst);
      NotVisited.erase(I);
    }
  }

  if (!NotVisited.empty())
    return false;

  BasicBlock *Preheader = CurLoop->getLoopPreheader();

  LAI = &LAIs.getInfo(*SubLoops[0]);

  if (LAI->getRuntimePointerChecking()->getChecks().empty())
    return false;

  transformStridedSpecialCases(OuterInductionVariable, InnerInductionVariable,
                               Stores[0], Preheader, ForLoop, OuterLoopHeader,
                               OuterLoopLatch, AlignmentInfo, UnrollSize);

  return true;
}
