//===-- AGXCodeGenPrepare.cpp - AGX IR preparation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IR transformations that AGX code generation relies on:
//
//  * AGXAlwaysInline marks every non-kernel function always_inline (and
//    internal, so it disappears once inlined), since there are no calls,
//    and removes the unused ones.
//  * AGXCodeGenPrepare expands what the hardware has no instruction or
//    library for: integer division, and memory intrinsics that are not
//    small enough to be expanded by instruction selection. There is no
//    floating point environment, so constrained floating point intrinsics
//    become their ordinary operations. It also unifies the function exits,
//    which the control flow structurizer expects.
//  * On subtargets where simd_shuffle merges the low index bits of each quad
//    (quad-shuffle-index-merge), shuffles with divergent indices are
//    expanded into four shuffles whose low index bits are uniform, like
//    Mesa's agx_nir_lower_subgroups.
//  * The hardware flushes 32-bit denormals but keeps 16-bit ones. Functions
//    whose denormal mode for half asks for flushing get explicit flushes of
//    the operands and results of their half operations.
//  * There is no double precision hardware: the double precision arithmetic
//    that AGXSoftF64 does not implement is diagnosed as unsupported.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXSubtarget.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/CallPromotionUtils.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

using namespace llvm;

#define DEBUG_TYPE "agx-codegenprepare"

//===----------------------------------------------------------------------===//
// AGXAlwaysInline
//===----------------------------------------------------------------------===//

namespace {

class AGXAlwaysInline : public ModulePass {
public:
  static char ID;
  AGXAlwaysInline() : ModulePass(ID) {}

  StringRef getPassName() const override { return "AGX Always Inline"; }
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

} // namespace

char AGXAlwaysInline::ID = 0;

INITIALIZE_PASS(AGXAlwaysInline, "agx-always-inline", "AGX Always Inline",
                false, false)

ModulePass *llvm::createAGXAlwaysInlinePass() { return new AGXAlwaysInline(); }

/// There are no calls in the hardware, direct or indirect: an indirect call
/// becomes a test of the callee against each address-taken function of its
/// type, with a direct call (inlined later) for each, and a trap if none
/// matches. The addresses of these functions, as values, become their
/// numbers (1, 2, ...), so that they can be stored, compared and passed.
static bool lowerIndirectCalls(Module &M) {
  SmallVector<CallBase *, 8> Indirect;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I); CB && CB->isIndirectCall() &&
                                             !CB->isInlineAsm())
        Indirect.push_back(CB);
  SmallVector<Function *, 8> Taken;
  for (Function &F : M)
    if (!F.isIntrinsic() && F.getCallingConv() != CallingConv::AGX_KERNEL &&
        F.hasAddressTaken(nullptr, /*IgnoreCallbackUses=*/false,
                          /*IgnoreAssumeLikeCalls=*/true,
                          /*IngoreLLVMUsed=*/true))
      Taken.push_back(&F);
  if (Indirect.empty() && Taken.empty())
    return false;
  for (CallBase *CB : Indirect) {
    CallBase *Rest = CB;
    for (Function *F : Taken) {
      if (F->getFunctionType() != CB->getFunctionType())
        continue;
      // The direct call is in the then block; the indirect one stays in the
      // else block.
      promoteCallWithIfThenElse(*Rest, F);
    }
    // No address-taken function of this type matches.
    IRBuilder<> B(Rest);
    B.CreateIntrinsic(Intrinsic::trap, {});
    if (!Rest->getType()->isVoidTy())
      Rest->replaceAllUsesWith(PoisonValue::get(Rest->getType()));
    Rest->eraseFromParent();
  }
  for (auto [N, F] : enumerate(Taken)) {
    Constant *Id = ConstantExpr::getIntToPtr(
        ConstantInt::get(Type::getInt64Ty(M.getContext()), N + 1), F->getType());
    F->replaceUsesWithIf(Id, [](Use &U) {
      auto *CB = dyn_cast<CallBase>(U.getUser());
      return !CB || !CB->isCallee(&U);
    });
  }
  return true;
}

bool AGXAlwaysInline::runOnModule(Module &M) {
  bool Changed = lowerIndirectCalls(M);
  // Modules without any agx_kernel function (for example tests of single
  // functions) compile every external function as an entry point.
  bool HasKernels = any_of(M, [](const Function &F) {
    return F.getCallingConv() == CallingConv::AGX_KERNEL;
  });

  SmallVector<Function *, 8> Dead;
  for (Function &F : M) {
    if (F.isDeclaration() || F.getCallingConv() == CallingConv::AGX_KERNEL)
      continue;
    if (!HasKernels && !F.hasLocalLinkage())
      continue;
    // With kernels in the module, other functions are only reachable from
    // them: internalize them so that they disappear once inlined, and drop
    // the unused ones (for example from linked device libraries).
    if (HasKernels && !F.hasLocalLinkage()) {
      F.setLinkage(GlobalValue::InternalLinkage);
      F.setVisibility(GlobalValue::DefaultVisibility);
      Changed = true;
    }
    if (F.use_empty()) {
      Dead.push_back(&F);
      continue;
    }
    if (F.hasFnAttribute(Attribute::NoInline))
      continue;
    F.addFnAttr(Attribute::AlwaysInline);
    Changed = true;
  }
  for (Function *F : Dead)
    F->eraseFromParent();
  Changed |= !Dead.empty();
  return Changed;
}

//===----------------------------------------------------------------------===//
// AGXCodeGenPrepare
//===----------------------------------------------------------------------===//

namespace {

class AGXCodeGenPrepare : public FunctionPass {
  bool expandDivisions(Function &F);
  bool expandMemIntrinsics(Function &F, const TargetTransformInfo &TTI);
  bool unifyExits(Function &F);
  bool lowerConstrainedFP(Function &F);
  bool lowerDivergentShuffles(Function &F, const UniformityInfo &UI);
  bool lowerMatrixFMA(Function &F);
  bool flushHalfDenormals(Function &F);
  bool flushFloatMinMax(Function &F);
  bool diagnoseDoubles(Function &F);

public:
  static char ID;
  AGXCodeGenPrepare() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX CodeGen Prepare"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<UniformityInfoWrapperPass>();
  }
};

} // namespace

char AGXCodeGenPrepare::ID = 0;

INITIALIZE_PASS_BEGIN(AGXCodeGenPrepare, DEBUG_TYPE, "AGX CodeGen Prepare",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_END(AGXCodeGenPrepare, DEBUG_TYPE, "AGX CodeGen Prepare",
                    false, false)

FunctionPass *llvm::createAGXCodeGenPreparePass() {
  return new AGXCodeGenPrepare();
}

/// Division by a power of two is a shift, and 32-bit division by another
/// constant is a multiply-high; both are left to instruction selection.
static bool isCheapDivision(const BinaryOperator &BO) {
  auto *C = dyn_cast<ConstantInt>(BO.getOperand(1));
  if (!C)
    return false;
  if (C->getValue().isPowerOf2())
    return true;
  return BO.getType()->isIntegerTy(32);
}

bool AGXCodeGenPrepare::expandDivisions(Function &F) {
  SmallVector<BinaryOperator *, 8> Worklist;
  for (Instruction &I : instructions(F)) {
    auto *BO = dyn_cast<BinaryOperator>(&I);
    if (!BO || !BO->getType()->isIntegerTy())
      continue;
    switch (BO->getOpcode()) {
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
      if (!isCheapDivision(*BO) &&
          BO->getType()->getIntegerBitWidth() <= 64)
        Worklist.push_back(BO);
      break;
    default:
      break;
    }
  }

  for (BinaryOperator *BO : Worklist) {
    bool IsRem = BO->getOpcode() == Instruction::URem ||
                 BO->getOpcode() == Instruction::SRem;
    bool Is64 = BO->getType()->getIntegerBitWidth() > 32;
    if (IsRem)
      Is64 ? expandRemainderUpTo64Bits(BO) : expandRemainderUpTo32Bits(BO);
    else
      Is64 ? expandDivisionUpTo64Bits(BO) : expandDivisionUpTo32Bits(BO);
  }
  return !Worklist.empty();
}

bool AGXCodeGenPrepare::expandMemIntrinsics(Function &F,
                                            const TargetTransformInfo &TTI) {
  // Instruction selection expands memory intrinsics of a known, small size
  // into loads and stores; everything else becomes a loop.
  constexpr uint64_t MaxInlineSize = 128;
  SmallVector<MemIntrinsic *, 4> Worklist;
  for (Instruction &I : instructions(F)) {
    auto *MI = dyn_cast<MemIntrinsic>(&I);
    if (!MI)
      continue;
    auto *Len = dyn_cast<ConstantInt>(MI->getLength());
    if (Len && Len->getZExtValue() <= MaxInlineSize)
      continue;
    Worklist.push_back(MI);
  }

  for (MemIntrinsic *MI : Worklist) {
    if (auto *Cpy = dyn_cast<MemCpyInst>(MI)) {
      expandMemCpyAsLoop(Cpy, TTI);
    } else if (auto *Move = dyn_cast<MemMoveInst>(MI)) {
      if (!expandMemMoveAsLoop(Move, TTI))
        continue;
    } else if (auto *Set = dyn_cast<MemSetInst>(MI)) {
      expandMemSetAsLoop(Set, TTI);
    } else {
      continue;
    }
    MI->eraseFromParent();
  }
  return !Worklist.empty();
}

bool AGXCodeGenPrepare::lowerConstrainedFP(Function &F) {
  SmallVector<ConstrainedFPIntrinsic *, 8> Worklist;
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<ConstrainedFPIntrinsic>(&I))
      Worklist.push_back(CI);

  for (ConstrainedFPIntrinsic *CI : Worklist) {
    IRBuilder<> B(CI);
    SmallVector<Value *, 4> Args(CI->args().begin(),
                                 CI->args().begin() +
                                     CI->getNonMetadataArgCount());
    Value *Res = nullptr;
    switch (CI->getIntrinsicID()) {
#define INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                        \
  case Intrinsic::INTRINSIC:                                                   \
    if (Instruction::isBinaryOp(Instruction::NAME))                            \
      Res = B.CreateBinOp(Instruction::BinaryOps(Instruction::NAME), Args[0],  \
                          Args[1]);                                            \
    else                                                                       \
      Res = B.CreateCast(Instruction::CastOps(Instruction::NAME), Args[0],     \
                         CI->getType());                                       \
    break;
#define CMP_INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC, DAGN)              \
  case Intrinsic::INTRINSIC:                                                   \
    Res = B.CreateFCmp(cast<ConstrainedFPCmpIntrinsic>(CI)->getPredicate(),    \
                       Args[0], Args[1]);                                      \
    break;
#define FUNCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                           \
  case Intrinsic::INTRINSIC: {                                                 \
    SmallVector<Type *, 4> ArgTys;                                             \
    for (Value *A : Args)                                                      \
      ArgTys.push_back(A->getType());                                          \
    SmallVector<Type *, 2> OverloadTys;                                        \
    FunctionType *FT = FunctionType::get(CI->getType(), ArgTys, false);        \
    if (Intrinsic::isSignatureValid(Intrinsic::NAME, FT, OverloadTys))         \
      Res = B.CreateCall(Intrinsic::getOrInsertDeclaration(                    \
                             F.getParent(), Intrinsic::NAME, OverloadTys),     \
                         Args);                                                \
    break;                                                                     \
  }
#include "llvm/IR/ConstrainedOps.def"
    default:
      break;
    }
    if (!Res)
      continue;
    Res->takeName(CI);
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
  }
  return !Worklist.empty();
}

/// Replace the returns, and unreachable terminators, by branches to a single
/// return block.
bool AGXCodeGenPrepare::unifyExits(Function &F) {
  SmallVector<Instruction *, 4> Exits;
  for (BasicBlock &BB : F) {
    Instruction *T = BB.getTerminator();
    if (isa<ReturnInst>(T) || isa<UnreachableInst>(T))
      Exits.push_back(T);
  }
  if (Exits.size() <= 1)
    return false;
  if (!F.getReturnType()->isVoidTy())
    return false;

  BasicBlock *Ret = BasicBlock::Create(F.getContext(), "unified.return", &F);
  ReturnInst::Create(F.getContext(), Ret);
  for (Instruction *T : Exits) {
    UncondBrInst::Create(Ret, T->getIterator());
    T->eraseFromParent();
  }
  return true;
}

/// simd_shuffle reads from each quad the thread at the OR of the low two
/// index bits of the quad's threads (applegpu documents this, Mesa works
/// around it). A shuffle whose index is the same in a quad (an immediate, or
/// low bits that are uniform) reads the thread it names. Divergent indices
/// are split into the four quad positions:
///
///   r = shuffle(v, idx & ~3); r = idx & 3 == k ? shuffle(v, idx & ~3 | k) : r
///
/// Quad shuffles take the four immediate positions instead. Relative
/// shuffles with divergent deltas become shuffles of the computed thread;
/// threads whose source is outside the SIMD-group (or quad) keep their own
/// value, as Metal documents for simd_shuffle_up and simd_shuffle_down.
bool AGXCodeGenPrepare::lowerDivergentShuffles(Function &F,
                                               const UniformityInfo &UI) {
  SmallVector<IntrinsicInst *, 8> Worklist;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II)
      continue;
    switch (II->getIntrinsicID()) {
    case Intrinsic::agx_simd_shuffle:
    case Intrinsic::agx_simd_shuffle_xor:
    case Intrinsic::agx_simd_shuffle_up:
    case Intrinsic::agx_simd_shuffle_down:
    case Intrinsic::agx_quad_shuffle:
    case Intrinsic::agx_quad_shuffle_xor:
    case Intrinsic::agx_quad_shuffle_up:
    case Intrinsic::agx_quad_shuffle_down:
      if (!isa<Constant>(II->getArgOperand(1)) &&
          UI.isDivergentAtUse(II->getArgOperandUse(1)))
        Worklist.push_back(II);
      break;
    default:
      break;
    }
  }

  for (IntrinsicInst *II : Worklist) {
    IRBuilder<> B(II);
    Value *V = II->getArgOperand(0);
    Value *Idx = II->getArgOperand(1);
    Type *Ty = V->getType();
    Intrinsic::ID ID = II->getIntrinsicID();
    bool Quad = ID == Intrinsic::agx_quad_shuffle ||
                ID == Intrinsic::agx_quad_shuffle_xor ||
                ID == Intrinsic::agx_quad_shuffle_up ||
                ID == Intrinsic::agx_quad_shuffle_down;
    unsigned Size = Quad ? 4 : 32;

    // The thread to read, relative to the start of the quad for quad
    // shuffles.
    Value *Lane = nullptr;
    if (ID != Intrinsic::agx_simd_shuffle && ID != Intrinsic::agx_quad_shuffle) {
      Lane = B.CreateIntrinsic(Intrinsic::agx_thread_index_in_simdgroup, {});
      if (Quad)
        Lane = B.CreateAnd(Lane, 3);
    }
    switch (ID) {
    case Intrinsic::agx_simd_shuffle_xor:
    case Intrinsic::agx_quad_shuffle_xor:
      Idx = B.CreateAnd(B.CreateXor(Lane, Idx), Size - 1);
      break;
    case Intrinsic::agx_simd_shuffle_up:
    case Intrinsic::agx_quad_shuffle_up:
      Idx = B.CreateSelect(B.CreateICmpUGE(Lane, Idx), B.CreateSub(Lane, Idx),
                           Lane);
      break;
    case Intrinsic::agx_simd_shuffle_down:
    case Intrinsic::agx_quad_shuffle_down:
      Idx = B.CreateSelect(
          B.CreateICmpULT(Idx, B.CreateSub(B.getInt32(Size), Lane)),
          B.CreateAdd(Lane, Idx), Lane);
      break;
    default:
      break;
    }

    Intrinsic::ID Shuffle =
        Quad ? Intrinsic::agx_quad_shuffle : Intrinsic::agx_simd_shuffle;
    Value *Low = B.CreateAnd(Idx, 3);
    Value *Base = Quad ? nullptr : B.CreateAnd(Idx, 0x1c);
    auto Position = [&](unsigned K) -> Value * {
      if (Quad)
        return B.getInt32(K);
      return K ? B.CreateOr(Base, K) : Base;
    };
    Value *Result = B.CreateIntrinsic(Shuffle, {Ty}, {V, Position(0)});
    for (unsigned K = 1; K < 4; ++K) {
      Value *R = B.CreateIntrinsic(Shuffle, {Ty}, {V, Position(K)});
      Result = B.CreateSelect(B.CreateICmpEQ(Low, B.getInt32(K)), R, Result);
    }
    Result->takeName(II);
    II->replaceAllUsesWith(Result);
    II->eraseFromParent();
  }
  return !Worklist.empty();
}

/// Whether I computes a floating point value from its operands (and so
/// reads and produces denormals according to the denormal mode).
static bool isFPArithmetic(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
    return true;
  default:
    break;
  }
  const auto *II = dyn_cast<IntrinsicInst>(&I);
  if (!II)
    return false;
  switch (II->getIntrinsicID()) {
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::sqrt:
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::minimum:
  case Intrinsic::maximum:
  case Intrinsic::minimumnum:
  case Intrinsic::maximumnum:
  case Intrinsic::exp2:
  case Intrinsic::log2:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::roundeven:
  case Intrinsic::canonicalize:
  case Intrinsic::agx_rcp:
  case Intrinsic::agx_rsqrt:
    return true;
  default:
    return false;
  }
}

static bool isHalf(const Type *Ty) { return Ty->getScalarType()->isHalfTy(); }

/// llvm.agx.simdgroup.matrix.fma: D = A * B + C on 8x8 matrices spread over
/// the SIMD-group. Thread t holds the elements (r, c) and (r, c + 1) of each
/// matrix, with r = 4 * (t / 16) + (t / 2) % 4 and c = 4 * ((t / 8) % 2) +
/// 2 * (t % 2). Each thread gathers its row of A and its two columns of B
/// with shuffles and accumulates in order of k with fused multiply-adds.
bool AGXCodeGenPrepare::lowerMatrixFMA(Function &F) {
  SmallVector<IntrinsicInst *, 2> Calls;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I);
        II && II->getIntrinsicID() == Intrinsic::agx_simdgroup_matrix_fma)
      Calls.push_back(II);
  for (IntrinsicInst *II : Calls) {
    IRBuilder<> B(II);
    Type *VT = II->getType();
    Value *A = II->getArgOperand(0), *Bm = II->getArgOperand(1),
          *C = II->getArgOperand(2);
    Value *T = B.CreateIntrinsic(Intrinsic::agx_thread_index_in_simdgroup, {});
    auto Bits = [&](Value *V, unsigned Shift, unsigned Mask) {
      return B.CreateAnd(B.CreateLShr(V, Shift), Mask);
    };
    // The thread holding the elements 2j and 2j + 1 of row r.
    auto Holder = [&](Value *R, Value *J) {
      Value *L = B.CreateShl(B.CreateLShr(R, 2), 4);
      L = B.CreateOr(L, B.CreateShl(B.CreateLShr(J, 1), 3));
      L = B.CreateOr(L, B.CreateShl(B.CreateAnd(R, 3), 1));
      return B.CreateOr(L, B.CreateAnd(J, 1));
    };
    auto Shuffle = [&](Value *V, unsigned Elt, Value *Lane) {
      Value *X = B.CreateExtractElement(V, Elt);
      return B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {X->getType()}, {X, Lane});
    };
    Value *Row = B.CreateOr(B.CreateShl(Bits(T, 4, 1), 2), Bits(T, 1, 3));
    Value *ColPair = B.CreateOr(B.CreateShl(Bits(T, 3, 1), 1), Bits(T, 0, 1));
    SmallVector<Value *, 8> RowA, ColB0, ColB1;
    for (unsigned J = 0; J < 4; ++J) {
      Value *Lane = Holder(Row, B.getInt32(J));
      RowA.push_back(Shuffle(A, 0, Lane));
      RowA.push_back(Shuffle(A, 1, Lane));
    }
    for (unsigned K = 0; K < 8; ++K) {
      Value *Lane = Holder(B.getInt32(K), ColPair);
      ColB0.push_back(Shuffle(Bm, 0, Lane));
      ColB1.push_back(Shuffle(Bm, 1, Lane));
    }
    Value *D0 = B.CreateExtractElement(C, uint64_t(0));
    Value *D1 = B.CreateExtractElement(C, 1);
    for (unsigned K = 0; K < 8; ++K) {
      D0 = B.CreateIntrinsic(Intrinsic::fma, {D0->getType()}, {RowA[K], ColB0[K], D0});
      D1 = B.CreateIntrinsic(Intrinsic::fma, {D1->getType()}, {RowA[K], ColB1[K], D1});
    }
    Value *R = B.CreateInsertElement(PoisonValue::get(VT), D0, uint64_t(0));
    R = B.CreateInsertElement(R, D1, 1);
    II->replaceAllUsesWith(R);
    II->eraseFromParent();
  }
  return !Calls.empty();
}

/// The hardware minimum and maximum return a denormal operand as it is, also
/// where float denormals are flushed: flush their results.
bool AGXCodeGenPrepare::flushFloatMinMax(Function &F) {
  if (F.getDenormalMode(APFloat::IEEEsingle()).Output == DenormalMode::IEEE)
    return false;
  SmallVector<IntrinsicInst *, 8> MinMax;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !II->getType()->isFloatTy() || II->hasMetadata("agx.ieee") ||
        II->hasMetadata("agx.exact"))
      continue;
    switch (II->getIntrinsicID()) {
    case Intrinsic::minnum:
    case Intrinsic::maxnum:
    case Intrinsic::minimum:
    case Intrinsic::maximum:
    case Intrinsic::minimumnum:
    case Intrinsic::maximumnum:
      MinMax.push_back(II);
      break;
    default:
      break;
    }
  }
  for (IntrinsicInst *II : MinMax) {
    IRBuilder<> B(II->getNextNode());
    SmallVector<Use *, 4> Uses;
    for (Use &U : II->uses())
      Uses.push_back(&U);
    Value *R = B.CreateUnaryIntrinsic(Intrinsic::canonicalize, II);
    for (Use *U : Uses)
      U->set(R);
  }
  return !MinMax.empty();
}

bool AGXCodeGenPrepare::flushHalfDenormals(Function &F) {
  DenormalMode Mode = F.getDenormalMode(APFloat::IEEEhalf());
  bool FlushInputs = Mode.Input != DenormalMode::IEEE &&
                     Mode.Input != DenormalMode::Dynamic;
  bool FlushOutputs = Mode.Output != DenormalMode::IEEE &&
                      Mode.Output != DenormalMode::Dynamic;
  if (!FlushInputs && !FlushOutputs)
    return false;
  bool PositiveZero = Mode.Output == DenormalMode::PositiveZero;

  // flush(x) = |x| < 2^-14 ? +-0 : x. NaNs compare false and pass through.
  SmallPtrSet<Value *, 16> Flushed;
  auto Flush = [&](IRBuilder<> &B, Value *X, bool Positive) -> Value * {
    if (Flushed.contains(X))
      return X;
    if (auto *C = dyn_cast<ConstantFP>(X))
      if (!C->getValueAPF().isDenormal())
        return X;
    Type *Ty = X->getType();
    Value *Abs = B.CreateUnaryIntrinsic(Intrinsic::fabs, X);
    Value *Small = B.CreateFCmpOLT(
        Abs, ConstantFP::get(Ty, APFloat::getSmallestNormalized(
                                     APFloat::IEEEhalf())));
    Value *Zero = Positive ? ConstantFP::get(Ty, 0.0)
                           : B.CreateBinaryIntrinsic(Intrinsic::copysign,
                                                     ConstantFP::get(Ty, 0.0),
                                                     X);
    Value *R = B.CreateSelect(Small, Zero, X);
    Flushed.insert(R);
    return R;
  };

  SmallVector<Instruction *, 32> Worklist;
  for (Instruction &I : instructions(F)) {
    // Operations with their own denormal handling (from AGXLowerFPEnv).
    if (I.hasMetadata("agx.ieee") || I.hasMetadata("agx.exact"))
      continue;
    bool Arith = isFPArithmetic(I) && isHalf(I.getType());
    bool Reads = Arith || (isa<FCmpInst>(I) &&
                           isHalf(I.getOperand(0)->getType())) ||
                 (isa<FPExtInst>(I) && isHalf(I.getOperand(0)->getType())) ||
                 ((isa<FPToSIInst>(I) || isa<FPToUIInst>(I)) &&
                  isHalf(I.getOperand(0)->getType()));
    bool Writes = Arith || (isa<FPTruncInst>(I) && isHalf(I.getType()));
    if (Reads || Writes)
      Worklist.push_back(&I);
  }

  for (Instruction *I : Worklist) {
    IRBuilder<> B(I);
    bool Arith = isFPArithmetic(*I) && isHalf(I->getType());
    bool Reads = !isa<FPTruncInst>(I);
    if (FlushInputs && Reads) {
      unsigned NumOps = isa<CallBase>(I) ? cast<CallBase>(I)->arg_size()
                                         : I->getNumOperands();
      for (unsigned Op = 0; Op < NumOps; ++Op)
        if (I->getOperand(Op)->getType()->isFPOrFPVectorTy() &&
            isHalf(I->getOperand(Op)->getType()))
          I->setOperand(Op, Flush(B, I->getOperand(Op), false));
    }
    if (auto *II = dyn_cast<IntrinsicInst>(I);
        II && II->getIntrinsicID() == Intrinsic::canonicalize) {
      // The operand is flushed already; nothing else to canonicalize.
      Value *Op = II->getArgOperand(0);
      if (FlushOutputs && !Flushed.contains(Op))
        Op = Flush(B, Op, PositiveZero);
      II->replaceAllUsesWith(Op);
      II->eraseFromParent();
      continue;
    }
    if (FlushOutputs && (Arith || isa<FPTruncInst>(I))) {
      B.SetInsertPoint(I->getNextNode());
      SmallVector<User *, 4> OldUsers(I->users());
      Value *R = Flush(B, I, PositiveZero);
      I->replaceUsesWithIf(R, [&](Use &U) {
        return is_contained(OldUsers, U.getUser());
      });
    }
  }
  // The flushes are explicit now. Later combines would otherwise assume that
  // the hardware flushes and fold them away.
  AttrBuilder AB(F.getContext());
  AB.addDenormalFPEnvAttr(DenormalFPEnv(
      DenormalMode::getIEEE(), F.getDenormalMode(APFloat::IEEEsingle())));
  F.addFnAttrs(AB);
  return true;
}

bool AGXCodeGenPrepare::diagnoseDoubles(Function &F) {
  SmallVector<Instruction *, 8> Unsupported;
  for (Instruction &I : instructions(F)) {
    bool Double = (isFPArithmetic(I) || isa<FCmpInst>(I) ||
                   isa<FPToSIInst>(I) || isa<FPToUIInst>(I) ||
                   isa<FPTruncInst>(I)) &&
                  I.getOperand(0)->getType()->getScalarType()->isDoubleTy();
    Double |= (isa<SIToFPInst>(I) || isa<UIToFPInst>(I) ||
               isa<FPExtInst>(I) || isFPArithmetic(I)) &&
              I.getType()->getScalarType()->isDoubleTy();
    if (Double)
      Unsupported.push_back(&I);
  }
  if (Unsupported.empty())
    return false;
  F.getContext().diagnose(DiagnosticInfoUnsupported(
      F, "double precision floating point arithmetic (AGX has no double "
         "precision hardware)",
      Unsupported.front()->getDebugLoc()));
  for (Instruction *I : Unsupported) {
    I->replaceAllUsesWith(PoisonValue::get(I->getType()));
    I->eraseFromParent();
  }
  return true;
}

bool AGXCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const TargetTransformInfo &TTI =
      getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  const TargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  const auto &ST = TM.getSubtarget<AGXSubtarget>(F);
  bool Changed = diagnoseDoubles(F);
  Changed |= lowerMatrixFMA(F);
  if (ST.hasQuadShuffleIndexMerge())
    Changed |= lowerDivergentShuffles(
        F, getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo());
  Changed |= lowerConstrainedFP(F);
  Changed |= flushHalfDenormals(F);
  Changed |= flushFloatMinMax(F);
  Changed |= expandDivisions(F);
  Changed |= expandMemIntrinsics(F, TTI);
  Changed |= unifyExits(F);
  return Changed;
}
