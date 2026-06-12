//===- MetalPrepare.cpp - Pre-serialization IR normalization --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRPrepare.h"
#include "AIR.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SCCPSolver.h"
#include <cstdlib>
#include <functional>

using namespace llvm;

#define DEBUG_TYPE "air-prepare"

namespace {
constexpr unsigned ASDevice = 1;
constexpr unsigned ASThreadgroup = 3;
constexpr unsigned PtrPhiLimit = 32;
} // namespace

// ── IRUtil helpers ───────────────────────────────────────────────────────
// Anonymous-TU statics; only the helpers used by the TG-global-GEP-rewrite
// logic below are kept here.

namespace {

// Strip identity-noise ops (xor X,0 / add X,0 / or X,0 / sub X,0, either
// operand order) that the Triton layout lowering threads through threadgroup
// byte offsets. They don't change the value but lengthen the use chain past
// computeKnownBits's default recursion depth, hiding provable alignment — which
// makes the byte-global retype path conservatively bail and leave an i8 array
// (and dynamic i8 GEPs) that the Metal GPU JIT then refuses to materialize.
static Value *stripIdentityIntOps(Value *V) {
  for (;;) {
    auto *BO = dyn_cast<BinaryOperator>(V);
    if (!BO)
      return V;
    unsigned Op = BO->getOpcode();
    if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1)))
      if (C->isZero() && (Op == Instruction::Xor || Op == Instruction::Add ||
                          Op == Instruction::Or || Op == Instruction::Sub)) {
        V = BO->getOperand(0);
        continue;
      }
    if (auto *C0 = dyn_cast<ConstantInt>(BO->getOperand(0)))
      if (C0->isZero() && (Op == Instruction::Xor || Op == Instruction::Add ||
                           Op == Instruction::Or)) {
        V = BO->getOperand(1);
        continue;
      }
    return V;
  }
}

// Provable power-of-two alignment (min count of trailing zero bits) of an
// integer SSA value. computeKnownBits caps recursion at depth 6, which the
// long index chains the Triton layout lowering emits routinely exceed — so the
// byte-global retype path conservatively bails and leaves dynamic i8 GEPs the
// Metal GPU JIT refuses to materialize. This walks only the alignment-relevant
// integer ops with a generous depth so the common strided-offset shapes
// (shl/mul-by-const, add/or/and of aligned terms, select, zext) are provable.
static unsigned minTrailingZeros(Value *V, unsigned Depth = 0) {
  Type *Ty = V->getType();
  if (!Ty->isIntegerTy())
    return 0;
  unsigned BitW = Ty->getIntegerBitWidth();
  if (auto *CI = dyn_cast<ConstantInt>(V))
    return CI->isZero() ? BitW : CI->getValue().countr_zero();
  if (Depth > 24)
    return 0;
  if (auto *BO = dyn_cast<BinaryOperator>(V)) {
    Value *A = BO->getOperand(0), *B = BO->getOperand(1);
    switch (BO->getOpcode()) {
    case Instruction::Shl:
      if (auto *C = dyn_cast<ConstantInt>(B))
        return std::min(BitW, minTrailingZeros(A, Depth + 1) +
                                  (unsigned)C->getZExtValue());
      return minTrailingZeros(A, Depth + 1);
    case Instruction::LShr:
    case Instruction::AShr:
      if (auto *C = dyn_cast<ConstantInt>(B)) {
        unsigned Sh = (unsigned)C->getZExtValue();
        unsigned TZA = minTrailingZeros(A, Depth + 1);
        return TZA > Sh ? TZA - Sh : 0;
      }
      return 0;
    case Instruction::Mul:
      return std::min(BitW, minTrailingZeros(A, Depth + 1) +
                                minTrailingZeros(B, Depth + 1));
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Or:
    case Instruction::Xor:
      return std::min(minTrailingZeros(A, Depth + 1),
                      minTrailingZeros(B, Depth + 1));
    case Instruction::And:
      // result trailing zeros ≥ max of the two operands' trailing zeros.
      return std::max(minTrailingZeros(A, Depth + 1),
                      minTrailingZeros(B, Depth + 1));
    default:
      return 0;
    }
  }
  if (auto *Sel = dyn_cast<SelectInst>(V))
    return std::min(minTrailingZeros(Sel->getTrueValue(), Depth + 1),
                    minTrailingZeros(Sel->getFalseValue(), Depth + 1));
  if (auto *ZE = dyn_cast<ZExtInst>(V))
    return minTrailingZeros(ZE->getOperand(0), Depth + 1);
  if (auto *TR = dyn_cast<TruncInst>(V))
    return std::min(BitW, minTrailingZeros(TR->getOperand(0), Depth + 1));
  return 0;
}

static void collectTGByteGlobals(Module &M,
                                 SmallVectorImpl<GlobalVariable *> &Out) {
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != ASThreadgroup)
      continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (AT && AT->getElementType()->isIntegerTy(8))
      Out.push_back(&GV);
  }
}

// Decide whether a set of constant byte offsets into a threadgroup arena
// represents genuine *buffer boundaries* (the triton allocator packs a few
// time-disjoint sub-buffers, each worth splitting into its own typed global)
// or merely *unrolled element addressing* (the mid-end constant-folds a
// strided access `base + k*elemSize` into many constant-offset GEPs — these
// all belong to ONE buffer and must NOT be split, or each tail-sized slice
// becomes its own global and the threadgroup budget explodes ~30x).
//
// The discriminator is structural, not a count threshold: unrolled addressing
// yields a DENSE run (every adjacent gap is a small element/vector stride, so
// the slots are physically contiguous in one buffer), whereas true sub-buffer
// boundaries are SPARSE — time-disjoint reuse buffers have unrelated sizes, so
// at least one gap is *large* (a jump no single element/vector access could
// span). The stride need NOT be uniform: a real unrolled run mixes 8-byte
// scalar, 16-byte vec, and 4-byte sub-element gaps. The signal is the presence
// of a large gap, not stride irregularity. `Offsets` must be sorted+deduped and
// exclude 0.
static bool offsetsAreBufferBoundaries(ArrayRef<int64_t> Offsets) {
  if (Offsets.empty())
    return false;
  if (Offsets.size() == 1)
    return true; // a single interior boundary is always a real split point
  // The widest natural threadgroup access is a 16-byte vec4. Any gap wider than
  // that cannot be two adjacent element slots of one buffer, so it marks a real
  // sub-buffer boundary. A run whose every gap is <= 16 bytes is dense unrolled
  // element addressing of ONE buffer and must NOT be split (else each
  // tail-sized slice becomes its own global and the threadgroup budget explodes
  // ~30x — see cummax scan2d: 27 contiguous i64 slots with mixed 4/8/16-byte
  // gaps would split into 27 globals = 124 KB).
  constexpr int64_t kMaxElementStride = 16;
  for (size_t i = 1; i < Offsets.size(); ++i)
    if (Offsets[i] - Offsets[i - 1] > kMaxElementStride)
      return true; // a wide gap => genuine sub-buffer boundary
  return false;    // all gaps small => dense unrolled run of one buffer
}

static void collectTGTypedGlobals(Module &M,
                                  SmallVectorImpl<GlobalVariable *> &Out) {
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != ASThreadgroup)
      continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (!AT || AT->getElementType()->isIntegerTy(8))
      continue;
    // Only genuine MMA operand scratch (__tg_dot_ab_*) is a valid merge target
    // for the byte arena. Other typed threadgroup globals - notably the
    // convert_layout scratch (__tg_cvt_*) - are independent live buffers; the
    // byte arena (e.g. a multi-warp scan's i64 index + f32 value partials) must
    // not be overlaid onto them, which would alias distinct live data and route
    // f32/f16 stores through an i64-typed global the Metal JIT cannot
    // materialize.
    if (!GV.getName().starts_with("__tg_dot_ab_"))
      continue;
    Out.push_back(&GV);
  }
}

static Type *inferElementType(Value *V) {
  for (auto *U : V->users()) {
    if (auto *SI = dyn_cast<StoreInst>(U))
      if (SI->getPointerOperand() == V)
        return SI->getValueOperand()->getType();
    if (auto *LI = dyn_cast<LoadInst>(U))
      return LI->getType();
    // A staging arena fed only by async-copy / simdgroup-matrix intrinsics has
    // no scalar loads or stores at all (the even-K fast path); the AIR pointer
    // suffix still pins the element type, and leaving it uninferred would skip
    // retyping while float GEPs index the byte global (PSO materialize fail).
    if (auto *CI = dyn_cast<CallInst>(U)) {
      if (Function *F = CI->getCalledFunction()) {
        StringRef Name = F->getName();
        if (Name.starts_with("air.simdgroup")) {
          auto &Ctx = V->getContext();
          if (Name.contains("bf16"))
            return Type::getBFloatTy(Ctx);
          if (Name.contains("f16"))
            return Type::getHalfTy(Ctx);
          return Type::getFloatTy(Ctx);
        }
      }
    }
    if (isa<GetElementPtrInst>(U) || isa<GEPOperator>(U) || isa<BitCastInst>(U))
      if (Type *T = inferElementType(U))
        return T;
  }
  return nullptr;
}

static void expandConstantExprUsers(GlobalVariable *GV) {
  SmallVector<std::pair<ConstantExpr *, Instruction *>, 4> ToExpand;
  for (auto *U : GV->users()) {
    auto *CE = dyn_cast<ConstantExpr>(U);
    if (!CE)
      continue;
    for (auto *CEU : CE->users())
      if (auto *I = dyn_cast<Instruction>(CEU))
        ToExpand.push_back({CE, I});
  }
  for (auto &[CE, I] : ToExpand) {
    auto *Inst = CE->getAsInstruction();
    Inst->insertBefore(I->getIterator());
    I->replaceUsesOfWith(CE, Inst);
  }
  SmallVector<ConstantExpr *, 4> Dead;
  for (auto *U : GV->users())
    if (auto *CE = dyn_cast<ConstantExpr>(U))
      if (CE->use_empty())
        Dead.push_back(CE);
  for (auto *CE : Dead)
    CE->destroyConstant();
}

static void collectI8Geps(Value *V, SmallVectorImpl<GetElementPtrInst *> &Out) {
  for (auto *U : V->users()) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (GEP->getSourceElementType()->isIntegerTy(8))
        Out.push_back(GEP);
      else
        collectI8Geps(GEP, Out);
    }
  }
}

// Walks loads/stores transitively reached through GEPs of V and inserts an
// identity bitcast immediately before each non-i8 access. Used by
// retypeByteGlobals when the byte global can't be safely retyped (mixed access
// types, or any unaligned dynamic byte GEP) — the writer still emits a single
// typed global, but per-site bitcasts preserve scalar/sub-element semantics.
static bool insertIdentityBitcastsAtNonByteAccesses(Value *Root) {
  bool Changed = false;
  std::function<void(Value *)> Walk = [&](Value *V) {
    for (auto *U : make_early_inc_range(V->users())) {
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getPointerOperand() == V &&
            !SI->getValueOperand()->getType()->isIntegerTy(8)) {
          auto *BC = new BitCastInst(V, V->getType(), "");
          BC->insertBefore(SI->getIterator());
          SI->setOperand(1, BC);
          Changed = true;
        }
      } else if (auto *LI = dyn_cast<LoadInst>(U)) {
        if (!LI->getType()->isIntegerTy(8)) {
          auto *BC = new BitCastInst(V, V->getType(), "");
          BC->insertBefore(LI->getIterator());
          LI->setOperand(0, BC);
          Changed = true;
        }
      } else if (isa<GetElementPtrInst>(U)) {
        Walk(U);
      }
    }
  };
  Walk(Root);
  return Changed;
}

static bool scalarizeVec1Users(Value *V, Type *I32Ty) {
  bool Changed = false;
  SmallVector<Instruction *, 8> Vec1Users;
  std::function<void(Value *)> FindVec1 = [&](Value *V) {
    for (auto *U : V->users()) {
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getPointerOperand() == V) {
          auto *VT =
              dyn_cast<FixedVectorType>(SI->getValueOperand()->getType());
          if (VT && VT->getNumElements() == 1)
            Vec1Users.push_back(SI);
        }
      } else if (auto *LI = dyn_cast<LoadInst>(U)) {
        auto *VT = dyn_cast<FixedVectorType>(LI->getType());
        if (VT && VT->getNumElements() == 1)
          Vec1Users.push_back(LI);
      } else if (isa<GetElementPtrInst>(U)) {
        FindVec1(U);
      }
    }
  };
  FindVec1(V);
  for (auto *I : Vec1Users) {
    if (auto *SI = dyn_cast<StoreInst>(I)) {
      IRBuilder<> B(SI);
      Value *Scalar = B.CreateExtractElement(SI->getValueOperand(),
                                             ConstantInt::get(I32Ty, 0));
      B.CreateAlignedStore(Scalar, SI->getPointerOperand(), SI->getAlign(),
                           SI->isVolatile());
      SI->eraseFromParent();
      Changed = true;
    } else if (auto *LI = dyn_cast<LoadInst>(I)) {
      IRBuilder<> B(LI);
      auto *VT = cast<FixedVectorType>(LI->getType());
      auto *Scalar =
          B.CreateAlignedLoad(VT->getElementType(), LI->getPointerOperand(),
                              LI->getAlign(), LI->isVolatile());
      Value *Vec = B.CreateInsertElement(UndefValue::get(VT), Scalar,
                                         ConstantInt::get(I32Ty, 0));
      LI->replaceAllUsesWith(Vec);
      LI->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

// Replace a `load <N x i1>` whose result is only consumed by constant-index
// `extractelement`s with scalar byte loads + bit extraction. Triton's bool
// reductions store individual bools into threadgroup memory and reload them as
// a wide packed-bool vector, then read back a few lanes. The Apple AGX
// AIR->ISA lowering hits a fatal error ("report_fatal_error", surfaced to the
// runtime as XPC_ERROR_CONNECTION_INTERRUPTED during PSO creation) on any
// `<N x i1>` threadgroup load.
//
// In LLVM's memory model a vector of i1 is BIT-PACKED (independent of the i1
// alloc size), so lane K lives in byte K/8 at bit K%8. We load the containing
// byte and extract the bit; the common case (bools stored one-per-byte and read
// back at lane multiples of 8) reduces to a plain byte load + low-bit trunc.
static bool narrowVectorI1Loads(Function &F) {
  bool Changed = false;
  SmallVector<LoadInst *, 4> Targets;
  for (Instruction &I : instructions(F)) {
    auto *LI = dyn_cast<LoadInst>(&I);
    if (!LI)
      continue;
    auto *VT = dyn_cast<FixedVectorType>(LI->getType());
    if (!VT || !VT->getElementType()->isIntegerTy(1))
      continue;
    // Every use must be a constant-index extractelement.
    bool OnlyConstExtract = !LI->use_empty();
    for (User *U : LI->users()) {
      auto *EE = dyn_cast<ExtractElementInst>(U);
      if (!EE || !isa<ConstantInt>(EE->getIndexOperand())) {
        OnlyConstExtract = false;
        break;
      }
    }
    if (OnlyConstExtract)
      Targets.push_back(LI);
  }
  Type *I8 = Type::getInt8Ty(F.getContext());
  for (LoadInst *LI : Targets) {
    Value *Base = LI->getPointerOperand();
    // One byte load per containing byte (lane/8), reused across lanes.
    DenseMap<uint64_t, Value *> ByteLoad;
    for (User *U : llvm::make_early_inc_range(LI->users())) {
      auto *EE = cast<ExtractElementInst>(U);
      uint64_t Lane = cast<ConstantInt>(EE->getIndexOperand())->getZExtValue();
      uint64_t ByteOff = Lane / 8, BitOff = Lane % 8;
      Value *&Byte = ByteLoad[ByteOff];
      IRBuilder<> B(LI);
      if (!Byte) {
        Value *Ptr = Base;
        if (ByteOff != 0)
          Ptr = B.CreateInBoundsGEP(I8, Base,
                                    B.getInt64(static_cast<int64_t>(ByteOff)));
        Byte = B.CreateAlignedLoad(I8, Ptr, Align(1), LI->isVolatile());
      }
      Value *Bit = Byte;
      if (BitOff != 0)
        Bit = B.CreateLShr(Byte, ConstantInt::get(I8, BitOff));
      Value *AsI1 = B.CreateTrunc(Bit, EE->getType());
      EE->replaceAllUsesWith(AsI1);
      EE->eraseFromParent();
    }
    LI->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

static bool narrowVectorI1Loads(Module &M) {
  bool Changed = false;
  for (Function &F : M)
    if (!F.isDeclaration())
      Changed |= narrowVectorI1Loads(F);
  return Changed;
}

static bool foldExtractInsert(Function &F) {
  bool Changed = false;
  for (auto &BB : F) {
    for (auto It = BB.begin(); It != BB.end();) {
      Instruction &I = *It++;
      if (auto *EE = dyn_cast<ExtractElementInst>(&I)) {
        if (auto *IE = dyn_cast<InsertElementInst>(EE->getVectorOperand())) {
          auto *VT = dyn_cast<FixedVectorType>(IE->getType());
          if (VT && VT->getNumElements() == 1) {
            EE->replaceAllUsesWith(IE->getOperand(1));
            EE->eraseFromParent();
            if (IE->use_empty())
              IE->eraseFromParent();
            Changed = true;
          }
        }
      }
    }
  }
  return Changed;
}

static bool foldExtractInsert(Module &M) {
  bool Changed = false;
  for (auto &F : M)
    Changed |= foldExtractInsert(F);
  return Changed;
}

} // namespace

// ── TG global GEP rewrite ─────────────────────────────────────────────────
// Retypes [N x i8] threadgroup globals into typed arrays based on usage
// inference, and rewrites byte-offset GEPs into element-index GEPs. Six
// sub-stages run in sequence (see comments below). Must run BEFORE the
// other MetalPrepare behaviors since retyping TG globals + their GEPs may
// produce new patterns the later stages need to normalize.

namespace {

// Shared by MergeMMA, Retype, and Strategy C. Rewrites GEPs on OldGV to use
// NewGV with element type ElemTy.
static bool rewriteByteGEPs(GlobalVariable *OldGV, GlobalVariable *NewGV,
                            ArrayType *OldAT, ArrayType *NewAT, Type *ElemTy,
                            unsigned ElemSize, LLVMContext &Ctx,
                            uint64_t ExtraElemOffset = 0) {
  bool Changed = false;
  // When the byte arena is concatenated after the MMA scratch, every element
  // index must be shifted by ExtraElemOffset so the arena lands at its slot in
  // the merged buffer.  addOff folds the shift in (constant-folded when the
  // index is constant).
  auto addOff = [&](IRBuilder<> &B, Value *Idx) -> Value * {
    if (ExtraElemOffset == 0)
      return Idx;
    Value *Off = ConstantInt::get(Idx->getType(), ExtraElemOffset);
    if (auto *CI = dyn_cast<ConstantInt>(Idx))
      return ConstantInt::get(Idx->getType(),
                              CI->getZExtValue() + ExtraElemOffset);
    return B.CreateAdd(Idx, Off);
  };
  SmallVector<GetElementPtrInst *, 16> Users;
  for (auto *U : OldGV->users())
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
      Users.push_back(GEP);

  for (auto *GEP : Users) {
    if (GEP->getPointerOperand() != OldGV)
      continue;
    IRBuilder<> B(GEP);
    Type *SrcTy = GEP->getSourceElementType();

    if (SrcTy == OldAT) {
      Value *ByteIdx = GEP->getNumIndices() >= 2
                           ? GEP->getOperand(2)
                           : ConstantInt::get(Type::getInt64Ty(Ctx), 0);
      Value *ElemIdx;
      if (auto *CI = dyn_cast<ConstantInt>(ByteIdx))
        ElemIdx =
            ConstantInt::get(CI->getType(), CI->getZExtValue() / ElemSize);
      else
        ElemIdx = B.CreateUDiv(ByteIdx,
                               ConstantInt::get(ByteIdx->getType(), ElemSize));
      ElemIdx = addOff(B, ElemIdx);
      auto *NewGEP = GetElementPtrInst::CreateInBounds(
          NewAT, NewGV, {ConstantInt::get(Type::getInt64Ty(Ctx), 0), ElemIdx},
          GEP->getName());
      NewGEP->insertBefore(B.GetInsertPoint());
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
    } else if (SrcTy->isIntegerTy(8)) {
      Value *ByteIdx = GEP->getOperand(1);
      Value *ElemIdx;
      if (auto *CI = dyn_cast<ConstantInt>(ByteIdx))
        ElemIdx =
            ConstantInt::get(CI->getType(), CI->getZExtValue() / ElemSize);
      else
        ElemIdx = B.CreateUDiv(ByteIdx,
                               ConstantInt::get(ByteIdx->getType(), ElemSize));
      ElemIdx = addOff(B, ElemIdx);
      auto *NewGEP = GetElementPtrInst::CreateInBounds(
          NewAT, NewGV, {ConstantInt::get(Type::getInt64Ty(Ctx), 0), ElemIdx},
          GEP->getName());
      NewGEP->insertBefore(B.GetInsertPoint());
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
    } else {
      // Direct base use (no GEP offset): point at element ExtraElemOffset.
      if (ExtraElemOffset == 0) {
        GEP->setOperand(0, NewGV);
      } else {
        IRBuilder<> B2(GEP);
        Value *Base = B2.CreateInBoundsGEP(
            NewAT, NewGV,
            {ConstantInt::get(Type::getInt64Ty(Ctx), 0),
             ConstantInt::get(Type::getInt64Ty(Ctx), ExtraElemOffset)});
        GEP->setOperand(0, Base);
      }
    }
    Changed = true;
  }

  // Redirect remaining direct (non-GEP) instruction users.
  SmallVector<Instruction *, 4> DirectUsers;
  for (auto *U : OldGV->users()) {
    auto *I = dyn_cast<Instruction>(U);
    if (!I || isa<GetElementPtrInst>(I))
      continue;
    DirectUsers.push_back(I);
  }
  for (auto *I : DirectUsers) {
    Value *Base = NewGV;
    if (ExtraElemOffset != 0) {
      IRBuilder<> B2(I);
      Base = B2.CreateInBoundsGEP(
          NewAT, NewGV,
          {ConstantInt::get(Type::getInt64Ty(Ctx), 0),
           ConstantInt::get(Type::getInt64Ty(Ctx), ExtraElemOffset)});
    }
    for (unsigned Op = 0; Op < I->getNumOperands(); Op++)
      if (I->getOperand(Op) == OldGV)
        I->setOperand(Op, Base);
    Changed = true;
  }
  return Changed;
}

// 14a: Split mixed-type byte globals at constant offsets.
static bool
splitMixedByteGlobals(Module &M,
                      SmallVectorImpl<GlobalVariable *> &ByteGlobals) {
  bool Changed = false;
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();

  for (size_t Gi = 0; Gi < ByteGlobals.size(); Gi++) {
    auto *GV = ByteGlobals[Gi];
    expandConstantExprUsers(GV);
    auto *OldAT = cast<ArrayType>(GV->getValueType());
    uint64_t TotalBytes = OldAT->getNumElements();

    SmallPtrSet<Type *, 4> AllScalarTypes;
    // Distinct *byte sizes* among the accesses. Same-size scalar types
    // (e.g. i16 / half / bfloat, or i32 / float) are type-pun views of the same
    // physical slots — the mid-end freely bitcasts between them — so they must
    // NOT trigger a type-split: doing so would scatter a store and its matching
    // load into separate typed globals at different threadgroup addresses,
    // breaking the buffer's store->load aliasing. A genuine mixed buffer (the
    // triton allocator packing time-disjoint buffers of different element
    // widths, e.g. an i64 index next to an f32 value) has >1 distinct size.
    SmallSet<uint64_t, 4> AllScalarSizes;
    SmallVector<int64_t, 4> ConstOffsets;
    // A wide (>1 byte) element type indexed by a runtime value directly off the
    // arena base (offset 0). The allocator overlaps time-disjoint buffers in
    // one byte arena, so this buffer's dynamic slots may run past an interior
    // constant offset that belongs to a later, time-disjoint reuse buffer.
    bool WideRuntimeBaseBuffer = false;
    // Helper: a wide (>1 byte) scalar access whose pointer is a dynamic
    // (runtime-indexed) slot of the offset-0 buffer means that buffer's dynamic
    // slots can run past an interior constant offset belonging to a later,
    // time-disjoint reuse buffer. Such a base buffer must be sized to span all
    // its slots, not truncated at the first split offset. Triton emits these as
    // byte (i8) GEPs feeding a typed load/store, so the access WIDTH lives on
    // the memory op, not the GEP source element type.
    auto flagWideBaseAccess = [&](Type *AccessTy, int64_t BaseOff,
                                  bool UnderDynamic) {
      if (BaseOff == 0 && UnderDynamic &&
          (AccessTy->isIntegerTy() || AccessTy->isFloatingPointTy()) &&
          DL.getTypeAllocSize(AccessTy) > 1)
        WideRuntimeBaseBuffer = true;
    };
    std::function<void(Value *, int64_t, bool)> CollectTypes =
        [&](Value *V, int64_t BaseOff, bool UnderDynamic) {
          for (auto *U : V->users()) {
            if (auto *SI = dyn_cast<StoreInst>(U)) {
              if (SI->getPointerOperand() == V) {
                Type *T = SI->getValueOperand()->getType();
                if (auto *VT = dyn_cast<FixedVectorType>(T))
                  T = VT->getElementType();
                if (T->isIntegerTy() || T->isFloatingPointTy()) {
                  AllScalarTypes.insert(T);
                  AllScalarSizes.insert(DL.getTypeAllocSize(T));
                  flagWideBaseAccess(T, BaseOff, UnderDynamic);
                }
              }
            } else if (auto *LI = dyn_cast<LoadInst>(U)) {
              Type *T = LI->getType();
              if (auto *VT = dyn_cast<FixedVectorType>(T))
                T = VT->getElementType();
              if (T->isIntegerTy() || T->isFloatingPointTy()) {
                AllScalarTypes.insert(T);
                AllScalarSizes.insert(DL.getTypeAllocSize(T));
                flagWideBaseAccess(T, BaseOff, UnderDynamic);
              }
            } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
              APInt Off(64, 0);
              if (GEP->accumulateConstantOffset(DL, Off)) {
                int64_t ByteOff = Off.getSExtValue();
                // A constant offset is a partition boundary only when reached
                // from the arena base by constants alone; chained on a dynamic
                // base it is plain element addressing (the optimizer's packed
                // multi-word reads), not a buffer boundary.
                if (ByteOff != 0 && !UnderDynamic)
                  ConstOffsets.push_back(ByteOff);
                CollectTypes(GEP, BaseOff + ByteOff, UnderDynamic);
              } else {
                Type *ST = GEP->getSourceElementType();
                if (BaseOff == 0 &&
                    (ST->isIntegerTy() || ST->isFloatingPointTy()) &&
                    DL.getTypeAllocSize(ST) > 1)
                  WideRuntimeBaseBuffer = true;
                CollectTypes(GEP, BaseOff, /*UnderDynamic=*/true);
              }
            } else if (isa<BitCastInst>(U)) {
              CollectTypes(U, BaseOff, UnderDynamic);
            }
          }
        };
    CollectTypes(GV, 0, /*UnderDynamic=*/false);

    // Split only a genuinely heterogeneous buffer: one that mixes scalar
    // accesses of *different byte widths*. A buffer touched by several scalar
    // types that all share one width (i16/half/bfloat, or i32/float) is a
    // single typed slot under bitcasts and must stay one global.
    if (AllScalarSizes.size() <= 1 || ConstOffsets.empty())
      continue;

    llvm::sort(ConstOffsets);
    ConstOffsets.erase(std::unique(ConstOffsets.begin(), ConstOffsets.end()),
                       ConstOffsets.end());

    // Only split when the constant offsets are genuine sub-buffer boundaries,
    // not the dense uniformly-strided run the mid-end emits when it unrolls a
    // strided access (splitting that would allocate a tail-sized global per
    // element slot — a ~30x threadgroup budget explosion).
    if (!offsetsAreBufferBoundaries(ConstOffsets))
      continue;

    // Size of the base (offset-0) typed global. Normally it ends at the first
    // constant offset. But when the offset-0 buffer is a wide runtime-indexed
    // buffer it may span past interior offsets that belong to time-disjoint
    // reuse buffers; size it to the last (largest) offset so its dynamic slots
    // are not truncated. The interior reuse buffers still split off into their
    // own typed globals, which Metal places at independent threadgroup
    // addresses, so the (source) byte overlap is harmless.
    int64_t BaseRegionEnd =
        WideRuntimeBaseBuffer ? ConstOffsets.back() : ConstOffsets.front();

    DenseMap<int64_t, GlobalVariable *> SplitMap;
    for (int64_t Off : ConstOffsets) {
      uint64_t RegionSize = TotalBytes - Off;
      if (RegionSize == 0)
        continue;
      auto *SplitAT = ArrayType::get(Type::getInt8Ty(Ctx), RegionSize);
      auto *SplitGV = new GlobalVariable(
          M, SplitAT, false, GV->getLinkage(), UndefValue::get(SplitAT),
          GV->getName() + "__off" + Twine(Off), GV,
          GlobalVariable::NotThreadLocal, ASThreadgroup);
      SplitGV->setAlignment(GV->getAlign());
      SplitMap[Off] = SplitGV;
    }

    auto *NewAT = ArrayType::get(Type::getInt8Ty(Ctx), BaseRegionEnd);
    auto *NewGV = new GlobalVariable(
        M, NewAT, false, GV->getLinkage(), UndefValue::get(NewAT),
        GV->getName().str(), GV, GlobalVariable::NotThreadLocal, ASThreadgroup);
    NewGV->setAlignment(GV->getAlign());

    SmallVector<GetElementPtrInst *, 8> Users;
    for (auto *U : GV->users())
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
        Users.push_back(GEP);

    for (auto *GEP : Users) {
      if (GEP->getPointerOperand() != GV)
        continue;
      APInt Off(64, 0);
      if (GEP->accumulateConstantOffset(DL, Off)) {
        int64_t ByteOff = Off.getSExtValue();
        if (ByteOff == 0) {
          GEP->setOperand(0, NewGV);
          if (GEP->getSourceElementType() == OldAT)
            GEP->setSourceElementType(NewAT);
        } else {
          auto Sit = SplitMap.find(ByteOff);
          if (Sit == SplitMap.end())
            continue;
          GEP->replaceAllUsesWith(Sit->second);
          GEP->eraseFromParent();
        }
      } else {
        GEP->setOperand(0, NewGV);
        if (GEP->getSourceElementType() == OldAT)
          GEP->setSourceElementType(NewAT);
      }
      Changed = true;
    }

    if (GV->use_empty())
      GV->eraseFromParent();
    ByteGlobals[Gi] = NewGV;
    SmallVector<int64_t, 8> SplitKeys;
    for (auto &Kv : SplitMap)
      SplitKeys.push_back(Kv.first);
    llvm::sort(SplitKeys);
    for (int64_t K : SplitKeys)
      ByteGlobals.push_back(SplitMap[K]);
    Changed = true;
  }
  return Changed;
}

// Returns true when the threadgroup byte arena (\p GV) is live at the SAME TIME
// as the MMA scratch buffer, so the two must NOT be overlapped at offset 0.
//
// Two cases make the byte arena concurrently live:
//
//   1. It is read/written directly by air.simdgroup_matrix_8x8_load/store
//      (the opt-in TRITON_SHARED_MMA path that feeds the MMA from shared).
//
//   2. It is the destination of an air.simdgroup_async_copy_2d.  That is the
//      software-pipeline (num_stages>=2) staging buffer: the prefetch DMA for
//      a future loop iteration writes the byte arena WHILE the current dot is
//      scattering its operands/accumulator into the MMA scratch.  With
//      multi-buffering (num_stages>=3) the arena holds more than one live slot,
//      so an MMA scratch overlapped at offset 0 stomps the prefetched operand
//      tile and corrupts the next iteration (bug #46).  The plain-load scatter
//      path reads the arena into registers, but those reads are NOT ordered
//      before the concurrent prefetch writes, so the regions still alias-clash;
//      keying off the async-copy destination captures every pipelined dot.
//
//   3. It is written by a plain StoreInst reached through a DYNAMIC-INDEX GEP.
//      That is the sync (TRITON_DMA_DISABLE / no-DMA) twin of case 2: when the
//      pipeline prefetch is lowered without async-copy, the per-element
//      prefetch writes the arena with a `store <N x T>` through the
//      rotating-slot GEP (a dynamic loop-carried index selects the live
//      buffer).  That store runs WHILE the current dot scatters into the MMA
//      scratch, so an offset-0 overlay stomps the prefetched tile exactly as in
//      case 2.  Keying off a dynamic-index GEP store (not a constant-offset
//      one) captures the rotating slot while leaving scan/conv overlays, which
//      write CONSTANT offsets, on the cheaper offset-0 path.
//
// In either case mergeByteMMA concatenates (MMA buffer AFTER the arena) instead
// of aliasing.  For non-pipelined kernels the arena has neither user and the
// cheaper offset-0 overlap is kept.
static bool concurrentWithMMAScratch(Value *V, SmallPtrSetImpl<Value *> &Seen,
                                     bool SawDynGEP) {
  if (!Seen.insert(V).second)
    return false;
  for (User *U : V->users()) {
    if (auto *CI = dyn_cast<CallInst>(U)) {
      if (auto *Callee = CI->getCalledFunction()) {
        StringRef N = Callee->getName();
        if (N.starts_with("air.simdgroup_matrix_8x8_load") ||
            N.starts_with("air.simdgroup_matrix_8x8_store"))
          return true;
        // Destination (arg 2) of the async-copy DMA = pipeline staging buffer.
        if (N.starts_with("air.simdgroup_async_copy")) {
          if (CI->arg_size() > 2 && CI->getArgOperand(2) == V)
            return true;
        }
      }
    } else if (auto *SI = dyn_cast<StoreInst>(U)) {
      // Sync rotating-slot prefetch store: a plain store into the arena through
      // a dynamic-index GEP is the live-arena signal that mirrors the
      // async-copy destination above.  Constant-offset stores (scan/conv
      // overlays) keep SawDynGEP false and stay on the offset-0 overlay.
      if (SawDynGEP && SI->getPointerOperand() == V)
        return true;
    } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (concurrentWithMMAScratch(GEP, Seen,
                                   SawDynGEP || !GEP->hasAllConstantIndices()))
        return true;
    } else if (isa<BitCastInst>(U)) {
      if (concurrentWithMMAScratch(U, Seen, SawDynGEP))
        return true;
    }
  }
  return false;
}

// 14b: Merge byte globals into MMA globals.
static bool mergeByteMMA(Module &M,
                         SmallVectorImpl<GlobalVariable *> &ByteGlobals,
                         SmallVectorImpl<GlobalVariable *> &MMAGlobals) {
  if (ByteGlobals.empty())
    return false;

  // Count convert_layout scratch (__tg_cvt_*) globals. collectTGTypedGlobals
  // lists only __tg_dot_ab_* as the MMA target, so a kernel that has one
  // genuine dot scratch PLUS one or more cvt buffers (e.g. conv and uint4x2
  // mixed-mm) reaches here with MMAGlobals.size()==1. Overlaying the byte arena
  // onto the dot scratch at offset 0 there aliases the concurrently-live cvt
  // buffers and corrupts results (conv produced NaNs), so that combination must
  // bail.
  //
  // But a scan (cummax) has NO __tg_dot_ab_* and exactly one cvt buffer that is
  // time-disjoint from the scan's byte arena (the convert finishes, a barrier,
  // then the scan reuses the same threadgroup region). Overlaying the arena's
  // index partials onto that cvt scratch is what kept these kernels under the
  // 32KB threadgroup budget. Re-enable that single-cvt overlay as the merge
  // target when there is no dot scratch, and require an EXACT element-type
  // match below so an f32/f16 value region is never routed through an i64-typed
  // cvt global (the cummin index miscompile: mismatched-width stores fail
  // Metal's materializeAll and alias distinct live data).
  GlobalVariable *CvtGV = nullptr;
  unsigned CvtCount = 0;
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != ASThreadgroup)
      continue;
    if (GV.getName().starts_with("__tg_cvt_")) {
      CvtCount++;
      CvtGV = &GV;
    }
  }

  bool CvtOverlay = false;
  if (MMAGlobals.size() != 1) {
    if (!MMAGlobals.empty() || CvtCount != 1)
      return false;
    // Only overlay onto a WIDE cvt scratch (element alloc size > 1 byte). The
    // cummax case this path exists for has a [N x i64] cvt buffer that is
    // time-disjoint from the scan byte arena, so overlaying the arena's i64
    // index region onto it is safe and keeps the kernel under the 32KB
    // threadgroup budget. A single-byte cvt buffer ([N x i8]) is the actively
    // live convert_layout staging for an int8 cast/cat/trans kernel, NOT
    // disjoint reuse scratch: its i8 element trivially "exactly matches" the
    // byte arena's inferred i8 element, so the merge would replaceAllUsesWith
    // the in-use cvt global and the Metal RAUW aborts. Gating on a wider-than-
    // byte cvt element excludes exactly those int8 crashers while keeping the
    // i64 cummax overlay, and leaves the dot+cvt bail (MMAGlobals.size()==1)
    // below untouched so conv / uint4x2 mixed-mm still bail.
    auto *CvtAT = dyn_cast<ArrayType>(CvtGV->getValueType());
    if (!CvtAT ||
        M.getDataLayout().getTypeAllocSize(CvtAT->getElementType()) <= 1)
      return false;
    MMAGlobals.push_back(CvtGV);
    CvtOverlay = true;
  } else if (CvtCount != 0) {
    return false;
  }

  bool Changed = false;
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();
  Type *I32 = Type::getInt32Ty(Ctx);

  auto *ByteGV = ByteGlobals[0];
  expandConstantExprUsers(ByteGV);

  // Check for wide vector stores.
  bool HasWideVec = false;
  {
    std::function<void(Value *)> Check = [&](Value *V) {
      for (auto *U : V->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U))
          if (SI->getPointerOperand() == V)
            if (auto *VT =
                    dyn_cast<FixedVectorType>(SI->getValueOperand()->getType()))
              if (VT->getNumElements() > 1)
                HasWideVec = true;
        if (isa<GetElementPtrInst>(U))
          Check(U);
      }
    };
    Check(ByteGV);
  }

  Changed |= scalarizeVec1Users(ByteGV, I32);
  Changed |= foldExtractInsert(M);

  if (HasWideVec)
    return Changed;

  auto *MMAGV = MMAGlobals[0];
  auto *MMAAT = cast<ArrayType>(MMAGV->getValueType());
  Type *MMAElemTy = MMAAT->getElementType();

  int BestIdx = -1;
  uint64_t BestBytes = 0;
  for (int I = 0; I < (int)ByteGlobals.size(); I++) {
    auto *BAT = cast<ArrayType>(ByteGlobals[I]->getValueType());
    uint64_t BBytes = BAT->getNumElements();
    Type *Inferred = inferElementType(ByteGlobals[I]);
    // The cvt overlay must be exact: only a byte region whose element type is
    // bit-identical to the cvt scratch may share its storage. The i32<->float
    // relaxation and the size-based fallback below are for genuine MMA operand
    // scratch (__tg_dot_ab_*), where every region is the same width as the dot
    // tile; allowing them for a cvt target is what overlaid a wide value region
    // onto a narrower-or-wider cvt buffer and corrupted the scan index.
    bool TypeMatch =
        Inferred && (Inferred == MMAElemTy ||
                     (!CvtOverlay &&
                      ((Inferred->isIntegerTy(32) && MMAElemTy->isFloatTy()) ||
                       (Inferred->isFloatTy() && MMAElemTy->isIntegerTy(32)))));
    if (TypeMatch && BBytes > BestBytes) {
      BestIdx = I;
      BestBytes = BBytes;
    }
  }
  if (BestIdx < 0 && !CvtOverlay) {
    for (int I = 0; I < (int)ByteGlobals.size(); I++) {
      auto *BAT = cast<ArrayType>(ByteGlobals[I]->getValueType());
      if (BAT->getNumElements() > BestBytes) {
        BestBytes = BAT->getNumElements();
        BestIdx = I;
      }
    }
  }

  // No exact-type byte region matched the cvt scratch: there is nothing safe to
  // overlay (e.g. cummin's offset-0 region is f32 while the cvt is i64). Leave
  // every buffer un-merged so the scan writes through correctly-typed globals.
  if (CvtOverlay && BestIdx < 0)
    return Changed;

  ByteGV = ByteGlobals[BestIdx >= 0 ? BestIdx : 0];
  auto *ByteAT = cast<ArrayType>(ByteGV->getValueType());
  uint64_t ByteBytes = ByteAT->getNumElements();
  unsigned MMAElemSize = DL.getTypeAllocSize(MMAAT->getElementType());
  uint64_t MMABytes = MMAAT->getNumElements() * MMAElemSize;

  Type *MergeElemTy = MMAElemTy;
  unsigned MergeElemSize = MMAElemSize;
  if (MergeElemSize == 0) {
    MergeElemTy = Type::getFloatTy(Ctx);
    MergeElemSize = 4;
  }

  // If the byte arena is live at the same time as the MMA scratch (pipeline
  // staging via async-copy, or the shared-direct MMA path), concatenate the two
  // regions instead of overlapping them at offset 0 (bug #46).  The MMA scratch
  // is kept at offset 0 (its air.simdgroup_matrix_8x8_load/store read a
  // CONSTANT global base; a non-zero constant-GEP base makes the Metal PSO
  // compiler fail to materialize), and the byte arena is shifted to start right
  // after it.  The arena's accesses are already dynamic GEPs, so a constant
  // element offset is harmless.  Non-pipelined kernels keep the cheaper
  // offset-0 overlap.
  SmallPtrSet<Value *, 16> SeenMMA;
  bool ByteIsMMA = concurrentWithMMAScratch(ByteGV, SeenMMA, false);
  uint64_t ByteElemCount = (ByteBytes + MergeElemSize - 1) / MergeElemSize;
  uint64_t MMAElemCount = (MMABytes + MergeElemSize - 1) / MergeElemSize;
  uint64_t ByteOffset = ByteIsMMA ? MMAElemCount : 0;
  uint64_t MergedElemCount =
      ByteIsMMA
          ? (ByteElemCount + MMAElemCount)
          : (std::max(ByteBytes, MMABytes) + MergeElemSize - 1) / MergeElemSize;

  auto *MergedAT = ArrayType::get(MergeElemTy, MergedElemCount);
  expandConstantExprUsers(ByteGV);

  auto *MergedGV =
      new GlobalVariable(M, MergedAT, false, ByteGV->getLinkage(),
                         UndefValue::get(MergedAT), ByteGV->getName().str(),
                         ByteGV, GlobalVariable::NotThreadLocal, ASThreadgroup);
  MergedGV->setAlignment(ByteGV->getAlign());

  // Rewrite the byte arena's accesses onto the merged buffer, shifted by
  // ByteOffset elements (0 in the overlap case, MMAElemCount in the concat case
  // so the arena sits right after the MMA scratch).
  Changed |= rewriteByteGEPs(ByteGV, MergedGV, ByteAT, MergedAT, MergeElemTy,
                             MergeElemSize, Ctx, ByteOffset);

  if (ByteGV->use_empty())
    ByteGV->eraseFromParent();
  // MMA scratch stays at offset 0 of the merged buffer (constant base) so its
  // air.simdgroup_matrix_8x8_load/store keep a materializable constant base.
  MMAGV->replaceAllUsesWith(MergedGV);
  MMAGV->eraseFromParent();

  if (BestIdx >= 0)
    ByteGlobals.erase(ByteGlobals.begin() + BestIdx);
  else
    ByteGlobals.clear();

  (void)Changed;
  return true;
}

// 14c: Retype [N x i8] -> [M x T].
static bool retypeByteGlobals(Module &M) {
  bool Changed = false;
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();
  Type *I32 = Type::getInt32Ty(Ctx);

  SmallVector<GlobalVariable *, 4> ByteGlobals;
  collectTGByteGlobals(M, ByteGlobals);

  for (auto *GV : ByteGlobals) {
    expandConstantExprUsers(GV);
    // A threadgroup byte-global with no remaining users is dead: some autotune
    // configs declare @global_smem but never touch it. Left in place it stays
    // an untyped [N x i8] addrspace(3) global, which metal-objdump rejects as a
    // truncated module. Erase it instead.
    if (GV->use_empty()) {
      GV->eraseFromParent();
      Changed = true;
      continue;
    }
    Type *StoreTy = inferElementType(GV);
    if (!StoreTy)
      continue;

    // Sub-track K (Task 3): mixed-access-types early-out (StoreTypes.size()>1
    // → bitcast bypass) had zero firings on the full MPS suite — current
    // Triton-MLIR lowering never produces a single byte-global with multiple
    // distinct scalar/vector access types post-MMA-merge. Dropped. Unaligned
    // byte-GEP fallback below is the remaining bitcast bypass.

    Changed |= scalarizeVec1Users(GV, I32);
    Changed |= foldExtractInsert(M);

    StoreTy = inferElementType(GV);
    if (!StoreTy)
      continue;

    auto *OldAT = cast<ArrayType>(GV->getValueType());
    uint64_t TotalBytes = OldAT->getNumElements();

    Type *ElemTy = StoreTy;
    if (ElemTy->isBFloatTy())
      ElemTy = Type::getHalfTy(Ctx);

    // When the inferred type is a vector (e.g. <4 x i32> / <8 x half>), Metal's
    // typed bitcode rejects a vector-element threadgroup global — and any mix
    // of a vector pointee with the scalar GEPs the same global is also accessed
    // by fails PSO creation. Always demote to a scalar element type. Prefer a
    // same-sized scalar type already used by a sibling GEP (so all GEPs share
    // one consistent pointee); otherwise fall back to the vector's own scalar.
    if (auto *VT = dyn_cast<FixedVectorType>(ElemTy)) {
      Type *ScalarTy = VT->getElementType();
      unsigned ScalarBytes = DL.getTypeAllocSize(ScalarTy);
      Type *SiblingScalar = nullptr;
      std::function<void(Value *)> CheckGEPs = [&](Value *V) {
        for (auto *U : V->users()) {
          if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
            Type *ST = GEP->getSourceElementType();
            if ((ST->isFloatingPointTy() || ST->isIntegerTy()) &&
                DL.getTypeAllocSize(ST) == ScalarBytes) {
              if (ST == ScalarTy)
                SiblingScalar = ScalarTy; // exact match wins
              else if (!SiblingScalar)
                SiblingScalar = ST;
            }
            CheckGEPs(GEP);
          }
        }
      };
      CheckGEPs(GV);
      ElemTy = SiblingScalar ? SiblingScalar : ScalarTy;
    }

    unsigned ElemSize = DL.getTypeAllocSize(ElemTy);
    if (ElemSize == 0)
      continue;
    uint64_t NumElems = TotalBytes / ElemSize;
    if (NumElems == 0)
      continue;

    // Classify byte GEPs: those whose offset is a known multiple of ElemSize
    // can be rewritten to typed GEPs; the rest must access the typed global
    // via an i8* alias so per-lane / sub-element scalar access stays correct.
    auto isAlignedByteGEP = [&](GetElementPtrInst *GEP) -> bool {
      if (ElemSize == 1)
        return true;
      if (GEP->getNumIndices() != 1)
        return false;
      Value *Idx = stripIdentityIntOps(GEP->getOperand(1));
      if (auto *CI = dyn_cast<ConstantInt>(Idx))
        return CI->getZExtValue() % ElemSize == 0;
      unsigned TZ = std::max(minTrailingZeros(Idx),
                             computeKnownBits(Idx, DL).countMinTrailingZeros());
      return (1u << TZ) >= ElemSize;
    };

    // If any byte GEP into GV has a dynamic index whose alignment to ElemSize
    // can't be proved, retyping would collapse distinct lanes onto the same
    // wide element (e.g. chained reductions). Fall back to the mixed-access
    // bitcast path: keep [N x i8], insert per-site bitcasts.
    {
      bool AnyUnaligned = false;
      std::function<bool(Value *)> Walk = [&](Value *V) -> bool {
        for (User *U : V->users()) {
          if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
            if (GEP->getSourceElementType()->isIntegerTy(8) &&
                !isAlignedByteGEP(GEP))
              return true;
            if (Walk(GEP))
              return true;
          }
        }
        return false;
      };
      AnyUnaligned = Walk(GV);
      if (AnyUnaligned) {
        Changed |= insertIdentityBitcastsAtNonByteAccesses(GV);
        continue;
      }
    }

    auto *NewAT = ArrayType::get(ElemTy, NumElems);
    auto *NewGV =
        new GlobalVariable(M, NewAT, GV->isConstant(), GV->getLinkage(),
                           UndefValue::get(NewAT), GV->getName() + ".typed", GV,
                           GlobalVariable::NotThreadLocal, ASThreadgroup);
    NewGV->setAlignment(GV->getAlign());

    Changed |= rewriteByteGEPs(GV, NewGV, OldAT, NewAT, ElemTy, ElemSize, Ctx);

    // Dead constant-expression GEPs into the old byte global keep use_empty()
    // false even though nothing reads through them; without pruning, the raw
    // global survives next to its .typed twin and double-counts the 32KB
    // threadgroup budget.
    GV->removeDeadConstantUsers();
    if (GV->use_empty())
      GV->eraseFromParent();

    // Clean up residual i8 GEPs: aligned ones become typed (index / ElemSize),
    // unaligned ones stay as i8 GEPs but rebase on an i8* alias of NewGV so
    // the Metal bitcode writer still emits a single typed global.
    SmallVector<GetElementPtrInst *, 8> ResidualI8;
    collectI8Geps(NewGV, ResidualI8);
    for (auto *GEP : ResidualI8) {
      IRBuilder<> B(GEP);
      Value *ByteIdx = GEP->getOperand(1);
      if (isAlignedByteGEP(GEP)) {
        Value *NewIdx;
        if (auto *CI = dyn_cast<ConstantInt>(ByteIdx))
          NewIdx =
              ConstantInt::get(CI->getType(), CI->getZExtValue() / ElemSize);
        else
          NewIdx = B.CreateUDiv(ByteIdx,
                                ConstantInt::get(ByteIdx->getType(), ElemSize));
        auto *NewGEP = B.CreateInBoundsGEP(ElemTy, GEP->getPointerOperand(),
                                           NewIdx, GEP->getName());
        GEP->replaceAllUsesWith(NewGEP);
      } else {
        // Keep i8 stride: GEP from an i8* alias of NewGV. The alias lets the
        // writer keep the global typed while preserving byte-granular access.
        auto *NewGEP = B.CreateInBoundsGEP(Type::getInt8Ty(Ctx), NewGV, ByteIdx,
                                           GEP->getName());
        GEP->replaceAllUsesWith(NewGEP);
      }
      GEP->eraseFromParent();
      Changed = true;
    }
  }

  // Strategy C: split remaining byte globals at constant offsets + retype.
  SmallVector<GlobalVariable *, 4> Remaining;
  collectTGByteGlobals(M, Remaining);

  for (auto *GV : Remaining) {
    expandConstantExprUsers(GV);
    auto *OldAT = cast<ArrayType>(GV->getValueType());
    uint64_t TotalBytes = OldAT->getNumElements();

    SmallVector<int64_t, 4> Offsets;
    bool HasDynamic = false;
    for (auto *U : GV->users()) {
      auto *GEP = dyn_cast<GetElementPtrInst>(U);
      if (!GEP) {
        // Splitting is only sound when every access provably stays inside
        // its constant-offset region. A non-GEP user (identity bitcast,
        // direct access, intrinsic operand) hides an access chain that may
        // span regions — treat like a dynamic offset and keep the global
        // whole.
        HasDynamic = true;
        break;
      }
      APInt Off(64, 0);
      if (GEP->accumulateConstantOffset(DL, Off)) {
        int64_t ByteOff = Off.getSExtValue();
        if (ByteOff != 0)
          Offsets.push_back(ByteOff);
      } else {
        HasDynamic = true;
      }
    }

    if (HasDynamic || Offsets.empty())
      continue;

    llvm::sort(Offsets);
    Offsets.erase(std::unique(Offsets.begin(), Offsets.end()), Offsets.end());

    // Same boundary-vs-unrolled-addressing discriminator as the mixed-byte
    // path above: only split on genuine sub-buffer boundaries.
    if (!offsetsAreBufferBoundaries(Offsets))
      continue;

    DenseMap<int64_t, GlobalVariable *> SplitMap;
    for (int64_t Off : Offsets) {
      uint64_t RegionSize = TotalBytes - Off;
      if (RegionSize == 0)
        continue;
      auto *SplitAT = ArrayType::get(Type::getInt8Ty(Ctx), RegionSize);
      auto *SplitGV = new GlobalVariable(
          M, SplitAT, false, GV->getLinkage(), UndefValue::get(SplitAT),
          GV->getName() + "__off" + Twine(Off), GV,
          GlobalVariable::NotThreadLocal, ASThreadgroup);
      SplitGV->setAlignment(GV->getAlign());
      SplitMap[Off] = SplitGV;
    }

    auto *NewAT = ArrayType::get(Type::getInt8Ty(Ctx), Offsets[0]);
    auto *NewGV = new GlobalVariable(
        M, NewAT, false, GV->getLinkage(), UndefValue::get(NewAT),
        GV->getName().str(), GV, GlobalVariable::NotThreadLocal, ASThreadgroup);
    NewGV->setAlignment(GV->getAlign());

    SmallVector<GetElementPtrInst *, 8> Users;
    for (auto *U : GV->users())
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
        Users.push_back(GEP);

    for (auto *GEP : Users) {
      if (GEP->getPointerOperand() != GV)
        continue;
      APInt Off(64, 0);
      if (!GEP->accumulateConstantOffset(DL, Off))
        continue;
      int64_t ByteOff = Off.getSExtValue();
      if (ByteOff == 0) {
        GEP->setOperand(0, NewGV);
        if (GEP->getSourceElementType() == OldAT)
          GEP->setSourceElementType(NewAT);
        Changed = true;
      } else {
        auto Sit = SplitMap.find(ByteOff);
        if (Sit == SplitMap.end())
          continue;
        GEP->replaceAllUsesWith(Sit->second);
        GEP->eraseFromParent();
        Changed = true;
      }
    }

    if (GV->use_empty())
      GV->eraseFromParent();

    SmallVector<GlobalVariable *, 4> ToRetype;
    ToRetype.push_back(NewGV);
    SmallVector<int64_t, 8> SplitKeys;
    for (auto &Kv : SplitMap)
      SplitKeys.push_back(Kv.first);
    llvm::sort(SplitKeys);
    for (int64_t K : SplitKeys)
      ToRetype.push_back(SplitMap[K]);
    for (auto *SplitGV : ToRetype) {
      Type *ElemTy = inferElementType(SplitGV);
      if (!ElemTy)
        continue;
      if (ElemTy->isBFloatTy())
        ElemTy = Type::getHalfTy(Ctx);
      unsigned ESize = DL.getTypeAllocSize(ElemTy);
      if (ESize == 0)
        continue;
      auto *SplitOldAT = cast<ArrayType>(SplitGV->getValueType());
      uint64_t NElems = SplitOldAT->getNumElements() / ESize;
      if (NElems == 0)
        continue;
      auto *TypedAT = ArrayType::get(ElemTy, NElems);
      auto *TypedGV = new GlobalVariable(
          M, TypedAT, false, SplitGV->getLinkage(), UndefValue::get(TypedAT),
          SplitGV->getName().str() + ".typed", SplitGV,
          GlobalVariable::NotThreadLocal, ASThreadgroup);
      TypedGV->setAlignment(SplitGV->getAlign());

      Changed |= rewriteByteGEPs(SplitGV, TypedGV, SplitOldAT, TypedAT, ElemTy,
                                 ESize, Ctx);
      if (SplitGV->use_empty())
        SplitGV->eraseFromParent();
    }
  }
  return Changed;
}

// 14d: Insert preamble GEPs for array TG globals.
static bool insertPreambleGEPs(Module &M) {
  bool Changed = false;
  auto &Ctx = M.getContext();

  SmallVector<GlobalVariable *, 8> AllTGGlobals;
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == ASThreadgroup &&
        isa<ArrayType>(GV.getValueType()))
      AllTGGlobals.push_back(&GV);

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    SmallPtrSet<GlobalVariable *, 4> UsedGlobals;
    for (auto &BB : F)
      for (auto &I : BB)
        for (auto &Op : I.operands())
          if (auto *GV = dyn_cast<GlobalVariable>(Op))
            if (GV->getAddressSpace() == ASThreadgroup)
              UsedGlobals.insert(GV);

    if (UsedGlobals.empty())
      continue;

    DenseMap<GlobalVariable *, Value *> PreambleMap;
    for (auto *GV : AllTGGlobals) {
      if (!UsedGlobals.count(GV))
        continue;
      if (!isa<ArrayType>(GV->getValueType()))
        continue;

      bool NeedsPreamble = false;
      for (auto *U : GV->users()) {
        auto *I = dyn_cast<Instruction>(U);
        if (!I || I->getFunction() != &F)
          continue;
        if (auto *GEPUser = dyn_cast<GetElementPtrInst>(U)) {
          if (GEPUser->getSourceElementType() != GV->getValueType())
            NeedsPreamble = true;
        } else {
          NeedsPreamble = true;
        }
      }
      if (!NeedsPreamble)
        continue;

      auto *AT = cast<ArrayType>(GV->getValueType());
      auto *GEP = GetElementPtrInst::CreateInBounds(
          AT, GV,
          {ConstantInt::get(Type::getInt64Ty(Ctx), 0),
           ConstantInt::get(Type::getInt64Ty(Ctx), 0)},
          "__base_" + GV->getName());
      GEP->insertBefore(F.getEntryBlock().getFirstInsertionPt());
      PreambleMap[GV] = GEP;
    }

    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (GEP->getName().starts_with("__base_"))
            continue;
        for (unsigned i = 0; i < I.getNumOperands(); i++) {
          auto *GV = dyn_cast<GlobalVariable>(I.getOperand(i));
          if (!GV)
            continue;
          auto Pit = PreambleMap.find(GV);
          if (Pit == PreambleMap.end())
            continue;
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
            if (i == 0 && GEP->getSourceElementType() == GV->getValueType())
              continue;
          // Insert identity bitcast so the writer's PTM sees the typed-
          // pointer transition explicitly (Metal v1 typed bitcode req;
          // otherwise PSO load fails on bfloat scan2d kernels). Two
          // shapes mirror the post-Prepare NormalizeAllocas re-run:
          // bfloat/half-source GEP off any typed base (case-5b), and
          // float-source GEP off a bfloat/half typed base (case-5a).
          Value *NewOp = Pit->second;
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I); GEP && i == 0) {
            Type *SrcTy = GEP->getSourceElementType();
            auto *AT = cast<ArrayType>(GV->getValueType());
            Type *GVElem = AT->getElementType();
            bool NeedsBitcast = SrcTy->isBFloatTy() || SrcTy->isHalfTy() ||
                                (SrcTy->isFloatTy() &&
                                 (GVElem->isBFloatTy() || GVElem->isHalfTy()));
            if (NeedsBitcast) {
              auto *BC =
                  CastInst::Create(Instruction::BitCast, NewOp,
                                   NewOp->getType(), "", I.getIterator());
              NewOp = BC;
            }
          } else if (auto *CI = dyn_cast<CallInst>(&I)) {
            // The TG global (here its float-typed base) is fed straight into an
            // MMA load/store intrinsic whose pointer pointee is half/bfloat
            // (p3f16/p1f16/p3bf16/p1bf16). The writer's PointeeTypeMap would
            // retag the shared base to that scalar to satisfy the MMA call,
            // which then disagrees with the float GEPs/loads that reuse the
            // same arena (load/store type != pointee type -> materializeAll
            // failure). Insert an identity bitcast so the MMA argument is a
            // distinct Value the PTM can retag in isolation, leaving the float
            // base untouched.
            Type *GVElem =
                cast<ArrayType>(GV->getValueType())->getElementType();
            Function *Callee = CI->getCalledFunction();
            if (Callee &&
                Callee->getName().starts_with("air.simdgroup_matrix_8x8_")) {
              StringRef Name = Callee->getName();
              bool MMAIsHalf = Name.contains("p3f16") || Name.contains("p1f16");
              bool MMAIsBF16 =
                  Name.contains("p3bf16") || Name.contains("p1bf16");
              bool TypeDiffers = (MMAIsHalf && !GVElem->isHalfTy()) ||
                                 (MMAIsBF16 && !GVElem->isBFloatTy());
              if (TypeDiffers) {
                auto *BC =
                    CastInst::Create(Instruction::BitCast, NewOp,
                                     NewOp->getType(), "", I.getIterator());
                NewOp = BC;
              }
            }
          }
          I.setOperand(i, NewOp);
          Changed = true;
        }
      }
    }
  }
  return Changed;
}

// 14e: Fix residual i8 GEPs on typed TG pointers.
// TODO(metal-revisit GH-XXX): the Metal GPU JIT rejects the type mismatch
// when an i8-source GEP is derived from a non-i8 typed GEP on a threadgroup
// pointer; rewrite the i8 GEP to use the producing GEP's element type.
static bool fixResidualI8GEPs(Module &M) {
  bool Changed = false;
  auto &DL = M.getDataLayout();

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<GetElementPtrInst *, 16> I8Geps;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (GEP->getSourceElementType()->isIntegerTy(8) &&
              GEP->getPointerAddressSpace() == ASThreadgroup) {
            auto *SrcGEP =
                dyn_cast<GetElementPtrInst>(GEP->getPointerOperand());
            if (SrcGEP && !SrcGEP->getSourceElementType()->isIntegerTy(8))
              I8Geps.push_back(GEP);
          }

    for (auto *GEP : I8Geps) {
      auto *SrcGEP = cast<GetElementPtrInst>(GEP->getPointerOperand());
      Type *ElemTy = SrcGEP->getSourceElementType();
      if (auto *AT = dyn_cast<ArrayType>(ElemTy))
        ElemTy = AT->getElementType();
      unsigned ElemSize = DL.getTypeAllocSize(ElemTy);

      // Vec-access value bitcasts (filled in for the byte-base case below).
      Type *AccessTy = nullptr;
      SmallVector<std::pair<Instruction *, Value *>, 2> MemUsers;

      // When the producing GEP is the [N x i8] array base (element i8, size 1),
      // the parent type carries no useful access width. Infer the real access
      // type from this i8 GEP's own load/store users instead: the Metal GPU JIT
      // rejects a dynamic-index i8-source TG GEP feeding a wider (e.g. <4 x
      // i32>) access, so retype the GEP to match the memory op. Only safe when
      // the byte index is provably a multiple of the access size.
      if (ElemSize <= 1) {
        // Look through identity ptr→ptr bitcasts (inserted by 14d) to reach the
        // real load/store (and its pointer alias) and learn its access width.
        std::function<void(Value *)> FindAccess = [&](Value *V) {
          for (auto *U : V->users()) {
            if (auto *LI = dyn_cast<LoadInst>(U)) {
              AccessTy = LI->getType();
              MemUsers.push_back({LI, V});
            } else if (auto *SI = dyn_cast<StoreInst>(U)) {
              if (SI->getPointerOperand() == V) {
                AccessTy = SI->getValueOperand()->getType();
                MemUsers.push_back({SI, V});
              }
            } else if (auto *BC = dyn_cast<BitCastInst>(U)) {
              if (BC->getType()->isPointerTy())
                FindAccess(BC);
            }
          }
        };
        FindAccess(GEP);
        if (!AccessTy)
          continue;
        unsigned AccessSize = DL.getTypeAllocSize(AccessTy);
        if (AccessSize <= 1)
          continue;

        // All GEPs that share this threadgroup byte-global base must use one
        // consistent scalar element type, else the Metal GPU JIT rejects the
        // pipeline (mixed float/i32/<N> pointee types on a single TG global).
        // Anchor on the dominant scalar element type already used by sibling
        // GEPs on the same base; default to the access's scalar type.
        Type *ScalarTy = AccessTy->getScalarType();
        DenseMap<Type *, unsigned> ScalarVotes;
        for (auto *U : SrcGEP->users())
          if (auto *Sib = dyn_cast<GetElementPtrInst>(U)) {
            Type *ST = Sib->getSourceElementType();
            if (ST->isFloatingPointTy() || ST->isIntegerTy())
              ScalarVotes[ST]++;
          }
        unsigned BestVotes = 0;
        for (auto &KV : ScalarVotes)
          if (DL.getTypeAllocSize(KV.first) == DL.getTypeAllocSize(ScalarTy) &&
              KV.second > BestVotes) {
            BestVotes = KV.second;
            ScalarTy = KV.first;
          }

        unsigned ScalarSize = DL.getTypeAllocSize(ScalarTy);
        if (ScalarSize == 0)
          continue;

        // Index must be provably a multiple of the scalar element size.
        Value *ByteIdx = GEP->getOperand(1);
        Value *AlignIdx = stripIdentityIntOps(ByteIdx);
        bool Aligned = false;
        if (auto *CI = dyn_cast<ConstantInt>(AlignIdx))
          Aligned = CI->getZExtValue() % ScalarSize == 0;
        else {
          unsigned TZ =
              std::max(minTrailingZeros(AlignIdx),
                       computeKnownBits(AlignIdx, DL).countMinTrailingZeros());
          Aligned = (1u << TZ) >= ScalarSize;
        }
        if (!Aligned)
          continue;
        ElemTy = ScalarTy;
        ElemSize = ScalarSize;
      }
      if (ElemSize == 0 || ElemSize == 1)
        continue;

      // The byte offset must be provably a multiple of the element size;
      // rescaling truncates otherwise (the optimizer emits sub-element
      // offsets, e.g. half data staged in a float-typed TG scratch). Leave
      // unaligned byte GEPs alone — the writer's identity-bitcast retype
      // keeps their records consistent.
      {
        Value *AlignIdx = stripIdentityIntOps(GEP->getOperand(1));
        bool Aligned = false;
        if (auto *CI = dyn_cast<ConstantInt>(AlignIdx))
          Aligned = CI->getZExtValue() % ElemSize == 0;
        else {
          unsigned TZ =
              std::max(minTrailingZeros(AlignIdx),
                       computeKnownBits(AlignIdx, DL).countMinTrailingZeros());
          Aligned = (1u << TZ) >= ElemSize;
        }
        if (!Aligned)
          continue;
      }

      IRBuilder<> B(GEP);
      Value *ByteIdx = GEP->getOperand(1);
      Value *ElemIdx;
      if (auto *CI = dyn_cast<ConstantInt>(ByteIdx))
        ElemIdx =
            ConstantInt::get(CI->getType(), CI->getZExtValue() / ElemSize);
      else
        ElemIdx = B.CreateUDiv(ByteIdx,
                               ConstantInt::get(ByteIdx->getType(), ElemSize));
      // When the new element type does not match the producing GEP's own
      // element type (access-driven anchor over an i8 base), retype the base
      // through an identity bitcast so the writer's usage inference gives it
      // the anchor type instead of the byte type.
      Value *Base = GEP->getPointerOperand();
      Type *ParentElem = SrcGEP->getSourceElementType();
      if (auto *PAT = dyn_cast<ArrayType>(ParentElem))
        ParentElem = PAT->getElementType();
      if (ParentElem != ElemTy)
        Base = CastInst::Create(Instruction::BitCast, Base, Base->getType(), "",
                                GEP->getIterator());
      auto *NewGEP = B.CreateInBoundsGEP(ElemTy, Base, ElemIdx, GEP->getName());

      // If the access type's scalar differs from the anchor element type (e.g.
      // <4 x i32> vector access through a float-anchored TG global), rewrite
      // the memory ops to load/store <N x ElemTy> and bitcast the value at the
      // leaf so every pointer derived from the global stays one consistent
      // type.
      if (AccessTy && AccessTy->getScalarType() != ElemTy) {
        unsigned NumElems = 1;
        if (auto *VT = dyn_cast<FixedVectorType>(AccessTy))
          NumElems = VT->getNumElements();
        Type *NewAccessTy = NumElems > 1
                                ? (Type *)FixedVectorType::get(ElemTy, NumElems)
                                : ElemTy;
        for (auto &[I, PtrAlias] : MemUsers) {
          if (auto *SI = dyn_cast<StoreInst>(I)) {
            IRBuilder<> SB(SI);
            Value *Val = SB.CreateBitCast(SI->getValueOperand(), NewAccessTy);
            SB.CreateAlignedStore(Val, NewGEP, SI->getAlign(),
                                  SI->isVolatile());
            SI->eraseFromParent();
          } else if (auto *LI = dyn_cast<LoadInst>(I)) {
            IRBuilder<> LB(LI);
            Value *NewLoad = LB.CreateAlignedLoad(
                NewAccessTy, NewGEP, LI->getAlign(), LI->isVolatile());
            Value *Casted = LB.CreateBitCast(NewLoad, LI->getType());
            LI->replaceAllUsesWith(Casted);
            LI->eraseFromParent();
          }
        }
      }

      GEP->replaceAllUsesWith(NewGEP);
      if (GEP->use_empty())
        GEP->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

// 14f: Fix mismatched-type GEPs on TG pointers.
// TODO(metal-revisit GH-XXX): the Metal GPU JIT crashes on non-float
// typed TG pointers when float-typed MMA globals coexist. Rewrite
// same-sized-type mismatches (e.g. gep i32 derived from a float TG
// pointer) to gep float + bitcast at the leaves.
static bool fixMismatchedTGGEPs(Module &M) {
  bool Changed = false;
  auto &DL = M.getDataLayout();
  auto &Ctx = M.getContext();

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    SmallVector<GetElementPtrInst *, 16> MismatchGeps;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (GEP->getPointerAddressSpace() == ASThreadgroup) {
            Type *GEPSrcTy = GEP->getSourceElementType();
            if (isa<ArrayType>(GEPSrcTy))
              continue;
            if (GEPSrcTy->isFloatTy())
              continue;
            Value *SrcPtr = GEP->getPointerOperand();
            while (auto *BC = dyn_cast<BitCastInst>(SrcPtr))
              SrcPtr = BC->getOperand(0);
            auto *SrcGEP = dyn_cast<GetElementPtrInst>(SrcPtr);
            if (!SrcGEP)
              continue;
            Type *ParentTy = SrcGEP->getSourceElementType();
            if (auto *AT = dyn_cast<ArrayType>(ParentTy))
              ParentTy = AT->getElementType();
            if (!ParentTy->isFloatTy())
              continue;
            unsigned GEPElemSize = DL.getTypeAllocSize(GEPSrcTy);
            unsigned FloatSize = DL.getTypeAllocSize(ParentTy);
            if (GEPElemSize == FloatSize)
              MismatchGeps.push_back(GEP);
          }

    for (auto *GEP : MismatchGeps) {
      IRBuilder<> B(GEP);
      Type *FloatTy = Type::getFloatTy(Ctx);
      Value *Idx = GEP->getOperand(1);
      auto *NewGEP = B.CreateInBoundsGEP(FloatTy, GEP->getPointerOperand(), Idx,
                                         GEP->getName());

      SmallVector<Instruction *, 8> Users;
      for (auto *U : GEP->users())
        if (auto *I = dyn_cast<Instruction>(U))
          Users.push_back(I);

      for (auto *U : Users) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getPointerOperand() == GEP) {
            IRBuilder<> SB(SI);
            Value *Val = SI->getValueOperand();
            if (!Val->getType()->isFloatTy())
              Val = SB.CreateBitCast(Val, FloatTy);
            SB.CreateAlignedStore(Val, NewGEP, SI->getAlign(),
                                  SI->isVolatile());
            SI->eraseFromParent();
          }
        } else if (auto *LI = dyn_cast<LoadInst>(U)) {
          if (!LI->getType()->isFloatTy()) {
            IRBuilder<> LB(LI);
            auto *NewLoad = LB.CreateAlignedLoad(
                FloatTy, NewGEP, LI->getAlign(), LI->isVolatile());
            Value *Casted = LB.CreateBitCast(NewLoad, LI->getType());
            LI->replaceAllUsesWith(Casted);
            LI->eraseFromParent();
          } else {
            LI->setOperand(0, NewGEP);
          }
        } else {
          for (unsigned I = 0; I < U->getNumOperands(); I++)
            if (U->getOperand(I) == GEP)
              U->setOperand(I, NewGEP);
        }
      }

      if (GEP->use_empty())
        GEP->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

// 14g: Scalarize wide-vector stores to a threadgroup global that is also
// accessed at a *different* vector width.
//
// Metal 4 / macOS 26 handles a `store <N x T>` to threadgroup memory fine when
// every access to that global uses the same width (the audited vec4/vec2
// fast-path). But when a wide-vector store (e.g. `store <4 x float>`) coexists
// on the same global with a narrower dynamic-indexed load (e.g.
// `load <1 x float>` / scalar `load float` through a `udiv`-derived index, as
// emitted by tile/combo reductions like var_mean), the Metal shader compiler
// fails `materializeAll` on the resulting metallib. Demoting only the wide
// stores on such mixed-width globals to a sequence of element stores fixes the
// materialization while leaving the audited same-width vec4/vec2 path intact.
static bool scalarizeMixedWidthTGVecStores(Module &M) {
  bool Changed = false;
  Type *I32 = Type::getInt32Ty(M.getContext());
  const DataLayout &DL = M.getDataLayout();

  // Walk the def-use chain from a TG global pointer through GEPs/bitcasts and
  // collect the loads/stores plus their (vector) element counts.
  auto collect = [&](GlobalVariable &GV, SmallVectorImpl<StoreInst *> &Stores,
                     unsigned &MaxStoreElems, unsigned &MinAccessElems,
                     bool &SawNarrowerAccess) {
    SmallVector<Value *, 16> Work{&GV};
    SmallPtrSet<Value *, 16> Seen;
    while (!Work.empty()) {
      Value *V = Work.pop_back_val();
      if (!Seen.insert(V).second)
        continue;
      for (User *U : V->users()) {
        if (isa<GetElementPtrInst>(U) || isa<BitCastInst>(U)) {
          Work.push_back(U);
          continue;
        }
        Type *AccTy = nullptr;
        StoreInst *SI = dyn_cast<StoreInst>(U);
        if (SI && SI->getPointerOperand() == V)
          AccTy = SI->getValueOperand()->getType();
        else if (auto *LI = dyn_cast<LoadInst>(U))
          AccTy = LI->getType();
        if (!AccTy)
          continue;
        unsigned Elems = 1;
        if (auto *VT = dyn_cast<FixedVectorType>(AccTy))
          Elems = VT->getNumElements();
        MinAccessElems = std::min(MinAccessElems, Elems);
        if (SI && SI->getPointerOperand() == V) {
          if (Elems > 1) {
            Stores.push_back(SI);
            MaxStoreElems = std::max(MaxStoreElems, Elems);
          }
        }
      }
    }
    SawNarrowerAccess = MinAccessElems < MaxStoreElems;
  };

  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != ASThreadgroup)
      continue;
    SmallVector<StoreInst *, 8> WideStores;
    unsigned MaxStoreElems = 1, MinAccessElems = ~0u;
    bool Mixed = false;
    collect(GV, WideStores, MaxStoreElems, MinAccessElems, Mixed);
    if (!Mixed || WideStores.empty())
      continue;

    for (StoreInst *SI : WideStores) {
      auto *VT = cast<FixedVectorType>(SI->getValueOperand()->getType());
      Type *ElemTy = VT->getElementType();
      IRBuilder<> B(SI);
      Value *Vec = SI->getValueOperand();
      Value *BasePtr = SI->getPointerOperand();
      Align A = SI->getAlign();
      for (unsigned i = 0, e = VT->getNumElements(); i != e; ++i) {
        Value *Elt = B.CreateExtractElement(Vec, ConstantInt::get(I32, i));
        Value *Ptr = i == 0 ? BasePtr
                            : B.CreateInBoundsGEP(ElemTy, BasePtr,
                                                  ConstantInt::get(I32, i));
        // Element i sits at byte offset i*sizeof(ElemTy); preserve alignment
        // only where it still holds (offset 0 keeps the vector alignment).
        Align EltAlign = i == 0 ? A : DL.getABITypeAlign(ElemTy);
        B.CreateAlignedStore(Elt, Ptr, EltAlign, SI->isVolatile());
      }
      SI->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

static bool rewriteTGGlobalGEPs(Module &M) {
  // Cheap early-out: nothing to do unless there is an array-typed TG global.
  bool HasArrayTG = false;
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == ASThreadgroup &&
        isa<ArrayType>(GV.getValueType())) {
      HasArrayTG = true;
      break;
    }
  if (!HasArrayTG)
    return false;

  bool Changed = false;

  SmallVector<GlobalVariable *, 4> ByteGlobals;
  SmallVector<GlobalVariable *, 4> MMAGlobals;
  collectTGByteGlobals(M, ByteGlobals);
  collectTGTypedGlobals(M, MMAGlobals);

  Changed |= splitMixedByteGlobals(M, ByteGlobals);

  // Wide-vector store scalarisation on TG byte globals (formerly gated on
  // MMAGlobals.size()==1) was a pre-Metal-4 workaround. The modern Apple
  // toolchain (xcrun metal on Metal 4 / macOS 26) emits `store <N x T>` on
  // threadgroup memory intact — vec4/vec2 float and vec4 int verified by
  // the sub-track C audit oracle. Removing the call dropped 576 firings on
  // the curated dot/reduce/scan/atomic suite; full Phase 1+2 of
  // run_mps_tests.sh showed zero new failures vs. baseline.
  // See PASS_GUARDS.md "Scalarisation audit (Sub-track C)".

  Changed |= mergeByteMMA(M, ByteGlobals, MMAGlobals);
  Changed |= retypeByteGlobals(M);
  Changed |= insertPreambleGEPs(M);

  // 14e: iterate; scalarized i8 GEPs form chains that peel one level per pass.
  // Bound is empirical (sub-track Q instrumentation): max observed depth is 1
  // on the sentinel + extended suite. Bound is defensive padding.
  for (int Iter = 0; Iter < 8; Iter++) {
    if (!fixResidualI8GEPs(M))
      break;
    Changed = true;
  }

  Changed |= fixMismatchedTGGEPs(M);
  // After the global is fully retyped and GEPs are normalized, demote wide
  // vector stores on any TG global that is also accessed at a narrower width
  // (mixed-width aliasing crashes Metal's materializeAll; see helper comment).
  Changed |= scalarizeMixedWidthTGVecStores(M);
  return Changed;
}

} // namespace

// ── i1 GEP normalization ────────────────────────────────────────────────────

static bool normalizeI1Pointers(Module &M) {
  bool Changed = false;
  Type *I8 = Type::getInt8Ty(M.getContext());

  // AIR has no 1-bit memory type; load/store i1 (e.g. mask values staged
  // through threadgroup memory) fails Metal PSO creation. Legalize i1 in memory
  // to i8: globals, GEP element types, and the load/store value types.

  // Retype `[N x i1]`/`i1` globals to i8 (collect first; erasing while
  // iterating M.globals() is invalid).
  SmallVector<GlobalVariable *, 4> I1Globals;
  for (GlobalVariable &GV : M.globals()) {
    Type *VT = GV.getValueType();
    if (VT->isIntegerTy(1))
      I1Globals.push_back(&GV);
    else if (auto *AT = dyn_cast<ArrayType>(VT))
      if (AT->getElementType()->isIntegerTy(1))
        I1Globals.push_back(&GV);
  }
  for (GlobalVariable *GV : I1Globals) {
    Type *VT = GV->getValueType();
    Type *NewVT =
        VT->isIntegerTy(1)
            ? I8
            : ArrayType::get(I8, cast<ArrayType>(VT)->getNumElements());
    auto *NewGV = new GlobalVariable(
        M, NewVT, GV->isConstant(), GV->getLinkage(),
        GV->hasInitializer() ? UndefValue::get(NewVT) : nullptr, "", GV,
        GV->getThreadLocalMode(), GV->getAddressSpace());
    NewGV->setAlignment(GV->getAlign().valueOrOne());
    NewGV->takeName(GV);
    GV->replaceAllUsesWith(NewGV);
    GV->eraseFromParent();
    Changed = true;
  }

  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        // GEP i1 element -> i8. Includes array-of-i1 source types (the
        // `__base_` preamble GEP synthesized off a `[N x i1]` global), which
        // otherwise reach the writer as an unmaterializable i1 array type.
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          Type *SrcTy = GEP->getSourceElementType();
          if (SrcTy->isIntegerTy(1)) {
            GEP->setSourceElementType(I8);
            GEP->setResultElementType(I8);
            Changed = true;
          } else if (auto *AT = dyn_cast<ArrayType>(SrcTy);
                     AT && AT->getElementType()->isIntegerTy(1)) {
            auto *NewAT = ArrayType::get(I8, AT->getNumElements());
            GEP->setSourceElementType(NewAT);
            if (GEP->getResultElementType() == AT)
              GEP->setResultElementType(NewAT);
            else if (GEP->getResultElementType()->isIntegerTy(1))
              GEP->setResultElementType(I8);
            Changed = true;
          }
          continue;
        }
        // store i1 -> store i8 (zext).
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          Value *V = SI->getValueOperand();
          if (V->getType()->isIntegerTy(1)) {
            IRBuilder<> B(SI);
            SI->setOperand(0, B.CreateZExt(V, I8));
            Changed = true;
          }
          continue;
        }
        // load i1 -> trunc(load i8).
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (LI->getType()->isIntegerTy(1)) {
            IRBuilder<> B(LI);
            auto *L8 = B.CreateLoad(I8, LI->getPointerOperand());
            L8->setAlignment(LI->getAlign());
            Value *Tr = B.CreateTrunc(L8, LI->getType());
            LI->replaceAllUsesWith(Tr);
            LI->eraseFromParent();
            Changed = true;
          }
          continue;
        }
      }
  return Changed;
}

// ── Conditional constant folding ────────────────────────────────────────────
//
// Triton emits loop-carried integer recurrences (loop counters, modulo
// double-buffer selectors) and the predicates/threadgroup-buffer indices
// derived from them in a form that is range- or constant-determinable only by
// sparse conditional constant propagation. Local simplification (InstSimplify,
// EarlyCSE, GVN) leaves them as live SSA values. Metal's PSO compiler crashes
// (XPC_ERROR_CONNECTION_INTERRUPTED) on some of those residual patterns. Run
// SCCP's solver and replace every integer value it proves constant; this is
// what InstCombine/SCCP/CorrelatedValuePropagation each independently do to
// remove the crash. CFG is left untouched (no block/edge removal) so the rest
// of the AIR pipeline sees the same control flow.

// True if Val is used, directly or through integer arithmetic / casts / selects
// / phis / GEP chains, as an index of a GEP into threadgroup (addrspace 3)
// memory.
static bool feedsThreadgroupGEPIndex(Value *Val) {
  SmallVector<Value *, 16> Work{Val};
  SmallPtrSet<Value *, 16> Seen;
  while (!Work.empty()) {
    Value *V = Work.pop_back_val();
    if (!Seen.insert(V).second)
      continue;
    for (User *U : V->users()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        if (GEP->getPointerAddressSpace() == ASThreadgroup)
          for (auto &Idx : GEP->indices())
            if (Idx.get() == V)
              return true;
        Work.push_back(GEP);
      } else if (isa<BinaryOperator>(U) || isa<CastInst>(U) ||
                 isa<SelectInst>(U) || isa<PHINode>(U)) {
        Work.push_back(U);
      }
    }
  }
  return false;
}

// SCCP's solver tracks a per-element lattice value for every aggregate it
// visits. For an N-element struct built by a chain of N insertvalue
// instructions (the pattern torch-inductor emits for wide reductions), each
// insertvalue re-merges all N element states, so solve() is O(N^2). On the
// conv1d_depthwise kernel (a 1024-element struct, 1024-long insertvalue chain)
// that is ~45s of pure aggregate lattice churn inside a single llc invocation.
//
// foldConditionalConstants only ever consumes INTEGER constants (the loop
// below skips any non-integer instruction), so none of that aggregate lattice
// work is ever read back. Skip the solver entirely for any function whose
// insertvalue aggregate work exceeds this bound. Such kernels are wide
// reduction bodies, not the small thread/lane/pid index recurrences that need
// the PSO-crash fold, so skipping the fold for them is safe: the byte-exact
// scalar integer patterns Metal's PSO compiler crashes on do not appear in a
// 1024-wide insertvalue aggregate. Small kernels stay below the bound and are
// folded exactly as before.
static constexpr uint64_t kMaxAggregateFoldWork = 4096;

static bool foldConditionalConstants(Module &M) {
  bool Changed = false;
  TargetLibraryInfoImpl TLIImpl(Triple(M.getTargetTriple()));
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    uint64_t AggregateWork = 0;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *IVI = dyn_cast<InsertValueInst>(&I))
          if (auto *STy = dyn_cast<StructType>(IVI->getType()))
            AggregateWork += STy->getNumElements();
    if (AggregateWork > kMaxAggregateFoldWork)
      continue;

    SCCPSolver Solver(
        M.getDataLayout(),
        [&](Function &Fn) -> const TargetLibraryInfo & {
          static TargetLibraryInfo TLI(TLIImpl, &Fn);
          return TLI;
        },
        M.getContext());

    Solver.markBlockExecutable(&F.front());
    for (Argument &A : F.args())
      Solver.trackValueOfArgument(&A);

    bool ResolvedUndefs = true;
    while (ResolvedUndefs) {
      Solver.solve();
      ResolvedUndefs = Solver.resolvedUndefsIn(F);
    }

    // Values that transitively derive from a thread-varying system-value
    // argument. MetalAIRSystemValues (run before this pass) lowers the thread/
    // lane-position intrinsics to kernel arguments: tid* / tidtg* / simdlane
    // are per-thread/per-lane, pid* / numprog* are threadgroup-uniform. SCCP
    // can "prove" a per-warp/per-lane value constant; folding such a value when
    // it indexes threadgroup memory is the bug below, so collect them here.
    SmallPtrSet<Value *, 32> ThreadVarying;
    {
      SmallVector<Value *, 32> Work;
      for (Argument &A : F.args()) {
        StringRef N = A.getName();
        if (N.starts_with("tid") || N.starts_with("simdlane"))
          Work.push_back(&A);
      }
      while (!Work.empty()) {
        Value *V = Work.pop_back_val();
        if (!ThreadVarying.insert(V).second)
          continue;
        for (User *U : V->users())
          if (isa<Instruction>(U))
            Work.push_back(U);
      }
    }

    for (BasicBlock &BB : F) {
      if (!Solver.isBlockExecutable(&BB))
        continue;
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        if (!I.getType()->isIntegerTy() || I.use_empty())
          continue;
        Constant *C = Solver.getConstantOrNull(&I);
        if (!C)
          continue;
        // Refuse to fold a thread-varying value that feeds a threadgroup GEP
        // index. SCCP can prove a per-warp/per-lane index constant for a single
        // thread, but it genuinely varies across the threadgroup. Replacing it
        // collapses the threadgroup address (every warp reads/writes warp 0's
        // slot), so a multi-warp cummax/cummin argmax scan stages its index
        // through the wrong threadgroup location and returns wrong results -
        // and the now-constant offset also makes the byte-global splitter
        // fragment the one index buffer into distinct per-offset globals.
        // Uniform folds (the ones that fix the fla chunk PSO crash) are not
        // thread-varying and proceed.
        if (ThreadVarying.count(&I) && feedsThreadgroupGEPIndex(&I))
          continue;
        I.replaceAllUsesWith(C);
        if (isInstructionTriviallyDead(&I))
          I.eraseFromParent();
        Changed = true;
      }
    }
  }
  return Changed;
}

// ── Struct-phi decomposition ────────────────────────────────────────────────
//
// The Metal GPU JIT cannot materialize struct-typed phi nodes (Triton wraps
// loop-carried values in singleton structs for runtime-bound loops, producing
// `phi { ptr }` etc.). Split each struct phi into one scalar phi per element,
// tracing the insertvalue chain for each incoming value and rewriting the
// extractvalue users. Runs BEFORE ptrPhiToI64 so the exposed bare ptr phis get
// the i64 treatment.

/// Trace an insertvalue chain to find the scalar value for element `idx`.
/// Returns nullptr if it cannot be resolved statically.
static Value *traceInsertValueElement(Value *V, unsigned Idx) {
  while (auto *IV = dyn_cast<InsertValueInst>(V)) {
    if (IV->getNumIndices() == 1 && IV->getIndices()[0] == Idx)
      return IV->getInsertedValueOperand();
    V = IV->getAggregateOperand();
  }
  if (isa<UndefValue>(V))
    return UndefValue::get(cast<StructType>(V->getType())->getElementType(Idx));
  if (isa<ConstantAggregateZero>(V))
    return Constant::getNullValue(
        cast<StructType>(V->getType())->getElementType(Idx));
  if (auto *C = dyn_cast<Constant>(V))
    if (auto *ST = dyn_cast<StructType>(V->getType()))
      if (Idx < ST->getNumElements())
        return C->getAggregateElement(Idx);
  return nullptr;
}

static bool decomposeStructPhis(Module &M,
                                SmallPtrSetImpl<Function *> &Decomposed) {
  bool Changed = false;

  for (Function &F : M) {
    // Collect struct phis (can't modify while iterating).
    SmallVector<PHINode *, 8> StructPhis;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (isa<StructType>(PN->getType()))
            StructPhis.push_back(PN);

    if (StructPhis.empty())
      continue;

    Decomposed.insert(&F);

    for (PHINode *PN : StructPhis) {
      auto *ST = cast<StructType>(PN->getType());
      unsigned NumElems = ST->getNumElements();
      IRBuilder<> B(PN);

      // Create one scalar phi per struct element.
      SmallVector<PHINode *, 4> ScalarPhis;
      for (unsigned i = 0; i < NumElems; i++)
        ScalarPhis.push_back(B.CreatePHI(ST->getElementType(i),
                                         PN->getNumIncomingValues(),
                                         PN->getName() + "_" + Twine(i)));

      // Populate scalar phi incoming values.
      for (unsigned Inc = 0; Inc < PN->getNumIncomingValues(); Inc++) {
        Value *InVal = PN->getIncomingValue(Inc);
        BasicBlock *InBB = PN->getIncomingBlock(Inc);

        for (unsigned i = 0; i < NumElems; i++) {
          Value *Elem = traceInsertValueElement(InVal, i);
          if (!Elem) {
            // Fallback: materialize an extractvalue in the predecessor,
            // before its terminator.
            IRBuilder<> PredB(InBB->getTerminator());
            Elem = PredB.CreateExtractValue(
                InVal, i, InVal->getName() + "_ext" + Twine(i));
          }
          ScalarPhis[i]->addIncoming(Elem, InBB);
        }
      }

      // Replace extractvalue users with the matching scalar phi.
      SmallVector<Instruction *, 8> ToRemove;
      for (User *U : PN->users())
        if (auto *EV = dyn_cast<ExtractValueInst>(U))
          if (EV->getNumIndices() == 1) {
            unsigned Idx = EV->getIndices()[0];
            if (Idx < NumElems) {
              EV->replaceAllUsesWith(ScalarPhis[Idx]);
              ToRemove.push_back(EV);
            }
          }
      for (Instruction *I : ToRemove)
        I->eraseFromParent();

      // Any remaining uses: rebuild the struct from the scalar phis.
      if (!PN->use_empty()) {
        IRBuilder<> AfterB(&*PN->getParent()->getFirstNonPHIIt());
        Value *Agg = UndefValue::get(ST);
        for (unsigned i = 0; i < NumElems; i++)
          Agg = AfterB.CreateInsertValue(Agg, ScalarPhis[i], i,
                                         PN->getName() + "_rebuild");
        PN->replaceAllUsesWith(Agg);
      }

      PN->eraseFromParent();
      Changed = true;
    }

    // Clean up dead insertvalue chains left behind.
    bool CleanedUp = true;
    while (CleanedUp) {
      CleanedUp = false;
      for (BasicBlock &BB : F)
        for (Instruction &I : llvm::make_early_inc_range(BB))
          if (auto *IV = dyn_cast<InsertValueInst>(&I))
            if (IV->use_empty()) {
              IV->eraseFromParent();
              CleanedUp = true;
            }
    }
  }

  return Changed;
}

// ── Pointer-phi to i64 lowering ─────────────────────────────────────────────

static bool hasUndefIncoming(PHINode *PN) {
  for (unsigned i = 0; i < PN->getNumIncomingValues(); i++)
    if (isa<UndefValue>(PN->getIncomingValue(i)))
      return true;
  return false;
}

static void convertPtrPhiToI64(PHINode *PN, Type *I64) {
  Type *PtrTy = PN->getType();

  PHINode *NewPhi =
      PHINode::Create(I64, PN->getNumIncomingValues(), PN->getName() + "_i64");
  NewPhi->insertBefore(PN->getIterator());

  for (unsigned i = 0; i < PN->getNumIncomingValues(); i++) {
    Value *InVal = PN->getIncomingValue(i);
    BasicBlock *InBB = PN->getIncomingBlock(i);

    Value *AsInt;
    if (isa<UndefValue>(InVal)) {
      AsInt = ConstantInt::get(I64, 0);
    } else if (isa<ConstantPointerNull>(InVal)) {
      AsInt = ConstantInt::get(I64, 0);
    } else {
      IRBuilder<> PredB(InBB->getTerminator());
      AsInt = PredB.CreatePtrToInt(InVal, I64, InVal->getName() + "_p2i");
    }
    NewPhi->addIncoming(AsInt, InBB);
  }

  BasicBlock *BB = PN->getParent();
  IRBuilder<> B(&*BB->getFirstNonPHIIt());
  Value *BackToPtr = B.CreateIntToPtr(NewPhi, PtrTy, PN->getName() + "_ptr");

  PN->replaceAllUsesWith(BackToPtr);
  PN->eraseFromParent();
}

static bool ptrPhiToI64(Module &M,
                        const SmallPtrSetImpl<Function *> &ForceConvert) {
  bool Changed = false;
  Type *I64 = Type::getInt64Ty(M.getContext());

  for (Function &F : M) {
    // Functions whose struct phis were just decomposed expose loop-carried
    // bare ptr phis. The Metal GPU JIT cannot materialize those either (they
    // were previously hidden inside the struct wrapper), so force-convert
    // every ptr phi in such functions to i64.
    bool ForceAll = ForceConvert.contains(&F);

    bool FunctionHasUndefPtrPhi = false;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (PN->getType()->isPointerTy() && hasUndefIncoming(PN)) {
            FunctionHasUndefPtrPhi = true;
            break;
          }

    for (BasicBlock &BB : F) {
      SmallVector<PHINode *, 16> PtrPhis;
      for (Instruction &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (PN->getType()->isPointerTy())
            PtrPhis.push_back(PN);

      if (!ForceAll && PtrPhis.size() <= PtrPhiLimit && !FunctionHasUndefPtrPhi)
        continue;

      for (PHINode *PN : PtrPhis) {
        convertPtrPhiToI64(PN, I64);
        Changed = true;
      }
    }
  }
  return Changed;
}

// ── Atomic intrinsic typed-pointer transition ───────────────────────────────
//
// The writer needs a fresh SSA pointer value before each `air.atomic.global.*`
// call so its pointee type in the side table can differ from the upstream
// GEP-result type (e.g. an i32 atomic on a float buffer needs an i32-typed
// pointer at the call site even though the GEP is typed float). The
// PointeeTypeMap built at write time picks up the inttoptr result via
// `inferFromUsage` and tags it with the intrinsic's expected pointee type.

static bool atomicTypedPointerFixup(Module &M) {
  bool Changed = false;
  Type *I64 = Type::getInt64Ty(M.getContext());

  SmallVector<CallInst *, 8> Fixups;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction())
          continue;
        StringRef Name = CI->getCalledFunction()->getName();
        if (!Name.starts_with("air.atomic.global."))
          continue;
        if (!Name.ends_with(".i32") && !Name.ends_with(".f32"))
          continue;
        Value *PtrArg = CI->getArgOperand(0);
        if (!PtrArg->getType()->isPointerTy())
          continue;
        unsigned AddrSpace = PtrArg->getType()->getPointerAddressSpace();
        if (AddrSpace != ASDevice && AddrSpace != ASThreadgroup)
          continue;
        // Only insert a transition when the pointer source is a GEP — the
        // typed-pointer mismatch this is fixing is exactly that case
        // (otherwise inferFromUsage already sees the atomic call directly
        // and would type the pointer to match).
        if (!isa<GetElementPtrInst>(PtrArg))
          continue;
        Fixups.push_back(CI);
      }

  for (CallInst *CI : Fixups) {
    Value *PtrArg = CI->getArgOperand(0);
    unsigned AddrSpace = PtrArg->getType()->getPointerAddressSpace();
    IRBuilder<> B(CI);
    Value *AsInt = B.CreatePtrToInt(PtrArg, I64);
    Value *NewPtr =
        B.CreateIntToPtr(AsInt, PointerType::get(M.getContext(), AddrSpace));
    CI->setArgOperand(0, NewPtr);
    Changed = true;
  }
  return Changed;
}

static bool airPrepare(Module &M) {
  bool Changed = false;
  // Fold conditionally-constant integer recurrences FIRST: Metal's PSO compiler
  // crashes on the unfolded loop-carried counters/selectors and the predicates
  // and threadgroup-buffer indices derived from them.
  // Then decompose struct-typed phi nodes: the Metal GPU JIT cannot materialize
  // them, and decomposing exposes bare ptr phis that the later ptrPhiToI64 /
  // i1 / TG-GEP stages must still normalize.
  SmallPtrSet<Function *, 4> DecomposedFns;
  Changed |= foldConditionalConstants(M);
  Changed |= decomposeStructPhis(M, DecomposedFns);
  Changed |= rewriteTGGlobalGEPs(M);
  Changed |= narrowVectorI1Loads(M);
  Changed |= normalizeI1Pointers(M);
  Changed |= ptrPhiToI64(M, DecomposedFns);
  Changed |= atomicTypedPointerFixup(M);
  return Changed;
}

PreservedAnalyses AIRPreparePass::run(Module &M, ModuleAnalysisManager &AM) {
  return airPrepare(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool AIRPrepareLegacy::runOnModule(Module &M) { return airPrepare(M); }

char AIRPrepareLegacy::ID = 0;

INITIALIZE_PASS(AIRPrepareLegacy, DEBUG_TYPE, "AIR Prepare", false, false)

ModulePass *llvm::createAIRPrepareLegacyPass() {
  return new AIRPrepareLegacy();
}
