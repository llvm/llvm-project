//===- BitcodeEmitter.cpp - AIR v1 bitcode emitter ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Top-level orchestrator for AIR v1 bitcode emission. Delegates to
// ValueEnumerator, TypeTableWriter, ConstantsWriter, MetadataWriter,
// and FunctionWriter.
//
//===----------------------------------------------------------------------===//

#include "BitcodeEmitter.h"
#include "BitcodeEncoding.h"
#include "LowerVectorSelect.h"
#include "MetadataWriter.h"
#include "AIRConstraints.h"
#include "AIRVersion.h"
#include "ValueEnumerator.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include <functional>

using namespace llvm;

namespace llvm {
namespace metal {

// The pointee type the AIR reader will attribute to a pointer value.
static Type *effectivePointee(Value *Base, const PointeeTypeMap &PTM) {
  if (auto *G = dyn_cast<GetElementPtrInst>(Base))
    return G->getResultElementType();
  return PTM.get(Base);
}

// Insert an identity bitcast carrying NewPointee in the PTM. See header above.
static BitCastInst *retypePointerVia(Value *Ptr, Type *NewPointee,
                                     Instruction *BeforeI,
                                     PointeeTypeMap &PTM) {
  auto *BC = cast<BitCastInst>(CastInst::Create(
      Instruction::BitCast, Ptr, Ptr->getType(), "", BeforeI->getIterator()));
  PTM.set(BC, NewPointee);
  return BC;
}

// Collect every instruction of type `Inst` in the module that satisfies
// `Pred`, into a worklist. The collect-then-rewrite split is mandatory for the
// scalarize/lower passes: each rewrites by inserting new instructions and
// erasing the original, so collecting first keeps the body iterators valid.
template <typename Inst, typename PredT>
static SmallVector<Inst *, 8> collectInsts(Module &M, PredT Pred) {
  SmallVector<Inst *, 8> Out;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *Cast = dyn_cast<Inst>(&I))
          if (Pred(Cast))
            Out.push_back(Cast);
  return Out;
}

// Lower all ConstantExpr operands in instructions to real instructions.
// AIR's GPU JIT doesn't handle constant expression records in bitcode,
// so they must be materialized as instructions before serialization.
//
// For byte-stride GEPs on threadgroup float globals (e.g.,
// gep i8, @tg_global, i64 byte_offset where @tg_global is [N x float]),
// converts to float-element GEPs (gep float, @base, i64 float_index)
// because AIR v1 typed-pointer bitcode requires GEP source type to match
// the pointer's pointee type.
void lowerConstantExprs(Module &M) {
  auto &Ctx = M.getContext();
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  unsigned FloatSize = M.getDataLayout().getTypeAllocSize(FloatTy);

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<std::pair<Instruction *, unsigned>, 32> Worklist;
    bool Changed = true;
    while (Changed) {
      Changed = false;
      Worklist.clear();
      for (auto &BB : F)
        for (auto &I : BB)
          for (unsigned J = 0; J < I.getNumOperands(); J++)
            if (isa<ConstantExpr>(I.getOperand(J)))
              Worklist.push_back({&I, J});
      for (auto &[I, OpIdx] : Worklist) {
        auto *CE = cast<ConstantExpr>(I->getOperand(OpIdx));
        Instruction *NewI = CE->getAsInstruction();

        // Convert byte-stride GEPs on TG float globals to float-element GEPs.
        if (auto *GEP = dyn_cast<GetElementPtrInst>(NewI)) {
          if (GEP->getSourceElementType()->isIntegerTy(8) &&
              GEP->getPointerAddressSpace() == metal::AS::Threadgroup &&
              GEP->getNumIndices() == 1) {
            // Check if base is a TG global with float array element type
            Value *Base = GEP->getPointerOperand();
            GlobalVariable *GV = dyn_cast<GlobalVariable>(Base);
            if (GV) {
              Type *ElemTy = nullptr;
              if (auto *AT = dyn_cast<ArrayType>(GV->getValueType()))
                ElemTy = AT->getElementType();
              if (ElemTy && ElemTy->isFloatTy()) {
                Value *ByteIdx = GEP->idx_begin()->get();
                if (auto *CI = dyn_cast<ConstantInt>(ByteIdx)) {
                  uint64_t ByteOff = CI->getZExtValue();
                  if (ByteOff % FloatSize == 0) {
                    // Find or create a base float* from the GV.
                    // Look for an existing gep [N x float], @GV, 0, 0 in the
                    // function's entry block.
                    Value *FloatBase = nullptr;
                    for (auto *U : GV->users()) {
                      auto *BaseGEP = dyn_cast<GetElementPtrInst>(U);
                      if (!BaseGEP || !BaseGEP->getParent())
                        continue;
                      if (BaseGEP->getFunction() == &F &&
                          BaseGEP->getSourceElementType() ==
                              GV->getValueType() &&
                          BaseGEP->getNumIndices() == 2) {
                        FloatBase = BaseGEP;
                        break;
                      }
                    }
                    if (!FloatBase) {
                      // Create base GEP: gep [N x float], @GV, 0, 0
                      auto *NewBaseGEP = GetElementPtrInst::CreateInBounds(
                          GV->getValueType(), GV,
                          {ConstantInt::get(I64Ty, 0),
                           ConstantInt::get(I64Ty, 0)});
                      NewBaseGEP->insertBefore(
                          F.getEntryBlock().getFirstInsertionPt());
                      FloatBase = NewBaseGEP;
                    }
                    // Create: gep float, %base, i64 (byteOff/4)
                    auto *FloatGEP = GetElementPtrInst::CreateInBounds(
                        ElemTy, FloatBase,
                        {ConstantInt::get(I64Ty, ByteOff / FloatSize)});
                    FloatGEP->insertBefore(I->getIterator());
                    I->setOperand(OpIdx, FloatGEP);
                    NewI->deleteValue(); // discard the byte GEP
                    Changed = true;
                    continue;
                  }
                }
              }
            }
          }
        }

        NewI->insertBefore(I->getIterator());
        I->setOperand(OpIdx, NewI);
        Changed = true;
      }
    }

    // Identity ptr-to-ptr bitcasts (same opaque type, different typed pointer
    // semantics) are kept - the FunctionWriter handles them by emitting a
    // bitcast to the correct typed pointer type inferred from PTM/usage.
  }
}

// (The shapes 1-3 GEP normalizers — formerly normalizeVectorGEPs,
// fixGEPTypeMismatches, normalizeByteArrayGEPs and their 3-call wrapper — are
// now folded into the single normalizeGEPs pass below. See its taxonomy
// comment for the per-shape classifier and rationale.)

// A pointer phi's record carries one pointee type; every incoming value must
// resolve to it. Globals (typed as their value type) and differently-typed
// GEP chains as incomings make the reader reject the record ("Invalid phi
// record"); wrap such incomings in an identity bitcast typed to the phi's
// pointee. The bitcast lands in the incoming block before its terminator.
static void fixPhiIncomingTypes(Module &M, PointeeTypeMap &PTM) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN)
          break; // phis are at block start
        if (!PN->getType()->isPointerTy())
          continue;
        // Only intervene when an incoming carries a CONCRETE pointee that the
        // phi record must be emitted against. If every incoming defaults to
        // the per-AS fallback (e.g. all-null, or args used only by the phi),
        // doing nothing keeps the phi, its incomings, and any consumer (e.g.
        // an insertelement into a <N x ptr>) all resolving to the same
        // default — touching it would create a spurious mismatch.
        Type *PhiPointee = PTM.get(PN);
        if (!PhiPointee) {
          for (unsigned J = 0; J < PN->getNumIncomingValues(); ++J) {
            Value *In = PN->getIncomingValue(J);
            if (isa<Constant>(In))
              continue;
            if (auto *GV = dyn_cast<GlobalVariable>(In))
              PhiPointee = GV->getValueType();
            else if (auto *G = dyn_cast<GetElementPtrInst>(In))
              PhiPointee = G->getResultElementType();
            else if (Type *T = PTM.get(In))
              PhiPointee = T;
            else
              continue;
            break;
          }
        }
        if (!PhiPointee)
          continue; // no concrete pointee anywhere — leave the phi alone
        for (unsigned J = 0; J < PN->getNumIncomingValues(); ++J) {
          Value *In = PN->getIncomingValue(J);
          Type *InPointee = nullptr;
          if (isa<ConstantPointerNull>(In)) {
            // A typed null is emitted via SETTYPE against its own pointer
            // type's default pointee; when that disagrees with the phi
            // pointee the record is invalid. Replace with an inttoptr(0)
            // pinned to the phi pointee so the incoming is a real typed value.
            auto &Ctx = M.getContext();
            auto *Zero = ConstantInt::get(Type::getInt64Ty(Ctx), 0);
            auto *I2P = new IntToPtrInst(
                Zero, In->getType(), "",
                PN->getIncomingBlock(J)->getTerminator()->getIterator());
            PTM.set(I2P, PhiPointee);
            PN->setIncomingValue(J, I2P);
            continue;
          }
          if (isa<Constant>(In))
            continue; // other constants: leave untouched
          if (auto *GV = dyn_cast<GlobalVariable>(In))
            InPointee = GV->getValueType();
          else if (auto *G = dyn_cast<GetElementPtrInst>(In))
            InPointee = G->getResultElementType();
          else
            InPointee = PTM.get(In);
          if (InPointee == PhiPointee)
            continue;
          PN->setIncomingValue(
              J,
              retypePointerVia(In, PhiPointee,
                               PN->getIncomingBlock(J)->getTerminator(), PTM));
        }
        // Pin the phi's own pointee so the record's type index matches the
        // (now-consistent) incomings.
        PTM.set(PN, PhiPointee);
      }
  }
}

// GEP source-element-type normalization (shapes 1-3, single pass).
//
// The AIR v1 reader requires a GEP's explicit source element type to equal
// the typed pointee it attributes to the base pointer; any disagreement is
// "Explicit gep type does not match pointee type of pointer operand". O3 emits
// four distinct GEP shapes that violate this. This pass classifies each GEP it
// walks and dispatches to the matching arm:
//
//   1. Vector-pointee GEP    `gep <NxT>, p, i, j`   (VectorCombine wide load)
//        -> linearize to `gep T, p, i*N+j`. The typed-pointer machinery has no
//           vector-pointee slot, so the two indices are flattened into one.
//   2. MMA element-mismatch   `gep half/i8/i32, p`  (device/TG MMA buffers)
//        -> collapse to float-element (same-size i32/float retype, no stride
//           change) or, for a smaller source (half/i8), an identity-bitcast
//           base typed to that source. MMA-modules only; a genuine i32 integer
//           buffer (PTM pointee == i32 source) is left alone.
//   3. Byte-stride GEP        `gep [Nxi8]/i8, p`     (non-global folded bytes)
//        -> rescale to the base's PTM pointee when the stride divides (pure
//           retype, or constant byte-offset rescale for plain i8), else an
//           identity-bitcast base typed to the source. Applies to every module,
//           not just MMA users. Global bases are skipped (shape 4's job).
//   4. Array-global GEP       `gep elem, @GV, i`     (handled post-constexpr by
//           normalizeArrayGlobalGEPs, see its call site — it must run after
//           lowerConstantExprs so materialized constexpr GEPs are covered).
//
// Ordering / why this is structured as three sequential phases:
//   - Phase 1, shape 1 (vector linearization) ERASES the 2-index GEP and emits
//     a new single-index `gep T` whose source type T (the vector element type)
//     may itself be half/i8 — i.e. a fresh shape-2/3 candidate. The original
//     three passes ran sequentially, so the byte/MMA retypers saw shape-1
//     output. We preserve that by doing all vector linearization first,
//     module-wide, before the type-mismatch arms run. (A single collect-once
//     walk would miss these freshly-created GEPs, changing behavior.)
//   - Phase 2, shape 2 (MMA element-mismatch) is a ONE-SHOT collect+rewrite
//     (NO fixpoint), exactly as the original fixGEPTypeMismatches: a half/i8
//     base-bitcast leaves the GEP source type unchanged, so re-running the
//     shape-2 predicate would re-fire and emit a second redundant bitcast.
//   - Phase 3, shape 3 (byte-stride) is iterated to a fixpoint, exactly as the
//     original normalizeByteArrayGEPs: retyping a GEP changes the pointee its
//     dependent GEPs see, so we re-scan until nothing changes.
//   - Precedence on overlap: a plain `i8`/half single-index GEP on a device/TG
//     pointer can match BOTH shape 2 and shape 3. In the original sequence
//     shape 2 ran first (whole pass) and claimed/rewrote it before shape 3's
//     pass even started. We preserve that by running phase 2 entirely before
//     phase 3, and by having classify() return MMATypeMismatch ahead of
//     ByteArray. Once shape 2 has retyped an i8 GEP's base, shape 3's
//     `Pointee != SrcTy` test no longer fires on it (Pointee now == i8 source),
//     matching the original cross-pass interaction. Shape 3 also requires a
//     non-global base, keeping it disjoint from shape 4.
//
// Shape 4 (normalizeArrayGlobalGEPs) is deliberately NOT folded in here: it
// must run LATER, after lowerConstantExprs materializes constexpr GEPs. It
// stays a separate call at its current call site.
static void normalizeGEPs(Module &M, PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  Type *FloatTy = Type::getFloatTy(Ctx);
  const DataLayout &DL = M.getDataLayout();

  bool HasMMA = false;
  for (auto &F : M)
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_"))
      HasMMA = true;

  // --- Phase 1: shape-1 vector-pointee linearization (module-wide first) ---
  // Must complete before the type-mismatch arms so the half/i8 single-index
  // GEPs it produces are visible to shapes 2/3 (see ordering note above).
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<GetElementPtrInst *, 8> ToFix;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (isa<FixedVectorType>(GEP->getSourceElementType()) &&
              GEP->getNumIndices() == 2)
            ToFix.push_back(GEP);
    for (auto *GEP : ToFix) {
      auto *VT = cast<FixedVectorType>(GEP->getSourceElementType());
      IRBuilder<> B(GEP);
      Value *I0 = B.CreateSExtOrTrunc(GEP->getOperand(1), B.getInt64Ty());
      Value *I1 = B.CreateSExtOrTrunc(GEP->getOperand(2), B.getInt64Ty());
      Value *Lin =
          B.CreateAdd(B.CreateMul(I0, B.getInt64(VT->getNumElements())), I1);
      auto *NewGEP = cast<GetElementPtrInst>(
          B.CreateGEP(VT->getElementType(), GEP->getPointerOperand(), Lin));
      NewGEP->setIsInBounds(GEP->isInBounds());
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
    }
  }

  // Shape-2 (MMA element-mismatch) predicate. MMA-modules only.
  auto isMMAMismatch = [&](GetElementPtrInst *GEP) -> bool {
    if (!HasMMA)
      return false;
    Type *SrcTy = GEP->getSourceElementType();
    if (SrcTy == FloatTy || GEP->getNumIndices() != 1)
      return false;
    if (!SrcTy->isIntegerTy() && !SrcTy->isHalfTy() && !SrcTy->isBFloatTy())
      return false;
    unsigned AS = GEP->getPointerAddressSpace();
    if (AS != metal::AS::Device && AS != metal::AS::Threadgroup)
      return false;
    // Don't float-ify a genuine integer buffer (e.g. the i32 output of an int8
    // dot, whose f32 accumulator is fptosi'd to i32). Collapse-to-float holds
    // only for buffers fed to float MMA intrinsics; rewriting an i32 buffer's
    // GEP source to float leaves its pointee i32 → "gep type does not match
    // pointee" → materializeAll failure.
    if (Type *Pointee = PTM.get(GEP->getPointerOperand()))
      if (Pointee->isIntegerTy() && Pointee == SrcTy)
        return false;
    return true;
  };

  // Shape-3 (byte-stride GEP on non-global bases) predicate.
  auto isByteArray = [&](GetElementPtrInst *GEP) -> bool {
    if (GEP->getNumIndices() != 1)
      return false;
    Type *SrcTy = GEP->getSourceElementType();
    auto *AT = dyn_cast<ArrayType>(SrcTy);
    bool IsByteArray = AT && AT->getElementType()->isIntegerTy(8);
    // Two byte-stride forms: `[N x i8]` (index = element count) and plain `i8`
    // (index = byte offset).
    if (!IsByteArray && !SrcTy->isIntegerTy(8))
      return false;
    // Array globals are normalizeArrayGlobalGEPs' job (shape 4, 2-index form).
    if (isa<GlobalVariable>(GEP->getPointerOperand()))
      return false;
    Type *Pointee = effectivePointee(GEP->getPointerOperand(), PTM);
    return Pointee && Pointee != SrcTy;
  };

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    // --- Phase 2: shape 2 (MMA element-mismatch) — one-shot, NO fixpoint.
    // Runs entirely before phase 3 to claim overlapping i8/half device/TG
    // GEPs first (original pass order). A half/i8 base-bitcast leaves the GEP
    // source unchanged, so re-running would emit a redundant second bitcast —
    // hence no fixpoint, exactly as the original fixGEPTypeMismatches.
    {
      SmallVector<GetElementPtrInst *, 8> ToFix;
      for (auto &BB : F)
        for (auto &I : BB)
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
            if (isMMAMismatch(GEP))
              ToFix.push_back(GEP);
      for (auto *GEP : ToFix) {
        Type *SrcTy = GEP->getSourceElementType();
        Value *Ptr = GEP->getPointerOperand();
        // Same-size types (i32 vs float, both 4 bytes): retype the GEP source
        // to float. Stride is identical so the arithmetic is unchanged.
        if (SrcTy->getPrimitiveSizeInBits() == 32) {
          GEP->setSourceElementType(FloatTy);
          GEP->setResultElementType(FloatTy);
          continue;
        }
        // Different-size (half=16, i8=8 vs float=32): stride differs, so we
        // can't retype the GEP itself — give it a base typed to its source.
        GEP->setOperand(0, retypePointerVia(Ptr, SrcTy, GEP, PTM));
      }
    }

    // --- Phase 3: shape 3 (byte-stride) — iterated to a fixpoint.
    // Rewriting a GEP changes the pointee its dependent GEPs see, so re-scan
    // until nothing changes (chains are shallow; converges in 2-3 rounds).
    bool Changed = true;
    while (Changed) {
      Changed = false;
      SmallVector<GetElementPtrInst *, 8> ToFix;
      for (auto &BB : F)
        for (auto &I : BB)
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
            if (isByteArray(GEP))
              ToFix.push_back(GEP);
      for (auto *GEP : ToFix) {
        Type *SrcTy = GEP->getSourceElementType();
        Type *Pointee = effectivePointee(GEP->getPointerOperand(), PTM);
        bool PointeeOK = Pointee->isSized() && !Pointee->isAggregateType() &&
                         DL.getTypeAllocSize(Pointee) > 0;
        if (auto *AT = dyn_cast<ArrayType>(SrcTy)) {
          // [N x i8]: same stride as the pointee iff alloc sizes match.
          if (PointeeOK &&
              DL.getTypeAllocSize(Pointee) == AT->getNumElements()) {
            GEP->setSourceElementType(Pointee);
            GEP->setResultElementType(Pointee);
            Changed = true;
            continue;
          }
        } else {
          // i8: a constant byte offset divisible by the pointee size can be
          // rescaled into an element-typed GEP.
          auto *CI = dyn_cast<ConstantInt>(GEP->idx_begin()->get());
          uint64_t ESz = PointeeOK ? DL.getTypeAllocSize(Pointee) : 0;
          if (CI && ESz && CI->getSExtValue() % (int64_t)ESz == 0) {
            GEP->setSourceElementType(Pointee);
            GEP->setResultElementType(Pointee);
            GEP->setOperand(1,
                            ConstantInt::get(CI->getType(), CI->getSExtValue() /
                                                                (int64_t)ESz));
            Changed = true;
            continue;
          }
        }
        // Fallback: give the GEP a base typed to its own source type.
        GEP->setOperand(
            0, retypePointerVia(GEP->getPointerOperand(), SrcTy, GEP, PTM));
        Changed = true;
      }
    }

    {
      SmallVector<GetElementPtrInst *, 8> ToFix;
      for (auto &BB : F)
        for (auto &I : BB)
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
            Value *Base = GEP->getPointerOperand();
            if (isa<GlobalVariable>(Base) || isa<BitCastInst>(Base))
              continue;
            Type *Pointee = effectivePointee(Base, PTM);
            if (Pointee && Pointee != GEP->getSourceElementType())
              ToFix.push_back(GEP);
          }
      for (auto *GEP : ToFix)
        GEP->setOperand(0, retypePointerVia(GEP->getPointerOperand(),
                                            GEP->getSourceElementType(), GEP,
                                            PTM));
    }
  }
}

// O3 canonicalizes 3-way comparison idioms (e.g. the if/elif/else `-1/0/1`
// ladder a `tl.map_elementwise` callback expands to) into the `llvm.scmp` /
// `llvm.ucmp` intrinsics. The AIR backend has no lowering for them, so
// the JIT fails with "Undefined symbols: llvm.scmp.*". Expand them inline:
// scmp(a,b) = zext(a > b) - zext(a < b) using signed/unsigned predicates per
// intrinsic, then sign-extended to the (possibly wider) result type.
static void lowerCmpIntrinsics(Module &M) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<CallInst *, 8> Calls;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Intrinsic::ID ID = CI->getIntrinsicID();
          if (ID == Intrinsic::scmp || ID == Intrinsic::ucmp)
            Calls.push_back(CI);
        }
    for (CallInst *CI : Calls) {
      bool Signed = CI->getIntrinsicID() == Intrinsic::scmp;
      IRBuilder<> B(CI);
      Value *A = CI->getArgOperand(0);
      Value *Bv = CI->getArgOperand(1);
      Value *Gt = Signed ? B.CreateICmpSGT(A, Bv) : B.CreateICmpUGT(A, Bv);
      Value *Lt = Signed ? B.CreateICmpSLT(A, Bv) : B.CreateICmpULT(A, Bv);
      Type *RetTy = CI->getType();
      Value *Res =
          B.CreateSub(B.CreateZExt(Gt, RetTy), B.CreateZExt(Lt, RetTy));
      CI->replaceAllUsesWith(Res);
      CI->eraseFromParent();
    }
  }
  // Drop the now-unused intrinsic declarations so no symbol is referenced.
  for (auto It = M.begin(); It != M.end();) {
    Function &F = *It++;
    Intrinsic::ID ID = F.getIntrinsicID();
    if ((ID == Intrinsic::scmp || ID == Intrinsic::ucmp) && F.use_empty())
      F.eraseFromParent();
  }
}

// O3's vectorizer can produce `<N x ptr addrspace(AS)>` values when it
// vectorizes a `tl.where`/select over pointer operands (e.g. the int-pointer
// payload of a masked gather). A vector-of-pointers POINTER element type has
// only one pointee slot per address space in the AIR type table, but the
// scalar pointers feeding the insertelement chain can carry conflicting
// pointees (an `i8` byte-array GEP base vs a `float`/`i64` typed null), so the
// emitted TYPE_CODE_VECTOR element disagrees with the scalar operands and the
// AIR reader rejects the module ("Invalid record"). These pointer vectors
// only ever exist to be `ptrtoint`'d to an integer vector and stored, so lower
// the whole `<N x ptr>` web to `<N x i64>` (pointer-width int): convert each
// scalar pointer operand with a scalar ptrtoint, keep the vector ops
// (insert/extract/phi/select) vectorized in integer space, and turn the
// trailing `ptrtoint <N x ptr>` into a plain truncation/passthrough. No
// pointee typing is needed for an integer vector, so the conflict disappears
// without scalarizing the bulk vector operations.
static void lowerVectorPointerToInt(Module &M) {
  auto isPtrVec = [](Type *T) -> FixedVectorType * {
    auto *VT = dyn_cast<FixedVectorType>(T);
    if (VT && VT->getElementType()->isPointerTy())
      return VT;
    return nullptr;
  };
  Type *I64 = Type::getInt64Ty(M.getContext());
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    // Collect every instruction that produces a vector-of-pointers.
    SmallVector<Instruction *, 16> PtrVecDefs;
    for (auto &BB : F)
      for (auto &I : BB)
        if (isPtrVec(I.getType()))
          PtrVecDefs.push_back(&I);
    if (PtrVecDefs.empty())
      continue;

    // Map each pointer-vector value to its integer-vector replacement.
    DenseMap<Value *, Value *> IntOf;
    auto intVecTy = [&](FixedVectorType *PVT) {
      return FixedVectorType::get(I64, PVT->getNumElements());
    };
    // Materialize the integer-vector form of an arbitrary pointer-vector
    // operand (constants/poison/undef and not-yet-rewritten defs).
    std::function<Value *(Value *, IRBuilder<> &)> asIntVec =
        [&](Value *V, IRBuilder<> &B) -> Value * {
      if (auto *It = IntOf.lookup(V))
        return It;
      auto *PVT = cast<FixedVectorType>(V->getType());
      if (isa<UndefValue>(V))
        return UndefValue::get(intVecTy(PVT));
      if (isa<ConstantAggregateZero>(V) ||
          (isa<Constant>(V) && cast<Constant>(V)->isNullValue()))
        return ConstantAggregateZero::get(intVecTy(PVT));
      // Fallback for any other constant/value: ptrtoint the whole vector.
      return B.CreatePtrToInt(V, intVecTy(PVT));
    };

    // First create placeholder integer phis so cycles resolve.
    for (Instruction *I : PtrVecDefs)
      if (auto *PN = dyn_cast<PHINode>(I)) {
        IRBuilder<> B(PN);
        auto *NewPN =
            B.CreatePHI(intVecTy(cast<FixedVectorType>(PN->getType())),
                        PN->getNumIncomingValues());
        IntOf[PN] = NewPN;
      }

    // Rewrite the non-phi defs in program order.
    for (Instruction *I : PtrVecDefs) {
      if (isa<PHINode>(I))
        continue;
      IRBuilder<> B(I);
      Value *Repl = nullptr;
      if (auto *IE = dyn_cast<InsertElementInst>(I)) {
        Value *Vec = asIntVec(IE->getOperand(0), B);
        Value *Sc = B.CreatePtrToInt(IE->getOperand(1), I64);
        Repl = B.CreateInsertElement(Vec, Sc, IE->getOperand(2));
      } else if (auto *SV = dyn_cast<ShuffleVectorInst>(I)) {
        Value *A = asIntVec(SV->getOperand(0), B);
        Value *Bv = asIntVec(SV->getOperand(1), B);
        Repl = B.CreateShuffleVector(A, Bv, SV->getShuffleMask());
      } else if (auto *Sel = dyn_cast<SelectInst>(I)) {
        Value *T = asIntVec(Sel->getTrueValue(), B);
        Value *Fv = asIntVec(Sel->getFalseValue(), B);
        Repl = B.CreateSelect(Sel->getCondition(), T, Fv);
      } else if (auto *BC = dyn_cast<BitCastInst>(I)) {
        // ptr-vec bitcast (e.g. addrspace-preserving): forward the int form.
        Repl = asIntVec(BC->getOperand(0), B);
      } else {
        // Unhandled producer: ptrtoint then back so users still see a ptr-vec.
        continue;
      }
      IntOf[I] = Repl;
    }

    // Fill phi incomings now that all defs have int forms.
    for (Instruction *I : PtrVecDefs)
      if (auto *PN = dyn_cast<PHINode>(I)) {
        auto *NewPN = cast<PHINode>(IntOf[PN]);
        for (unsigned J = 0; J < PN->getNumIncomingValues(); ++J) {
          IRBuilder<> B(PN->getIncomingBlock(J)->getTerminator());
          NewPN->addIncoming(asIntVec(PN->getIncomingValue(J), B),
                             PN->getIncomingBlock(J));
        }
      }

    // Redirect users: consumers of the pointer-vector now read the int form.
    // ptrtoint <N x ptr>->ivec becomes the int form (with width fixups);
    // extractelement yields a scalar int turned back into a pointer;
    // everything else gets an inttoptr-rebuilt vector so it stays valid.
    for (Instruction *I : PtrVecDefs) {
      Value *Int = IntOf.lookup(I);
      if (!Int)
        continue;
      SmallVector<Use *, 8> Uses;
      for (Use &U : I->uses())
        Uses.push_back(&U);
      for (Use *U : Uses) {
        auto *User = cast<Instruction>(U->getUser());
        if (IntOf.count(User))
          continue; // already rewritten to consume the int form
        IRBuilder<> B(User);
        if (auto *P2I = dyn_cast<PtrToIntInst>(User)) {
          Value *V = Int;
          if (P2I->getType() != Int->getType())
            V = B.CreateZExtOrTrunc(Int, P2I->getType());
          P2I->replaceAllUsesWith(V);
          continue; // P2I now dead; cleaned up below
        }
        if (auto *EE = dyn_cast<ExtractElementInst>(User)) {
          Value *Sc = B.CreateExtractElement(Int, EE->getIndexOperand());
          EE->replaceAllUsesWith(B.CreateIntToPtr(Sc, EE->getType()));
          continue;
        }
        // Generic consumer still expecting a pointer vector: rebuild one.
        U->set(B.CreateIntToPtr(Int, I->getType()));
      }
    }

    // Erase the now-dead pointer-vector defs and orphaned ptrtoints.
    for (Instruction *I : reverse(PtrVecDefs)) {
      if (!IntOf.count(I))
        continue;
      if (!I->use_empty())
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
    }
    // Drop ptrtoint/extractelement consumers that were replaced.
    SmallVector<Instruction *, 8> Dead;
    for (auto &BB : F)
      for (auto &I : BB)
        if ((isa<PtrToIntInst>(I) || isa<ExtractElementInst>(I)) &&
            I.use_empty() && isPtrVec(I.getOperand(0)->getType()) &&
            IntOf.count(I.getOperand(0)))
          Dead.push_back(&I);
    for (Instruction *I : Dead)
      I->eraseFromParent();
    for (Instruction *I : reverse(PtrVecDefs))
      if (IntOf.count(I) && I->use_empty())
        I->eraseFromParent();
  }
}

// The mid-end emits >64-bit integer arithmetic for overflow-free closed
// forms (e.g. SCEV's `trunc((zext(a) * zext(b)) >> 1)` triangular sums as
// i65). The AGX JIT cannot legalize any iN > 64; expand such chains into
// (lo, hi) i64 limb pairs. Unsupported wide ops fail loud.
static void expandWideIntegers(Module &M) {
  auto isWide = [](Type *T) {
    return T->isIntegerTy() && T->getIntegerBitWidth() > 64 &&
           T->getIntegerBitWidth() <= 128;
  };
  Type *I64 = Type::getInt64Ty(M.getContext());
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    DenseMap<Value *, std::pair<Value *, Value *>> Limbs; // wide -> (lo, hi)
    SmallVector<Instruction *, 8> Wide;
    for (auto &BB : F)
      for (auto &I : BB)
        if (isWide(I.getType()) || llvm::any_of(I.operands(), [&](Value *Op) {
              return isWide(Op->getType());
            }))
          Wide.push_back(&I);
    if (Wide.empty())
      continue;
    auto umulh = [&](IRBuilder<> &B, Value *A, Value *Bv) -> Value * {
      Value *Mask = ConstantInt::get(I64, 0xffffffffull);
      Value *AL = B.CreateAnd(A, Mask), *AH = B.CreateLShr(A, 32);
      Value *BL = B.CreateAnd(Bv, Mask), *BH = B.CreateLShr(Bv, 32);
      Value *LL = B.CreateMul(AL, BL);
      Value *LH = B.CreateMul(AL, BH);
      Value *HL = B.CreateMul(AH, BL);
      Value *HH = B.CreateMul(AH, BH);
      Value *Mid =
          B.CreateAdd(B.CreateAdd(B.CreateLShr(LL, 32), B.CreateAnd(LH, Mask)),
                      B.CreateAnd(HL, Mask));
      return B.CreateAdd(B.CreateAdd(HH, B.CreateAdd(B.CreateLShr(LH, 32),
                                                     B.CreateLShr(HL, 32))),
                         B.CreateLShr(Mid, 32));
    };
    for (Instruction *I : Wide) {
      IRBuilder<> B(I);
      auto limbsOf = [&](Value *V) -> std::pair<Value *, Value *> {
        auto It = Limbs.find(V);
        if (It != Limbs.end())
          return It->second;
        if (auto *C = dyn_cast<ConstantInt>(V)) {
          APInt A = C->getValue();
          return {ConstantInt::get(I64, A.trunc(64)),
                  ConstantInt::get(I64, A.lshr(64).trunc(64))};
        }
        report_fatal_error("AIRWriter: unmapped wide integer operand");
      };
      if (auto *ZE = dyn_cast<ZExtInst>(I)) {
        Limbs[I] = {B.CreateZExtOrTrunc(ZE->getOperand(0), I64),
                    ConstantInt::get(I64, 0)};
      } else if (auto *SE2 = dyn_cast<SExtInst>(I)) {
        Value *Lo = B.CreateSExtOrTrunc(SE2->getOperand(0), I64);
        Limbs[I] = {Lo, B.CreateAShr(Lo, 63)};
      } else if (auto *BO = dyn_cast<BinaryOperator>(I)) {
        auto [L1, H1] = limbsOf(BO->getOperand(0));
        switch (BO->getOpcode()) {
        case Instruction::Mul: {
          auto [L2, H2] = limbsOf(BO->getOperand(1));
          Value *Lo = B.CreateMul(L1, L2);
          Value *Hi =
              B.CreateAdd(umulh(B, L1, L2), B.CreateAdd(B.CreateMul(L1, H2),
                                                        B.CreateMul(H1, L2)));
          Limbs[I] = {Lo, Hi};
          break;
        }
        case Instruction::Add: {
          auto [L2, H2] = limbsOf(BO->getOperand(1));
          Value *Lo = B.CreateAdd(L1, L2);
          Value *Carry = B.CreateZExt(B.CreateICmpULT(Lo, L1), I64);
          Limbs[I] = {Lo, B.CreateAdd(B.CreateAdd(H1, H2), Carry)};
          break;
        }
        case Instruction::Sub: {
          auto [L2, H2] = limbsOf(BO->getOperand(1));
          Value *Lo = B.CreateSub(L1, L2);
          Value *Borrow = B.CreateZExt(B.CreateICmpULT(L1, L2), I64);
          Limbs[I] = {Lo, B.CreateSub(B.CreateSub(H1, H2), Borrow)};
          break;
        }
        case Instruction::LShr: {
          auto *CI = dyn_cast<ConstantInt>(BO->getOperand(1));
          if (!CI)
            report_fatal_error("AIRWriter: wide lshr by non-constant");
          uint64_t Sh = CI->getZExtValue();
          if (Sh == 0) {
            Limbs[I] = {L1, H1};
          } else if (Sh < 64) {
            Limbs[I] = {
                B.CreateOr(B.CreateLShr(L1, Sh), B.CreateShl(H1, 64 - Sh)),
                B.CreateLShr(H1, Sh)};
          } else {
            Limbs[I] = {B.CreateLShr(H1, Sh - 64), ConstantInt::get(I64, 0)};
          }
          break;
        }
        default:
          report_fatal_error(Twine("AIRWriter: unhandled wide integer op '") +
                             BO->getOpcodeName() + "'");
        }
      } else if (auto *TR = dyn_cast<TruncInst>(I)) {
        auto [Lo, Hi] = limbsOf(TR->getOperand(0));
        (void)Hi;
        Value *R = B.CreateZExtOrTrunc(Lo, TR->getType());
        TR->replaceAllUsesWith(R);
      } else {
        report_fatal_error(Twine("AIRWriter: unhandled wide integer user '") +
                           I->getOpcodeName() + "'");
      }
    }
    for (auto It = Wide.rbegin(); It != Wide.rend(); ++It)
      (*It)->eraseFromParent();
  }
}

static bool isI1VecScalarBitcast(BitCastInst *BC) {
  auto *SV = dyn_cast<FixedVectorType>(BC->getSrcTy());
  auto *DV = dyn_cast<FixedVectorType>(BC->getDestTy());
  return (SV && SV->getElementType()->isIntegerTy(1) &&
          BC->getDestTy()->isIntegerTy(SV->getNumElements())) ||
         (DV && DV->getElementType()->isIntegerTy(1) &&
          BC->getSrcTy()->isIntegerTy(DV->getNumElements()));
}

static bool isI1VecToSubByteVecBitcast(BitCastInst *BC) {
  auto *SV = dyn_cast<FixedVectorType>(BC->getSrcTy());
  auto *DV = dyn_cast<FixedVectorType>(BC->getDestTy());
  if (!SV || !DV || !SV->getElementType()->isIntegerTy(1) ||
      !DV->getElementType()->isIntegerTy())
    return false;
  unsigned K = DV->getElementType()->getIntegerBitWidth();
  return SV->getNumElements() == DV->getNumElements() * K;
}

static void scalarizeBoolVectorCasts(Module &M) {
  auto Casts = collectInsts<BitCastInst>(M, [](BitCastInst *BC) {
    return isI1VecScalarBitcast(BC) || isI1VecToSubByteVecBitcast(BC);
  });
  for (auto *BC : Casts) {
    IRBuilder<> B(BC);
    Value *R;
    if (isI1VecToSubByteVecBitcast(BC)) {
      auto *DV = cast<FixedVectorType>(BC->getDestTy());
      Type *ElemTy = DV->getElementType();
      unsigned K = ElemTy->getIntegerBitWidth();
      R = UndefValue::get(DV);
      for (unsigned E = 0; E < DV->getNumElements(); ++E) {
        Value *Packed = ConstantInt::get(ElemTy, 0);
        for (unsigned K0 = 0; K0 < K; ++K0) {
          Value *Bit = B.CreateZExt(
              B.CreateExtractElement(BC->getOperand(0), B.getInt64(E * K + K0)),
              ElemTy);
          Packed = B.CreateOr(Packed, B.CreateShl(Bit, K0));
        }
        R = B.CreateInsertElement(R, Packed, B.getInt64(E));
      }
    } else if (auto *SV = dyn_cast<FixedVectorType>(BC->getSrcTy())) {
      R = ConstantInt::get(BC->getDestTy(), 0);
      for (unsigned L = 0; L < SV->getNumElements(); ++L) {
        Value *Bit = B.CreateZExt(
            B.CreateExtractElement(BC->getOperand(0), B.getInt64(L)),
            BC->getDestTy());
        R = B.CreateOr(R, B.CreateShl(Bit, L));
      }
    } else {
      auto *DV = cast<FixedVectorType>(BC->getDestTy());
      R = UndefValue::get(DV);
      for (unsigned L = 0; L < DV->getNumElements(); ++L) {
        Value *Bit =
            B.CreateTrunc(B.CreateLShr(BC->getOperand(0), L), B.getInt1Ty());
        R = B.CreateInsertElement(R, Bit, B.getInt64(L));
      }
    }
    BC->replaceAllUsesWith(R);
    BC->eraseFromParent();
  }
}

// `zext nneg` is definitionally equal to `sext`; emit the sext form. The
// AGX JIT widens zext-fed 64-bit multiplies into 65-bit operations it then
// fails to legalize (PSO abort "unable to legalize instruction ... s65"),
// while the sext form is its long-proven path.
static void canonicalizeNNegZExt(Module &M) {
  SmallVector<ZExtInst *, 16> Zexts;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *ZE = dyn_cast<ZExtInst>(&I))
          if (ZE->hasNonNeg())
            Zexts.push_back(ZE);
  for (auto *ZE : Zexts) {
    auto *SE = CastInst::Create(Instruction::SExt, ZE->getOperand(0),
                                ZE->getType(), "", ZE->getIterator());
    SE->takeName(ZE);
    ZE->replaceAllUsesWith(SE);
    ZE->eraseFromParent();
  }
}

// AIR v1 bitcode has no freeze opcode. Replacing freeze with its operand is
// a legal refinement (freeze only matters for poison/undef inputs, where any
// fixed value is a valid choice).
static void lowerFreezeInsts(Module &M) {
  SmallVector<FreezeInst *, 8> Frozen;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *FI = dyn_cast<FreezeInst>(&I))
          Frozen.push_back(FI);
  for (auto *FI : Frozen) {
    FI->replaceAllUsesWith(FI->getOperand(0));
    FI->eraseFromParent();
  }
}

// AIR v1 bitcode has no aggregate load: the reader cannot enumerate an
// `[N x T]`-typed LOAD record. The mid-end produces them when it widens a
// small fixed-count gather into one load. Expand into per-element
// GEP+load+insertvalue so every load is scalar/vector-typed.
static void scalarizeAggregateLoads(Module &M, PointeeTypeMap &PTM) {
  // Field-based (not lane-based): decompose along the aggregate's elements, so
  // every emitted load is scalar/vector-typed. Distinct from the vector
  // scalarizers above, which decompose along SIMD lanes.
  auto Aggs = collectInsts<LoadInst>(
      M, [](LoadInst *LI) { return isa<ArrayType>(LI->getType()); });
  for (LoadInst *LI : Aggs) {
    auto *AT = cast<ArrayType>(LI->getType());
    Type *ElemTy = AT->getElementType();
    Value *Ptr = LI->getPointerOperand();
    if (effectivePointee(Ptr, PTM) != ElemTy &&
        (isa<ConstantPointerNull>(Ptr) || isa<IntToPtrInst>(Ptr)))
      Ptr = retypePointerVia(Ptr, ElemTy, LI, PTM);
    Value *Agg = UndefValue::get(AT);
    IRBuilder<> B(LI);
    for (uint64_t E = 0; E < AT->getNumElements(); ++E) {
      Value *EP = B.CreateGEP(ElemTy, Ptr, B.getInt64(E));
      Value *EV = B.CreateLoad(ElemTy, EP);
      Agg = B.CreateInsertValue(Agg, EV, {unsigned(E)});
    }
    LI->replaceAllUsesWith(Agg);
    LI->eraseFromParent();
  }
}

static void scalarizeAggregateStores(Module &M, PointeeTypeMap &PTM) {
  auto Aggs = collectInsts<StoreInst>(M, [](StoreInst *SI) {
    return isa<ArrayType>(SI->getValueOperand()->getType());
  });
  for (StoreInst *SI : Aggs) {
    Value *Val = SI->getValueOperand();
    auto *AT = cast<ArrayType>(Val->getType());
    Type *ElemTy = AT->getElementType();
    Value *Ptr = SI->getPointerOperand();
    if (effectivePointee(Ptr, PTM) != ElemTy &&
        (isa<ConstantPointerNull>(Ptr) || isa<IntToPtrInst>(Ptr)))
      Ptr = retypePointerVia(Ptr, ElemTy, SI, PTM);
    IRBuilder<> B(SI);
    for (uint64_t E = 0; E < AT->getNumElements(); ++E) {
      Value *EP = B.CreateGEP(ElemTy, Ptr, B.getInt64(E));
      Value *EV = B.CreateExtractValue(Val, {unsigned(E)});
      B.CreateStore(EV, EP);
    }
    SI->eraseFromParent();
  }
}

static void fixSelectPointerArms(Module &M, PointeeTypeMap &PTM) {
  auto pointeeOf = [&](Value *V) -> Type * {
    if (isa<ConstantPointerNull>(V))
      return nullptr;
    return effectivePointee(V, PTM);
  };
  auto Selects = collectInsts<SelectInst>(M, [&](SelectInst *S) {
    if (!S->getType()->isPointerTy())
      return false;
    Value *T = S->getTrueValue(), *F = S->getFalseValue();
    if (isa<ConstantPointerNull>(T) || isa<ConstantPointerNull>(F))
      return true;
    Type *Use = PointeeTypeMap::inferFromUsage(S);
    return pointeeOf(T) != pointeeOf(F) ||
           (Use && (pointeeOf(T) != Use || pointeeOf(F) != Use));
  });
  for (auto *S : Selects) {
    Value *T = S->getTrueValue(), *F = S->getFalseValue();
    Type *Pointee = PointeeTypeMap::inferFromUsage(S);
    if (!Pointee)
      Pointee = pointeeOf(T);
    if (!Pointee)
      Pointee = pointeeOf(F);
    if (!Pointee)
      continue;
    if (pointeeOf(T) != Pointee || isa<ConstantPointerNull>(T))
      S->setOperand(1, retypePointerVia(T, Pointee, S, PTM));
    if (pointeeOf(F) != Pointee || isa<ConstantPointerNull>(F))
      S->setOperand(2, retypePointerVia(F, Pointee, S, PTM));
    PTM.set(S, Pointee);
  }
}

// Make every load/store's recorded pointer pointee equal its access type.
// The optimizer's folded byte-GEPs and reused buffers leave e.g. a float store
// through an i8-typed pointer; the reader rejects this with "Explicit
// load/store type does not match pointee type of pointer operand". Route such
// accesses through an identity bitcast pinned to the access type.
static void fixAccessTypeMismatch(Module &M, PointeeTypeMap &PTM) {
  // The access type a load/store demands of its pointer operand, or null if
  // this instruction is neither.
  auto accessTypeOf = [](Instruction *I) -> Type * {
    if (auto *LI = dyn_cast<LoadInst>(I))
      return LI->getType();
    if (auto *SI = dyn_cast<StoreInst>(I))
      return SI->getValueOperand()->getType();
    return nullptr;
  };
  auto pointerOf = [](Instruction *I) -> Value * {
    if (auto *LI = dyn_cast<LoadInst>(I))
      return LI->getPointerOperand();
    return cast<StoreInst>(I)->getPointerOperand();
  };
  auto Fix = collectInsts<Instruction>(M, [&](Instruction *I) {
    Type *AccessTy = accessTypeOf(I);
    if (!AccessTy)
      return false;
    Value *Ptr = pointerOf(I);
    if (isa<BitCastInst>(Ptr))
      return false;
    // Vector accesses always get the retype (legacy behavior). Scalar
    // accesses only when the pointer's pointee provably disagrees.
    if (!AccessTy->isVectorTy()) {
      // inttoptr-derived pointers get their typed pointer from the shared
      // per-type default, which other values' inference can claim first —
      // always retype them to the access type.
      if (!isa<IntToPtrInst>(Ptr) && !isa<ConstantPointerNull>(Ptr)) {
        Type *Pointee = effectivePointee(Ptr, PTM);
        if (!Pointee || Pointee == AccessTy)
          return false;
      }
    }
    return true;
  });
  for (Instruction *I : Fix) {
    if (auto *LI = dyn_cast<LoadInst>(I))
      LI->setOperand(
          0, retypePointerVia(LI->getPointerOperand(), LI->getType(), LI, PTM));
    else {
      auto *SI = cast<StoreInst>(I);
      SI->setOperand(1, retypePointerVia(SI->getPointerOperand(),
                                         SI->getValueOperand()->getType(), SI,
                                         PTM));
    }
  }
}

// Identity-bitcast a simdgroup-matrix intrinsic's pointer arg so its pointee
// matches the intrinsic's element suffix. A float-typed TG/device pointer
// passed straight into a p3f16/p1f16 load (or any suffix mismatch) emits an
// invalid typed record (PSO "Failed to materializeAll"); the identity bitcast
// plus a PTM entry retypes it, same retypePointerVia convention as
// normalizeGEPs.
static void fixMMAPointerSuffixMismatch(Module &M, PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  auto Calls = collectInsts<CallInst>(M, [](CallInst *CI) {
    return CI->getCalledFunction() &&
           CI->getCalledFunction()->getName().starts_with(
               "air.simdgroup_matrix_8x8_");
  });
  for (auto *CI : Calls) {
    StringRef Name = CI->getCalledFunction()->getName();
    Type *Elem = nullptr;
    if (Name.contains("f16") && !Name.contains("bf16"))
      Elem = Type::getHalfTy(Ctx);
    else if (Name.contains("bf16"))
      Elem = Type::getBFloatTy(Ctx);
    else if (Name.contains("f32"))
      Elem = Type::getFloatTy(Ctx);
    if (!Elem)
      continue;
    for (unsigned J = 0; J < CI->arg_size(); J++) {
      Value *Op = CI->getArgOperand(J);
      if (!Op->getType()->isPointerTy())
        continue;
      if (Elem->isFloatTy() && !isa<Constant>(Op)) {
        // Float-suffix operands are normally float-typed already; wrap
        // only when the pointee provably disagrees (the optimizer's
        // byte-GEP chains leave i8-typed pointers feeding p1f32/p3f32).
        Type *Pointee = effectivePointee(Op, PTM);
        if (!Pointee || Pointee == Elem)
          continue;
      }
      if (isa<BitCastInst>(Op) || isa<AllocaInst>(Op))
        continue;
      CI->setArgOperand(J, retypePointerVia(Op, Elem, CI, PTM));
    }
  }
}

static bool isAtomicDeviceArg(Argument *Arg) {
  SmallVector<Value *, 8> Work{Arg};
  SmallPtrSet<Value *, 8> Seen;
  while (!Work.empty()) {
    Value *V = Work.pop_back_val();
    if (!Seen.insert(V).second)
      continue;
    for (User *U : V->users()) {
      if (auto *CI = dyn_cast<CallInst>(U)) {
        Function *Callee = CI->getCalledFunction();
        if (Callee && Callee->getName().starts_with("air.atomic.global.") &&
            CI->arg_size() > 0 && CI->getArgOperand(0) == V)
          return true;
        continue;
      }
      if (isa<BitCastInst>(U) || isa<GetElementPtrInst>(U) ||
          isa<AddrSpaceCastInst>(U))
        Work.push_back(U);
    }
  }
  return false;
}

static void fixKernelArgMetadata(Module &M, const PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  auto *AirKernel = M.getNamedMetadata("air.kernel");
  if (!AirKernel)
    return;

  // Canonical metal::_atomic arg metadata pairs with air.version >= (2,9,0) /
  // MSL 4.1 (macOS 26), the level the Metal compiler accepts the 8-arg
  // device-atomic form.
  bool AllowAtomicStruct =
      AIRVersion::fromTriple(M.getTargetTriple().str()).OSMajor >= 16;

  for (unsigned K = 0; K < AirKernel->getNumOperands(); K++) {
    auto *KernelMD = AirKernel->getOperand(K);
    if (KernelMD->getNumOperands() < 3)
      continue;
    // KernelMD: {fn, attrs, argDescs}
    auto *ArgDescs = dyn_cast_or_null<MDNode>(KernelMD->getOperand(2));
    if (!ArgDescs)
      continue;

    auto *FnVAM = dyn_cast_or_null<ValueAsMetadata>(KernelMD->getOperand(0));
    if (!FnVAM)
      continue;
    auto *Fn = dyn_cast<Function>(FnVAM->getValue());
    if (!Fn)
      continue;

    for (unsigned A = 0; A < ArgDescs->getNumOperands(); A++) {
      auto *ArgMD = dyn_cast_or_null<MDNode>(ArgDescs->getOperand(A));
      if (!ArgMD || ArgMD->getNumOperands() < 2)
        continue;

      // Check if this is a buffer arg (has "air.buffer" string)
      bool IsBuffer = false;
      for (unsigned I = 1; I < ArgMD->getNumOperands(); I++)
        if (auto *S = dyn_cast_or_null<MDString>(ArgMD->getOperand(I)))
          if (S->getString() == "air.buffer") {
            IsBuffer = true;
            break;
          }
      if (!IsBuffer)
        continue;

      // Get the arg index from the first operand
      auto *IdxVAM = dyn_cast_or_null<ValueAsMetadata>(ArgMD->getOperand(0));
      if (!IdxVAM)
        continue;
      auto *IdxCI = dyn_cast<ConstantInt>(IdxVAM->getValue());
      if (!IdxCI)
        continue;
      unsigned ArgIdx = IdxCI->getZExtValue();
      if (ArgIdx >= Fn->arg_size())
        continue;

      // Infer pointee type from PTM, following through bitcasts
      Argument *Arg = Fn->getArg(ArgIdx);
      if (!Arg->getType()->isPointerTy())
        continue;
      Type *Pointee = nullptr;
      if (auto *Ty = PTM.get(Arg))
        Pointee = Ty;
      if (!Pointee)
        Pointee = PointeeTypeMap::inferFromUsage(Arg);
      // Follow through bitcasts if inference failed on the arg directly
      if (!Pointee || Pointee->isFloatTy()) {
        for (auto *U : Arg->users()) {
          if (auto *BC = dyn_cast<BitCastInst>(U)) {
            if (auto *Ty = PTM.get(BC)) {
              if (!Ty->isFloatTy()) {
                Pointee = Ty;
                break;
              }
            }
            Type *BcTy = PointeeTypeMap::inferFromUsage(BC);
            if (BcTy && !BcTy->isFloatTy()) {
              Pointee = BcTy;
              break;
            }
          }
        }
      }
      if (!Pointee)
        continue;

      // Determine correct type name, size, and alignment
      StringRef TypeName;
      unsigned TypeSize = 0, TypeAlign = 0;
      if (Pointee->isBFloatTy()) {
        TypeName = "bfloat";
        TypeSize = 2;
        TypeAlign = 2;
      } else if (Pointee->isFloatTy()) {
        TypeName = "float";
        TypeSize = 4;
        TypeAlign = 4;
      } else if (Pointee->isHalfTy()) {
        TypeName = "half";
        TypeSize = 2;
        TypeAlign = 2;
      } else if (Pointee->isIntegerTy(8)) {
        TypeName = "char";
        TypeSize = 1;
        TypeAlign = 1;
      } else if (Pointee->isIntegerTy(16)) {
        TypeName = "short";
        TypeSize = 2;
        TypeAlign = 2;
      } else if (Pointee->isIntegerTy(32)) {
        TypeName = "int";
        TypeSize = 4;
        TypeAlign = 4;
      } else {
        continue; // Unknown type, don't change
      }

      // Device-atomic arg: type as metal::_atomic { i32 } + air.struct_type_info,
      // matching Apple's front end. The 8-arg cmpxchg PSO-crashes without it.
      bool IsAtomic = AllowAtomicStruct && Pointee->isIntegerTy(32) &&
                      isAtomicDeviceArg(Arg);
      Metadata *StructTypeInfo = nullptr;
      if (IsAtomic) {
        TypeName = "metal::_atomic";
        auto *I32 = Type::getInt32Ty(Ctx);
        StructTypeInfo = MDNode::get(
            Ctx, {ValueAsMetadata::get(ConstantInt::get(I32, 0)),
                  ValueAsMetadata::get(ConstantInt::get(I32, 4)),
                  ValueAsMetadata::get(ConstantInt::get(I32, 0)),
                  MDString::get(Ctx, "int"), MDString::get(Ctx, "__s")});
      }

      // Rebuild the metadata node with corrected values
      SmallVector<Metadata *, 16> NewOps;
      for (unsigned I = 0; I < ArgMD->getNumOperands(); I++) {
        Metadata *Op = ArgMD->getOperand(I);
        if (I + 1 < ArgMD->getNumOperands()) {
          if (auto *PrevS = dyn_cast_or_null<MDString>(ArgMD->getOperand(I))) {
            if (PrevS->getString() == "air.arg_type_name" &&
                I + 1 < ArgMD->getNumOperands()) {
              NewOps.push_back(Op);
              NewOps.push_back(MDString::get(Ctx, TypeName));
              I++; // skip original type name
              continue;
            }
            if (PrevS->getString() == "air.struct_type_info" &&
                I + 1 < ArgMD->getNumOperands()) {
              I++;
              continue;
            }
            if (PrevS->getString() == "air.arg_type_size" &&
                I + 1 < ArgMD->getNumOperands()) {
              if (StructTypeInfo) {
                NewOps.push_back(MDString::get(Ctx, "air.struct_type_info"));
                NewOps.push_back(StructTypeInfo);
              }
              NewOps.push_back(Op);
              NewOps.push_back(ValueAsMetadata::get(
                  ConstantInt::get(Type::getInt32Ty(Ctx), TypeSize)));
              I++; // skip original size
              continue;
            }
            if (PrevS->getString() == "air.arg_type_align_size" &&
                I + 1 < ArgMD->getNumOperands()) {
              NewOps.push_back(Op);
              NewOps.push_back(ValueAsMetadata::get(
                  ConstantInt::get(Type::getInt32Ty(Ctx), TypeAlign)));
              I++; // skip original align
              continue;
            }
          }
        }
        NewOps.push_back(Op);
      }
      auto *NewArgMD = MDNode::get(Ctx, NewOps);
      ArgDescs->replaceOperandWith(A, NewArgMD);
    }
  }
}

// Map LLVM's in-memory AttrKind to the AIR-v1 bitcode attr-kind encoding.
// The two numbering spaces diverged long ago, and the mid-end optimizer
// attaches modern attributes (captures, noundef, memory, ...) that Apple's
// reader rejects as "Unknown attribute kind" — anything outside this
// whitelist is dropped (attributes are hints; dropping is always sound).
static std::optional<uint64_t> airEnumAttrKind(Attribute::AttrKind K) {
  switch (K) {
  case Attribute::NoAlias:
    return 9;
  case Attribute::NoUnwind:
    return 18;
  case Attribute::ReadNone:
    return 20;
  case Attribute::ReadOnly:
    return 21;
  case Attribute::NonNull:
    return 39;
  case Attribute::Convergent:
    return 43;
  case Attribute::WriteOnly:
    return 52;
  case Attribute::WillReturn:
    return 61;
  case Attribute::NoFree:
    return 62;
  case Attribute::NoSync:
    return 63;
  case Attribute::MustProgress:
    return 70;
  default:
    return std::nullopt;
  }
}

static std::optional<uint64_t> airIntAttrKind(Attribute::AttrKind K) {
  switch (K) {
  case Attribute::Alignment:
    return 1;
  case Attribute::Dereferenceable:
    return 41;
  case Attribute::DereferenceableOrNull:
    return 42;
  default:
    return std::nullopt;
  }
}

// Append the AIR-bitcode encoding of Attr to Grp (no-op for attributes
// outside the whitelist). Returns true if the attribute was encoded.
// captures(none) maps back to the legacy valueless nocapture kind.
static bool encodeAirAttr(const Attribute &Attr,
                          SmallVectorImpl<uint64_t> *Grp) {
  if (Attr.isEnumAttribute()) {
    if (auto BK = airEnumAttrKind(Attr.getKindAsEnum())) {
      if (Grp) {
        Grp->push_back(0);
        Grp->push_back(*BK);
      }
      return true;
    }
    return false;
  }
  if (Attr.isIntAttribute()) {
    if (Attr.getKindAsEnum() == Attribute::Captures) {
      if (!capturesNothing(Attr.getCaptureInfo()))
        return false;
      if (Grp) {
        Grp->push_back(0);
        Grp->push_back(11); // nocapture
      }
      return true;
    }
    if (auto BK = airIntAttrKind(Attr.getKindAsEnum())) {
      if (Grp) {
        Grp->push_back(1);
        Grp->push_back(*BK);
        Grp->push_back(Attr.getValueAsInt());
      }
      return true;
    }
    return false;
  }
  return false;
}

// True if at least one attribute in AS survives the AIR whitelist (string
// attributes always pass through).
static bool hasAirAttrs(const AttributeSet &AS) {
  for (Attribute Attr : AS) {
    if (Attr.isStringAttribute())
      return true;
    if (encodeAirAttr(Attr, nullptr))
      return true;
  }
  return false;
}

// Forward declarations (defined in separate .cpp files)
void emitTypeBlock(BitstreamWriter &W, ValueEnumerator &E);
void emitConstantsBlock(BitstreamWriter &W, ValueEnumerator &E,
                        ArrayRef<const Constant *> Constants,
                        unsigned CodeSize);
void emitMetadataKindBlock(BitstreamWriter &W);
void emitMetadataBlock(BitstreamWriter &W, Module &M, ValueEnumerator &E,
                       MetadataEnumerator &MD);
void emitOperandBundleTagsBlock(BitstreamWriter &W);
void emitSinglethreadBlock(BitstreamWriter &W);
void emitFunctionBlock(BitstreamWriter &W, const Function &F,
                       ValueEnumerator &E, const MetadataEnumerator &MD);

// Remove truly redundant ptr-to-ptr bitcasts where the PTM has the SAME
// pointee type on both sides. Bitcasts that serve as typed pointer
// transitions (where PTM records different types) must be kept.
// Only removes bitcasts where BOTH sides have the same PTM entry.
static void removeRedundantBitcasts(Module &M, PointeeTypeMap &PTM) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<BitCastInst *, 16> ToRemove;
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *BC = dyn_cast<BitCastInst>(&I);
        if (!BC || BC->getSrcTy() != BC->getDestTy())
          continue;
        // Only remove if BOTH sides have the same PTM-recorded type.
        // If either side has no PTM entry, keep the bitcast (it may serve
        // as a type transition that the serializer needs).
        Type *SrcPT = PTM.get(BC->getOperand(0));
        Type *DstPT = PTM.get(BC);
        if (!SrcPT || !DstPT)
          continue; // Unknown - keep
        if (SrcPT != DstPT)
          continue; // Different - keep
        ToRemove.push_back(BC);
      }
    }
    for (auto *BC : ToRemove) {
      PTM.remove(BC);
      BC->replaceAllUsesWith(BC->getOperand(0));
      BC->eraseFromParent();
    }
  }
}

// normalizeGEPs shape 4 (array-global element GEP). Runs after
// lowerConstantExprs (see call site) — keep it out of normalizeGEPs.
static void normalizeArrayGlobalGEPs(Module &M) {
  Type *I64Ty = Type::getInt64Ty(M.getContext());
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    // Loads/stores directly through an array global (no GEP, e.g. after a
    // single-incoming phi folds away) have element access type vs array
    // pointee; route them through an element-0 GEP.
    SmallVector<Instruction *, 4> DirectAccess;
    for (auto &BB : F)
      for (auto &I : BB) {
        Value *Ptr = nullptr;
        if (auto *LI = dyn_cast<LoadInst>(&I))
          Ptr = LI->getPointerOperand();
        else if (auto *SI = dyn_cast<StoreInst>(&I))
          Ptr = SI->getPointerOperand();
        if (!Ptr)
          continue;
        auto *GV = dyn_cast<GlobalVariable>(Ptr);
        if (!GV || !isa<ArrayType>(GV->getValueType()))
          continue;
        DirectAccess.push_back(&I);
      }
    for (Instruction *I : DirectAccess) {
      auto *GV = cast<GlobalVariable>(
          isa<LoadInst>(I) ? cast<LoadInst>(I)->getPointerOperand()
                           : cast<StoreInst>(I)->getPointerOperand());
      auto *Base = GetElementPtrInst::CreateInBounds(
          GV->getValueType(), GV,
          {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 0)});
      Base->insertBefore(I->getIterator());
      if (isa<LoadInst>(I))
        cast<LoadInst>(I)->setOperand(0, Base);
      else
        cast<StoreInst>(I)->setOperand(1, Base);
    }
    SmallVector<GetElementPtrInst *, 8> ToFix;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          if (GEP->getNumIndices() != 1)
            continue;
          auto *GV = dyn_cast<GlobalVariable>(GEP->getPointerOperand());
          if (!GV)
            continue;
          auto *AT = dyn_cast<ArrayType>(GV->getValueType());
          if (!AT)
            continue;
          if (GEP->getSourceElementType() == AT)
            continue;
          ToFix.push_back(GEP);
        }

    for (auto *GEP : ToFix) {
      auto *GV = cast<GlobalVariable>(GEP->getPointerOperand());
      auto *AT = cast<ArrayType>(GV->getValueType());
      Type *ElemTy = AT->getElementType();
      Type *SrcTy = GEP->getSourceElementType();
      Value *Idx = GEP->idx_begin()->get();

      Value *ElemIdx = nullptr;
      if (SrcTy == ElemTy) {
        ElemIdx = Idx;
      } else {
        uint64_t SrcSize = M.getDataLayout().getTypeAllocSize(SrcTy);
        uint64_t ElemSize = M.getDataLayout().getTypeAllocSize(ElemTy);
        auto *CI = dyn_cast<ConstantInt>(Idx);
        if (!CI || ElemSize == 0)
          continue;
        uint64_t ByteOff = CI->getZExtValue() * SrcSize;
        if (ByteOff % ElemSize != 0)
          continue;
        ElemIdx = ConstantInt::get(I64Ty, ByteOff / ElemSize);
      }

      auto *NewGEP = GetElementPtrInst::Create(
          AT, GV, {ConstantInt::get(I64Ty, 0), ElemIdx}, "",
          GEP->getIterator());
      NewGEP->setIsInBounds(GEP->isInBounds());
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
    }
  }
}

std::vector<uint8_t> emitAIRBitcode(Module &M, PointeeTypeMap &PTM) {
  SmallVector<char, 0> Buf;
  {
    BitstreamWriter W(Buf);

    // BC magic
    W.Emit('B', 8);
    W.Emit('C', 8);
    W.Emit(0xC0, 8);
    W.Emit(0xDE, 8);

    // IDENTIFICATION
    W.EnterSubblock(bitc::IDENTIFICATION_BLOCK_ID, 5);
    emitString(W, bitc::IDENTIFICATION_CODE_STRING, "AIRIR");
    {
      SmallVector<uint64_t, 1> V = {0};
      W.EmitRecord(bitc::IDENTIFICATION_CODE_EPOCH, V);
    }
    W.ExitBlock();

    // Pre-serialization IR fixups. These bring the O3-optimized module into
    // the subset the AIR v1 typed-pointer bitcode + AGX JIT accept, and
    // refine the PTM (pointer->pointee map) in place. Four stages, ordered:
    //
    //   A. Legalize / lower constructs the AGX JIT can't take (wide ints,
    //      freeze, nneg-zext, bool-vector casts, scmp/ucmp, vector-of-ptr,
    //      vector-cond selects). These run first because they rewrite the IR
    //      shape the later type fixups inspect.
    //   B. Normalize GEP source types (shapes 1-3; see normalizeGEPs).
    //   C. Make pointer pointees agree across phis / MMA calls / aggregate
    //      loads / loads+stores (the identity-bitcast type-agreement fixups).
    //   D. Materialize ConstantExprs, then the post-constexpr GEP shape 4 and
    //      kernel-arg metadata.
    //
    // --- Stage A: legalize / lower ---
    expandWideIntegers(M);
    lowerFreezeInsts(M);
    canonicalizeNNegZExt(M);
    scalarizeBoolVectorCasts(M);
    lowerCmpIntrinsics(M);
    lowerVectorPointerToInt(M);
    lowerVectorSelects(M);
    removeRedundantBitcasts(M, PTM);
    // --- Stage B: GEP source-type normalization ---
    // Shapes 1-3 (see normalizeGEPs). Runs after the Stage-A passes that
    // create new GEPs and before the Stage-C fixups that read GEP result types.
    normalizeGEPs(M, PTM);
    // --- Stage C: pointer-pointee type agreement ---
    fixPhiIncomingTypes(M, PTM);
    fixMMAPointerSuffixMismatch(M, PTM);
    fixSelectPointerArms(M, PTM);
    scalarizeAggregateLoads(M, PTM);
    scalarizeAggregateStores(M, PTM);
    fixAccessTypeMismatch(M, PTM);

    // --- Stage D: materialize constexprs, then post-constexpr fixups ---
    // Lower ConstantExpr operands to real instructions before enumeration.
    lowerConstantExprs(M);
    normalizeArrayGlobalGEPs(M);

    // Fix kernel argument metadata to match actual pointee types.
    fixKernelArgMetadata(M, PTM);

    ValueEnumerator E(M, PTM);

    // MODULE_BLOCK (CodeSize=4)
    W.EnterSubblock(bitc::MODULE_BLOCK_ID, 4);

    {
      SmallVector<uint64_t, 1> V = {1};
      W.EmitRecord(bitc::MODULE_CODE_VERSION, V);
    }

    // PARAMATTR blocks BEFORE TYPE_BLOCK (AIR requires this order).
    //
    // E2c — general path: walk every Function, collect its parameter
    // AttributeSets and emit one PARAMATTR_GRP per unique (paramIdx, AS) tuple
    // plus one PARAMATTR list per function. Group ID 1 is reserved for the
    // legacy MMA-load nocapture+readonly entry, which lives on the call site
    // (the CallInst's first paramattr operand), not the function declaration.
    struct GroupKey {
      unsigned ListIdx;
      AttributeSet AS;
      bool operator==(const GroupKey &O) const {
        return ListIdx == O.ListIdx && AS == O.AS;
      }
    };
    struct GroupKeyInfo {
      static GroupKey getEmptyKey() {
        return {~0u, DenseMapInfo<AttributeSet>::getEmptyKey()};
      }
      static GroupKey getTombstoneKey() {
        return {~0u - 1, DenseMapInfo<AttributeSet>::getTombstoneKey()};
      }
      static unsigned getHashValue(const GroupKey &K) {
        return hash_combine(K.ListIdx,
                            DenseMapInfo<AttributeSet>::getHashValue(K.AS));
      }
      static bool isEqual(const GroupKey &A, const GroupKey &B) {
        return A.ListIdx == B.ListIdx && A.AS == B.AS;
      }
    };

    DenseMap<GroupKey, unsigned, GroupKeyInfo> GroupID;
    SmallVector<GroupKey, 8> GroupOrder;
    auto getGroupID = [&](unsigned ListIdx, AttributeSet AS) -> unsigned {
      GroupKey K{ListIdx, AS};
      auto It = GroupID.find(K);
      if (It != GroupID.end())
        return It->second;
      unsigned ID = GroupID.size() + 1;
      GroupID[K] = ID;
      GroupOrder.push_back(K);
      return ID;
    };

    DenseMap<const Function *, unsigned> FnAttrListID;
    SmallVector<SmallVector<unsigned, 4>, 8> AttrLists;

    // Synthesize attribute groups on the simdgroup-matrix intrinsic declarations
    // to match Apple's `xcrun metal` AIR — the macOS 13/14/15 AIR driver
    // rejects the metallib otherwise. The groups must be encoded with bitcode
    // ATTR_KIND_* values, which do NOT match LLVM's in-memory Attribute::AttrKind
    // enum (e.g. convergent is 6 in-memory but 43 in bitcode).
    enum : uint64_t {
      BK_NO_CAPTURE = 11,
      BK_NO_UNWIND = 18,
      BK_READ_ONLY = 21,
      BK_CONVERGENT = 43,
      BK_WRITEONLY = 52,
      BK_WILLRETURN = 61,
      BK_NOFREE = 62,
      BK_MUSTPROGRESS = 70,
    };
    // A synthesized group: target index (~0u=function, N=param N) + bitcode
    // enum attr-kinds.
    struct SynthGroup {
      uint64_t Index; // ~0u for function attrs, param index (1-based) otherwise
      SmallVector<uint64_t, 6> EnumKinds;
    };
    // Pending synthesized groups, in emission order, with their assigned IDs.
    SmallVector<std::pair<unsigned, SynthGroup>, 8> SynthGroups;
    // Functions that should be marked local_unnamed_addr in their record.
    DenseSet<const Function *> LocalUnnamedFns;
    auto addSynthGroup = [&](SynthGroup G) -> unsigned {
      unsigned ID = GroupID.size() + 1 + SynthGroups.size();
      SynthGroups.push_back({ID, std::move(G)});
      return ID;
    };

    bool HasMMALoad = false;
    for (auto &F : M) {
      if (F.getName().starts_with("air.simdgroup_matrix_8x8_load")) {
        HasMMALoad = true;
        break;
      }
    }
    if (HasMMALoad) {
      // Reserve group ID 1 + list ID 1 for the MMA call-site paramattr (the
      // FunctionWriter unconditionally threads `paramattr=1` into those calls).
      GroupKey K{/*ListIdx=*/1, AttributeSet()};
      GroupID[K] = 1;
      GroupOrder.push_back(K);
    }

    for (auto &F : M) {
      if (F.isDeclaration())
        continue;
      AttributeList AL = F.getAttributes();
      SmallVector<unsigned, 4> GroupIDs;
      for (unsigned i = 0; i < F.arg_size(); i++) {
        AttributeSet AS = AL.getParamAttrs(i);
        if (!AS.hasAttributes() || !hasAirAttrs(AS))
          continue;
        GroupIDs.push_back(getGroupID(i + 1, AS));
      }
      AttributeSet RetAS = AL.getRetAttrs();
      if (RetAS.hasAttributes() && hasAirAttrs(RetAS))
        GroupIDs.push_back(getGroupID(0, RetAS));
      if (GroupIDs.empty())
        continue;
      unsigned ListID = AttrLists.size() + 1;
      AttrLists.push_back(std::move(GroupIDs));
      FnAttrListID[&F] = ListID;
    }

    // Now synthesize declaration attribute groups for the simdgroup intrinsics.
    for (auto &F : M) {
      if (!F.isDeclaration())
        continue;
      StringRef Name = F.getName();
      if (!Name.starts_with("air.simdgroup_matrix_8x8_"))
        continue;
      bool IsLoad = Name.contains("_load");
      bool IsStore = Name.contains("_store");

      // Function-attr group (matches Apple). Order matches Apple's textual AIR.
      SynthGroup FnG;
      FnG.Index = ~0u;
      FnG.EnumKinds.push_back(BK_CONVERGENT);
      FnG.EnumKinds.push_back(BK_MUSTPROGRESS);
      if (IsLoad)
        FnG.EnumKinds.push_back(BK_NOFREE);
      FnG.EnumKinds.push_back(BK_NO_UNWIND);
      if (IsLoad)
        FnG.EnumKinds.push_back(BK_READ_ONLY);
      FnG.EnumKinds.push_back(BK_WILLRETURN);
      if (IsStore)
        FnG.EnumKinds.push_back(BK_WRITEONLY);
      unsigned FnGID = addSynthGroup(std::move(FnG));

      SmallVector<unsigned, 4> GroupIDs;
      GroupIDs.push_back(FnGID);

      // Pointer-arg group: load ptr is arg 0, store ptr is arg 1
      // (`nocapture readonly` / `nocapture writeonly`).
      if (IsLoad || IsStore) {
        SynthGroup PG;
        PG.Index = IsStore ? 2u : 1u; // 1-based param index
        PG.EnumKinds.push_back(BK_NO_CAPTURE);
        PG.EnumKinds.push_back(IsStore ? BK_WRITEONLY : BK_READ_ONLY);
        GroupIDs.push_back(addSynthGroup(std::move(PG)));
      }

      unsigned ListID = AttrLists.size() + 1;
      AttrLists.push_back(std::move(GroupIDs));
      FnAttrListID[&F] = ListID;
      LocalUnnamedFns.insert(&F);
    }

    if (!GroupID.empty() || !SynthGroups.empty()) {
      W.EnterSubblock(bitc::PARAMATTR_GROUP_BLOCK_ID, 4);
      for (auto &K : GroupOrder) {
        unsigned ID = GroupID.lookup(K);
        SmallVector<uint64_t, 16> Grp;
        Grp.push_back(ID);
        Grp.push_back(K.ListIdx);
        if (HasMMALoad && ID == 1 && K.AS.getNumAttributes() == 0) {
          // Legacy MMA call-site group: param 1 nocapture + readonly.
          Grp.push_back(0);
          Grp.push_back(11);
          Grp.push_back(0);
          Grp.push_back(21);
        } else {
          for (Attribute Attr : K.AS) {
            if (Attr.isStringAttribute()) {
              StringRef Key = Attr.getKindAsString();
              StringRef Val = Attr.getValueAsString();
              // 3 = string-key-only, 4 = key + value (matches upstream
              // writeAttributeGroupTable encoding).
              Grp.push_back(Val.empty() ? 3 : 4);
              for (char C : Key)
                Grp.push_back((unsigned char)C);
              Grp.push_back(0);
              if (!Val.empty()) {
                for (char C : Val)
                  Grp.push_back((unsigned char)C);
                Grp.push_back(0);
              }
            } else {
              encodeAirAttr(Attr, &Grp);
            }
          }
        }
        W.EmitRecord(bitc::PARAMATTR_GRP_CODE_ENTRY, Grp);
      }
      // Synthesized declaration groups (simdgroup-matrix intrinsics). Each entry
      // is (ID, Index, [0, kind]...) — every attr here is a plain bitcode enum
      // (record-code 0), with the bitcode ATTR_KIND_* values set above.
      for (auto &IDG : SynthGroups) {
        SmallVector<uint64_t, 16> Grp;
        Grp.push_back(IDG.first);        // group ID
        Grp.push_back(IDG.second.Index); // ~0u = function, else param index
        for (uint64_t Kind : IDG.second.EnumKinds) {
          Grp.push_back(0); // 0 = well-known enum attribute (no value)
          Grp.push_back(Kind);
        }
        W.EmitRecord(bitc::PARAMATTR_GRP_CODE_ENTRY, Grp);
      }
      W.ExitBlock();

      W.EnterSubblock(bitc::PARAMATTR_BLOCK_ID, 4);
      if (HasMMALoad) {
        SmallVector<uint64_t, 2> List;
        List.push_back(1);
        W.EmitRecord(2, List);
      }
      for (auto &AL : AttrLists) {
        SmallVector<uint64_t, 8> List;
        for (unsigned ID : AL)
          List.push_back(ID);
        W.EmitRecord(2, List);
      }
      W.ExitBlock();
    }

    emitTypeBlock(W, E);

    // Emit target triple - AIR GPU JIT expects it for proper codegen.
    // Use module value if set, otherwise default AIR triple.
    {
      std::string T = M.getTargetTriple().str();
      if (T.empty() || T == "air")
        // Derive the canonical AIR triple from whatever OS info is present
        // (falls back to macOS 16 / 26-era), rather than hardcoding it.
        T = metal::AIRVersion::fromTriple(M.getTargetTriple().str())
                .tripleString();
      emitString(W, bitc::MODULE_CODE_TRIPLE, T);
    }
    // Emit data layout - AIR GPU JIT uses this for type size/alignment.
    {
      auto DLStr = M.getDataLayoutStr();
      if (!DLStr.empty()) {
        emitString(W, bitc::MODULE_CODE_DATALAYOUT, DLStr);
      } else {
        emitString(W, bitc::MODULE_CODE_DATALAYOUT,
                   "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
                   "-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32"
                   "-v48:64:64-v64:64:64-v96:128:128-v128:128:128"
                   "-v192:256:256-v256:256:256-v512:512:512"
                   "-v1024:1024:1024-n8:16:32");
      }
    }

    if (!M.getSourceFileName().empty())
      emitString(W, bitc::MODULE_CODE_SOURCE_FILENAME, M.getSourceFileName());

    // GLOBALVAR and FUNCTION records - emit in globalValues order
    // (globals first, then functions, matching value ID assignment)
    for (auto *V : E.globalValues) {
      if (auto *G = dyn_cast<GlobalVariable>(V)) {
        SmallVector<uint64_t, 14> Ops;
        Ops.push_back(E.globalPtrTypeIdx(G)); // ptr-to-valueType
        Ops.push_back(G->isConstant() ? 1 : 0);
        Ops.push_back(
            G->hasInitializer() ? E.moduleConstIdx(G->getInitializer()) + 1 : 0);
        Ops.push_back(encodeLinkage(G->getLinkage()));
        Ops.push_back(G->getAlign() ? Log2_32(G->getAlign()->value()) + 1 : 0);
        for (int J = 0; J < 3; J++)
          Ops.push_back(0);
        Ops.push_back(G->hasGlobalUnnamedAddr() ? 1 : 0);
        Ops.push_back(G->isExternallyInitialized() ? 1 : 0);
        Ops.push_back(0);
        Ops.push_back(0);
        Ops.push_back(G->getAddressSpace());
        Ops.push_back(0);
        W.EmitRecord(bitc::MODULE_CODE_GLOBALVAR, Ops);
      } else if (auto *Fn = dyn_cast<Function>(V)) {
        SmallVector<uint64_t, 17> Ops;
        Ops.push_back(E.typeIdx(Fn->getFunctionType()));
        Ops.push_back(Fn->getCallingConv());
        Ops.push_back(Fn->isDeclaration() ? 1 : 0);
        Ops.push_back(encodeLinkage(Fn->getLinkage()));
        // paramattr: attribute-list ID for this function (0 = none).
        // Per-function list IDs start at 2 when HasMMALoad reserves list 1
        // for the MMA call site, else at 1.
        unsigned ListID = 0;
        auto It = FnAttrListID.find(Fn);
        if (It != FnAttrListID.end())
          ListID = It->second + (HasMMALoad ? 1 : 0);
        Ops.push_back(ListID);
        Ops.push_back(0); // align
        // Function record fields 6..15: section, visibility, gc, unnamed_addr,
        // prologuedata, dllstorageclass, comdat, prefixdata, personalityfn, ...
        // Field 9 is unnamed_addr. Bitcode encoding (getEncodedUnnamedAddr):
        // None=0, Global=1, Local=2. Apple's simdgroup intrinsic decls are
        // `local_unnamed_addr` (=2); the macOS-14 driver expects this to match.
        Ops.push_back(0);                                      // 6: section
        Ops.push_back(0);                                      // 7: visibility
        Ops.push_back(0);                                      // 8: gc
        Ops.push_back(LocalUnnamedFns.contains(Fn) ? 2u : 0u); // 9: unnamed_addr
        for (int J = 10; J < 16; J++)
          Ops.push_back(0);
        Ops.push_back(Fn->getAddressSpace());
        W.EmitRecord(bitc::MODULE_CODE_FUNCTION, Ops);
      }
    }

    emitConstantsBlock(W, E, E.moduleConstants, 5);
    emitMetadataKindBlock(W);

    // Share one MetadataEnumerator between the module-level METADATA_BLOCK
    // (where the nodes are emitted) and per-function attachment blocks
    // (where they are referenced by ID).
    MetadataEnumerator MDEnum;
    MDEnum.collect(M, E);
    emitMetadataBlock(W, M, E, MDEnum);
    emitOperandBundleTagsBlock(W);
    emitSinglethreadBlock(W);

    for (auto *V : E.globalValues)
      if (auto *F = dyn_cast<Function>(V))
        if (!F->isDeclaration())
          emitFunctionBlock(W, *F, E, MDEnum);

    // VALUE_SYMTAB
    W.EnterSubblock(bitc::VALUE_SYMTAB_BLOCK_ID, 4);
    for (unsigned I = 0; I < E.globalValues.size(); I++) {
      if (!E.globalValues[I]->hasName())
        continue;
      SmallVector<uint64_t, 32> NV;
      NV.push_back(I);
      for (char C : E.globalValues[I]->getName())
        NV.push_back((uint64_t)(unsigned char)C);
      W.EmitRecord(bitc::VST_CODE_ENTRY, NV);
    }
    W.ExitBlock();

    W.ExitBlock(); // MODULE_BLOCK
  }

  return std::vector<uint8_t>(Buf.begin(), Buf.end());
}

} // namespace metal
} // namespace llvm
