//===- PointeeTypeMap.cpp - Typed-pointer reconstruction --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reconstructs per-value pointee types for opaque LLVM pointers so the AIR
// bitcode writer can emit typed POINTER records (code 8) that the AIR GPU
// JIT requires. The map is populated by walking memory ops and GEPs, then
// consulted by BitcodeEmitter / TypeTableWriter / FunctionWriter when they
// need a concrete pointee for a Value or function-parameter type.
//
//===----------------------------------------------------------------------===//

#include "PointeeTypeMap.h"
#include "AIRConstraints.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace llvm {
namespace metal {

AnalysisKey PointeeTypeAnalysis::Key;

static bool isIntegerDevicePointer(Value *Ptr);

static Type *atomicPointeeFromUsers(Value *Ptr) {
  for (auto *U : Ptr->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || !CI->getCalledFunction())
      continue;
    StringRef Name = CI->getCalledFunction()->getName();
    if (!Name.starts_with("air.atomic."))
      continue;
    if (CI->arg_size() == 0 || CI->getArgOperand(0) != Ptr)
      continue;
    if (Name.ends_with(".i32"))
      return Type::getInt32Ty(Ptr->getContext());
    if (Name.ends_with(".f32"))
      return Type::getFloatTy(Ptr->getContext());
  }
  return nullptr;
}

// ── Infer pointee type from usage ────────────────────────────────────────
//
// Recurses through load/store/GEP usage, then falls back to GEP source type
// and atomic intrinsic name inference.

Type *PointeeTypeMap::inferFromUsage(Value *Ptr) {
  SmallPtrSet<Value *, 8> Visited;
  return inferFromUsage(Ptr, Visited);
}

Type *PointeeTypeMap::inferFromUsage(Value *Ptr,
                                     SmallPtrSetImpl<Value *> &Visited) {
  if (!Visited.insert(Ptr).second)
    return nullptr;
  if (!Ptr->hasUseList())
    return nullptr;
  // Prioritize load/store types over GEP source types.
  // Recurse through GEP chains to find the ultimate store/load type.
  // NOTE: Do NOT follow atomic intrinsic calls through GEP chains.
  // When a float buffer pointer goes through a float GEP to a CAS call
  // (which operates on i32), the GEP source type (float) must win.
  // The atomic type mismatch is resolved separately by AIRPrepare's
  // ptrtoint+inttoptr insertion before atomic intrinsic calls.
  Type *GepType = nullptr;
  for (auto *U : Ptr->users()) {
    if (auto *LI = dyn_cast<LoadInst>(U))
      return LI->getType();
    if (auto *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getPointerOperand() == Ptr)
        return SI->getValueOperand()->getType();
    }
    // Follow identity pointer bitcasts on DEVICE (addrspace 1) pointers: a
    // device output buffer's store often sits past a no-op ptr→ptr bitcast the
    // mid-end leaves between a byte-form GEP and the typed store, so the base
    // would otherwise infer the [N x i8] GEP source type instead of the stored
    // element type (div7's i64 outputs). Restricted to addrspace(1):
    // threadgroup (addrspace 3) smem uses array-typed globals (`[N x float]`)
    // whose GEP chain must keep its array element type — following identity
    // smem bitcasts there makes the base infer a scalar pointee that disagrees
    // with the array GEP and the AIR writer rejects it ("Explicit gep type does
    // not match pointee type").
    if (auto *BC = dyn_cast<BitCastInst>(U)) {
      if (BC->getType()->isPointerTy() &&
          BC->getType()->getPointerAddressSpace() == AS::Device)
        if (Type *T = inferFromUsage(BC, Visited)) {
          auto *GEPPtr = dyn_cast<GetElementPtrInst>(Ptr);
          auto *VT = dyn_cast<VectorType>(T);
          if (GEPPtr && VT &&
              VT->getElementType() == GEPPtr->getResultElementType())
            return GEPPtr->getResultElementType();
          return T;
        }
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (Type *T = inferFromUsage(GEP, Visited)) {
        if (auto *VT = dyn_cast<VectorType>(T))
          if (VT->getElementType() == GEP->getSourceElementType())
            return GEP->getSourceElementType();
        return T;
      }
      if (!GepType)
        GepType = GEP->getSourceElementType();
    }
    // Follow pointer-typed select users so both arms get a typed-pointer slot
    // (the writer cannot emit a typed `select ptr` otherwise).
    if (auto *Sel = dyn_cast<SelectInst>(U)) {
      if (Sel->getType()->isPointerTy())
        if (Type *T = inferFromUsage(Sel, Visited))
          return T;
    }
    // Follow pointer-typed phi users (loop-carried operand pointers): the
    // loads often happen only through the phi, and the phi's own pointee is
    // force-set elsewhere — without this the base infers the byte-GEP type
    // and the phi record's operand types disagree. Visited guards cycles.
    if (auto *PN = dyn_cast<PHINode>(U)) {
      if (PN->getType()->isPointerTy())
        if (Type *T = inferFromUsage(PN, Visited))
          return T;
    }
    if (auto *CI = dyn_cast<CallInst>(U)) {
      if (auto *Callee = CI->getCalledFunction()) {
        StringRef Name = Callee->getName();
        // The simdgroup-matrix intrinsic's pointer suffix is definitive
        // element-type evidence — byte-form GEP source types are mere
        // addressing artifacts and must not outvote it (bf16 before f16:
        // substring overlap).
        if (Name.starts_with("air.simdgroup_matrix_8x8_")) {
          auto &Ctx = Ptr->getContext();
          if (Name.contains("p1bf16") || Name.contains("p3bf16"))
            return Type::getBFloatTy(Ctx);
          if (Name.contains("p1f16") || Name.contains("p3f16"))
            return Type::getHalfTy(Ctx);
          if (Name.contains("p1i8") || Name.contains("p3i8"))
            return Type::getInt8Ty(Ctx);
          if (Name.contains("p1f32") || Name.contains("p3f32"))
            return Type::getFloatTy(Ctx);
        }
        if (Name.starts_with("air.init_strided_private_tensor") ||
            Name.starts_with("air.slice_private_tensor") ||
            Name.starts_with("air.get_extent_private_tensor")) {
          unsigned ArgNo = ~0u;
          for (unsigned J = 0; J < CI->arg_size(); ++J)
            if (CI->getArgOperand(J) == Ptr) {
              ArgNo = J;
              break;
            }
          bool IsHandle =
              (ArgNo == 0) ||
              (ArgNo == 1 && Name.starts_with("air.slice_private_tensor"));
          if (IsHandle) {
            auto &Ctx = Ptr->getContext();
            StructType *TT = StructType::getTypeByName(Ctx, "struct._tensor_t");
            if (!TT)
              TT = StructType::create(Ctx, "struct._tensor_t");
            return TT;
          }
          return Type::getInt8Ty(Ptr->getContext());
        }
        if (Name.starts_with("air.get_descriptor_size_tensor"))
          return Type::getInt8Ty(Ptr->getContext());
        if (Name.starts_with("__tensorops_impl_matmul2d")) {
          if (CI->arg_size() && CI->getArgOperand(0) == Ptr) {
            if (auto *AI = dyn_cast<AllocaInst>(Ptr->stripPointerCasts()))
              return AI->getAllocatedType();
          }
          return Type::getInt8Ty(Ptr->getContext());
        }
        // Only use atomic type if the pointer is NOT a GEP result.
        // GEP results must keep their source element type for consistency;
        // the atomic type mismatch is handled by inserting ptrtoint+inttoptr.
        if (!isa<GetElementPtrInst>(Ptr) && Name.starts_with("air.atomic.")) {
          if (Name.ends_with(".i32"))
            return Type::getInt32Ty(Ptr->getContext());
          else if (Name.ends_with(".f32"))
            return Type::getFloatTy(Ptr->getContext());
        }
      }
    }
  }
  return GepType;
}

// ── Collapse device pointers to float* ───────────────────────────────────
//
// When MMA intrinsics (simdgroup_multiply_accumulate) are present, the AIR
// GPU JIT crashes on ANY non-float device pointer. This collapses all
// addrspace(1) entries to float*.

void PointeeTypeMap::collapseDevicePointersToFloat(Module &M) {
  Type *F32 = Type::getFloatTy(M.getContext());
  for (auto &[Ptr, Ty] : map) {
    // Check if this is a device pointer (addrspace 1)
    auto *PtrTy = Ptr->getType();
    if (auto *PT = dyn_cast<PointerType>(PtrTy)) {
      if (PT->getAddressSpace() == AS::Device && !isIntegerDevicePointer(Ptr))
        Ty = F32;
    }
  }
}

// ── Remap i1 → i8 ───────────────────────────────────────────────────────
//
// AIR has no i1 memory type. Pointers to i1 crash the GPU JIT.
// Remap to i8 (booleans are i8 in AIR memory).

void PointeeTypeMap::remapI1ToI8(Module &M) {
  Type *I8 = Type::getInt8Ty(M.getContext());
  for (auto &[Ptr, Ty] : map) {
    if (Ty && Ty->isIntegerTy(1))
      Ty = I8;
  }
}

// ── Initial analysis: scan all pointers and infer types ──────────────────
//
// This analysis MUST be self-contained - it may be re-run after pipeline
// passes invalidate it. All AIR-specific overrides (MMA, async copy)
// must be here, not only in InferTypedPointersPass.

// MMA intrinsic names are shared with InferTypedPointersPass via
// PointeeTypeMap.h (namespace mma_intrinsics). See the "Two-stage design"
// note in that header for why the override logic is duplicated.
static constexpr const char *kMMALoad = mma_intrinsics::kLoad;
static constexpr const char *kMMAStore = mma_intrinsics::kStore;
static constexpr const char *kMMALoadDev = mma_intrinsics::kLoadDev;
static constexpr const char *kMMAStoreDev = mma_intrinsics::kStoreDev;

static StructType *argBufferStructType(const Argument *Arg) {
  const Function *F = Arg->getParent();
  if (Arg->getArgNo() != 0 || !F->getMetadata("air.argbuf.member_as"))
    return nullptr;
  for (const BasicBlock &BB : *F)
    for (const Instruction &I : BB)
      if (const auto *GEP = dyn_cast<GetElementPtrInst>(&I))
        if (GEP->getPointerOperand() == Arg)
          return dyn_cast<StructType>(GEP->getSourceElementType());
  return nullptr;
}

static bool functionUsesMMA(const Function &F) {
  for (const auto &BB : F)
    for (const auto &I : BB)
      if (const auto *CI = dyn_cast<CallInst>(&I))
        if (const auto *Callee = CI->getCalledFunction())
          if (Callee->getName().starts_with(mma_intrinsics::kPrefix))
            return true;
  return false;
}

static bool isNonFloatScalarDevicePointer(Value *Ptr) {
  Type *Ty = PointeeTypeMap::inferFromUsage(Ptr);
  if (!Ty)
    return false;
  if (Ty->isIntegerTy() && !Ty->isIntegerTy(1))
    return true;
  if (Ty->isHalfTy() || Ty->isBFloatTy())
    return true;
  return false;
}

static bool isIntegerDevicePointer(Value *Ptr) {
  return isNonFloatScalarDevicePointer(Ptr);
}

PointeeTypeMap buildPointeeTypeMap(Module &M) {
  PointeeTypeMap PTM;
  Type *F32 = Type::getFloatTy(M.getContext());
  Type *I8 = Type::getInt8Ty(M.getContext());

  // Detect MMA and async copy presence
  bool HasMMA = false;
  bool HasAsyncCopy = false;
  DenseMap<const Function *, bool> FunctionHasMMA;
  for (auto &F : M) {
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_"))
      HasMMA = true;
    if (F.getName().starts_with("air.simdgroup_async_copy_2d."))
      HasAsyncCopy = true;
    if (!F.isDeclaration())
      FunctionHasMMA[&F] = functionUsesMMA(F);
  }

  // Phase 1: Function parameters - infer from usage
  for (auto &F : M)
    for (auto &Arg : F.args())
      if (Arg.getType()->isPointerTy()) {
        if (auto *ST = argBufferStructType(&Arg)) {
          PTM.set(&Arg, ST);
          continue;
        }
        if (auto *Ty = PointeeTypeMap::inferFromUsage(&Arg))
          PTM.set(&Arg, Ty);
      }

  // Phase 2: Instructions that produce pointers
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        if (!I.getType()->isPointerTy())
          continue;
        if (auto *Ty = PointeeTypeMap::inferFromUsage(&I))
          PTM.set(&I, Ty);
      }

  for (auto &F : M)
    if (F.getMetadata("air.argbuf.member_as"))
      for (auto &BB : F)
        for (auto &I : BB)
          if (I.getType()->isPointerTy() &&
              I.getType()->getPointerAddressSpace() == AS::Device &&
              !isIntegerDevicePointer(&I))
            PTM.set(&I, F32);

  // Phase 2b: Force float* for device pointer phi nodes
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN || !PN->getType()->isPointerTy())
          continue;
        if (PN->getType()->getPointerAddressSpace() != AS::Device)
          continue;
        if (Type *AtomicTy = atomicPointeeFromUsers(PN))
          PTM.set(PN, AtomicTy);
        else
          PTM.set(PN, F32);
      }

  // Phase 3: Global variables
  for (auto &GV : M.globals())
    if (GV.getType()->isPointerTy())
      PTM.set(&GV, GV.getValueType());

  // Phase 4: Fill gaps - phi incoming, inttoptr, GEP
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        if (!I.getType()->isPointerTy() || PTM.has(&I))
          continue;
        if (auto *PHI = dyn_cast<PHINode>(&I))
          for (unsigned J = 0; J < PHI->getNumIncomingValues(); ++J)
            if (auto *Ty = PTM.get(PHI->getIncomingValue(J))) {
              PTM.set(&I, Ty);
              break;
            }
        if (isa<IntToPtrInst>(&I))
          if (auto *Ty = PointeeTypeMap::inferFromUsage(&I))
            PTM.set(&I, Ty);
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          PTM.set(&I, GEP->getResultElementType());
        if (auto *AI = dyn_cast<AllocaInst>(&I))
          PTM.set(&I, AI->getAllocatedType());
      }

  // Phase 5: i1* → i8*
  PTM.remapI1ToI8(M);

  // Phase 6: MMA - collapse device pointers to float* only in functions that
  // actually use MMA intrinsics.
  if (HasMMA) {
    for (auto &F : M) {
      if (F.isDeclaration() || !FunctionHasMMA.lookup(&F))
        continue;

      for (auto &Arg : F.args())
        if (Arg.getType()->isPointerTy() &&
            Arg.getType()->getPointerAddressSpace() == AS::Device &&
            !argBufferStructType(&Arg) && !isIntegerDevicePointer(&Arg))
          PTM.set(&Arg, F32);

      for (auto &BB : F)
        for (auto &I : BB)
          if (I.getType()->isPointerTy() &&
              I.getType()->getPointerAddressSpace() == AS::Device &&
              !isIntegerDevicePointer(&I))
            PTM.set(&I, F32);

      for (auto &Arg : F.args())
        if (Arg.getType()->isPointerTy() &&
            Arg.getType()->getPointerAddressSpace() == AS::Device &&
            !PTM.has(&Arg) && !argBufferStructType(&Arg) &&
            !isIntegerDevicePointer(&Arg))
          PTM.set(&Arg, F32);
    }

    // MMA declaration params → typed pointer (float*/half*/bfloat*)
    {
      Type *F16 = Type::getHalfTy(M.getContext());
      Type *BF16 = Type::getBFloatTy(M.getContext());
      for (auto &F : M) {
        if (!F.isDeclaration())
          continue;
        StringRef Name = F.getName();
        Type *PtrPointee = nullptr;
        if (Name == kMMALoad || Name == kMMAStore || Name == kMMALoadDev ||
            Name == kMMAStoreDev)
          PtrPointee = F32;
        else if (Name.contains("p1f16") &&
                 Name.starts_with("air.simdgroup_matrix_8x8_"))
          PtrPointee = F16;
        else if ((Name.contains("p1bf16") || Name.contains("p3bf16")) &&
                 Name.starts_with("air.simdgroup_matrix_8x8_"))
          PtrPointee = BF16;
        else if ((Name.contains("p1f16") || Name.contains("p3f16")) &&
                 Name.starts_with("air.simdgroup_matrix_8x8_"))
          PtrPointee = F16;
        if (PtrPointee)
          for (auto &Arg : F.args())
            if (Arg.getType()->isPointerTy())
              PTM.set(&Arg, PtrPointee);
      }
    }

    // MMA kernel device pointer args → float*
    for (auto &F : M)
      if (!F.isDeclaration() && FunctionHasMMA.lookup(&F))
        for (auto &Arg : F.args())
          if (Arg.getType()->isPointerTy() &&
              Arg.getType()->getPointerAddressSpace() == AS::Device &&
              !argBufferStructType(&Arg) && !isIntegerDevicePointer(&Arg))
            PTM.set(&Arg, F32);

    // MMA call site pointer operands → typed pointer
    for (auto &F : M)
      for (auto &BB : F)
        for (auto &I : BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI || !CI->getCalledFunction())
            continue;
          StringRef Name = CI->getCalledFunction()->getName();
          if (!Name.starts_with("air.simdgroup_matrix_8x8_"))
            continue;
          Type *PtrPointee = nullptr;
          if (Name == kMMALoad || Name == kMMAStore || Name == kMMALoadDev ||
              Name == kMMAStoreDev)
            PtrPointee = F32;
          else if (Name.contains("p1f16") || Name.contains("p3f16"))
            PtrPointee = Type::getHalfTy(M.getContext());
          else if (Name.contains("p1bf16") || Name.contains("p3bf16"))
            PtrPointee = Type::getBFloatTy(M.getContext());
          else if (Name.contains("p3f32"))
            PtrPointee = F32;
          if (PtrPointee)
            for (unsigned J = 0; J < CI->arg_size(); J++)
              if (CI->getArgOperand(J)->getType()->isPointerTy())
                PTM.set(CI->getArgOperand(J), PtrPointee);
        }
  }

  // Phase 7: Async copy overrides (AFTER MMA collapse, re-applies i8*)
  if (HasAsyncCopy) {
    StructType *EventTy = StructType::getTypeByName(M.getContext(), "event_t");
    if (!EventTy)
      EventTy = StructType::create(M.getContext(), "event_t");

    for (auto &F : M) {
      if (!F.isDeclaration())
        continue;
      StringRef Name = F.getName();

      if (Name.starts_with("air.simdgroup_async_copy_2d.")) {
        // Return type → event_t (set on call results)
        for (auto &FN : M)
          for (auto &BB : FN)
            for (auto &I : BB)
              if (auto *CI = dyn_cast<CallInst>(&I))
                if (CI->getCalledFunction() == &F)
                  PTM.set(CI, EventTy);
        // Pointer params → i8* (byte copy)
        for (unsigned J = 0; J < F.arg_size(); J++) {
          auto &Arg = *F.getArg(J);
          if (!Arg.getType()->isPointerTy())
            continue;
          PTM.set(&Arg, I8);
          for (auto *U : F.users())
            if (auto *CI = dyn_cast<CallInst>(U))
              if (J < CI->arg_size())
                PTM.set(CI->getArgOperand(J), I8);
        }
      }

      if (Name == "air.wait_simdgroup_events") {
        // Param 1: pointer to event_t pointer storage
        Type *PtrAs3 = PointerType::get(M.getContext(), 3);
        if (F.arg_size() >= 2) {
          auto &Arg = *F.getArg(1);
          if (Arg.getType()->isPointerTy()) {
            PTM.set(&Arg, PtrAs3);
            for (auto *U : F.users())
              if (auto *CI = dyn_cast<CallInst>(U))
                if (CI->arg_size() >= 2)
                  PTM.set(CI->getArgOperand(1), PtrAs3);
          }
        }
      }
    }

    // Event alloca: stores event_t pointers
    Type *PtrAs3 = PointerType::get(M.getContext(), 3);
    for (auto &F : M)
      for (auto &BB : F)
        for (auto &I : BB)
          if (auto *AI = dyn_cast<AllocaInst>(&I))
            if (AI->getAllocatedType()->isPointerTy() &&
                AI->getAllocatedType()->getPointerAddressSpace() == 3)
              PTM.set(AI, PtrAs3);

    // Propagate event_t through identity bitcasts
    for (auto &F : M)
      for (auto &BB : F)
        for (auto &I : BB) {
          auto *BC = dyn_cast<BitCastInst>(&I);
          if (!BC || BC->getSrcTy() != BC->getDestTy())
            continue;
          if (!BC->getType()->isPointerTy())
            continue;
          if (auto *SrcTy = PTM.get(BC->getOperand(0)))
            if (SrcTy == EventTy || isa<PointerType>(SrcTy))
              PTM.set(BC, SrcTy);
        }
  }

  // Phase 8: Fix up ptr-to-ptr bitcasts for typed pointer transitions.
  // Some upstream passes leave identity bitcasts (ptr→ptr) before non-float
  // device loads from phi pointers; MMA collapse clobbers their PTM to float*.
  // Re-infer from load/store usage.
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *BC = dyn_cast<BitCastInst>(&I);
        if (!BC || !BC->getType()->isPointerTy())
          continue;
        if (BC->getSrcTy() != BC->getDestTy())
          continue;
        if (BC->getType()->getPointerAddressSpace() != AS::Device)
          continue;
        for (auto *U : BC->users()) {
          if (auto *LI = dyn_cast<LoadInst>(U)) {
            if (!LI->getType()->isFloatTy()) {
              PTM.set(BC, LI->getType());
              break;
            }
          }
          if (auto *SI = dyn_cast<StoreInst>(U)) {
            if (SI->getPointerOperand() == BC &&
                !SI->getValueOperand()->getType()->isFloatTy()) {
              PTM.set(BC, SI->getValueOperand()->getType());
              break;
            }
          }
        }
      }

  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *Sel = dyn_cast<SelectInst>(&I);
        if (!Sel || !Sel->getType()->isPointerTy())
          continue;
        Value *TV = Sel->getTrueValue();
        Value *FV = Sel->getFalseValue();
        Type *TT = PTM.get(TV);
        Type *FT = PTM.get(FV);
        if (TT == FT)
          continue;
        Type *Unified = nullptr;
        if (TT && !isa<IntToPtrInst>(TV))
          Unified = TT;
        else if (FT && !isa<IntToPtrInst>(FV))
          Unified = FT;
        else
          Unified = TT ? TT : FT;
        if (!Unified)
          continue;
        PTM.set(TV, Unified);
        PTM.set(FV, Unified);
        PTM.set(Sel, Unified);
      }

  // Phase 9: Function pointer
  for (auto &F : M)
    if (!F.isDeclaration()) {
      PTM.set(&F, F.getFunctionType());
      break;
    }

  return PTM;
}

PointeeTypeMap PointeeTypeAnalysis::run(Module &M, ModuleAnalysisManager &) {
  return buildPointeeTypeMap(M);
}

} // namespace metal
} // namespace llvm
