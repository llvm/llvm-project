//===-- AGXPromoteKernelArgs.cpp - Kernel arguments in uniform registers --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kernel arguments passed by reference (`ptr addrspace(4) byref(T)`, as
// front ends like ZLUDA's emit PTX .param arguments) would cost a dependent
// device load before their first use. When every use of such an argument is
// a load at a constant offset, this pass passes its bytes instead, in
// consecutive uniform registers, and rebuilds each load from them.
// Arguments with other uses keep their pointer.
//
// The kernel's signature changes, so the pass records, for each original
// argument, how it is passed in the function attribute "agx-kernel-args":
// a comma-separated list of `<kind><bytes>:<first argument>`, where kind is
// `v` (the value) or `p` (a pointer to it). The asm printer turns this into
// the argument table of the kernel's .agx.kernels record.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "agx-promote-kernel-args"

/// The uniform file: 256 32-bit registers, all of which the hardware preloads
/// with no measurable launch cost (ptx-re-blockers B9: 20 to 512 halves).
static constexpr uint64_t UniformBytes = 1024;
/// What the uniforms hold besides the arguments: threadgroups_per_grid (12
/// bytes) and the global data buffer address (8).
static constexpr uint64_t ImplicitBytes = 12 + 8;

namespace {

class AGXPromoteKernelArgs : public ModulePass {
public:
  static char ID;
  AGXPromoteKernelArgs() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "AGX Promote Kernel Arguments";
  }
  bool runOnModule(Module &M) override;
};

struct LoadAt {
  LoadInst *Load;
  uint64_t Offset;
};

} // namespace

char AGXPromoteKernelArgs::ID = 0;

INITIALIZE_PASS(AGXPromoteKernelArgs, DEBUG_TYPE,
                "AGX Promote Kernel Arguments", false, false)

ModulePass *llvm::createAGXPromoteKernelArgsPass() {
  return new AGXPromoteKernelArgs();
}

/// The loads of A at constant offsets within [0, Bytes), or false if A has
/// another use.
static bool collectLoads(Value *Ptr, uint64_t Base, uint64_t Bytes,
                         const DataLayout &DL,
                         SmallVectorImpl<LoadAt> &Loads) {
  for (User *U : Ptr->users()) {
    if (auto *L = dyn_cast<LoadInst>(U)) {
      uint64_t Size = DL.getTypeStoreSize(L->getType());
      if (!L->isSimple() || L->getPointerOperand() != Ptr ||
          Base + Size > Bytes || L->getType()->isAggregateType())
        return false;
      Loads.push_back({L, Base});
      continue;
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      APInt Off(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
      if (GEP->getPointerOperand() != Ptr ||
          !GEP->accumulateConstantOffset(DL, Off) || Off.isNegative())
        return false;
      if (!collectLoads(GEP, Base + Off.getZExtValue(), Bytes, DL, Loads))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

/// A piece of a promoted argument: an i64 argument for each 8 bytes and an i32
/// one for a last 4. Both are aligned to 32 bits in the uniforms, so the
/// argument's bytes stay contiguous and in order.
struct Chunk {
  Value *V;
  uint64_t Offset;
  uint64_t Size;
};

static SmallVector<uint64_t, 4> chunkSizes(uint64_t Bytes) {
  SmallVector<uint64_t, 4> Sizes;
  for (uint64_t Left = alignTo(Bytes, 4); Left; Left -= Sizes.back())
    Sizes.push_back(Left >= 8 ? 8 : 4);
  return Sizes;
}

/// Rebuilds the value of Ty at byte Offset from the argument's chunks.
static Value *loadFromChunks(IRBuilder<> &B, ArrayRef<Chunk> Chunks,
                             uint64_t Offset, Type *Ty, const DataLayout &DL) {
  uint64_t Size = DL.getTypeStoreSize(Ty);
  SmallVector<const Chunk *, 4> Covering;
  for (const Chunk &C : Chunks)
    if (C.Offset < Offset + Size && Offset < C.Offset + C.Size)
      Covering.push_back(&C);
  uint64_t Base = Covering.front()->Offset;
  uint64_t End = Covering.back()->Offset + Covering.back()->Size;
  Value *V = Covering.front()->V;
  if (Covering.size() > 1) {
    IntegerType *WideTy = B.getIntNTy((End - Base) * 8);
    V = B.CreateZExt(V, WideTy);
    for (const Chunk *C : ArrayRef(Covering).drop_front())
      V = B.CreateOr(V, B.CreateShl(B.CreateZExt(C->V, WideTy),
                                    (C->Offset - Base) * 8));
  }
  if (Offset != Base)
    V = B.CreateLShr(V, (Offset - Base) * 8);
  if (End - Base != Size)
    V = B.CreateTrunc(V, B.getIntNTy(Size * 8));
  if (Ty->isPtrOrPtrVectorTy())
    return B.CreateIntToPtr(V, Ty);
  if (DL.getTypeSizeInBits(Ty) != Size * 8) // i1 and other odd sizes
    return B.CreateTrunc(V, Ty);
  return B.CreateBitCast(V, Ty);
}

static bool promote(Function &F) {
  if (F.getCallingConv() != CallingConv::AGX_KERNEL || F.isDeclaration() ||
      !F.use_empty() || F.isVarArg())
    return false;
  const DataLayout &DL = F.getDataLayout();
  LLVMContext &Ctx = F.getContext();

  struct Plan {
    bool Promote = false;
    bool ByRef = false;
    uint64_t Bytes = 0;
    SmallVector<LoadAt, 4> Loads;
  };
  SmallVector<Plan, 8> Plans(F.arg_size());
  // The uniforms left once every argument takes its unpromoted place (a
  // by-reference argument its 64-bit pointer, others their bytes rounded to
  // 32 bits, as the lowering aligns them); promoting an argument gives back
  // its pointer.
  uint64_t Fixed = ImplicitBytes;
  for (Argument &A : F.args())
    Fixed += A.getType()->isPointerTy() && A.getParamByRefType()
                 ? 8
                 : alignTo(DL.getTypeStoreSize(A.getType()), 4);
  uint64_t Budget = Fixed < UniformBytes ? UniformBytes - Fixed : 0;
  bool Any = false;
  for (Argument &A : F.args()) {
    Plan &P = Plans[A.getArgNo()];
    Type *T = A.getType()->isPointerTy() ? A.getParamByRefType() : nullptr;
    if (!T) {
      P.Bytes = DL.getTypeStoreSize(A.getType());
      continue;
    }
    P.ByRef = true;
    P.Bytes = DL.getTypeAllocSize(T);
    uint64_t Words = divideCeil(P.Bytes, 4);
    if (P.Bytes == 0 || Words * 4 > Budget + 8 ||
        !collectLoads(&A, 0, P.Bytes, DL, P.Loads)) {
      P.Loads.clear();
      continue;
    }
    P.Promote = true;
    Budget = Budget + 8 - Words * 4;
    Any = true;
  }
  if (!Any)
    return false;

  // The new signature and parameter attributes.
  SmallVector<Type *, 8> Params;
  SmallVector<AttributeSet, 8> ParamAttrs;
  SmallVector<unsigned, 8> FirstNew;
  AttributeList Attrs = F.getAttributes();
  for (Argument &A : F.args()) {
    const Plan &P = Plans[A.getArgNo()];
    FirstNew.push_back(Params.size());
    if (!P.Promote) {
      Params.push_back(A.getType());
      ParamAttrs.push_back(Attrs.getParamAttrs(A.getArgNo()));
      continue;
    }
    for (uint64_t Size : chunkSizes(P.Bytes)) {
      Params.push_back(Type::getIntNTy(Ctx, Size * 8));
      ParamAttrs.push_back(AttributeSet());
    }
  }
  FunctionType *NFTy = FunctionType::get(F.getReturnType(), Params, false);
  Function *NF = Function::Create(NFTy, F.getLinkage(), F.getAddressSpace());
  NF->copyAttributesFrom(&F);
  NF->setAttributes(AttributeList::get(Ctx, Attrs.getFnAttrs(),
                                       Attrs.getRetAttrs(), ParamAttrs));
  NF->copyMetadata(&F, 0);
  NF->takeName(&F);
  F.getParent()->getFunctionList().insert(F.getIterator(), NF);
  NF->splice(NF->begin(), &F);

  std::string Desc;
  raw_string_ostream OS(Desc);
  bool FirstItem = true;
  for (Argument &A : F.args()) {
    unsigned I = A.getArgNo();
    const Plan &P = Plans[I];
    // The hidden arguments of checked mode are not source arguments: the
    // record describes them separately (AGXCheckedAccesses).
    if (!Attrs.getParamAttr(I, "agx-checked").isValid()) {
      OS << (FirstItem ? "" : ",") << (P.ByRef && !P.Promote ? 'p' : 'v')
         << P.Bytes << ':' << FirstNew[I];
      FirstItem = false;
    }
    if (!P.Promote) {
      Argument *NA = NF->getArg(FirstNew[I]);
      NA->takeName(&A);
      A.replaceAllUsesWith(NA);
      continue;
    }
    SmallVector<Chunk, 8> Chunks;
    uint64_t Offset = 0;
    for (auto [K, Size] : enumerate(chunkSizes(P.Bytes))) {
      Argument *NA = NF->getArg(FirstNew[I] + K);
      NA->setName(A.getName() + "." + Twine(K));
      Chunks.push_back({NA, Offset, Size});
      Offset += Size;
    }
    for (const LoadAt &L : P.Loads) {
      IRBuilder<> B(L.Load);
      Value *V = loadFromChunks(B, Chunks, L.Offset, L.Load->getType(), DL);
      V->takeName(L.Load);
      L.Load->replaceAllUsesWith(V);
      L.Load->eraseFromParent();
    }
    // The address computations are now dead.
    SmallVector<WeakTrackingVH, 8> Dead(A.users().begin(), A.users().end());
    for (WeakTrackingVH &V : Dead)
      if (auto *I = dyn_cast_or_null<Instruction>(V))
        RecursivelyDeleteTriviallyDeadInstructions(I);
  }
  NF->addFnAttr("agx-kernel-args", Desc);
  F.eraseFromParent();
  return true;
}

bool AGXPromoteKernelArgs::runOnModule(Module &M) {
  bool Changed = false;
  for (Function &F : make_early_inc_range(M))
    Changed |= promote(F);
  return Changed;
}
