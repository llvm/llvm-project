//===- llvm/CodeGen/TileShapeInfo.h - ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Pass to transform <256 x i32>
/// <256 x i32> is mapped to AMX tile register on X86, AMX instruction set only
/// provides simple operation on tile register. The basic elementwise operation
/// is not supported by AMX. Since we define the AMX tile as vector <256 x i32>
/// and only AMX intrinsics can operate on the type, we need transform
/// load/store <256 x i32> instruction to AMX load/store. Besides, we split
/// <256 x i32> to 2 <128 x i32> if the vector is not used or defined by AMX
/// intrinsics, so that in instruction selection it can be lowered to proper
/// size which HW can support.
//
//===----------------------------------------------------------------------===//
//
#include "X86.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "lower-amx-type"

namespace {
class X86LowerAMXType {
  Function &Func;
  const DataLayout &DL;
  DenseSet<Instruction *> LDSet;
  DenseSet<Instruction *> STSet;
  DenseMap<Value *, std::pair<LoadInst *, LoadInst *>> LoadMap;

public:
  X86LowerAMXType(Function &F) : Func(F), DL(F.getParent()->getDataLayout()) {}
  bool visit();
  bool visitLD();
  bool visitST();
  void splitST(Instruction *Inst);
  void splitLD(Instruction *Inst);
};

// Split v256i32 load/store to 2 v128i32, so that ISel can
// lower it to proper vector size.
void X86LowerAMXType::splitST(Instruction *Inst) {
  StoreInst *ST = dyn_cast<StoreInst>(Inst);
  IRBuilder<> Builder(ST);
  LLVMContext &Ctx = Builder.getContext();
  Type *Ty = ST->getValueOperand()->getType();
  EVT VT = EVT::getEVT(Ty);
  EVT HalfVT = VT.getHalfNumVectorElementsVT(Ctx);
  Type *HalfTy = HalfVT.getTypeForEVT(Ctx);

  LoadInst *Lo, *Hi;
  std::tie(Lo, Hi) = LoadMap[ST->getValueOperand()];
  Value *Ptr = ST->getPointerOperand();
  PointerType *HalfPtrTy = HalfTy->getPointerTo(ST->getPointerAddressSpace());
  Value *HalfPtr = Builder.CreateBitCast(Ptr, HalfPtrTy);
  // The HW require the alignment for AMX tile is 64, but front-end generate
  // code for the vector alignment which is the vector size.
  uint64_t HalfTySize = HalfTy->getPrimitiveSizeInBits().getFixedSize() / 8;
  Align Alignment = std::min(Lo->getAlign(), Align(HalfTySize));
  Builder.CreateAlignedStore(Lo, HalfPtr, Alignment, ST->isVolatile());

  HalfPtr = Builder.CreateGEP(HalfTy, HalfPtr, Builder.getInt32(1));
  Builder.CreateAlignedStore(Hi, HalfPtr, Alignment, ST->isVolatile());
}

bool X86LowerAMXType::visitST() {
  if (STSet.empty())
    return false;
  for (auto *Inst : STSet) {
    Value *Row, *Col;
    const IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst->getOperand(0));
    if (!II)
      Row = Col = nullptr;
    else {
      switch (II->getIntrinsicID()) {
      default:
        Row = Col = nullptr;
        break;
      case Intrinsic::x86_tileloadd64_internal:
      case Intrinsic::x86_tdpbssd_internal: {
        Row = II->getArgOperand(0);
        Col = II->getArgOperand(1);
        break;
      }
      }
    }
    if (!Row) {
      splitST(Inst);
      continue;
    }
    IRBuilder<> Builder(Inst);
    LLVMContext &Ctx = Builder.getContext();
    // Use the maximun column as stride. It must be the same with load stride.
    Value *Stride = Builder.getInt64(64);
    Value *I8Ptr =
        Builder.CreateBitCast(Inst->getOperand(1), Type::getInt8PtrTy(Ctx));
    std::array<Value *, 5> Args = {Row, Col, I8Ptr, Stride,
                                   Inst->getOperand(0)};

    Builder.CreateIntrinsic(Intrinsic::x86_tilestored64_internal, None, Args);
  }
  return true;
}

void X86LowerAMXType::splitLD(Instruction *Inst) {
  LoadInst *LD = dyn_cast<LoadInst>(Inst);
  IRBuilder<> Builder(LD);
  LLVMContext &Ctx = Builder.getContext();
  Type *Ty = LD->getType();
  EVT VT = EVT::getEVT(Ty);
  EVT HalfVT = VT.getHalfNumVectorElementsVT(Ctx);
  Type *HalfTy = HalfVT.getTypeForEVT(Ctx);

  Value *Ptr = LD->getPointerOperand();
  PointerType *HalfPtrTy = HalfTy->getPointerTo(LD->getPointerAddressSpace());
  Value *HalfPtr = Builder.CreateBitCast(Ptr, HalfPtrTy);
  // The HW require the alignment for AMX tile is 64, but front-end generate
  // code for the vector alignment which is the vector size.
  uint64_t HalfTySize = HalfTy->getPrimitiveSizeInBits().getFixedSize() / 8;
  Align Alignment = std::min(LD->getAlign(), Align(HalfTySize));
  auto *Lo =
      Builder.CreateAlignedLoad(HalfTy, HalfPtr, Alignment, LD->isVolatile());

  HalfPtr = Builder.CreateGEP(HalfTy, HalfPtr, Builder.getInt32(1));
  auto *Hi =
      Builder.CreateAlignedLoad(HalfTy, HalfPtr, Alignment, LD->isVolatile());

  LoadMap[Inst] = std::make_pair(Lo, Hi);
}

bool X86LowerAMXType::visitLD() {
  if (LDSet.empty())
    return false;
  for (auto &Inst : LDSet) {
    int Count = 0;
    Value *NewInst = nullptr;
    // The user should be all AMX intrinsics or all LLVM instruction.
    // Don't support it is used by both AMX intrinsics and LLVM instructions.
    for (auto I = Inst->use_begin(), E = Inst->use_end(); I != E;) {
      Use &U = *I++;
      const IntrinsicInst *II = dyn_cast<IntrinsicInst>(U.getUser());
      if (!II) {
        Count++;
        continue;
      }
      if (NewInst)
        continue;
      Value *Row, *Col;
      switch (II->getIntrinsicID()) {
      default:
        report_fatal_error("Non-AMX intrinsic use tile type.");
        break;
      case Intrinsic::x86_tdpbssd_internal: {
        unsigned OpNo = U.getOperandNo();
        switch (OpNo) {
        case 3:
          Row = II->getArgOperand(0);
          Col = II->getArgOperand(1);
          break;
        case 4:
          Row = II->getArgOperand(0);
          Col = II->getArgOperand(2);
          break;
        case 5:
          Row = II->getArgOperand(2);
          Col = II->getArgOperand(1);
          break;
        }
        break;
      }
      case Intrinsic::x86_tilestored64_internal: {
        Row = II->getArgOperand(0);
        Col = II->getArgOperand(1);
        break;
      }
      }
      assert(Count == 0 && "Can NOT mix amx intrinsic and LLVM instruction");
      // FIXME: The shape def should be ahead of load.
      IRBuilder<> Builder(Inst);
      LLVMContext &Ctx = Builder.getContext();
      // Use the maximun column as stride.
      Value *Stride = Builder.getInt64(64);
      Value *I8Ptr =
          Builder.CreateBitCast(Inst->getOperand(0), Type::getInt8PtrTy(Ctx));
      std::array<Value *, 4> Args = {Row, Col, I8Ptr, Stride};

      NewInst = Builder.CreateIntrinsic(Intrinsic::x86_tileloadd64_internal,
                                        None, Args);

      Inst->replaceAllUsesWith(NewInst);
    }
    if (!NewInst)
      splitLD(Inst);
  }
  return true;
}

bool X86LowerAMXType::visit() {
  bool C;
  auto IsAMXType = [](FixedVectorType *VTy) {
    if (!VTy)
      return false;
    if (!VTy->getScalarType()->isIntegerTy(32))
      return false;
    if (VTy->getNumElements() != 256)
      return false;

    return true;
  };

  for (BasicBlock &BB : Func) {
    for (Instruction &Inst : BB) {
      LoadInst *LD = dyn_cast<LoadInst>(&Inst);
      // Check load instruction.
      // %3 = load <256 x i32>, <256 x i32>* %1, align 64
      if (LD) {
        FixedVectorType *VTy = dyn_cast<FixedVectorType>(Inst.getType());
        if (!IsAMXType(VTy))
          continue;
        LDSet.insert(&Inst);
        continue;
      }
      // Check store instruction.
      // store <256 x i32> %3, <256 x i32>* %2, align 64
      StoreInst *ST = dyn_cast<StoreInst>(&Inst);
      if (!ST)
        continue;
      FixedVectorType *VTy =
          dyn_cast<FixedVectorType>(ST->getOperand(0)->getType());
      if (!IsAMXType(VTy))
        continue;
      STSet.insert(&Inst);
    }
  }

  C = visitLD() | visitST();
  for (auto *Inst : STSet)
    Inst->eraseFromParent();
  for (auto *Inst : LDSet)
    Inst->eraseFromParent();
  return C;
}
} // anonymous namespace

namespace {

class X86LowerAMXTypeLegacyPass : public FunctionPass {
public:
  static char ID;

  X86LowerAMXTypeLegacyPass() : FunctionPass(ID) {
    initializeX86LowerAMXTypeLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    X86LowerAMXType LAT(F);
    bool C = LAT.visit();
    return C;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

} // anonymous namespace

static const char PassName[] = "Lower AMX type for load/store";
char X86LowerAMXTypeLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(X86LowerAMXTypeLegacyPass, DEBUG_TYPE, PassName, false,
                      false)
INITIALIZE_PASS_END(X86LowerAMXTypeLegacyPass, DEBUG_TYPE, PassName, false,
                    false)

FunctionPass *llvm::createX86LowerAMXTypePass() {
  return new X86LowerAMXTypeLegacyPass();
}
