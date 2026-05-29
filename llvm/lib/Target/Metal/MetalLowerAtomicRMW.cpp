//===- MetalLowerAtomicRMW.cpp - Lower atomicrmw to AIR intrinsics --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalLowerAtomicRMW.h"
#include "Metal.h"
#include "MetalAddressSpaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-lower-atomic-rmw"

// Metal address spaces. Inlined here (rather than pulled from a separate
// header) since this pass only needs the threadgroup constant.

// Build an AIR atomic intrinsic name following AIR's sign-prefix rules.
static std::string buildAtomicName(bool IsGlobal, AtomicRMWInst::BinOp Op,
                                   bool IsFloat) {
  std::string Name = "air.atomic.";
  Name += IsGlobal ? "global" : "local";
  Name += '.';

  switch (Op) {
  case AtomicRMWInst::Xchg:
    Name += "xchg";
    break;
  case AtomicRMWInst::Add:
  case AtomicRMWInst::FAdd:
    Name += "add.s";
    break;
  case AtomicRMWInst::Sub:
    Name += "sub.s";
    break;
  case AtomicRMWInst::Max:
    Name += "max.s";
    break;
  case AtomicRMWInst::Min:
    Name += "min.s";
    break;
  case AtomicRMWInst::UMax:
    Name += "umax.u";
    break;
  case AtomicRMWInst::UMin:
    Name += "umin.u";
    break;
  case AtomicRMWInst::And:
    Name += "and.s";
    break;
  case AtomicRMWInst::Or:
    Name += "or.s";
    break;
  case AtomicRMWInst::Xor:
    Name += "xor.s";
    break;
  default:
    return {};
  }

  Name += '.';
  Name += IsFloat ? "f32" : "i32";
  return Name;
}

static bool lowerAtomicRMW(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I1 = Type::getInt1Ty(Ctx);
  bool Changed = false;

  SmallVector<AtomicRMWInst *, 16> Atomics;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
          Atomics.push_back(RMW);

  for (AtomicRMWInst *RMW : Atomics) {
    unsigned AddrSpace =
        RMW->getPointerOperand()->getType()->getPointerAddressSpace();
    bool IsGlobal = AddrSpace != metal::AS::Threadgroup;

    Type *ValTy = RMW->getValOperand()->getType();
    bool IsFloat = ValTy->isFloatTy();
    std::string Name = buildAtomicName(IsGlobal, RMW->getOperation(), IsFloat);
    if (Name.empty())
      continue;

    auto *PtrTy = PointerType::get(Ctx, AddrSpace);
    FunctionType *FTy =
        FunctionType::get(ValTy, {PtrTy, ValTy, I32, I32, I1}, false);
    FunctionCallee FC = M.getOrInsertFunction(Name, FTy);

    IRBuilder<> B(RMW);
    Value *Result =
        B.CreateCall(FC,
                     {RMW->getPointerOperand(), RMW->getValOperand(),
                      ConstantInt::get(I32, 0), ConstantInt::get(I32, 1),
                      ConstantInt::get(I1, 1)},
                     RMW->getName());
    RMW->replaceAllUsesWith(Result);
    RMW->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

PreservedAnalyses MetalLowerAtomicRMWPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  return lowerAtomicRMW(M) ? PreservedAnalyses::none()
                           : PreservedAnalyses::all();
}

bool MetalLowerAtomicRMWLegacy::runOnModule(Module &M) {
  return lowerAtomicRMW(M);
}

char MetalLowerAtomicRMWLegacy::ID = 0;

INITIALIZE_PASS(MetalLowerAtomicRMWLegacy, DEBUG_TYPE, "Metal Lower AtomicRMW",
                false, false)

ModulePass *llvm::createMetalLowerAtomicRMWLegacyPass() {
  return new MetalLowerAtomicRMWLegacy();
}
