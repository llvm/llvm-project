//===- DXILOpLower.cpp - Lowering LLVM intrinsic to DIXLOp function -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains passes and utilities to lower llvm intrinsic call
/// to DXILOp function call.
//===----------------------------------------------------------------------===//

#include "DXILConstants.h"
#include "DXILIntrinsicExpansion.h"
#include "DXILOpBuilder.h"
#include "DirectX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "dxil-op-lower"

using namespace llvm;
using namespace llvm::dxil;

static void lowerIntrinsic(dxil::OpCode DXILOp, Function &F, Module &M) {
  IRBuilder<> B(M.getContext());
  Value *DXILOpArg = B.getInt32(static_cast<unsigned>(DXILOp));
  DXILOpBuilder DXILB(M, B);
  Type *OverloadTy = DXILB.getOverloadTy(DXILOp, F.getFunctionType());
  for (User *U : make_early_inc_range(F.users())) {
    CallInst *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;

    SmallVector<Value *> Args;
    Args.emplace_back(DXILOpArg);
    Args.append(CI->arg_begin(), CI->arg_end());
    B.SetInsertPoint(CI);
    CallInst *DXILCI = DXILB.createDXILOpCall(DXILOp, OverloadTy, CI->args());

    CI->replaceAllUsesWith(DXILCI);
    CI->eraseFromParent();
  }
  if (F.user_empty())
    F.eraseFromParent();
}

static bool lowerIntrinsics(Module &M) {
  bool Updated = false;

#define DXIL_OP_INTRINSIC_MAP
#include "DXILOperation.inc"
#undef DXIL_OP_INTRINSIC_MAP

  for (Function &F : make_early_inc_range(M.functions())) {
    if (!F.isDeclaration())
      continue;
    Intrinsic::ID ID = F.getIntrinsicID();
    if (ID == Intrinsic::not_intrinsic)
      continue;
    auto LowerIt = LowerMap.find(ID);
    if (LowerIt == LowerMap.end())
      continue;
    lowerIntrinsic(LowerIt->second, F, M);
    Updated = true;
  }
  return Updated;
}

namespace {
/// A pass that transforms external global definitions into declarations.
class DXILOpLowering : public PassInfoMixin<DXILOpLowering> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    if (lowerIntrinsics(M))
      return PreservedAnalyses::none();
    return PreservedAnalyses::all();
  }
};
} // namespace

namespace {
class DXILOpLoweringLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override { return lowerIntrinsics(M); }
  StringRef getPassName() const override { return "DXIL Op Lowering"; }
  DXILOpLoweringLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    // Specify the passes that your pass depends on
    AU.addRequired<DXILIntrinsicExpansionLegacy>();
  }
};
char DXILOpLoweringLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILOpLoweringLegacy, DEBUG_TYPE, "DXIL Op Lowering",
                      false, false)
INITIALIZE_PASS_END(DXILOpLoweringLegacy, DEBUG_TYPE, "DXIL Op Lowering", false,
                    false)

ModulePass *llvm::createDXILOpLoweringLegacyPass() {
  return new DXILOpLoweringLegacy();
}
