//===- SPIRVLowerOCLBlocks.cpp - OCL Utilities ----------------------------===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2018 Intel Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Intel Corporation, nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// SPIR-V specification doesn't allow function pointers, so SPIR-V translator
// is designed to fail if a value with function type (except calls) is occured.
// Currently there is only two cases, when function pointers are generating in
// LLVM IR in OpenCL - block calls and device side enqueue built-in calls.
//
// In both cases values with function type used as intermediate representation
// for block literal structure.
//
// This pass is designed to find such cases and simplify them to avoid any
// function pointer types occurrences in LLVM IR in 4 steps.
//
// 1. Find all function pointer allocas, like
//      %block = alloca void () *
//
//    Then find a single store to that alloca:
//      %blockLit = alloca <{ i32, i32, ...}>, align 4
//      %0 = bitcast <{ i32, i32, ... }>* %blockLit to void ()*
//    > store void ()* %0, void ()** %block, align 4
//
//    And replace the alloca users by new instructions which used stored value
//    %blockLit itself instead of function pointer alloca %block.
//
// 2. Find consecutive casts from block literal type to i8 addrspace(4)*
//    used function pointers as an intermediate type:
//      %0 = bitcast <{ i32, i32 }> %block to void() *
//      %1 = addrspacecast void() * %0 to i8 addrspace(4)*
//    And simplify them:
//      %2 = addrspacecast <{ i32, i32 }> %block to i8 addrspace(4)*
//
// 3. Find all unused instructions with function pointer type occured after
//    pp.1-2 and remove them.
//
// 4. Find unused globals with function pointer type, like
//    @block = constant void ()*
//             bitcast ({ i32, i32 }* @__block_literal_global to void ()*
//
//    And remove them.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spv-lower-ocl-blocks"

#include "OCLUtil.h"
#include "SPIRVInternal.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

namespace {

static void
removeUnusedFunctionPtrInst(Instruction *I,
                            SmallSetVector<Instruction *, 16> &FuncPtrInsts) {
  for (unsigned OpIdx = 0, Ops = I->getNumOperands(); OpIdx != Ops; ++OpIdx) {
    Instruction *OpI = dyn_cast<Instruction>(I->getOperand(OpIdx));
    I->setOperand(OpIdx, nullptr);
    if (OpI && OpI != I && OpI->user_empty())
      FuncPtrInsts.insert(OpI);
  }
  I->eraseFromParent();
}

static bool isFuncPtrAlloca(const AllocaInst *AI) {
  auto *ET = dyn_cast<PointerType>(AI->getAllocatedType());
  return ET && ET->getElementType()->isFunctionTy();
}

static bool hasFuncPtrType(const Value *V) {
  auto *PT = dyn_cast<PointerType>(V->getType());
  return PT && PT->getElementType()->isFunctionTy();
}

static bool isFuncPtrInst(const Instruction *I) {
  if (auto *AI = dyn_cast<AllocaInst>(I))
    return isFuncPtrAlloca(AI);

  for (auto &Op : I->operands()) {
    if (auto *AI = dyn_cast<AllocaInst>(Op))
      return isFuncPtrAlloca(AI);

    auto *OpI = dyn_cast<Instruction>(&Op);
    if (OpI && OpI != I && hasFuncPtrType(OpI))
      return true;
  }
  return false;
}

static StoreInst *findSingleStore(AllocaInst *AI) {
  StoreInst *Store = nullptr;
  for (auto *U : AI->users()) {
    if (!isa<StoreInst>(U))
      continue; // not a store
    if (Store)
      return nullptr; // there are more than one stores
    Store = dyn_cast<StoreInst>(U);
  }
  return Store;
}

static void fixFunctionPtrAllocaUsers(AllocaInst *AI) {
  // Find and remove a single store to alloca
  auto *SingleStore = findSingleStore(AI);
  assert(SingleStore && "More than one store to the function pointer alloca");
  auto *StoredVal = SingleStore->getValueOperand();
  SingleStore->eraseFromParent();

  // Find loads from the alloca and replace thier users
  for (auto *U : AI->users()) {
    auto *LI = dyn_cast<LoadInst>(U);
    if (!LI)
      continue;

    for (auto *U : LI->users()) {
      auto *UInst = cast<Instruction>(U);
      auto *Cast = CastInst::CreatePointerBitCastOrAddrSpaceCast(
          StoredVal, UInst->getType(), "", UInst);
      UInst->replaceAllUsesWith(Cast);
    }
  }
}

static int getBlockLiteralIdx(const Function &F) {
  StringRef FName = F.getName();
  if (isEnqueueKernelBI(FName))
    return FName.contains("events") ? 7 : 4;
  if (isKernelQueryBI(FName))
    return FName.contains("for_ndrange") ? 2 : 1;
  if (FName.startswith("__") && FName.contains("_block_invoke"))
    return F.hasStructRetAttr() ? 1 : 0;

  return -1; // No block literal argument
}

static bool hasBlockLiteralArg(const Function &F) {
  return getBlockLiteralIdx(F) != -1;
}

static bool simplifyFunctionPtrCasts(Function &F) {
  bool Changed = false;
  int BlockLiteralIdx = getBlockLiteralIdx(F);
  for (auto *U : F.users()) {
    auto *Call = dyn_cast<CallInst>(U);
    if (!Call)
      continue;
    if (Call->getFunction()->getName() == F.getName().str() + "_kernel")
      continue; // Skip block invoke function calls inside block invoke kernels

    const DataLayout &DL = F.getParent()->getDataLayout();
    auto *BlockLiteral = Call->getOperand(BlockLiteralIdx);
    auto *BlockLiteralVal = GetUnderlyingObject(BlockLiteral, DL);
    if (isa<GlobalVariable>(BlockLiteralVal))
      continue; // nothing to do with globals

    auto *BlockLiteralAlloca = cast<AllocaInst>(BlockLiteralVal);
    assert(!BlockLiteralAlloca->getAllocatedType()->isFunctionTy() &&
           "Function type shouldn't be there");

    auto *NewBlockLiteral = CastInst::CreatePointerBitCastOrAddrSpaceCast(
        BlockLiteralAlloca, BlockLiteral->getType(), "", Call);
    BlockLiteral->replaceAllUsesWith(NewBlockLiteral);
    Changed |= true;
  }
  return Changed;
}

static void
findFunctionPtrAllocas(Module &M,
                       SmallVectorImpl<AllocaInst *> &FuncPtrAllocas) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    for (auto &I : instructions(F)) {
      auto *AI = dyn_cast<AllocaInst>(&I);
      if (!AI || !isFuncPtrAlloca(AI))
        continue;
      FuncPtrAllocas.push_back(AI);
    }
  }
}

static void
findUnusedFunctionPtrInsts(Module &M,
                           SmallSetVector<Instruction *, 16> &FuncPtrInsts) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    for (auto &I : instructions(F))
      if (I.user_empty() && isFuncPtrInst(&I))
        FuncPtrInsts.insert(&I);
  }
}

static void
findUnusedFunctionPtrGlbs(Module &M,
                          SmallVectorImpl<GlobalVariable *> &FuncPtrGlbs) {
  for (auto &GV : M.globals()) {
    if (!GV.user_empty())
      continue;
    auto *GVType = dyn_cast<PointerType>(GV.getType()->getElementType());
    if (GVType && GVType->getElementType()->isFunctionTy())
      FuncPtrGlbs.push_back(&GV);
  }
}

class SPIRVLowerOCLBlocks : public ModulePass {

public:
  SPIRVLowerOCLBlocks() : ModulePass(ID) {}

  bool runOnModule(Module &M) {
    bool Changed = false;

    // 1. Find function pointer allocas and fix their users
    SmallVector<AllocaInst *, 16> FuncPtrAllocas;
    findFunctionPtrAllocas(M, FuncPtrAllocas);

    Changed |= !FuncPtrAllocas.empty();
    for (auto *AI : FuncPtrAllocas)
      fixFunctionPtrAllocaUsers(AI);

    // 2. Simplify consecutive casts which use function pointer types
    for (auto &F : M)
      if (hasBlockLiteralArg(F))
        Changed |= simplifyFunctionPtrCasts(F);

    // 3. Cleanup unused instructions with function pointer type
    // which are occured after pp. 1-2
    SmallSetVector<Instruction *, 16> FuncPtrInsts;
    findUnusedFunctionPtrInsts(M, FuncPtrInsts);

    Changed |= !FuncPtrInsts.empty();
    while (!FuncPtrInsts.empty()) {
      Instruction *I = FuncPtrInsts.pop_back_val();
      removeUnusedFunctionPtrInst(I, FuncPtrInsts);
    }

    // 4. Find and remove unused global variables with function pointer type
    SmallVector<GlobalVariable *, 16> FuncPtrGlbs;
    findUnusedFunctionPtrGlbs(M, FuncPtrGlbs);

    Changed |= !FuncPtrGlbs.empty();
    for (auto *GV : FuncPtrGlbs)
      GV->eraseFromParent();

    return Changed;
  }

  static char ID;
}; // class SPIRVLowerOCLBlocks

char SPIRVLowerOCLBlocks::ID = 0;

} // namespace

INITIALIZE_PASS(
    SPIRVLowerOCLBlocks, "spv-lower-ocl-blocks",
    "Remove function pointers occured in case of using OpenCL blocks", false,
    false)

llvm::ModulePass *llvm::createSPIRVLowerOCLBlocks() {
  return new SPIRVLowerOCLBlocks();
}
