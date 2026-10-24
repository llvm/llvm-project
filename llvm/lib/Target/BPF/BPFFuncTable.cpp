//===-- BPFFuncTable.cpp - Call functions through a table -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A program that keeps its data in arena keeps pointers to functions there too:
// vtables, closures, Box<dyn Trait>. Arena is shared with user space, so what
// is loaded from it is a number and the kernel does not let callx take it.
//
// Make the pointer to a function a number to begin with: the index of the
// function in a table that is read-only data.
//   &f         index of f
//   (*p)(...)  __bpf_func_table[min(p - first, cnt - 1) + first](...)
// The functions of one type are next to each other in the table: 'first' and
// 'cnt' are those of the type of the call. A number that is not an index of
// such function calls the last one of them. That is not what the program
// meant, but it is a function that takes what the call passes.
//
// The kernel knows the pointers to functions in read-only data. A load at
// an offset that is not constant is fine as long as there is such a pointer at
// every offset it can be at. So callx gets a pointer to a function again and
// there is nothing for a loader or the kernel to do.
//
// 0 is not an index, so that null stays null. When there is one function of
// the type the call is direct. When there is none it is a trap.
//
// A function that is passed to a helper or to a kfunc stays a function.
//
// All functions whose address is taken have to be in the module.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-func-table"

namespace {

/// The functions of one type: where they start in the table.
struct FuncRange {
  uint64_t First = 0;
  SmallVector<Function *, 4> Funcs;
};

} // namespace

/// callx takes static functions of .text only. The address of a function in
/// another section is the address of a program, e.g. in struct_ops.
static bool isCallable(const Function &F) {
  return !F.isDeclaration() && F.hasLocalLinkage() && !F.hasSection();
}

/// The use is the address of the function as a value.
static bool isAddressUse(const Use &U) {
  const User *Usr = U.getUser();

  if (const auto *CB = dyn_cast<CallBase>(Usr)) {
    if (CB->isCallee(&U))
      return false;
    // A callback: bpf_loop(), bpf_timer_set_callback() and co. The kernel calls
    // it and wants the function. A helper is a number, a kfunc a declaration.
    const Value *Callee = CB->getCalledOperand()->stripPointerCasts();
    const auto *F = dyn_cast<Function>(Callee);
    return F ? !F->isDeclaration() || F->isIntrinsic() : !isa<Constant>(Callee);
  }
  // Personality, prefix and prologue data, aliases.
  return !isa<BlockAddress, NoCFIValue, Function, GlobalAlias, GlobalIFunc>(
      Usr);
}

static void replaceAddress(Function &F, Constant *Index) {
  SmallSetVector<Constant *, 4> Constants;

  for (Use &U : make_early_inc_range(F.uses())) {
    if (!isAddressUse(U))
      continue;
    // Constants are uniqued: the operand cannot be set.
    auto *C = dyn_cast<Constant>(U.getUser());
    if (C && !isa<GlobalValue>(C))
      Constants.insert(C);
    else
      U.set(Index);
  }
  for (Constant *C : Constants)
    C->handleOperandChange(&F, Index);
}

static bool runFuncTable(Module &M) {
  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *I64 = Type::getInt64Ty(C);
  PointerType *Ptr = PointerType::getUnqual(C);
  MapVector<FunctionType *, FuncRange> Types;
  SmallVector<CallBase *, 16> Calls;
  SmallVector<GlobalValue *, 4> Used, CompilerUsed;
  SmallVector<Constant *, 16> Table;

  // They hold functions, not their addresses.
  if (GlobalVariable *GV = collectUsedGlobalVariables(M, Used, false))
    GV->eraseFromParent();
  if (GlobalVariable *GV = collectUsedGlobalVariables(M, CompilerUsed, true))
    GV->eraseFromParent();
  bool Changed = !Used.empty() || !CompilerUsed.empty();

  for (Function &F : M) {
    if (isCallable(F) && any_of(F.uses(), isAddressUse))
      Types[F.getFunctionType()].Funcs.push_back(&F);
    for (Instruction &I : instructions(F)) {
      auto *CB = dyn_cast<CallBase>(&I);
      // A constant is a function or the number of a helper.
      if (CB && !CB->isInlineAsm() && !isa<Constant>(CB->getCalledOperand()))
        Calls.push_back(CB);
    }
  }

  Table.push_back(ConstantPointerNull::get(Ptr));
  for (auto &[FTy, R] : Types) {
    R.First = Table.size();
    for (Function *F : R.Funcs) {
      replaceAddress(*F, ConstantExpr::getIntToPtr(
                             ConstantInt::get(I64, Table.size()), Ptr));
      Table.push_back(F);
    }
  }

  appendToUsed(M, Used);
  appendToCompilerUsed(M, CompilerUsed);
  if (Calls.empty() && Types.empty())
    return Changed;

  ArrayType *TableTy = ArrayType::get(Ptr, Table.size());
  GlobalVariable *GV = nullptr;

  for (CallBase *CB : Calls) {
    const FuncRange &R = Types.lookup(CB->getFunctionType());
    uint64_t Cnt = R.Funcs.size();

    // Nothing to call. The kernel rejects the trap if it is reachable. A call
    // of null alone is undefined behaviour: what leads to nothing else is
    // removed with it, the exit of a loop for example.
    if (!Cnt) {
      IRBuilder<>(CB).CreateIntrinsic(Intrinsic::trap, {});
      CB->setCalledOperand(ConstantPointerNull::get(Ptr));
      continue;
    }
    if (Cnt == 1) {
      CB->setCalledOperand(R.Funcs[0]);
      continue;
    }
    if (!GV)
      GV = new GlobalVariable(M, TableTy, true, GlobalValue::InternalLinkage,
                              ConstantArray::get(TableTy, Table),
                              "__bpf_func_table");

    IRBuilder<> B(CB);
    Value *Idx = B.CreatePtrToInt(CB->getCalledOperand(), I64);
    Idx = B.CreateSub(Idx, ConstantInt::get(I64, R.First));
    Idx = B.CreateBinaryIntrinsic(Intrinsic::umin, Idx,
                                  ConstantInt::get(I64, Cnt - 1));
    Constant *First = ConstantExpr::getInBoundsPtrAdd(
        GV, ConstantInt::get(I64, R.First * DL.getPointerSize()));
    Value *Slot = B.CreateInBoundsGEP(Ptr, First, Idx);
    CB->setCalledOperand(B.CreateLoad(Ptr, Slot));
  }
  return true;
}

namespace {

struct BPFFuncTableLegacy : public ModulePass {
  static char ID;
  BPFFuncTableLegacy() : ModulePass(ID) {}
  StringRef getPassName() const override { return "BPF Function Table"; }
  bool runOnModule(Module &M) override { return runFuncTable(M); }
};

} // namespace

char BPFFuncTableLegacy::ID = 0;

INITIALIZE_PASS(BPFFuncTableLegacy, DEBUG_TYPE, "BPF Function Table", false,
                false)

ModulePass *llvm::createBPFFuncTablePass() { return new BPFFuncTableLegacy(); }

PreservedAnalyses BPFFuncTablePass::run(Module &M, ModuleAnalysisManager &AM) {
  return runFuncTable(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
