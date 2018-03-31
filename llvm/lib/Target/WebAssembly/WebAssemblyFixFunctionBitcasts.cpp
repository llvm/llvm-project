//===-- WebAssemblyFixFunctionBitcasts.cpp - Fix function bitcasts --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Fix bitcasted functions.
///
/// WebAssembly requires caller and callee signatures to match, however in LLVM,
/// some amount of slop is vaguely permitted. Detect mismatch by looking for
/// bitcasts of functions and rewrite them to use wrapper functions instead.
///
/// This doesn't catch all cases, such as when a function's address is taken in
/// one place and casted in another, but it works for many common cases.
///
/// Note that LLVM already optimizes away function bitcasts in common cases by
/// dropping arguments as needed, so this pass only ends up getting used in less
/// common cases.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-fix-function-bitcasts"

static cl::opt<bool> TemporaryWorkarounds(
  "wasm-temporary-workarounds",
  cl::desc("Apply certain temporary workarounds"),
  cl::init(true), cl::Hidden);

namespace {
class FixFunctionBitcasts final : public ModulePass {
  StringRef getPassName() const override {
    return "WebAssembly Fix Function Bitcasts";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    ModulePass::getAnalysisUsage(AU);
  }

  bool runOnModule(Module &M) override;

public:
  static char ID;
  FixFunctionBitcasts() : ModulePass(ID) {}
};
} // End anonymous namespace

char FixFunctionBitcasts::ID = 0;
INITIALIZE_PASS(FixFunctionBitcasts, DEBUG_TYPE,
                "Fix mismatching bitcasts for WebAssembly", false, false)

ModulePass *llvm::createWebAssemblyFixFunctionBitcasts() {
  return new FixFunctionBitcasts();
}

// Recursively descend the def-use lists from V to find non-bitcast users of
// bitcasts of V.
static void FindUses(Value *V, Function &F,
                     SmallVectorImpl<std::pair<Use *, Function *>> &Uses,
                     SmallPtrSetImpl<Constant *> &ConstantBCs) {
  for (Use &U : V->uses()) {
    if (BitCastOperator *BC = dyn_cast<BitCastOperator>(U.getUser()))
      FindUses(BC, F, Uses, ConstantBCs);
    else if (U.get()->getType() != F.getType()) {
      CallSite CS(U.getUser());
      if (!CS)
        // Skip uses that aren't immediately called
        continue;
      Value *Callee = CS.getCalledValue();
      if (Callee != V)
        // Skip calls where the function isn't the callee
        continue;
      if (isa<Constant>(U.get())) {
        // Only add constant bitcasts to the list once; they get RAUW'd
        auto c = ConstantBCs.insert(cast<Constant>(U.get()));
        if (!c.second)
          continue;
      }
      Uses.push_back(std::make_pair(&U, &F));
    }
  }
}

// Create a wrapper function with type Ty that calls F (which may have a
// different type). Attempt to support common bitcasted function idioms:
//  - Call with more arguments than needed: arguments are dropped
//  - Call with fewer arguments than needed: arguments are filled in with undef
//  - Return value is not needed: drop it
//  - Return value needed but not present: supply an undef
//
// For now, return nullptr without creating a wrapper if the wrapper cannot
// be generated due to incompatible types.
static Function *CreateWrapper(Function *F, FunctionType *Ty) {
  Module *M = F->getParent();

  Function *Wrapper =
      Function::Create(Ty, Function::PrivateLinkage, "bitcast", M);
  BasicBlock *BB = BasicBlock::Create(M->getContext(), "body", Wrapper);

  // Determine what arguments to pass.
  SmallVector<Value *, 4> Args;
  Function::arg_iterator AI = Wrapper->arg_begin();
  Function::arg_iterator AE = Wrapper->arg_end();
  FunctionType::param_iterator PI = F->getFunctionType()->param_begin();
  FunctionType::param_iterator PE = F->getFunctionType()->param_end();
  for (; AI != AE && PI != PE; ++AI, ++PI) {
    if (AI->getType() != *PI) {
      Wrapper->eraseFromParent();
      return nullptr;
    }
    Args.push_back(&*AI);
  }
  for (; PI != PE; ++PI)
    Args.push_back(UndefValue::get(*PI));
  if (F->isVarArg())
    for (; AI != AE; ++AI)
      Args.push_back(&*AI);

  CallInst *Call = CallInst::Create(F, Args, "", BB);

  // Determine what value to return.
  if (Ty->getReturnType()->isVoidTy())
    ReturnInst::Create(M->getContext(), BB);
  else if (F->getFunctionType()->getReturnType()->isVoidTy())
    ReturnInst::Create(M->getContext(), UndefValue::get(Ty->getReturnType()),
                       BB);
  else if (F->getFunctionType()->getReturnType() == Ty->getReturnType())
    ReturnInst::Create(M->getContext(), Call, BB);
  else {
    Wrapper->eraseFromParent();
    return nullptr;
  }

  return Wrapper;
}

bool FixFunctionBitcasts::runOnModule(Module &M) {
  Function *Main = nullptr;
  CallInst *CallMain = nullptr;
  SmallVector<std::pair<Use *, Function *>, 0> Uses;
  SmallPtrSet<Constant *, 2> ConstantBCs;

  // Collect all the places that need wrappers.
  for (Function &F : M) {
    FindUses(&F, F, Uses, ConstantBCs);

    // If we have a "main" function, and its type isn't
    // "int main(int argc, char *argv[])", create an artificial call with it
    // bitcasted to that type so that we generate a wrapper for it, so that
    // the C runtime can call it.
    if (!TemporaryWorkarounds && !F.isDeclaration() && F.getName() == "main") {
      Main = &F;
      LLVMContext &C = M.getContext();
      Type *MainArgTys[] = {
        PointerType::get(Type::getInt8PtrTy(C), 0),
        Type::getInt32Ty(C)
      };
      FunctionType *MainTy = FunctionType::get(Type::getInt32Ty(C), MainArgTys,
                                               /*isVarArg=*/false);
      if (F.getFunctionType() != MainTy) {
        Value *Args[] = {
          UndefValue::get(MainArgTys[0]),
          UndefValue::get(MainArgTys[1])
        };
        Value *Casted = ConstantExpr::getBitCast(Main,
                                                 PointerType::get(MainTy, 0));
        CallMain = CallInst::Create(Casted, Args, "call_main");
        Use *UseMain = &CallMain->getOperandUse(2);
        Uses.push_back(std::make_pair(UseMain, &F));
      }
    }
  }

  DenseMap<std::pair<Function *, FunctionType *>, Function *> Wrappers;

  for (auto &UseFunc : Uses) {
    Use *U = UseFunc.first;
    Function *F = UseFunc.second;
    PointerType *PTy = cast<PointerType>(U->get()->getType());
    FunctionType *Ty = dyn_cast<FunctionType>(PTy->getElementType());

    // If the function is casted to something like i8* as a "generic pointer"
    // to be later casted to something else, we can't generate a wrapper for it.
    // Just ignore such casts for now.
    if (!Ty)
      continue;

    // Bitcasted vararg functions occur in Emscripten's implementation of
    // EM_ASM, so suppress wrappers for them for now.
    if (TemporaryWorkarounds && (Ty->isVarArg() || F->isVarArg()))
      continue;

    auto Pair = Wrappers.insert(std::make_pair(std::make_pair(F, Ty), nullptr));
    if (Pair.second)
      Pair.first->second = CreateWrapper(F, Ty);

    Function *Wrapper = Pair.first->second;
    if (!Wrapper)
      continue;

    if (isa<Constant>(U->get()))
      U->get()->replaceAllUsesWith(Wrapper);
    else
      U->set(Wrapper);
  }

  // If we created a wrapper for main, rename the wrapper so that it's the
  // one that gets called from startup.
  if (CallMain) {
    Main->setName("__original_main");
    Function *MainWrapper =
        cast<Function>(CallMain->getCalledValue()->stripPointerCasts());
    MainWrapper->setName("main");
    MainWrapper->setLinkage(Main->getLinkage());
    MainWrapper->setVisibility(Main->getVisibility());
    Main->setLinkage(Function::PrivateLinkage);
    Main->setVisibility(Function::DefaultVisibility);
    delete CallMain;
  }

  return true;
}
