//===- AIRKernelParams.cpp - Kernel values passed down to callees ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRKernelParams.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

StringRef llvm::airArgName(const Argument &A) {
  Attribute Name =
      A.getParent()->getAttributes().getParamAttr(A.getArgNo(), "air-arg-name");
  return Name.isValid() ? Name.getValueAsString() : A.getName();
}

void llvm::setAIRArgName(Argument &A, StringRef Name) {
  A.setName(Name);
  if (A.getName() != Name) // a context that discards value names
    A.addAttr(Attribute::get(A.getContext(), "air-arg-name", Name));
}

bool llvm::isCalledFunction(const Function &F) {
  for (const User *U : F.users())
    if (auto *CB = dyn_cast<CallBase>(U))
      if (CB->getCalledOperand() == &F)
        return true;
  return false;
}

AIRKernelParams llvm::addKernelParams(
    Module &M, ArrayRef<AIRKernelParamRequest> Requests,
    function_ref<Argument *(Function &, StringRef Key)> Existing) {
  AIRKernelParams R;

  // The keys, in the order of the requests, with their types and names.
  struct KeyInfo {
    std::string Key;
    Type *Ty;
    std::string Name;
  };
  SmallVector<KeyInfo, 4> Keys;
  auto KeyIndex = [&](const AIRKernelParamRequest &Req) {
    for (auto [I, K] : enumerate(Keys))
      if (K.Key == Req.Key)
        return unsigned(I);
    Keys.push_back({Req.Key.str(), Req.Ty, Req.Name.str()});
    return unsigned(Keys.size() - 1);
  };

  // The keys each function needs: its own, and those of the functions it
  // calls.
  MapVector<Function *, SmallVector<unsigned, 4>> Need;
  SmallVector<std::pair<Function *, unsigned>, 16> Work;
  auto Add = [&](Function *F, unsigned K) {
    SmallVector<unsigned, 4> &Ks = Need[F];
    if (is_contained(Ks, K))
      return;
    Ks.push_back(K);
    Work.push_back({F, K});
  };
  for (const AIRKernelParamRequest &Req : Requests)
    Add(Req.F, KeyIndex(Req));
  while (!Work.empty()) {
    auto [F, K] = Work.pop_back_val();
    for (User *U : F->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledOperand() != F)
        reportFatalUsageError(("AIR: function '" + F->getName() +
                               "' needs its kernel's " + Keys[K].Name +
                               " but is used other than by direct calls")
                                  .str()
                                  .c_str());
      Add(CI->getFunction(), K);
    }
  }
  for (auto &[F, Ks] : Need)
    llvm::sort(Ks);

  // Give each function the parameters it has not got yet.
  SmallVector<std::pair<Function *, Function *>, 8> Rewritten;
  for (auto &[F, Ks] : Need) {
    StringMap<Argument *> Have;
    SmallVector<unsigned, 4> New;
    for (unsigned K : Ks) {
      if (Argument *A = Existing ? Existing(*F, Keys[K].Key) : nullptr)
        Have[Keys[K].Key] = A;
      else
        New.push_back(K);
    }
    if (New.empty()) {
      R.Args[F] = std::move(Have);
      continue;
    }

    FunctionType *FT = F->getFunctionType();
    SmallVector<Type *, 8> Params(FT->params());
    for (unsigned K : New)
      Params.push_back(Keys[K].Ty);
    // (In F's place, keeping the module's order.)
    Function *NewF = Function::Create(
        FunctionType::get(FT->getReturnType(), Params, FT->isVarArg()),
        F->getLinkage(), F->getAddressSpace(), "");
    M.getFunctionList().insert(F->getIterator(), NewF);
    NewF->copyAttributesFrom(F);
    NewF->copyMetadata(F, 0);
    NewF->setCallingConv(F->getCallingConv());
    NewF->splice(NewF->begin(), F);
    for (auto [Old, NewArg] : zip(F->args(), NewF->args())) {
      NewArg.setName(Old.getName());
      Old.replaceAllUsesWith(&NewArg);
    }
    for (auto &E : Have)
      E.second = NewF->getArg(E.second->getArgNo());
    bool Kernel = !isCalledFunction(*F);
    for (auto [I, K] : enumerate(New)) {
      Argument *A = NewF->getArg(FT->getNumParams() + I);
      setAIRArgName(*A, Keys[K].Name);
      Have[Keys[K].Key] = A;
      if (Kernel)
        R.Added.push_back({NewF, Keys[K].Key, A});
    }
    // Metadata naming the function (a pre-baked !air.kernel) follows it.
    ValueAsMetadata::handleRAUW(F, NewF);
    NewF->takeName(F);
    R.Args[NewF] = std::move(Have);
    R.Replaced[F] = NewF;
    Rewritten.push_back({F, NewF});
  }

  // Each call of a function given parameters passes its caller's.
  for (auto [Old, NewF] : Rewritten) {
    SmallVector<CallInst *, 8> Calls;
    for (User *U : Old->users())
      Calls.push_back(cast<CallInst>(U));
    unsigned NumOld = Old->arg_size();
    for (CallInst *CI : Calls) {
      const StringMap<Argument *> &CallerArgs = R.Args[CI->getFunction()];
      SmallVector<Value *, 8> Args(CI->args());
      for (unsigned I = NumOld; I < NewF->arg_size(); ++I) {
        // The new parameters' keys, in order.
        Value *V = nullptr;
        for (auto &E : R.Args[NewF])
          if (E.second == NewF->getArg(I))
            V = CallerArgs.lookup(E.first());
        assert(V && "the caller has every key its callee has");
        Args.push_back(V);
      }
      auto *NC = CallInst::Create(NewF->getFunctionType(), NewF, Args, "",
                                  CI->getIterator());
      NC->setCallingConv(CI->getCallingConv());
      NC->setAttributes(CI->getAttributes());
      NC->setTailCallKind(CI->getTailCallKind());
      NC->copyMetadata(*CI);
      NC->takeName(CI);
      CI->replaceAllUsesWith(NC);
      CI->eraseFromParent();
    }
    Old->eraseFromParent();
  }
  return R;
}
