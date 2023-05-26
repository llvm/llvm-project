//===---- IndirectThunks.h - Indirect Thunk Base Class ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Contains a base class for Passes that inject an MI thunk.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_INDIRECTTHUNKS_H
#define LLVM_CODEGEN_INDIRECTTHUNKS_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace llvm {

template <typename Derived, typename InsertedThunksTy = bool>
class ThunkInserter {
  Derived &getDerived() { return *static_cast<Derived *>(this); }

protected:
  // A variable used to track whether (and possible which) thunks have been
  // inserted so far. InsertedThunksTy is usually a bool, but can be other types
  // to represent more than one type of thunk. Requires an |= operator to
  // accumulate results.
  InsertedThunksTy InsertedThunks;
  void doInitialization(Module &M) {}
  void createThunkFunction(MachineModuleInfo &MMI, StringRef Name,
                           bool Comdat = true, StringRef TargetAttrs = "");

public:
  void init(Module &M) {
    InsertedThunks = InsertedThunksTy{};
    getDerived().doInitialization(M);
  }
  // return `true` if `MMI` or `MF` was modified
  bool run(MachineModuleInfo &MMI, MachineFunction &MF);
};

template <typename Derived, typename InsertedThunksTy>
void ThunkInserter<Derived, InsertedThunksTy>::createThunkFunction(
    MachineModuleInfo &MMI, StringRef Name, bool Comdat,
    StringRef TargetAttrs) {
  assert(Name.startswith(getDerived().getThunkPrefix()) &&
         "Created a thunk with an unexpected prefix!");

  Module &M = const_cast<Module &>(*MMI.getModule());
  LLVMContext &Ctx = M.getContext();
  auto Type = FunctionType::get(Type::getVoidTy(Ctx), false);
  Function *F = Function::Create(Type,
                                 Comdat ? GlobalValue::LinkOnceODRLinkage
                                        : GlobalValue::InternalLinkage,
                                 Name, &M);
  if (Comdat) {
    F->setVisibility(GlobalValue::HiddenVisibility);
    F->setComdat(M.getOrInsertComdat(Name));
  }

  // Add Attributes so that we don't create a frame, unwind information, or
  // inline.
  AttrBuilder B(Ctx);
  B.addAttribute(llvm::Attribute::NoUnwind);
  B.addAttribute(llvm::Attribute::Naked);
  if (TargetAttrs != "")
    B.addAttribute("target-features", TargetAttrs);
  F->addFnAttrs(B);

  // Populate our function a bit so that we can verify.
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", F);
  IRBuilder<> Builder(Entry);

  Builder.CreateRetVoid();

  // MachineFunctions aren't created automatically for the IR-level constructs
  // we already made. Create them and insert them into the module.
  MachineFunction &MF = MMI.getOrCreateMachineFunction(*F);
  // A MachineBasicBlock must not be created for the Entry block; code
  // generation from an empty naked function in C source code also does not
  // generate one.  At least GlobalISel asserts if this invariant isn't
  // respected.

  // Set MF properties. We never use vregs...
  MF.getProperties().set(MachineFunctionProperties::Property::NoVRegs);
}

template <typename Derived, typename InsertedThunksTy>
bool ThunkInserter<Derived, InsertedThunksTy>::run(MachineModuleInfo &MMI,
                                                   MachineFunction &MF) {
  // If MF is not a thunk, check to see if we need to insert a thunk.
  if (!MF.getName().startswith(getDerived().getThunkPrefix())) {
    // Only add a thunk if one of the functions has the corresponding feature
    // enabled in its subtarget, and doesn't enable external thunks. The target
    // can use InsertedThunks to detect whether relevant thunks have already
    // been inserted.
    // FIXME: Conditionalize on indirect calls so we don't emit a thunk when
    // nothing will end up calling it.
    // FIXME: It's a little silly to look at every function just to enumerate
    // the subtargets, but eventually we'll want to look at them for indirect
    // calls, so maybe this is OK.
    if (!getDerived().mayUseThunk(MF, InsertedThunks))
      return false;

    InsertedThunks |= getDerived().insertThunks(MMI, MF);
    return true;
  }

  // If this *is* a thunk function, we need to populate it with the correct MI.
  getDerived().populateThunk(MF);
  return true;
}

} // namespace llvm

#endif
