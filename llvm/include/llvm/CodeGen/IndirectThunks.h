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
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// ThunkInserter provides a common interface for injecting thunk functions into
/// a module.
///
/// A thunk is a small piece of code whose purpose is to provide an alternative
/// calling convention or security mitigation. The target-specific subclasses
/// will typically iterate over all functions in the module and call
/// `createThunkFunction` to ensure the required thunks exist.
template <typename Derived> class ThunkInserter {
  Derived &getDerived() { return *static_cast<Derived *>(this); }

protected:
  // Interface for subclasses to use.

  /// Create an empty thunk function. If the function already exists, this is a
  /// no-op.
  ///
  /// The new function will eventually be passed to populateThunk.
  void createThunkFunction(MachineModuleInfo &MMI, StringRef Name,
                           bool Comdat = true, StringRef TargetAttrs = "");

protected:
  // Interface for subclasses to implement.
  // Note: all functions are non-virtual and are called via getDerived().

  /// Returns common prefix for thunk function's names.
  const char *getThunkPrefix(); // undefined

  /// Checks if MF may use thunks (true - maybe, false - definitely not).
  bool mayUseThunk(const MachineFunction &MF); // undefined

  /// Rewrites the function if necessary, returns true if the current
  /// MachineFunction was modified.
  bool insertThunks(MachineModuleInfo &MMI, MachineFunction &MF); // undefined

  /// Populate the thunk function with instructions.
  void populateThunk(MachineFunction &MF); // undefined

public:
  // return `true` if `MMI` or `MF` was modified
  bool run(MachineModuleInfo &MMI, MachineFunction &MF);
};

template <typename Derived>
void ThunkInserter<Derived>::createThunkFunction(MachineModuleInfo &MMI,
                                                 StringRef Name, bool Comdat,
                                                 StringRef TargetAttrs) {
  assert(Name.starts_with(getDerived().getThunkPrefix()) &&
         "Created a thunk with an unexpected prefix!");

  Module &M = const_cast<Module &>(*MMI.getModule());
  if (M.getFunction(Name))
    return;

  LLVMContext &Ctx = M.getContext();
  auto *Type = FunctionType::get(Type::getVoidTy(Ctx), false);
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

  MachineFunction &MF = MMI.getOrCreateMachineFunction(*F);

  // Set MF properties. We never use vregs.
  MF.getProperties().setNoVRegs();
}

template <typename Derived>
bool ThunkInserter<Derived>::run(MachineModuleInfo &MMI, MachineFunction &MF) {
  // If MF is not a thunk, check to see if we need to insert a thunk.
  if (!MF.getName().starts_with(getDerived().getThunkPrefix())) {
    // Only add thunks if one of the functions may use them.
    if (!getDerived().mayUseThunk(MF))
      return false;

    return getDerived().insertThunks(MMI, MF);
  }

  // If this *is* a thunk function, we need to populate it with the correct MI.
  getDerived().populateThunk(MF);
  return true;
}

/// Generic Legacy Pass implementation wrapping one or more `ThunkInserter`s.
template <typename... Inserters>
class ThunkInserterLegacyPass : public MachineFunctionPass {
protected:
  std::tuple<Inserters...> TIs;

  ThunkInserterLegacyPass(char &ID) : MachineFunctionPass(ID) {}

public:
  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
    return runTIs(MMI, MF, TIs);
  }

private:
  template <typename... ThunkInserterT>
  static bool runTIs(MachineModuleInfo &MMI, MachineFunction &MF,
                     std::tuple<ThunkInserterT...> &ThunkInserters) {
    return (0 | ... | std::get<ThunkInserterT>(ThunkInserters).run(MMI, MF));
  }
};

/// Generic New Pass Manager implementation wrapping one or more
/// `ThunkInserter`s.
template <typename Derived, typename... Inserters>
class ThunkInserterPass : public PassInfoMixin<Derived> {
protected:
  std::tuple<Inserters...> TIs;

public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM) {
    auto &MMI = MFAM.getResult<MachineModuleAnalysis>(MF).getMMI();
    bool Changed = (0 | ... | std::get<Inserters>(TIs).run(MMI, MF));
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // namespace llvm

#endif
