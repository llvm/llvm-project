//===- FatLtoCleanup.cpp - clean up IR for the FatLTO pipeline --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines operations used to clean up IR for the FatLTO pipeline.
// Instrumentation that is beneficial for bitcode sections used in LTO may
// need to be cleaned up to finish non-LTO compilation. llvm.checked.load is
// an example of an instruction that we want to preserve for LTO, but is
// incorrect to leave unchanged during the per-TU compilation in FatLTO.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/FatLTOCleanup.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Use.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "fatlto-cleanup"

namespace {
// Replaces uses of llvm.type.checked.load instructions with unchecked loads.
// In essence, we're undoing the frontends instrumentation, since it isn't
// correct for the non-LTO part of a FatLTO object.
//
// llvm.type.checked.load instruction sequences always have a particular form:
//
// clang-format off
//
//   %0 = tail call { ptr, i1 } @llvm.type.checked.load(ptr %vtable, i32 0, metadata !"foo"), !nosanitize !0
//   %1 = extractvalue { ptr, i1 } %0, 1, !nosanitize !0
//   br i1 %1, label %cont2, label %trap1, !nosanitize !0
//
// trap1:                                            ; preds = %entry
//   tail call void @llvm.ubsantrap(i8 2) #3, !nosanitize !0
//   unreachable, !nosanitize !0
//
// cont2:                                            ; preds = %entry
//   %2 = extractvalue { ptr, i1 } %0, 0, !nosanitize !0
//   %call = tail call noundef i64 %2(ptr noundef nonnull align 8 dereferenceable(8) %p1) #4
//
// clang-format on
//
// In this sequence, the vtable pointer is first loaded and checked against some
// metadata. The result indicates failure, then the program traps. On the
// success path, the pointer is used to make an indirect call to the function
// pointer loaded from the vtable.
//
// Since we won't be able to lower this correctly later in non-LTO builds, we
// need to drop the special load and trap, and emit a normal load of the
// function pointer from the vtable.
//
// This is straight forward, since the checked load can be replaced w/ a load
// of the vtable pointer and a GEP instruction to index into the vtable and get
// the correct method/function pointer. We replace the "check" with a constant
// indicating success, which allows later passes to simplify control flow and
// remove any now dead instructions.
//
// This logic holds for both llvm.type.checked.load and
// llvm.type.checked.load.relative instructions.
static bool cleanUpTypeCheckedLoad(Module &M, Function &CheckedLoadFn,
                                   bool IsRelative) {
  bool Changed = false;
  for (User *User : llvm::make_early_inc_range(CheckedLoadFn.users())) {
    Instruction *I = dyn_cast<Instruction>(User);
    if (!I)
      continue;
    IRBuilder<> IRB(I);
    Value *Ptr = I->getOperand(0);
    Value *Offset = I->getOperand(1);
    Type *PtrTy = I->getType()->getStructElementType(0);
    ConstantInt *True = ConstantInt::getTrue(M.getContext());
    Instruction *Load;
    if (IsRelative) {
      Load =
          IRB.CreateIntrinsic(Intrinsic::load_relative, {Offset->getType()},
                              {Ptr, Offset}, /*FMFSource=*/nullptr, "rel_load");
    } else {
      Value *PtrAdd = IRB.CreatePtrAdd(Ptr, Offset);
      Load = IRB.CreateLoad(PtrTy, PtrAdd, "vfunc");
    }

    Value *Replacement = PoisonValue::get(I->getType());
    Replacement = IRB.CreateInsertValue(Replacement, True, {1});
    Replacement = IRB.CreateInsertValue(Replacement, Load, {0});
    I->replaceAllUsesWith(Replacement);

    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": erase " << *I << "\n");
    I->eraseFromParent();
    Changed = true;
  }
  if (Changed)
    CheckedLoadFn.eraseFromParent();
  return Changed;
}
} // namespace

PreservedAnalyses FatLtoCleanup::run(Module &M, ModuleAnalysisManager &AM) {
  Function *TypeCheckedLoadFn =
      Intrinsic::getDeclarationIfExists(&M, Intrinsic::type_checked_load);
  Function *TypeCheckedLoadRelFn = Intrinsic::getDeclarationIfExists(
      &M, Intrinsic::type_checked_load_relative);

  bool Changed = false;
  if (TypeCheckedLoadFn)
    Changed |= cleanUpTypeCheckedLoad(M, *TypeCheckedLoadFn, false);
  if (TypeCheckedLoadRelFn)
    Changed |= cleanUpTypeCheckedLoad(M, *TypeCheckedLoadRelFn, true);

  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
