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
#include "llvm/Transforms/IPO/LowerTypeTests.h"

using namespace llvm;

#define DEBUG_TYPE "fatlto-cleanup"

PreservedAnalyses FatLtoCleanup::run(Module &M, ModuleAnalysisManager &AM) {
  // Undo the frontend checked-load instrumentation: it is not correct for
  // the non-LTO part of a FatLTO object, where whole-program metadata is
  // unavailable, and the backend cannot select these intrinsics anyway.
  Function *TypeCheckedLoadFn =
      Intrinsic::getDeclarationIfExists(&M, Intrinsic::type_checked_load);
  Function *TypeCheckedLoadRelFn = Intrinsic::getDeclarationIfExists(
      &M, Intrinsic::type_checked_load_relative);

  bool Changed = false;
  if (TypeCheckedLoadFn)
    Changed |=
        lowertypetests::cleanUpTypeCheckedLoad(M, *TypeCheckedLoadFn, false);
  if (TypeCheckedLoadRelFn)
    Changed |=
        lowertypetests::cleanUpTypeCheckedLoad(M, *TypeCheckedLoadRelFn, true);

  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
