//===-- LowerCommentStringPass.cpp - Lower Comment string metadata -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass lowers the module-level comment string metadata emitted by Clang:
//
//     !comment_string.loadtime = !{!"Copyright ..."}
//
// into concrete, translation-unit-weak-hidden globals.
// This Pass is enabled only for AIX.
// For each module (translation unit), the pass performs the following:
//
//   1. Creates a null-terminated, weak_odr hidden constant string global
//      (`__loadtime_comment_str`) containing the copyright text. The backend
//      places this in the .text section of the object file.
//
//   2. Marks the string in `llvm.compiler.used` so it cannot be dropped by
//      optimization or LTO.
//
//   3. Attaches `!implicit.ref` metadata referencing the string to every
//      defined function in the module. The PowerPC AIX backend recognizes
//      this metadata and emits a `.ref` directive from the function to the
//      string, creating a concrete relocation that prevents the linker from
//      discarding the string (as long as the referencing symbol is kept).
//
//  Input IR:
//     !comment_string.loadtime = !{!"Copyright"}
//  Output IR:
//     @__loadtime_comment_str_HASH = weak_odr constant [N x i8]
//     c"Copyright\00"
//     @llvm.compiler.used = appending global [1 x ptr] [ptr
//     @__loadtime_comment_str_HASH]
//
//     define i32 @func() !implicit.ref !5 { ... }
//     !5 = !{ptr @__loadtime_comment_str_HASH}
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LowerCommentStringPass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#define DEBUG_TYPE "lower-comment-string"

using namespace llvm;

static cl::opt<bool>
    DisableCopyrightMetadata("disable-lower-comment-string", cl::ReallyHidden,
                             cl::desc("Disable LowerCommentString pass."),
                             cl::init(false));

static bool isSupportedTarget(const Module &M) {
  Triple T{M.getTargetTriple()};
  return T.isOSAIX();
}

PreservedAnalyses LowerCommentStringPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  if (DisableCopyrightMetadata || !isSupportedTarget(M))
    return PreservedAnalyses::all();

  LLVMContext &Ctx = M.getContext();

  // Collect all globals marked with !loadtime_comment metadata.
  SmallVector<GlobalValue *, 4> LoadTimeCommentGlobals;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.hasMetadata("loadtime_comment"))
      LoadTimeCommentGlobals.push_back(&GV);
  }

  if (LoadTimeCommentGlobals.empty())
    return PreservedAnalyses::all();

  // Add implicit.ref from every function to each loadtime comment global.
  for (GlobalValue *GV : LoadTimeCommentGlobals) {
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      Metadata *Ops[] = {ConstantAsMetadata::get(GV)};
      MDNode *NewMD = MDNode::get(Ctx, Ops);
      F.addMetadata(LLVMContext::MD_implicit_ref, *NewMD);

      LLVM_DEBUG(
          dbgs() << "[loadtime-comment] attached implicit.ref to function: "
                 << F.getName() << " for global: " << GV->getName() << "\n");
    }
  }

  LLVM_DEBUG(dbgs() << "[loadtime-comment] processed "
                    << LoadTimeCommentGlobals.size()
                    << " loadtime comment globals\n");

  return PreservedAnalyses::all();
}
