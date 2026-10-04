//===- LowerCommentStringPass.cpp - Lower loadtime comment strings --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass keeps loadtime identifying strings alive through linking.
//
// A loadtime identifying string is a global variable carrying
// !loadtime_comment metadata. Clang produces such globals from two sources:
//
//  * #pragma comment(copyright, "..."): CodeGen creates a weak_odr hidden
//    unnamed_addr constant named __loadtime_comment_str_<hash> in the
//    __loadtime_comment section.
//
//  * -mloadtime-comment-vars=<names>: Sema attaches an implicit attribute to
//    each listed string variable, and CodeGen tags the variable's ordinary
//    definition. Its name, linkage, and section are unchanged.
//
// Both producers also add the global to llvm.compiler.used, which keeps it
// through IR optimization but not through linking.
//
// This pass attaches !implicit.ref metadata naming every such global to each
// function defined in the module. The PowerPC backend for XCOFF lowers the
// metadata to a .ref directive, which creates a relocation from the function's
// csect to the string's csect. The linker then retains the string for as long
// as it retains any function from the module.
//
// The pass runs only for XCOFF targets; elsewhere it is a no-op.
//
// Input IR (the pragma producer is shown; a -mloadtime-comment-vars global
// differs only in name, linkage, and section):
//   @__loadtime_comment_str_HASH = weak_odr hidden unnamed_addr constant
//     [N x i8] c"Copyright\00", section "__loadtime_comment", align 1,
//     !loadtime_comment !0
//   @llvm.compiler.used = appending global [1 x ptr]
//     [ptr @__loadtime_comment_str_HASH], section "llvm.metadata"
//
// Output IR: the globals are unchanged and every defined function gains
//   define i32 @func() !implicit.ref !1 { ... }
//   !1 = !{ptr @__loadtime_comment_str_HASH}
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#define DEBUG_TYPE "lower-comment-string"

using namespace llvm;

static cl::opt<bool>
    DisableLowerCommentString("disable-lower-comment-string", cl::ReallyHidden,
                              cl::desc("Disable LowerCommentString pass."),
                              cl::init(false));

static bool isSupportedTarget(const Module &M) {
  // The pass runs only for XCOFF targets; elsewhere it is a no-op.
  Triple T{M.getTargetTriple()};
  return T.isOSAIX();
}

PreservedAnalyses LowerCommentStringPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  if (DisableLowerCommentString || !isSupportedTarget(M))
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
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (GlobalValue *GV : LoadTimeCommentGlobals) {
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
