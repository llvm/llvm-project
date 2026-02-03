//===-- LowerCommentStringPass.cpp - Lower Comment string metadata -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// LowerCommentStringPass pass lowers module-level comment string metadata
// emitted by Clang:
//
//     !comment_string.loadtime = !{!"Copyright ..."}
//
// into concrete, translation-unit–local globals.
// This Pass is enabled only for AIX.
// For each module (translation unit), the pass performs the following:
//
//   1. Creates a null-terminated, internal constant string global
//      (`__loadtime_comment_str`) containing the copyright text in
//      `__loadtime_comment` section.
//
//   2. Marks the string in `llvm.used` so it cannot be dropped by
//      optimization or LTO.
//
//   3. Attaches `!implicit.ref` metadata referencing the string to every
//      defined function in the module. The PowerPC AIX backend recognizes
//      this metadata and emits a `.ref` directive from the function to the
//      string, creating a concrete relocation that prevents the linker from
//      discarding it (as long as the referencing symbol is kept).
//
//  Input IR:
//     !comment_string.loadtime = !{!"Copyright"}
//  Output IR:
//     @__loadtime_comment_str = internal constant [N x i8] c"Copyright\00",
//                          section "__loadtime_comment"
//     @llvm.used = appending global [1 x ptr] [ptr @__loadtime_comment_str]
//
//     define i32 @func() !implicit.ref !5 { ... }
//     !5 = !{ptr @__loadtime_comment_str}
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LowerCommentStringPass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
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

static bool isAIXTriple(const Module &M) {
  return Triple(M.getTargetTriple()).isOSAIX();
}

PreservedAnalyses LowerCommentStringPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  if (DisableCopyrightMetadata || !isAIXTriple(M))
    return PreservedAnalyses::all();

  LLVMContext &Ctx = M.getContext();

  // Single-metadata: !comment_string.loadtime = !{!0}
  // Each operand node is expected to have one MDString operand.
  NamedMDNode *MD = M.getNamedMetadata("comment_string.loadtime");
  if (!MD || MD->getNumOperands() == 0)
    return PreservedAnalyses::all();

  // At this point we are guarateed that one TU contains a single copyright
  // metadata entry. Create TU-local string global for that metadata entry.
  MDNode *MdNode = MD->getOperand(0);
  if (!MdNode || MdNode->getNumOperands() == 0)
    return PreservedAnalyses::all();

  auto *MdString = dyn_cast_or_null<MDString>(MdNode->getOperand(0));
  if (!MdString)
    return PreservedAnalyses::all();

  StringRef Text = MdString->getString();
  if (Text.empty())
    return PreservedAnalyses::all();

  // 1. Create a single NULL-terminated string global
  Constant *StrInit = ConstantDataArray::getString(Ctx, Text, /*AddNull=*/true);

  // Internal, constant, TU-local--avoids duplicate symbol issues across TUs.
  auto *StrGV = new GlobalVariable(M, StrInit->getType(),
                                   /*isConstant=*/true,
                                   GlobalValue::InternalLinkage, StrInit,
                                   /*Name=*/"__loadtime_comment_str");
  // Set unnamed_addr to allow the linker to merge identical strings
  StrGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  StrGV->setAlignment(Align(1));
  // Place in the "__loadtime_comment" section.
  // The GV is constant, so we expect a read-only section.
  StrGV->setSection("__loadtime_comment");

  // 2. Add the string to llvm.used to prevent LLVM optimization/LTO passes from
  // removing it.
  appendToUsed(M, {StrGV});

  // 3. Attach !implicit ref to every defined function
  // Create a metadata node pointing to the copyright string:
  //   !N = !{ptr @__loadtime_comment_str}
  Metadata *Ops[] = {ConstantAsMetadata::get(StrGV)};
  MDNode *ImplicitRefMD = MDNode::get(Ctx, Ops);

  // Lambda to attach implicit.ref metadata to a function.
  auto AddImplicitRef = [&](Function &F) {
    if (F.isDeclaration())
      return;
    // Attach the implicit.ref metadata to the function
    F.setMetadata("implicit.ref", ImplicitRefMD);
    LLVM_DEBUG(dbgs() << "[copyright] attached implicit.ref to function:  "
                      << F.getName() << "\n");
  };

  // Process all functions in the module
  for (Function &F : M)
    AddImplicitRef(F);

  // Cleanup the processed metadata.
  MD->eraseFromParent();
  LLVM_DEBUG(dbgs() << "[copyright] created string and anchor for module\n");

  return PreservedAnalyses::all();
}
