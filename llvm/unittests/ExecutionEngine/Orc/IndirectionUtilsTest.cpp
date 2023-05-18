//===- LazyEmittingLayerTest.cpp - Unit tests for the lazy emitting layer -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "OrcTestCommon.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(IndirectionUtilsTest, MakeStub) {
  LLVMContext Context;
  ModuleBuilder MB(Context, "x86_64-apple-macosx10.10", "");
  StructType *ArgTy = getDummyStructTy(Context);
  Type *ArgPtrTy = PointerType::getUnqual(ArgTy);
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Context), {ArgPtrTy, ArgPtrTy}, false);
  Function *F = MB.createFunctionDecl(FTy, "");
  AttributeSet FnAttrs = AttributeSet::get(
      Context, AttrBuilder(Context).addAttribute(Attribute::NoUnwind));
  AttributeSet RetAttrs; // None
  AttributeSet ArgAttrs[2] = {
      AttributeSet::get(Context, AttrBuilder(Context).addStructRetAttr(ArgTy)),
      AttributeSet::get(Context, AttrBuilder(Context).addByValAttr(ArgTy)),
  };
  F->setAttributes(AttributeList::get(Context, FnAttrs, RetAttrs, ArgAttrs));

  auto ImplPtr = orc::createImplPointer(*F->getType(), *MB.getModule(), "", nullptr);
  orc::makeStub(*F, *ImplPtr);

  auto II = F->getEntryBlock().begin();
  EXPECT_TRUE(isa<LoadInst>(*II)) << "First instruction of stub should be a load.";
  auto *Call = dyn_cast<CallInst>(std::next(II));
  EXPECT_TRUE(Call != nullptr) << "Second instruction of stub should be a call.";
  EXPECT_TRUE(Call->isTailCall()) << "Indirect call from stub should be tail call.";
  EXPECT_TRUE(Call->hasStructRetAttr())
    << "makeStub should propagate sret attr on 1st argument.";
  EXPECT_TRUE(Call->paramHasAttr(1U, Attribute::ByVal))
    << "makeStub should propagate byval attr on 2nd argument.";
}

}
