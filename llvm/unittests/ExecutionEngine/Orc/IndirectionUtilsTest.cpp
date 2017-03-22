//===- LazyEmittingLayerTest.cpp - Unit tests for the lazy emitting layer -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(IndirectionUtilsTest, MakeStub) {
  LLVMContext Context;
  ModuleBuilder MB(Context, "x86_64-apple-macosx10.10", "");
  Function *F = MB.createFunctionDecl<void(DummyStruct, DummyStruct)>("");
  SmallVector<AttributeList, 4> Attrs;
  Attrs.push_back(
      AttributeList::get(MB.getModule()->getContext(), 1U,
                         AttrBuilder().addAttribute(Attribute::StructRet)));
  Attrs.push_back(
      AttributeList::get(MB.getModule()->getContext(), 2U,
                         AttrBuilder().addAttribute(Attribute::ByVal)));
  Attrs.push_back(
      AttributeList::get(MB.getModule()->getContext(), ~0U,
                         AttrBuilder().addAttribute(Attribute::NoUnwind)));
  F->setAttributes(AttributeList::get(MB.getModule()->getContext(), Attrs));

  auto ImplPtr = orc::createImplPointer(*F->getType(), *MB.getModule(), "", nullptr);
  orc::makeStub(*F, *ImplPtr);

  auto II = F->getEntryBlock().begin();
  EXPECT_TRUE(isa<LoadInst>(*II)) << "First instruction of stub should be a load.";
  auto *Call = dyn_cast<CallInst>(std::next(II));
  EXPECT_TRUE(Call != nullptr) << "Second instruction of stub should be a call.";
  EXPECT_TRUE(Call->isTailCall()) << "Indirect call from stub should be tail call.";
  EXPECT_TRUE(Call->hasStructRetAttr())
    << "makeStub should propagate sret attr on 1st argument.";
  EXPECT_TRUE(Call->paramHasAttr(2U, Attribute::ByVal))
    << "makeStub should propagate byval attr on 2nd argument.";
}

}
