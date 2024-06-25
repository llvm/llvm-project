//===- llvm/unittest/IR/AttributesTest.cpp - Attributes unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Attributes.h"
#include "llvm-c/Core.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

TEST(Attributes, Uniquing) {
  LLVMContext C;

  Attribute AttrA = Attribute::get(C, Attribute::AlwaysInline);
  Attribute AttrB = Attribute::get(C, Attribute::AlwaysInline);
  EXPECT_EQ(AttrA, AttrB);

  AttributeList ASs[] = {AttributeList::get(C, 1, Attribute::ZExt),
                         AttributeList::get(C, 2, Attribute::SExt)};

  AttributeList SetA = AttributeList::get(C, ASs);
  AttributeList SetB = AttributeList::get(C, ASs);
  EXPECT_EQ(SetA, SetB);
}

TEST(Attributes, Ordering) {
  LLVMContext C;

  Attribute Align4 = Attribute::get(C, Attribute::Alignment, 4);
  Attribute Align5 = Attribute::get(C, Attribute::Alignment, 5);
  Attribute Deref4 = Attribute::get(C, Attribute::Dereferenceable, 4);
  Attribute Deref5 = Attribute::get(C, Attribute::Dereferenceable, 5);
  EXPECT_TRUE(Align4 < Align5);
  EXPECT_TRUE(Align4 < Deref4);
  EXPECT_TRUE(Align4 < Deref5);
  EXPECT_TRUE(Align5 < Deref4);

  Attribute ByVal = Attribute::get(C, Attribute::ByVal, Type::getInt32Ty(C));
  EXPECT_FALSE(ByVal < Attribute::get(C, Attribute::ZExt));
  EXPECT_TRUE(ByVal < Align4);
  EXPECT_FALSE(ByVal < ByVal);

  AttributeList ASs[] = {AttributeList::get(C, 2, Attribute::ZExt),
                         AttributeList::get(C, 1, Attribute::SExt)};

  AttributeList SetA = AttributeList::get(C, ASs);
  AttributeList SetB =
      SetA.removeParamAttributes(C, 0, ASs[1].getParamAttrs(0));
  EXPECT_NE(SetA, SetB);
}

TEST(Attributes, AddAttributes) {
  LLVMContext C;
  AttributeList AL;
  AttrBuilder B(C);
  B.addAttribute(Attribute::NoReturn);
  AL = AL.addFnAttributes(C, AttrBuilder(C, AttributeSet::get(C, B)));
  EXPECT_TRUE(AL.hasFnAttr(Attribute::NoReturn));
  B.clear();
  B.addAttribute(Attribute::SExt);
  AL = AL.addRetAttributes(C, B);
  EXPECT_TRUE(AL.hasRetAttr(Attribute::SExt));
  EXPECT_TRUE(AL.hasFnAttr(Attribute::NoReturn));
}

TEST(Attributes, RemoveAlign) {
  LLVMContext C;

  Attribute AlignAttr = Attribute::getWithAlignment(C, Align(8));
  Attribute StackAlignAttr = Attribute::getWithStackAlignment(C, Align(32));
  AttrBuilder B_align_readonly(C);
  B_align_readonly.addAttribute(AlignAttr);
  B_align_readonly.addAttribute(Attribute::ReadOnly);
  AttributeMask B_align;
  B_align.addAttribute(AlignAttr);
  AttrBuilder B_stackalign_optnone(C);
  B_stackalign_optnone.addAttribute(StackAlignAttr);
  B_stackalign_optnone.addAttribute(Attribute::OptimizeNone);
  AttributeMask B_stackalign;
  B_stackalign.addAttribute(StackAlignAttr);

  AttributeSet AS = AttributeSet::get(C, B_align_readonly);
  EXPECT_TRUE(AS.getAlignment() == MaybeAlign(8));
  EXPECT_TRUE(AS.hasAttribute(Attribute::ReadOnly));
  AS = AS.removeAttribute(C, Attribute::Alignment);
  EXPECT_FALSE(AS.hasAttribute(Attribute::Alignment));
  EXPECT_TRUE(AS.hasAttribute(Attribute::ReadOnly));
  AS = AttributeSet::get(C, B_align_readonly);
  AS = AS.removeAttributes(C, B_align);
  EXPECT_TRUE(AS.getAlignment() == std::nullopt);
  EXPECT_TRUE(AS.hasAttribute(Attribute::ReadOnly));

  AttributeList AL;
  AL = AL.addParamAttributes(C, 0, B_align_readonly);
  AL = AL.addRetAttributes(C, B_stackalign_optnone);
  EXPECT_TRUE(AL.hasRetAttrs());
  EXPECT_TRUE(AL.hasRetAttr(Attribute::StackAlignment));
  EXPECT_TRUE(AL.hasRetAttr(Attribute::OptimizeNone));
  EXPECT_TRUE(AL.getRetStackAlignment() == MaybeAlign(32));
  EXPECT_TRUE(AL.hasParamAttrs(0));
  EXPECT_TRUE(AL.hasParamAttr(0, Attribute::Alignment));
  EXPECT_TRUE(AL.hasParamAttr(0, Attribute::ReadOnly));
  EXPECT_TRUE(AL.getParamAlignment(0) == MaybeAlign(8));

  AL = AL.removeParamAttribute(C, 0, Attribute::Alignment);
  EXPECT_FALSE(AL.hasParamAttr(0, Attribute::Alignment));
  EXPECT_TRUE(AL.hasParamAttr(0, Attribute::ReadOnly));
  EXPECT_TRUE(AL.hasRetAttr(Attribute::StackAlignment));
  EXPECT_TRUE(AL.hasRetAttr(Attribute::OptimizeNone));
  EXPECT_TRUE(AL.getRetStackAlignment() == MaybeAlign(32));

  AL = AL.removeRetAttribute(C, Attribute::StackAlignment);
  EXPECT_FALSE(AL.hasParamAttr(0, Attribute::Alignment));
  EXPECT_TRUE(AL.hasParamAttr(0, Attribute::ReadOnly));
  EXPECT_FALSE(AL.hasRetAttr(Attribute::StackAlignment));
  EXPECT_TRUE(AL.hasRetAttr(Attribute::OptimizeNone));

  AttributeList AL2;
  AL2 = AL2.addParamAttributes(C, 0, B_align_readonly);
  AL2 = AL2.addRetAttributes(C, B_stackalign_optnone);

  AL2 = AL2.removeParamAttributes(C, 0, B_align);
  EXPECT_FALSE(AL2.hasParamAttr(0, Attribute::Alignment));
  EXPECT_TRUE(AL2.hasParamAttr(0, Attribute::ReadOnly));
  EXPECT_TRUE(AL2.hasRetAttr(Attribute::StackAlignment));
  EXPECT_TRUE(AL2.hasRetAttr(Attribute::OptimizeNone));
  EXPECT_TRUE(AL2.getRetStackAlignment() == MaybeAlign(32));

  AL2 = AL2.removeRetAttributes(C, B_stackalign);
  EXPECT_FALSE(AL2.hasParamAttr(0, Attribute::Alignment));
  EXPECT_TRUE(AL2.hasParamAttr(0, Attribute::ReadOnly));
  EXPECT_FALSE(AL2.hasRetAttr(Attribute::StackAlignment));
  EXPECT_TRUE(AL2.hasRetAttr(Attribute::OptimizeNone));
}

TEST(Attributes, AddMatchingAlignAttr) {
  LLVMContext C;
  AttributeList AL;
  AL = AL.addParamAttribute(C, 0, Attribute::getWithAlignment(C, Align(8)));
  AL = AL.addParamAttribute(C, 1, Attribute::getWithAlignment(C, Align(32)));
  EXPECT_EQ(Align(8), AL.getParamAlignment(0));
  EXPECT_EQ(Align(32), AL.getParamAlignment(1));

  AttrBuilder B(C);
  B.addAttribute(Attribute::NonNull);
  B.addAlignmentAttr(8);
  AL = AL.addParamAttributes(C, 0, B);
  EXPECT_EQ(Align(8), AL.getParamAlignment(0));
  EXPECT_EQ(Align(32), AL.getParamAlignment(1));
  EXPECT_TRUE(AL.hasParamAttr(0, Attribute::NonNull));
}

TEST(Attributes, EmptyGet) {
  LLVMContext C;
  AttributeList EmptyLists[] = {AttributeList(), AttributeList()};
  AttributeList AL = AttributeList::get(C, EmptyLists);
  EXPECT_TRUE(AL.isEmpty());
}

TEST(Attributes, OverflowGet) {
  LLVMContext C;
  std::pair<unsigned, Attribute> Attrs[] = { { AttributeList::ReturnIndex, Attribute::get(C, Attribute::SExt) },
                                             { AttributeList::FunctionIndex, Attribute::get(C, Attribute::ReadOnly) } };
  AttributeList AL = AttributeList::get(C, Attrs);
  EXPECT_EQ(2U, AL.getNumAttrSets());
}

TEST(Attributes, StringRepresentation) {
  LLVMContext C;
  StructType *Ty = StructType::create(Type::getInt32Ty(C), "mystruct");

  // Insufficiently careful printing can result in byval(%mystruct = { i32 })
  Attribute A = Attribute::getWithByValType(C, Ty);
  EXPECT_EQ(A.getAsString(), "byval(%mystruct)");

  A = Attribute::getWithByValType(C, Type::getInt32Ty(C));
  EXPECT_EQ(A.getAsString(), "byval(i32)");
}

TEST(Attributes, HasParentContext) {
  LLVMContext C1, C2;

  {
    Attribute Attr1 = Attribute::get(C1, Attribute::AlwaysInline);
    Attribute Attr2 = Attribute::get(C2, Attribute::AlwaysInline);
    EXPECT_TRUE(Attr1.hasParentContext(C1));
    EXPECT_FALSE(Attr1.hasParentContext(C2));
    EXPECT_FALSE(Attr2.hasParentContext(C1));
    EXPECT_TRUE(Attr2.hasParentContext(C2));
  }

  {
    AttributeSet AS1 = AttributeSet::get(
        C1, ArrayRef(Attribute::get(C1, Attribute::NoReturn)));
    AttributeSet AS2 = AttributeSet::get(
        C2, ArrayRef(Attribute::get(C2, Attribute::NoReturn)));
    EXPECT_TRUE(AS1.hasParentContext(C1));
    EXPECT_FALSE(AS1.hasParentContext(C2));
    EXPECT_FALSE(AS2.hasParentContext(C1));
    EXPECT_TRUE(AS2.hasParentContext(C2));
  }

  {
    AttributeList AL1 = AttributeList::get(C1, 1, Attribute::ZExt);
    AttributeList AL2 = AttributeList::get(C2, 1, Attribute::ZExt);
    EXPECT_TRUE(AL1.hasParentContext(C1));
    EXPECT_FALSE(AL1.hasParentContext(C2));
    EXPECT_FALSE(AL2.hasParentContext(C1));
    EXPECT_TRUE(AL2.hasParentContext(C2));
  }
}

TEST(Attributes, AttributeListPrinting) {
  LLVMContext C;

  {
    std::string S;
    raw_string_ostream OS(S);
    AttributeList AL;
    AL.addFnAttribute(C, Attribute::AlwaysInline).print(OS);
    EXPECT_EQ(S, "AttributeList[\n"
                 "  { function => alwaysinline }\n"
                 "]\n");
  }

  {
    std::string S;
    raw_string_ostream OS(S);
    AttributeList AL;
    AL.addRetAttribute(C, Attribute::SExt).print(OS);
    EXPECT_EQ(S, "AttributeList[\n"
                 "  { return => signext }\n"
                 "]\n");
  }

  {
    std::string S;
    raw_string_ostream OS(S);
    AttributeList AL;
    AL.addParamAttribute(C, 5, Attribute::ZExt).print(OS);
    EXPECT_EQ(S, "AttributeList[\n"
                 "  { arg(5) => zeroext }\n"
                 "]\n");
  }
}

TEST(Attributes, MismatchedABIAttrs) {
  const char *IRString = R"IR(
    declare void @f1(i32* byval(i32))
    define void @g() {
      call void @f1(i32* null)
      ret void
    }
    declare void @f2(i32* preallocated(i32))
    define void @h() {
      call void @f2(i32* null)
      ret void
    }
    declare void @f3(i32* inalloca(i32))
    define void @i() {
      call void @f3(i32* null)
      ret void
    }
  )IR";

  SMDiagnostic Err;
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssemblyString(IRString, Err, Context);
  ASSERT_TRUE(M);

  {
    auto *I = cast<CallBase>(&M->getFunction("g")->getEntryBlock().front());
    ASSERT_TRUE(I->isByValArgument(0));
    ASSERT_TRUE(I->getParamByValType(0));
  }
  {
    auto *I = cast<CallBase>(&M->getFunction("h")->getEntryBlock().front());
    ASSERT_TRUE(I->getParamPreallocatedType(0));
  }
  {
    auto *I = cast<CallBase>(&M->getFunction("i")->getEntryBlock().front());
    ASSERT_TRUE(I->isInAllocaArgument(0));
    ASSERT_TRUE(I->getParamInAllocaType(0));
  }
}

TEST(Attributes, RemoveParamAttributes) {
  LLVMContext C;
  AttributeList AL;
  AL = AL.addParamAttribute(C, 1, Attribute::NoUndef);
  EXPECT_EQ(AL.getNumAttrSets(), 4U);
  AL = AL.addParamAttribute(C, 3, Attribute::NonNull);
  EXPECT_EQ(AL.getNumAttrSets(), 6U);
  AL = AL.removeParamAttributes(C, 3);
  EXPECT_EQ(AL.getNumAttrSets(), 4U);
  AL = AL.removeParamAttribute(C, 1, Attribute::NoUndef);
  EXPECT_EQ(AL.getNumAttrSets(), 0U);
}

TEST(Attributes, ConstantRangeAttributeCAPI) {
  LLVMContext C;
  {
    const unsigned NumBits = 8;
    const uint64_t LowerWords[] = {0};
    const uint64_t UpperWords[] = {42};

    ConstantRange Range(APInt(NumBits, ArrayRef(LowerWords)),
                        APInt(NumBits, ArrayRef(UpperWords)));

    Attribute RangeAttr = Attribute::get(C, Attribute::Range, Range);
    auto OutAttr = unwrap(LLVMCreateConstantRangeAttribute(
        wrap(&C), Attribute::Range, NumBits, LowerWords, UpperWords));
    EXPECT_EQ(OutAttr, RangeAttr);
  }
  {
    const unsigned NumBits = 128;
    const uint64_t LowerWords[] = {1, 1};
    const uint64_t UpperWords[] = {42, 42};

    ConstantRange Range(APInt(NumBits, ArrayRef(LowerWords)),
                        APInt(NumBits, ArrayRef(UpperWords)));

    Attribute RangeAttr = Attribute::get(C, Attribute::Range, Range);
    auto OutAttr = unwrap(LLVMCreateConstantRangeAttribute(
        wrap(&C), Attribute::Range, NumBits, LowerWords, UpperWords));
    EXPECT_EQ(OutAttr, RangeAttr);
  }
}

TEST(Attributes, CalleeAttributes) {
  const char *IRString = R"IR(
    declare void @f1(i32 %i)
    declare void @f2(i32 range(i32 1, 2) %i)

    define void @g1(i32 %i) {
      call void @f1(i32 %i)
      ret void
    }
    define void @g2(i32 %i) {
      call void @f2(i32 %i)
      ret void
    }
    define void @g3(i32 %i) {
      call void @f1(i32 range(i32 3, 4) %i)
      ret void
    }
    define void @g4(i32 %i) {
      call void @f2(i32 range(i32 3, 4) %i)
      ret void
    }
  )IR";

  SMDiagnostic Err;
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssemblyString(IRString, Err, Context);
  ASSERT_TRUE(M);

  {
    auto *I = cast<CallBase>(&M->getFunction("g1")->getEntryBlock().front());
    ASSERT_FALSE(I->getParamAttr(0, Attribute::Range).isValid());
  }
  {
    auto *I = cast<CallBase>(&M->getFunction("g2")->getEntryBlock().front());
    ASSERT_TRUE(I->getParamAttr(0, Attribute::Range).isValid());
  }
  {
    auto *I = cast<CallBase>(&M->getFunction("g3")->getEntryBlock().front());
    ASSERT_TRUE(I->getParamAttr(0, Attribute::Range).isValid());
  }
  {
    auto *I = cast<CallBase>(&M->getFunction("g4")->getEntryBlock().front());
    ASSERT_TRUE(I->getParamAttr(0, Attribute::Range).isValid());
  }
}

} // end anonymous namespace
