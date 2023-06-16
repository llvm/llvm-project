//===- unittests/rocm-gdb-symbols/MetadataTest.cpp - Metadata unit tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

class MetadataTest : public testing::Test {
public:
  MetadataTest() {}

protected:
  LLVMContext Context;
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *Int64Ty = Type::getInt64Ty(Context);
};

typedef MetadataTest DIExprBuilderTest;

TEST_F(DIExprBuilderTest, EmptyBuilderGet) {
  DIExprBuilder BuilderA(Context);
  DIExpr *ExprA = BuilderA.intoExpr();
  EXPECT_NE(ExprA, nullptr);
  DIExprBuilder BuilderB(Context);
  EXPECT_EQ(ExprA, BuilderB.intoExpr());
}

TEST_F(DIExprBuilderTest, NonUnique) {
  DIExprBuilder BuilderA(Context);
  BuilderA.append<DIOp::Referrer>(Int32Ty);
  DIExprBuilder BuilderB(Context);
  BuilderB.append<DIOp::Referrer>(Int32Ty);
  EXPECT_EQ(BuilderA.intoExpr(), BuilderB.intoExpr());
}

TEST_F(DIExprBuilderTest, Unique) {
  DIExprBuilder BuilderA(Context);
  BuilderA.append<DIOp::Referrer>(Int32Ty);
  DIExprBuilder BuilderB(Context);
  BuilderB.append<DIOp::Referrer>(Int64Ty);
  EXPECT_NE(BuilderA.intoExpr(), BuilderB.intoExpr());
}

TEST_F(DIExprBuilderTest, EquivalentAppends) {
  DIExprBuilder BuilderA(Context);
  BuilderA.append<DIOp::Referrer>(Int64Ty);
  DIExprBuilder BuilderB(Context);
  BuilderB.append(DIOp::Referrer(Int64Ty));
  DIExprBuilder BuilderC(Context);
  BuilderC.append(DIOp::Variant(DIOp::Referrer(Int64Ty)));
  DIExpr *ExprA = BuilderA.intoExpr();
  DIExpr *ExprB = BuilderB.intoExpr();
  DIExpr *ExprC = BuilderC.intoExpr();
  EXPECT_EQ(ExprA, ExprB);
  EXPECT_EQ(ExprB, ExprC);
}

TEST_F(DIExprBuilderTest, Iterator) {
  DIExprBuilder Builder(Context);
  DIOp::Variant Op(std::in_place_type<DIOp::Referrer>, Int32Ty);
  DIExpr *Expr = Builder.append(Op).intoExpr();
  DIExprBuilder ViewBuilder = Expr->builder();
  DIExprBuilder::Iterator I = ViewBuilder.begin();
  DIExprBuilder::Iterator E = ViewBuilder.end();
  EXPECT_EQ(*I, Op);
  ++I;
  EXPECT_EQ(I, E);
  --I;
  EXPECT_EQ(*I, Op);
  I++;
  EXPECT_EQ(I, E);
  I--;
  EXPECT_EQ(*I, Op);
}

TEST_F(DIExprBuilderTest, IteratorRange) {
  SmallVector<DIOp::Variant> Ops{
      DIOp::Variant{std::in_place_type<DIOp::Arg>, 0, Int64Ty},
      DIOp::Variant{std::in_place_type<DIOp::Arg>, 1, Int64Ty},
      DIOp::Variant{std::in_place_type<DIOp::Add>}};
  DIExprBuilder Builder(Context);
  for (auto &Op : Ops)
    Builder.append(Op);
  DIExpr *Expr = Builder.intoExpr();
  SmallVector<DIOp::Variant> Ops2;
  for (auto &Op : Expr->builder())
    Ops2.push_back(Op);
  EXPECT_EQ(Ops, Ops2);
}

TEST_F(DIExprBuilderTest, InitializerList) {
  std::initializer_list<DIOp::Variant> IL{
      DIOp::Variant{std::in_place_type<DIOp::Arg>, 0, Int64Ty},
      DIOp::Variant{std::in_place_type<DIOp::Arg>, 1, Int64Ty},
      DIOp::Variant{std::in_place_type<DIOp::Add>}};
  DIExprBuilder BuilderA(Context, IL);

  DIExprBuilder BuilderB(Context);
  BuilderB.insert(BuilderB.begin(), IL);

  DIExprBuilder BuilderC(Context);
  for (auto &Op : IL)
    BuilderC.append(Op);

  DIExpr *ExprA = BuilderA.intoExpr();
  DIExpr *ExprB = BuilderB.intoExpr();
  DIExpr *ExprC = BuilderC.intoExpr();

  EXPECT_EQ(ExprA, ExprB);
  EXPECT_EQ(ExprB, ExprC);
}

TEST_F(DIExprBuilderTest, InsertByValue) {
  DIOp::Variant V(std::in_place_type<DIOp::Sub>);

  {
    DIExprBuilder BuilderA(Context);
    BuilderA.insert(BuilderA.begin(), V);
    DIExprBuilder ExpectedA(Context, {DIOp::Variant{std::in_place_type<DIOp::Sub>}});
    EXPECT_EQ(BuilderA.intoExpr(), ExpectedA.intoExpr());
  }

  {
    DIExprBuilder BuilderB(Context, {DIOp::Variant{std::in_place_type<DIOp::Mul>},
                                     DIOp::Variant{std::in_place_type<DIOp::Div>}});
    BuilderB.insert(BuilderB.begin() + 1, V);
    DIExprBuilder ExpectedB(Context, {DIOp::Variant{std::in_place_type<DIOp::Mul>},
                                      DIOp::Variant{std::in_place_type<DIOp::Sub>},
                                      DIOp::Variant{std::in_place_type<DIOp::Div>}});
    EXPECT_EQ(BuilderB.intoExpr(), ExpectedB.intoExpr());
  }
}

TEST_F(DIExprBuilderTest, InsertEmplace) {
  {
    DIExprBuilder BuilderA(Context);
    BuilderA.insert<DIOp::Sub>(BuilderA.begin());
    DIExprBuilder ExpectedA(Context, {DIOp::Variant{std::in_place_type<DIOp::Sub>}});
    EXPECT_EQ(BuilderA.intoExpr(), ExpectedA.intoExpr());
  }

  {
    DIExprBuilder BuilderB(Context, {DIOp::Variant{std::in_place_type<DIOp::Mul>},
                                     DIOp::Variant{std::in_place_type<DIOp::Div>}});
    BuilderB.insert<DIOp::Sub>(BuilderB.begin() + 1);
    DIExprBuilder ExpectedB(Context, {DIOp::Variant{std::in_place_type<DIOp::Mul>},
                                      DIOp::Variant{std::in_place_type<DIOp::Sub>},
                                      DIOp::Variant{std::in_place_type<DIOp::Div>}});
    EXPECT_EQ(BuilderB.intoExpr(), ExpectedB.intoExpr());
  }
}

TEST_F(DIExprBuilderTest, InsertRange) {
  SmallVector<DIOp::Variant> Vs{DIOp::Variant{std::in_place_type<DIOp::Sub>},
                                DIOp::Variant{std::in_place_type<DIOp::Add>}};

  {
    DIExprBuilder BuilderA(Context);
    BuilderA.insert(BuilderA.begin(), Vs);
    DIExprBuilder ExpectedA(Context, {DIOp::Variant{std::in_place_type<DIOp::Sub>},
                                      DIOp::Variant{std::in_place_type<DIOp::Add>}});
    EXPECT_EQ(BuilderA.intoExpr(), ExpectedA.intoExpr());
  }

  {
    DIExprBuilder BuilderB(Context, {DIOp::Variant{std::in_place_type<DIOp::Mul>},
                                     DIOp::Variant{std::in_place_type<DIOp::Div>}});
    BuilderB.insert(BuilderB.begin() + 1, Vs);
    DIExprBuilder ExpectedB(Context, {DIOp::Variant{std::in_place_type<DIOp::Mul>},
                                      DIOp::Variant{std::in_place_type<DIOp::Sub>},
                                      DIOp::Variant{std::in_place_type<DIOp::Add>},
                                      DIOp::Variant{std::in_place_type<DIOp::Div>}});
    EXPECT_EQ(BuilderB.intoExpr(), ExpectedB.intoExpr());
  }
}

TEST_F(DIExprBuilderTest, InsertFromTo) {
  SmallVector<DIOp::Variant> Vs{DIOp::Variant{std::in_place_type<DIOp::Sub>},
                                DIOp::Variant{std::in_place_type<DIOp::Add>}};

  {
    DIExprBuilder BuilderA(Context);
    BuilderA.insert(BuilderA.begin(), Vs.begin(), Vs.end());
    DIExprBuilder ExpectedA(Context, {DIOp::Variant{std::in_place_type<DIOp::Sub>},
                                      DIOp::Variant{std::in_place_type<DIOp::Add>}});
    EXPECT_EQ(BuilderA.intoExpr(), ExpectedA.intoExpr());
  }

  {
    DIExprBuilder BuilderB(Context, {DIOp::Variant{std::in_place_type<DIOp::Mul>},
                                     DIOp::Variant{std::in_place_type<DIOp::Div>}});
    BuilderB.insert(BuilderB.begin() + 1, Vs.begin(), Vs.end());
    DIExprBuilder ExpectedB(Context, {DIOp::Variant{std::in_place_type<DIOp::Mul>},
                                      DIOp::Variant{std::in_place_type<DIOp::Sub>},
                                      DIOp::Variant{std::in_place_type<DIOp::Add>},
                                      DIOp::Variant{std::in_place_type<DIOp::Div>}});
    EXPECT_EQ(BuilderB.intoExpr(), ExpectedB.intoExpr());
  }
}

TEST_F(DIExprBuilderTest, Erase) {
  DIExprBuilder BuilderA(
      Context, {DIOp::Variant{std::in_place_type<DIOp::Referrer>, Int64Ty},
                DIOp::Variant{std::in_place_type<DIOp::Arg>, 0, Int64Ty},
                DIOp::Variant{std::in_place_type<DIOp::Add>},
                DIOp::Variant{std::in_place_type<DIOp::Mul>}});
  ASSERT_TRUE(
      std::holds_alternative<DIOp::Add>(*BuilderA.erase(++BuilderA.begin())));
  ASSERT_TRUE(
      std::holds_alternative<DIOp::Add>(*BuilderA.erase(BuilderA.begin())));
  auto I = BuilderA.erase(--BuilderA.end());
  ASSERT_EQ(I, BuilderA.end());
}

TEST_F(DIExprBuilderTest, Contains) {
  DIExprBuilder ExprBuilder0(Context);
  EXPECT_EQ(ExprBuilder0.contains<DIOp::Add>(), false);

  DIExprBuilder ExprBuilder1(
      Context, {DIOp::Variant{std::in_place_type<DIOp::Add>}});
  EXPECT_EQ(ExprBuilder1.contains<DIOp::Add>(), true);
  EXPECT_EQ(ExprBuilder1.contains<DIOp::Mul>(), false);

  DIExprBuilder ExprBuilder2(
      Context, {DIOp::Variant{std::in_place_type<DIOp::Add>},
                DIOp::Variant{std::in_place_type<DIOp::Mul>},
                DIOp::Variant{std::in_place_type<DIOp::Add>}});
  EXPECT_EQ(ExprBuilder2.contains<DIOp::Add>(), true);
  EXPECT_EQ(ExprBuilder2.contains<DIOp::Mul>(), true);
  EXPECT_EQ(ExprBuilder2.contains<DIOp::Select>(), false);
}

TEST_F(DIExprBuilderTest, Visitor) {
  DIOp::Variant Op(std::in_place_type<DIOp::Referrer>, Int32Ty);
  visit(makeVisitor([](DIOp::Referrer) {}, [](auto) { FAIL(); }), Op);
}

typedef MetadataTest DIExprOpsTest;

TEST_F(DIExprOpsTest, Referrer) {
  DIOp::Variant V{std::in_place_type<DIOp::Referrer>, Int64Ty};
  ASSERT_TRUE(std::holds_alternative<DIOp::Referrer>(V));
  ASSERT_EQ(std::get<DIOp::Referrer>(V).getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, Arg) {
  DIOp::Variant V{std::in_place_type<DIOp::Arg>, 1u, Int64Ty};
  ASSERT_TRUE(std::holds_alternative<DIOp::Arg>(V));
  ASSERT_EQ(std::get<DIOp::Arg>(V).getIndex(), 1u);
  ASSERT_EQ(std::get<DIOp::Arg>(V).getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, TypeObject) {
  DIOp::Variant V{std::in_place_type<DIOp::TypeObject>, Int64Ty};
  ASSERT_TRUE(std::holds_alternative<DIOp::TypeObject>(V));
  ASSERT_EQ(std::get<DIOp::TypeObject>(V).getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, Constant) {
  DIOp::Variant V{std::in_place_type<DIOp::Constant>,
                  ConstantFP::get(Context, APFloat(2.0f))};
  ASSERT_TRUE(std::holds_alternative<DIOp::Constant>(V));
  ASSERT_EQ(std::get<DIOp::Constant>(V).getLiteralValue(),
            ConstantFP::get(Context, APFloat(2.0f)));
}

TEST_F(DIExprOpsTest, Convert) {
  DIOp::Variant V{std::in_place_type<DIOp::Convert>, Int64Ty};
  ASSERT_TRUE(std::holds_alternative<DIOp::Convert>(V));
  ASSERT_EQ(std::get<DIOp::Convert>(V).getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, Reinterpret) {
  DIOp::Variant V{std::in_place_type<DIOp::Reinterpret>, Int64Ty};
  ASSERT_TRUE(std::holds_alternative<DIOp::Reinterpret>(V));
  ASSERT_EQ(std::get<DIOp::Reinterpret>(V).getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, BitOffset) {
  DIOp::Variant V{std::in_place_type<DIOp::BitOffset>, Int64Ty};
  ASSERT_TRUE(std::holds_alternative<DIOp::BitOffset>(V));
  ASSERT_EQ(std::get<DIOp::BitOffset>(V).getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, ByteOffset) {
  DIOp::Variant V{std::in_place_type<DIOp::ByteOffset>, Int64Ty};
  ASSERT_TRUE(std::holds_alternative<DIOp::ByteOffset>(V));
  ASSERT_EQ(std::get<DIOp::ByteOffset>(V).getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, Composite) {
  DIOp::Variant V{std::in_place_type<DIOp::Composite>, 4u, Int64Ty};
  ASSERT_TRUE(std::holds_alternative<DIOp::Composite>(V));
  ASSERT_EQ(std::get<DIOp::Composite>(V).getCount(), 4u);
  ASSERT_EQ(std::get<DIOp::Composite>(V).getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, Extend) {
  DIOp::Variant V{std::in_place_type<DIOp::Extend>, 16u};
  ASSERT_TRUE(std::holds_alternative<DIOp::Extend>(V));
  ASSERT_EQ(std::get<DIOp::Extend>(V).getCount(), 16u);
}

TEST_F(DIExprOpsTest, Select) {
  DIOp::Variant V{std::in_place_type<DIOp::Select>};
  ASSERT_TRUE(std::holds_alternative<DIOp::Select>(V));
}

TEST_F(DIExprOpsTest, AddrOf) {
  DIOp::Variant V{std::in_place_type<DIOp::AddrOf>, 16u};
  ASSERT_TRUE(std::holds_alternative<DIOp::AddrOf>(V));
  ASSERT_EQ(std::get<DIOp::AddrOf>(V).getAddressSpace(), 16u);
}

TEST_F(DIExprOpsTest, Deref) {
  DIOp::Variant V{std::in_place_type<DIOp::Deref>, Int64Ty};
  ASSERT_TRUE(std::holds_alternative<DIOp::Deref>(V));
  ASSERT_EQ(std::get<DIOp::Deref>(V).getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, Read) {
  DIOp::Variant V{std::in_place_type<DIOp::Read>};
  ASSERT_TRUE(std::holds_alternative<DIOp::Read>(V));
}

TEST_F(DIExprOpsTest, Add) {
  DIOp::Variant V{std::in_place_type<DIOp::Add>};
  ASSERT_TRUE(std::holds_alternative<DIOp::Add>(V));
}

TEST_F(DIExprOpsTest, Sub) {
  DIOp::Variant V{std::in_place_type<DIOp::Sub>};
  ASSERT_TRUE(std::holds_alternative<DIOp::Sub>(V));
}

TEST_F(DIExprOpsTest, Mul) {
  DIOp::Variant V{std::in_place_type<DIOp::Mul>};
  ASSERT_TRUE(std::holds_alternative<DIOp::Mul>(V));
}

TEST_F(DIExprOpsTest, Div) {
  DIOp::Variant V{std::in_place_type<DIOp::Div>};
  ASSERT_TRUE(std::holds_alternative<DIOp::Div>(V));
}

TEST_F(DIExprOpsTest, Shr) {
  DIOp::Variant V{std::in_place_type<DIOp::Shr>};
  ASSERT_TRUE(std::holds_alternative<DIOp::Shr>(V));
}

TEST_F(DIExprOpsTest, Shl) {
  DIOp::Variant V{std::in_place_type<DIOp::Shl>};
  ASSERT_TRUE(std::holds_alternative<DIOp::Shl>(V));
}

TEST_F(DIExprOpsTest, PushLane) {
  DIOp::Variant V{std::in_place_type<DIOp::PushLane>, Int64Ty};
  ASSERT_TRUE(std::holds_alternative<DIOp::PushLane>(V));
  ASSERT_EQ(std::get<DIOp::PushLane>(V).getResultType(), Int64Ty);
}

typedef MetadataTest DIExprTest;

TEST_F(DIExprTest, setLocation) {
  DIExpr *Original = DIExprBuilder(Context, {}).intoExpr();
  DIExpr *Replacement =
      DIExprBuilder(Context, {DIOp::Variant{std::in_place_type<DIOp::Sub>}})
          .intoExpr();
  DIFragment *Fragment = DIFragment::getDistinct(Context);
  DILifetime *Lifetime = DILifetime::getDistinct(Context, Fragment, Original);
  EXPECT_EQ(Lifetime->getLocation(), Original);
  Lifetime->setLocation(Replacement);
  EXPECT_EQ(Lifetime->getLocation(), Replacement);
}

} // end namespace
