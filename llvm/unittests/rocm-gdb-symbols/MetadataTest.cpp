//===- unittests/IR/MetadataTest.cpp - Metadata unit tests ----------------===//
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
  DIOp::Variant Op(in_place_type<DIOp::Referrer>, Int32Ty);
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
      DIOp::Variant{in_place_type<DIOp::Arg>, 0, Int64Ty},
      DIOp::Variant{in_place_type<DIOp::Arg>, 1, Int64Ty},
      DIOp::Variant{in_place_type<DIOp::Add>}};
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
      DIOp::Variant{in_place_type<DIOp::Arg>, 0, Int64Ty},
      DIOp::Variant{in_place_type<DIOp::Arg>, 1, Int64Ty},
      DIOp::Variant{in_place_type<DIOp::Add>}};
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
  DIOp::Variant V(in_place_type<DIOp::Sub>);

  {
    DIExprBuilder BuilderA(Context);
    BuilderA.insert(BuilderA.begin(), V);
    DIExprBuilder ExpectedA(Context, {DIOp::Variant{in_place_type<DIOp::Sub>}});
    EXPECT_EQ(BuilderA.intoExpr(), ExpectedA.intoExpr());
  }

  {
    DIExprBuilder BuilderB(Context, {DIOp::Variant{in_place_type<DIOp::Mul>},
                                     DIOp::Variant{in_place_type<DIOp::Div>}});
    BuilderB.insert(BuilderB.begin() + 1, V);
    DIExprBuilder ExpectedB(Context, {DIOp::Variant{in_place_type<DIOp::Mul>},
                                      DIOp::Variant{in_place_type<DIOp::Sub>},
                                      DIOp::Variant{in_place_type<DIOp::Div>}});
    EXPECT_EQ(BuilderB.intoExpr(), ExpectedB.intoExpr());
  }
}

TEST_F(DIExprBuilderTest, InsertEmplace) {
  {
    DIExprBuilder BuilderA(Context);
    BuilderA.insert<DIOp::Sub>(BuilderA.begin());
    DIExprBuilder ExpectedA(Context, {DIOp::Variant{in_place_type<DIOp::Sub>}});
    EXPECT_EQ(BuilderA.intoExpr(), ExpectedA.intoExpr());
  }

  {
    DIExprBuilder BuilderB(Context, {DIOp::Variant{in_place_type<DIOp::Mul>},
                                     DIOp::Variant{in_place_type<DIOp::Div>}});
    BuilderB.insert<DIOp::Sub>(BuilderB.begin() + 1);
    DIExprBuilder ExpectedB(Context, {DIOp::Variant{in_place_type<DIOp::Mul>},
                                      DIOp::Variant{in_place_type<DIOp::Sub>},
                                      DIOp::Variant{in_place_type<DIOp::Div>}});
    EXPECT_EQ(BuilderB.intoExpr(), ExpectedB.intoExpr());
  }
}

TEST_F(DIExprBuilderTest, InsertRange) {
  SmallVector<DIOp::Variant> Vs{DIOp::Variant{in_place_type<DIOp::Sub>},
                                DIOp::Variant{in_place_type<DIOp::Add>}};

  {
    DIExprBuilder BuilderA(Context);
    BuilderA.insert(BuilderA.begin(), Vs);
    DIExprBuilder ExpectedA(Context, {DIOp::Variant{in_place_type<DIOp::Sub>},
                                      DIOp::Variant{in_place_type<DIOp::Add>}});
    EXPECT_EQ(BuilderA.intoExpr(), ExpectedA.intoExpr());
  }

  {
    DIExprBuilder BuilderB(Context, {DIOp::Variant{in_place_type<DIOp::Mul>},
                                     DIOp::Variant{in_place_type<DIOp::Div>}});
    BuilderB.insert(BuilderB.begin() + 1, Vs);
    DIExprBuilder ExpectedB(Context, {DIOp::Variant{in_place_type<DIOp::Mul>},
                                      DIOp::Variant{in_place_type<DIOp::Sub>},
                                      DIOp::Variant{in_place_type<DIOp::Add>},
                                      DIOp::Variant{in_place_type<DIOp::Div>}});
    EXPECT_EQ(BuilderB.intoExpr(), ExpectedB.intoExpr());
  }
}

TEST_F(DIExprBuilderTest, InsertFromTo) {
  SmallVector<DIOp::Variant> Vs{DIOp::Variant{in_place_type<DIOp::Sub>},
                                DIOp::Variant{in_place_type<DIOp::Add>}};

  {
    DIExprBuilder BuilderA(Context);
    BuilderA.insert(BuilderA.begin(), Vs.begin(), Vs.end());
    DIExprBuilder ExpectedA(Context, {DIOp::Variant{in_place_type<DIOp::Sub>},
                                      DIOp::Variant{in_place_type<DIOp::Add>}});
    EXPECT_EQ(BuilderA.intoExpr(), ExpectedA.intoExpr());
  }

  {
    DIExprBuilder BuilderB(Context, {DIOp::Variant{in_place_type<DIOp::Mul>},
                                     DIOp::Variant{in_place_type<DIOp::Div>}});
    BuilderB.insert(BuilderB.begin() + 1, Vs.begin(), Vs.end());
    DIExprBuilder ExpectedB(Context, {DIOp::Variant{in_place_type<DIOp::Mul>},
                                      DIOp::Variant{in_place_type<DIOp::Sub>},
                                      DIOp::Variant{in_place_type<DIOp::Add>},
                                      DIOp::Variant{in_place_type<DIOp::Div>}});
    EXPECT_EQ(BuilderB.intoExpr(), ExpectedB.intoExpr());
  }
}

TEST_F(DIExprBuilderTest, Erase) {
  DIExprBuilder BuilderA(
      Context, {DIOp::Variant{in_place_type<DIOp::Referrer>, Int64Ty},
                DIOp::Variant{in_place_type<DIOp::Arg>, 0, Int64Ty},
                DIOp::Variant{in_place_type<DIOp::Add>},
                DIOp::Variant{in_place_type<DIOp::Mul>}});
  ASSERT_TRUE(
      BuilderA.erase(++BuilderA.begin())->holdsAlternative<DIOp::Add>());
  ASSERT_TRUE(BuilderA.erase(BuilderA.begin())->holdsAlternative<DIOp::Add>());
  auto I = BuilderA.erase(--BuilderA.end());
  ASSERT_EQ(I, BuilderA.end());
}

TEST_F(DIExprBuilderTest, Contains) {
  DIExprBuilder ExprBuilder0(Context);
  EXPECT_EQ(ExprBuilder0.contains<DIOp::Add>(), false);

  DIExprBuilder ExprBuilder1(
      Context, {DIOp::Variant{in_place_type<DIOp::Add>}});
  EXPECT_EQ(ExprBuilder1.contains<DIOp::Add>(), true);
  EXPECT_EQ(ExprBuilder1.contains<DIOp::Mul>(), false);

  DIExprBuilder ExprBuilder2(
      Context, {DIOp::Variant{in_place_type<DIOp::Add>},
                DIOp::Variant{in_place_type<DIOp::Mul>},
                DIOp::Variant{in_place_type<DIOp::Add>}});
  EXPECT_EQ(ExprBuilder2.contains<DIOp::Add>(), true);
  EXPECT_EQ(ExprBuilder2.contains<DIOp::Mul>(), true);
  EXPECT_EQ(ExprBuilder2.contains<DIOp::Select>(), false);
}

TEST_F(DIExprBuilderTest, Visitor) {
  DIOp::Variant Op(in_place_type<DIOp::Referrer>, Int32Ty);
  visit(makeVisitor([](DIOp::Referrer) {}, [](auto) { FAIL(); }), Op);
}

typedef MetadataTest DIExprOpsTest;

TEST_F(DIExprOpsTest, Referrer) {
  DIOp::Variant V{in_place_type<DIOp::Referrer>, Int64Ty};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Referrer>());
  ASSERT_EQ(V.get<DIOp::Referrer>().getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, Arg) {
  DIOp::Variant V{in_place_type<DIOp::Arg>, 1u, Int64Ty};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Arg>());
  ASSERT_EQ(V.get<DIOp::Arg>().getIndex(), 1u);
  ASSERT_EQ(V.get<DIOp::Arg>().getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, TypeObject) {
  DIOp::Variant V{in_place_type<DIOp::TypeObject>, Int64Ty};
  ASSERT_TRUE(V.holdsAlternative<DIOp::TypeObject>());
  ASSERT_EQ(V.get<DIOp::TypeObject>().getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, Constant) {
  DIOp::Variant V{in_place_type<DIOp::Constant>,
                  ConstantFP::get(Context, APFloat(2.0f))};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Constant>());
  ASSERT_EQ(V.get<DIOp::Constant>().getLiteralValue(),
            ConstantFP::get(Context, APFloat(2.0f)));
}

TEST_F(DIExprOpsTest, Convert) {
  DIOp::Variant V{in_place_type<DIOp::Convert>, Int64Ty};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Convert>());
  ASSERT_EQ(V.get<DIOp::Convert>().getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, Reinterpret) {
  DIOp::Variant V{in_place_type<DIOp::Reinterpret>, Int64Ty};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Reinterpret>());
  ASSERT_EQ(V.get<DIOp::Reinterpret>().getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, BitOffset) {
  DIOp::Variant V{in_place_type<DIOp::BitOffset>, Int64Ty};
  ASSERT_TRUE(V.holdsAlternative<DIOp::BitOffset>());
  ASSERT_EQ(V.get<DIOp::BitOffset>().getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, ByteOffset) {
  DIOp::Variant V{in_place_type<DIOp::ByteOffset>, Int64Ty};
  ASSERT_TRUE(V.holdsAlternative<DIOp::ByteOffset>());
  ASSERT_EQ(V.get<DIOp::ByteOffset>().getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, Composite) {
  DIOp::Variant V{in_place_type<DIOp::Composite>, 4u, Int64Ty};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Composite>());
  ASSERT_EQ(V.get<DIOp::Composite>().getCount(), 4u);
  ASSERT_EQ(V.get<DIOp::Composite>().getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, Extend) {
  DIOp::Variant V{in_place_type<DIOp::Extend>, 16u};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Extend>());
  ASSERT_EQ(V.get<DIOp::Extend>().getCount(), 16u);
}

TEST_F(DIExprOpsTest, Select) {
  DIOp::Variant V{in_place_type<DIOp::Select>};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Select>());
}

TEST_F(DIExprOpsTest, AddrOf) {
  DIOp::Variant V{in_place_type<DIOp::AddrOf>, 16u};
  ASSERT_TRUE(V.holdsAlternative<DIOp::AddrOf>());
  ASSERT_EQ(V.get<DIOp::AddrOf>().getAddressSpace(), 16u);
}

TEST_F(DIExprOpsTest, Deref) {
  DIOp::Variant V{in_place_type<DIOp::Deref>, Int64Ty};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Deref>());
  ASSERT_EQ(V.get<DIOp::Deref>().getResultType(), Int64Ty);
}

TEST_F(DIExprOpsTest, Read) {
  DIOp::Variant V{in_place_type<DIOp::Read>};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Read>());
}

TEST_F(DIExprOpsTest, Add) {
  DIOp::Variant V{in_place_type<DIOp::Add>};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Add>());
}

TEST_F(DIExprOpsTest, Sub) {
  DIOp::Variant V{in_place_type<DIOp::Sub>};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Sub>());
}

TEST_F(DIExprOpsTest, Mul) {
  DIOp::Variant V{in_place_type<DIOp::Mul>};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Mul>());
}

TEST_F(DIExprOpsTest, Div) {
  DIOp::Variant V{in_place_type<DIOp::Div>};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Div>());
}

TEST_F(DIExprOpsTest, Shr) {
  DIOp::Variant V{in_place_type<DIOp::Shr>};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Shr>());
}

TEST_F(DIExprOpsTest, Shl) {
  DIOp::Variant V{in_place_type<DIOp::Shl>};
  ASSERT_TRUE(V.holdsAlternative<DIOp::Shl>());
}

TEST_F(DIExprOpsTest, PushLane) {
  DIOp::Variant V{in_place_type<DIOp::PushLane>, Int64Ty};
  ASSERT_TRUE(V.holdsAlternative<DIOp::PushLane>());
  ASSERT_EQ(V.get<DIOp::PushLane>().getResultType(), Int64Ty);
}

typedef MetadataTest DIExprTest;

TEST_F(DIExprTest, setLocation) {
  DIExpr *Original = DIExprBuilder(Context, {}).intoExpr();
  DIExpr *Replacement =
      DIExprBuilder(Context, {DIOp::Variant{in_place_type<DIOp::Sub>}})
          .intoExpr();
  DIFragment *Fragment = DIFragment::getDistinct(Context);
  DILifetime *Lifetime = DILifetime::getDistinct(Context, Fragment, Original);
  EXPECT_EQ(Lifetime->getLocation(), Original);
  Lifetime->setLocation(Replacement);
  EXPECT_EQ(Lifetime->getLocation(), Replacement);
}

} // end namespace
