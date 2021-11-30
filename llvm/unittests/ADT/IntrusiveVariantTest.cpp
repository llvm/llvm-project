//===- llvm/unittest/Support/AnyTest.cpp - Any tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/IntrusiveVariant.h"
#include "gtest/gtest.h"
#include <cstdlib>

using namespace llvm;

namespace {

class A {
  DECLARE_INTRUSIVE_ALTERNATIVE
};

class B {
  DECLARE_INTRUSIVE_ALTERNATIVE
};

TEST(IntrusiveVariantTest, SingleAlternative) { IntrusiveVariant<A> V; }

TEST(IntrusiveVariantTest, ZeroArgConstructionAndAssignment) {
  IntrusiveVariant<A, B> V;
  ASSERT_TRUE(V.holdsAlternative<A>());
  visit(makeVisitor([](A) {}, [](B) { FAIL(); }), V);
  visit(makeVisitor([](B) { FAIL(); }, [](A) {}), V);
  visit(makeVisitor([](A &) {}, [](B &) { FAIL(); }), V);
  visit(makeVisitor([](A) {}, [](B &) { FAIL(); }), V);
  visit(makeVisitor([](A &) {}, [](B) { FAIL(); }), V);
  visit(makeVisitor([](auto &&) {}), V);

  V.emplace<B>();
  ASSERT_TRUE(V.holdsAlternative<B>());

  IntrusiveVariant<A, B> W{V};
  ASSERT_TRUE(W.holdsAlternative<B>());

  const IntrusiveVariant<A, B> X{V};
  ASSERT_TRUE(X.holdsAlternative<B>());
}

template <typename T> class Alt {
  DECLARE_INTRUSIVE_ALTERNATIVE
  T Val;

public:
  Alt(T Val) : Val(Val) {}
  T getVal() const { return Val; }
  friend bool operator==(const Alt &LHS, const Alt &RHS) {
    return LHS.getVal() == RHS.getVal();
  }
  friend bool operator!=(const Alt &LHS, const Alt &RHS) {
    return LHS.getVal() != RHS.getVal();
  }
  friend bool operator<(const Alt &LHS, const Alt &RHS) {
    return LHS.getVal() < RHS.getVal();
  }
  friend bool operator>(const Alt &LHS, const Alt &RHS) {
    return LHS.getVal() > RHS.getVal();
  }
  friend bool operator<=(const Alt &LHS, const Alt &RHS) {
    return LHS.getVal() <= RHS.getVal();
  }
  friend bool operator>=(const Alt &LHS, const Alt &RHS) {
    return LHS.getVal() >= RHS.getVal();
  }
};
using I = Alt<int>;
using F = Alt<float>;
using D = Alt<double>;

TEST(IntrusiveVariantTest, ConstructionAndAssignment) {
  IntrusiveVariant<I, F, D> V{in_place_type<F>, 2.0f};
  visit(makeVisitor([](I) { FAIL(); }, [](F X) { EXPECT_EQ(X.getVal(), 2.0f); },
                    [](D) { FAIL(); }),
        V);
  IntrusiveVariant<I, F, D> W{V};
  visit(makeVisitor([](I) { FAIL(); }, [](F X) { EXPECT_EQ(X.getVal(), 2.0f); },
                    [](D) { FAIL(); }),
        W);
  W.emplace<I>(42);
  visit(makeVisitor([](I X) { EXPECT_EQ(X.getVal(), 42); }, [](F) { FAIL(); },
                    [](D) { FAIL(); }),
        W);
  W = V;
  visit(makeVisitor([](I) { FAIL(); }, [](F X) { EXPECT_EQ(X.getVal(), 2.0f); },
                    [](D) { FAIL(); }),
        W);
}

TEST(IntrusiveVariantTest, Comparison) {
  IntrusiveVariant<I, F, D> V{in_place_type<I>, 1};
  IntrusiveVariant<I, F, D> W{in_place_type<F>, 2.0f};
  IntrusiveVariant<I, F, D> X{in_place_type<F>, 2.0f};
  IntrusiveVariant<I, F, D> Y{in_place_type<F>, 3.0f};
  IntrusiveVariant<I, F, D> Z{in_place_type<D>, 3.0};
  EXPECT_NE(V, W);
  EXPECT_LT(V, W);
  EXPECT_LE(V, W);
  EXPECT_GT(W, V);
  EXPECT_GE(W, V);
  EXPECT_EQ(W, X);
  EXPECT_LE(W, X);
  EXPECT_GE(W, X);
  EXPECT_NE(W, Y);
  EXPECT_NE(X, Y);
  EXPECT_LT(X, Y);
  EXPECT_LE(X, Y);
  EXPECT_GT(Y, X);
  EXPECT_GE(Y, X);
  EXPECT_NE(Y, Z);
  std::swap(X, Y);
  EXPECT_EQ(W, Y);
  EXPECT_NE(W, X);
}

TEST(IntrusiveVariantTest, IntrusiveVariantSize) {
  constexpr auto One = IntrusiveVariantSize<IntrusiveVariant<I>>::value;
  EXPECT_EQ(One, 1u);
  constexpr auto Two = IntrusiveVariantSize<IntrusiveVariant<I, F>>::value;
  EXPECT_EQ(Two, 2u);
  constexpr auto Three = IntrusiveVariantSize<IntrusiveVariant<I, F, D>>::value;
  EXPECT_EQ(Three, 3u);
}

TEST(IntrusiveVariantTest, HoldsAlternative) {
  IntrusiveVariant<I, F, D> V{in_place_type<D>, 2.0};
  EXPECT_FALSE(V.holdsAlternative<I>());
  EXPECT_FALSE(V.holdsAlternative<F>());
  EXPECT_TRUE(V.holdsAlternative<D>());
  V.emplace<I>(1);
  EXPECT_TRUE(V.holdsAlternative<I>());
  EXPECT_FALSE(V.holdsAlternative<F>());
  EXPECT_FALSE(V.holdsAlternative<D>());
  const IntrusiveVariant<I, F, D> C{in_place_type<F>, 2.0f};
  EXPECT_FALSE(C.holdsAlternative<I>());
  EXPECT_TRUE(C.holdsAlternative<F>());
  EXPECT_FALSE(C.holdsAlternative<D>());
}

TEST(IntrusiveVariantTest, Get) {
  IntrusiveVariant<I, F, D> V{in_place_type<D>, 2.0};
  EXPECT_EQ(V.get<D>(), D{2.0});
  EXPECT_EQ(V.get<D>(), *V.getIf<D>());
  EXPECT_EQ(&V.get<D>(), V.getIf<D>());
  V.emplace<I>(1);
  EXPECT_EQ(V.get<I>(), I{1});
  EXPECT_EQ(V.get<I>(), *V.getIf<I>());
  EXPECT_EQ(&V.get<I>(), V.getIf<I>());
  const IntrusiveVariant<I, F, D> C{in_place_type<D>, 2.0};
  EXPECT_EQ(C.get<D>(), D{2.0});
  EXPECT_EQ(C.get<D>(), *C.getIf<D>());
  EXPECT_EQ(&C.get<D>(), C.getIf<D>());
}

TEST(IntrusiveVariantTest, GetIf) {
  IntrusiveVariant<I, F, D> V{in_place_type<D>, 2.0};
  EXPECT_EQ(V.getIf<I>(), nullptr);
  EXPECT_EQ(V.getIf<F>(), nullptr);
  EXPECT_NE(V.getIf<D>(), nullptr);
  V.emplace<I>(1);
  EXPECT_NE(V.getIf<I>(), nullptr);
  EXPECT_EQ(V.getIf<F>(), nullptr);
  EXPECT_EQ(V.getIf<D>(), nullptr);
  const IntrusiveVariant<I, F, D> C{in_place_type<F>, 2.0f};
  EXPECT_EQ(C.getIf<I>(), nullptr);
  EXPECT_NE(C.getIf<F>(), nullptr);
  EXPECT_EQ(C.getIf<D>(), nullptr);
}

struct IntA {
  DECLARE_INTRUSIVE_ALTERNATIVE
  int Val;
  IntA(int Val) : Val(Val) {}
  friend hash_code hash_value(const IntA &IA) { return hash_value(IA.Val); }
};

struct IntB {
  DECLARE_INTRUSIVE_ALTERNATIVE
  int Val;
  IntB(int Val) : Val(Val) {}
  friend hash_code hash_value(const IntB &IB) { return hash_value(IB.Val); }
};

TEST(IntrusiveVariantTest, HashValue) {
  IntrusiveVariant<IntA, IntB> ATwo{in_place_type<IntA>, 2};
  IntrusiveVariant<IntA, IntB> AThree{in_place_type<IntA>, 3};
  IntrusiveVariant<IntA, IntB> BTwo{in_place_type<IntB>, 2};
  EXPECT_EQ(hash_value(ATwo), hash_value(ATwo));
  EXPECT_NE(hash_value(ATwo), hash_value(AThree));
  EXPECT_NE(hash_value(ATwo), hash_value(BTwo));
}

class AltInt {
  DECLARE_INTRUSIVE_ALTERNATIVE
  int Int;

public:
  AltInt() : Int(0) {}
  AltInt(int Int) : Int(Int) {}
  int getInt() const { return Int; }
  void setInt(int Int) { this->Int = Int; }
};

class AltDouble {
  DECLARE_INTRUSIVE_ALTERNATIVE
  double Double;

public:
  AltDouble(double Double) : Double(Double) {}
  double getDouble() const { return Double; }
  void setDouble(double Double) { this->Double = Double; }
};

class AltComplexInt {
  DECLARE_INTRUSIVE_ALTERNATIVE
  int Real;
  int Imag;

public:
  AltComplexInt(int Real, int Imag) : Real(Real), Imag(Imag) {}
  int getReal() const { return Real; }
  void setReal(int Real) { this->Real = Real; }
  int getImag() const { return Imag; }
  void setImag(int Imag) { this->Imag = Imag; }
};

TEST(IntrusiveVariantTest, HeaderExample) {
  using MyVariant = IntrusiveVariant<AltInt, AltDouble, AltComplexInt>;

  MyVariant DefaultConstructedVariant;
  ASSERT_TRUE(DefaultConstructedVariant.holdsAlternative<AltInt>());
  ASSERT_EQ(DefaultConstructedVariant.get<AltInt>().getInt(), 0);
  MyVariant Variant{in_place_type<AltComplexInt>, 4, 2};
  ASSERT_TRUE(Variant.holdsAlternative<AltComplexInt>());
  int NonSense = visit(
      makeVisitor(
          [](AltInt &AI) { return AI.getInt(); },
          [](AltDouble &AD) { return static_cast<int>(AD.getDouble()); },
          [](AltComplexInt &ACI) { return ACI.getReal() + ACI.getImag(); }),
      Variant);
  ASSERT_EQ(NonSense, 6);
  Variant.emplace<AltDouble>(2.0);
  ASSERT_TRUE(Variant.holdsAlternative<AltDouble>());
  Variant.get<AltDouble>().setDouble(3.0);
  AltDouble AD = Variant.get<AltDouble>();
  double D = AD.getDouble();
  ASSERT_EQ(D, 3.0);
  Variant.emplace<AltComplexInt>(4, 5);
  ASSERT_EQ(Variant.get<AltComplexInt>().getReal(), 4);
  ASSERT_EQ(Variant.get<AltComplexInt>().getImag(), 5);
}

} // anonymous namespace
