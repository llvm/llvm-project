//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains the tests for the mlir::NonFloatComplex type.
///
//===----------------------------------------------------------------------===//

#include "mlir/Support/Complex.h"
#include "gtest/gtest.h"

namespace mlir {
// Provide ostream operator so that tests pretty print NonFloatComplex values
template <typename T>
static std::ostream &operator<<(std::ostream &os, const NonFloatComplex<T> c) {
  os << "(" << c.real() << "," << c.imag() << ")";
  return os;
}

} // namespace mlir

// The majority of these tests just check that NonFloatComplex does exactly the
// same as std::complex<float>.

TEST(ComplexTest, Typedef) {
  EXPECT_TRUE((std::is_same_v<mlir::Complex<float>, std::complex<float>>));

  EXPECT_TRUE((std::is_same_v<mlir::Complex<int>, mlir::NonFloatComplex<int>>));
}

TEST(ComplexTest, DefaultConstructor) {
  mlir::NonFloatComplex<float> mc;
  std::complex<float> sc;
  EXPECT_EQ(mc, sc);
}

TEST(ComplexTest, RealConstructor) {
  mlir::NonFloatComplex<float> mc{10};
  std::complex<float> sc{10};
  EXPECT_EQ(mc, sc);
}

TEST(ComplexTest, MemberConstructor) {
  mlir::NonFloatComplex<float> mc{10, 20};
  std::complex<float> sc{10, 20};
  EXPECT_EQ(mc, sc);
}

TEST(ComplexTest, ExplicitCopyConstructor) {
  std::complex<double> sc{5, 10};
  mlir::NonFloatComplex<float> mc{sc};
  EXPECT_EQ(mc, sc);

  // check the explicit constructors were used
  EXPECT_FALSE((std::is_convertible_v<decltype(sc), decltype(mc)>));
}

TEST(ComplexTest, ImplicitCopyConstructor) {
  std::complex<float> sc{};
  mlir::NonFloatComplex<float> mc = sc;
  EXPECT_EQ(mc, sc);

  // check the implicit constructors were used
  EXPECT_TRUE((std::is_convertible_v<decltype(sc), decltype(mc)>));
}

TEST(ComplexTest, RealAccessor) {
  mlir::NonFloatComplex<float> mc{5};
  std::complex<float> sc{5};
  EXPECT_EQ(mc.real(), sc.real());
}

TEST(ComplexTest, RealSetter) {
  mlir::NonFloatComplex<float> mc{5};
  mc.real(7);
  std::complex<float> sc{5};
  sc.real(7);
  EXPECT_EQ(mc.real(), sc.real());
}

TEST(ComplexTest, ImagAccessor) {
  mlir::NonFloatComplex<int> mc{2, 5};
  std::complex<float> sc{2, 5};
  EXPECT_EQ(mc.imag(), sc.imag());
}

TEST(ComplexTest, ImagSetter) {
  mlir::NonFloatComplex<int> mc{2, 5};
  mc.imag(8);
  std::complex<float> sc{2, 5};
  sc.imag(8);
  EXPECT_EQ(mc.imag(), sc.imag());
}

TEST(ComplexTest, CopyAssignment) {
  mlir::NonFloatComplex<int> mc{2, 5};
  mlir::NonFloatComplex<int> mc2 = mc;

  EXPECT_EQ(mc, mc2);
}

TEST(ComplexTest, StdCopyAssignment) {
  std::complex<float> sc{2, 5};
  mlir::NonFloatComplex<float> mc = sc;

  EXPECT_EQ(mc, sc);
}

TEST(ComplexTest, RealAssignment) {
  std::complex<float> sc = 2.f;
  mlir::NonFloatComplex<float> mc = 2.f;

  EXPECT_EQ(mc, sc);
}

TEST(ComplexTest, PlusEqualsReal) {
  mlir::NonFloatComplex<float> mc{2, 5};
  mc += 7;
  std::complex<float> sc{2, 5};
  sc += 7;

  EXPECT_EQ(mc, sc);
}

TEST(ComplexTest, MinusEqualsReal) {
  mlir::NonFloatComplex<float> mc{3, 6};
  mc -= 8;
  std::complex<float> sc{3, 6};
  sc -= 8;

  EXPECT_EQ(mc, sc);
}

TEST(ComplexTest, TimesEqualsReal) {
  mlir::NonFloatComplex<float> mc{1, 4};
  mc *= 2;
  std::complex<float> sc{1, 4};
  sc *= 2;

  EXPECT_EQ(mc, sc);
}

TEST(ComplexTest, DivideEqualsReal) {
  mlir::NonFloatComplex<float> mc{1, 4};
  mc /= 2;
  std::complex<float> sc{1, 4};
  sc /= 2;

  EXPECT_EQ(mc, sc);
}

TEST(ComplexTest, AssignmentOp) {
  mlir::NonFloatComplex<int> mc{2, 5};
  mlir::NonFloatComplex<int> mc2 = mc;

  EXPECT_EQ(mc, mc2);
}

TEST(ComplexTest, StdAssignmentOp) {

  std::complex<float> sc{2, 5};
  mlir::NonFloatComplex<float> mc = sc;

  EXPECT_EQ(mc, sc);
}

TEST(ComplexTest, AddOp) {
  mlir::NonFloatComplex<float> mc1{2, 5};
  mlir::NonFloatComplex<float> mc2{3, 7};
  std::complex<float> sc1{2, 5};
  std::complex<float> sc2{3, 7};

  EXPECT_EQ(mc1 + mc2, sc1 + sc2);
  EXPECT_EQ(mc1 + 5.f, sc1 + 5.f);
}

TEST(ComplexTest, MinusOp) {
  mlir::NonFloatComplex<float> mc1{2, 5};
  mlir::NonFloatComplex<float> mc2{3, 7};
  std::complex<float> sc1{2, 5};
  std::complex<float> sc2{3, 7};

  EXPECT_EQ(mc1 - mc2, sc1 - sc2);
  EXPECT_EQ(mc1 - 5.f, sc1 - 5.f);
}

TEST(ComplexTest, TimesOp) {
  mlir::NonFloatComplex<float> mc1{2, 5};
  mlir::NonFloatComplex<float> mc2{3, 7};
  std::complex<float> sc1{2, 5};
  std::complex<float> sc2{3, 7};

  EXPECT_EQ(mc1 * mc2, sc1 * sc2);
  EXPECT_EQ(mc1 * 5.f, sc1 * 5.f);
}

TEST(ComplexTest, DivideOp) {
  mlir::NonFloatComplex<float> mc1{5, 10};
  mlir::NonFloatComplex<float> mc2{3, 4};
  std::complex<float> sc1{5, 10};
  std::complex<float> sc2{3, 4};

  EXPECT_EQ(mc1 / mc2, sc1 / sc2);
  EXPECT_EQ(mc1 / 5.f, sc1 / 5.f);
}

TEST(ComplexTest, EqualityOp) {
  mlir::NonFloatComplex<float> mc1{3, 4};
  mlir::NonFloatComplex<float> mc2{3, 4};

  EXPECT_EQ(mc1, mc2);
  EXPECT_EQ(mc2, mc1);
}

TEST(ComplexTest, StdEqualityOp) {
  mlir::NonFloatComplex<float> mc{7, 8};
  std::complex<float> sc{7, 8};

  EXPECT_EQ(mc, sc);
  EXPECT_EQ(sc, mc);
}

TEST(ComplexTest, InequalityOp) {
  mlir::NonFloatComplex<float> mc1{3, 4};
  mlir::NonFloatComplex<float> mc2{7, 8};

  EXPECT_NE(mc1, mc2);
  EXPECT_NE(mc2, mc1);
}

TEST(ComplexTest, StdInequalityOp) {
  mlir::NonFloatComplex<float> mc{7, 8};
  std::complex<float> sc{3, 4};

  EXPECT_NE(mc, sc);
  EXPECT_NE(sc, mc);
}

TEST(ComplexTest, RealFn) {
  mlir::NonFloatComplex<float> mc{4, 6};
  std::complex<float> sc{4, 6};

  EXPECT_EQ(real(mc), real(sc));
}

TEST(ComplexTest, ImagFn) {
  mlir::NonFloatComplex<float> mc{4, 6};
  std::complex<float> sc{4, 6};

  EXPECT_EQ(imag(mc), imag(sc));
}
