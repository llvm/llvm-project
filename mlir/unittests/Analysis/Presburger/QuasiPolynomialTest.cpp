//===- MatrixTest.cpp - Tests for QuasiPolynomial -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/QuasiPolynomial.h"
#include "./Utils.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

// Test the arithmetic operations on QuasiPolynomials;
// addition, subtraction, multiplication, and division
// by a constant.
// Two QPs of 3 parameters each were generated randomly
// and their sum, difference, and product computed by hand.
TEST(QuasiPolynomialTest, arithmetic) {
  QuasiPolynomial qp1(
      3, {Fraction(1, 3), Fraction(1, 1), Fraction(1, 2)},
      {{{Fraction(1, 1), Fraction(-1, 2), Fraction(4, 5), Fraction(0, 1)},
        {Fraction(2, 3), Fraction(3, 4), Fraction(-1, 1), Fraction(5, 7)}},
       {{Fraction(1, 2), Fraction(1, 1), Fraction(4, 5), Fraction(1, 1)}},
       {{Fraction(-3, 2), Fraction(1, 1), Fraction(5, 6), Fraction(7, 5)},
        {Fraction(1, 4), Fraction(2, 1), Fraction(6, 5), Fraction(-9, 8)},
        {Fraction(3, 2), Fraction(2, 5), Fraction(-7, 4), Fraction(0, 1)}}});
  QuasiPolynomial qp2(
      3, {Fraction(1, 1), Fraction(2, 1)},
      {{{Fraction(1, 2), Fraction(0, 1), Fraction(-1, 3), Fraction(5, 3)},
        {Fraction(2, 1), Fraction(5, 4), Fraction(9, 7), Fraction(-1, 5)}},
       {{Fraction(1, 3), Fraction(-2, 3), Fraction(1, 1), Fraction(0, 1)}}});

  QuasiPolynomial sum = qp1 + qp2;
  EXPECT_EQ_REPR_QUASIPOLYNOMIAL(
      sum,
      QuasiPolynomial(
          3,
          {Fraction(1, 3), Fraction(1, 1), Fraction(1, 2), Fraction(1, 1),
           Fraction(2, 1)},
          {{{Fraction(1, 1), Fraction(-1, 2), Fraction(4, 5), Fraction(0, 1)},
            {Fraction(2, 3), Fraction(3, 4), Fraction(-1, 1), Fraction(5, 7)}},
           {{Fraction(1, 2), Fraction(1, 1), Fraction(4, 5), Fraction(1, 1)}},
           {{Fraction(-3, 2), Fraction(1, 1), Fraction(5, 6), Fraction(7, 5)},
            {Fraction(1, 4), Fraction(2, 1), Fraction(6, 5), Fraction(-9, 8)},
            {Fraction(3, 2), Fraction(2, 5), Fraction(-7, 4), Fraction(0, 1)}},
           {{Fraction(1, 2), Fraction(0, 1), Fraction(-1, 3), Fraction(5, 3)},
            {Fraction(2, 1), Fraction(5, 4), Fraction(9, 7), Fraction(-1, 5)}},
           {{Fraction(1, 3), Fraction(-2, 3), Fraction(1, 1),
             Fraction(0, 1)}}}));

  QuasiPolynomial diff = qp1 - qp2;
  EXPECT_EQ_REPR_QUASIPOLYNOMIAL(
      diff,
      QuasiPolynomial(
          3,
          {Fraction(1, 3), Fraction(1, 1), Fraction(1, 2), Fraction(-1, 1),
           Fraction(-2, 1)},
          {{{Fraction(1, 1), Fraction(-1, 2), Fraction(4, 5), Fraction(0, 1)},
            {Fraction(2, 3), Fraction(3, 4), Fraction(-1, 1), Fraction(5, 7)}},
           {{Fraction(1, 2), Fraction(1, 1), Fraction(4, 5), Fraction(1, 1)}},
           {{Fraction(-3, 2), Fraction(1, 1), Fraction(5, 6), Fraction(7, 5)},
            {Fraction(1, 4), Fraction(2, 1), Fraction(6, 5), Fraction(-9, 8)},
            {Fraction(3, 2), Fraction(2, 5), Fraction(-7, 4), Fraction(0, 1)}},
           {{Fraction(1, 2), Fraction(0, 1), Fraction(-1, 3), Fraction(5, 3)},
            {Fraction(2, 1), Fraction(5, 4), Fraction(9, 7), Fraction(-1, 5)}},
           {{Fraction(1, 3), Fraction(-2, 3), Fraction(1, 1),
             Fraction(0, 1)}}}));

  QuasiPolynomial prod = qp1 * qp2;
  EXPECT_EQ_REPR_QUASIPOLYNOMIAL(
      prod,
      QuasiPolynomial(
          3,
          {Fraction(1, 3), Fraction(2, 3), Fraction(1, 1), Fraction(2, 1),
           Fraction(1, 2), Fraction(1, 1)},
          {{{Fraction(1, 1), Fraction(-1, 2), Fraction(4, 5), Fraction(0, 1)},
            {Fraction(2, 3), Fraction(3, 4), Fraction(-1, 1), Fraction(5, 7)},
            {Fraction(1, 2), Fraction(0, 1), Fraction(-1, 3), Fraction(5, 3)},
            {Fraction(2, 1), Fraction(5, 4), Fraction(9, 7), Fraction(-1, 5)}},
           {{Fraction(1, 1), Fraction(-1, 2), Fraction(4, 5), Fraction(0, 1)},
            {Fraction(2, 3), Fraction(3, 4), Fraction(-1, 1), Fraction(5, 7)},
            {Fraction(1, 3), Fraction(-2, 3), Fraction(1, 1), Fraction(0, 1)}},
           {{Fraction(1, 2), Fraction(1, 1), Fraction(4, 5), Fraction(1, 1)},
            {Fraction(1, 2), Fraction(0, 1), Fraction(-1, 3), Fraction(5, 3)},
            {Fraction(2, 1), Fraction(5, 4), Fraction(9, 7), Fraction(-1, 5)}},
           {{Fraction(1, 2), Fraction(1, 1), Fraction(4, 5), Fraction(1, 1)},
            {Fraction(1, 3), Fraction(-2, 3), Fraction(1, 1), Fraction(0, 1)}},
           {{Fraction(-3, 2), Fraction(1, 1), Fraction(5, 6), Fraction(7, 5)},
            {Fraction(1, 4), Fraction(2, 1), Fraction(6, 5), Fraction(-9, 8)},
            {Fraction(3, 2), Fraction(2, 5), Fraction(-7, 4), Fraction(0, 1)},
            {Fraction(1, 2), Fraction(0, 1), Fraction(-1, 3), Fraction(5, 3)},
            {Fraction(2, 1), Fraction(5, 4), Fraction(9, 7), Fraction(-1, 5)}},
           {{Fraction(-3, 2), Fraction(1, 1), Fraction(5, 6), Fraction(7, 5)},
            {Fraction(1, 4), Fraction(2, 1), Fraction(6, 5), Fraction(-9, 8)},
            {Fraction(3, 2), Fraction(2, 5), Fraction(-7, 4), Fraction(0, 1)},
            {Fraction(1, 3), Fraction(-2, 3), Fraction(1, 1),
             Fraction(0, 1)}}}));

  QuasiPolynomial quot = qp1 / 2;
  EXPECT_EQ_REPR_QUASIPOLYNOMIAL(
      quot,
      QuasiPolynomial(
          3, {Fraction(1, 6), Fraction(1, 2), Fraction(1, 4)},
          {{{Fraction(1, 1), Fraction(-1, 2), Fraction(4, 5), Fraction(0, 1)},
            {Fraction(2, 3), Fraction(3, 4), Fraction(-1, 1), Fraction(5, 7)}},
           {{Fraction(1, 2), Fraction(1, 1), Fraction(4, 5), Fraction(1, 1)}},
           {{Fraction(-3, 2), Fraction(1, 1), Fraction(5, 6), Fraction(7, 5)},
            {Fraction(1, 4), Fraction(2, 1), Fraction(6, 5), Fraction(-9, 8)},
            {Fraction(3, 2), Fraction(2, 5), Fraction(-7, 4),
             Fraction(0, 1)}}}));
}

// Test the simplify() operation on QPs, which removes terms that
// are identically zero. A random QP was generated and terms were
// changed to account for each condition in simplify() – 
// the term coefficient being zero, or all the coefficients in some
// affine term in the product being zero.
TEST(QuasiPolynomialTest, simplify) {
  QuasiPolynomial qp(2,
                     {Fraction(2, 3), Fraction(0, 1), Fraction(1, 1),
                      Fraction(1, 2), Fraction(0, 1)},
                     {{{Fraction(1, 1), Fraction(3, 4), Fraction(5, 3)},
                       {Fraction(2, 1), Fraction(0, 1), Fraction(0, 1)}},
                      {{Fraction(1, 3), Fraction(8, 5), Fraction(2, 5)}},
                      {{Fraction(2, 7), Fraction(9, 5), Fraction(0, 1)},
                       {Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)}},
                      {{Fraction(1, 1), Fraction(4, 5), Fraction(6, 5)}},
                      {{Fraction(1, 3), Fraction(4, 3), Fraction(7, 8)}}});
  EXPECT_EQ_REPR_QUASIPOLYNOMIAL(
      qp.simplify(),
      QuasiPolynomial(2, {Fraction(2, 3), Fraction(1, 2)},
                      {{{Fraction(1, 1), Fraction(3, 4), Fraction(5, 3)},
                        {Fraction(2, 1), Fraction(0, 1), Fraction(0, 1)}},
                       {{Fraction(1, 1), Fraction(4, 5), Fraction(6, 5)}}}));
}