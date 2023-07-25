//===- Barvinok.h - MLIR Barvinok's Algorithm -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions and classes for Barvinok's algorithm in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_BARVINOK_H
#define MLIR_ANALYSIS_PRESBURGER_BARVINOK_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>


using PolyhedronH = IntegerRelation;
using PolyhedronV = Matrix;
using ConeH = PolyhedronH;
using Point = SmallVector<MPInt>

namespace mlir {
namespace presburger {

// Describe the type of generating function
// used to enumerate the integer points in a polytope.
// Consists of a set of terms, each having
// * a sign, ±1
// * a numerator, of the form x^{n}
// * a denominator, of the form (1 - x^{d1})...(1 - x^{dn})
class GeneratingFunction
{
public:
    GeneratingFunction(SmallVector<int> s, SmallVector<Point> nums, SmallVector<SmallVector<Point>> dens)
        : signs(s), numerators(nums), denominators(dens) {};

private:
    SmallVector<int> signs;
    SmallVector<Point> numerators;
    SmallVector<SmallVector<Point>> denominators;
};

// Get the index of a cone.
// If it has more rays than the dimension, return 0.
MPInt getIndex(ConeH);

// Get the smallest vector in the basis described by the rays of the cone,
// and the coefficients needed to express it in that basis.
std::pair<Point, SmallVector<MPInt>> getSamplePoint(ConeH);

// Get the dual of a cone, returning the V-representation of it.
PolyhedronV getDual(ConeH);

// Decompose a cone into unimodular cones,
// triangulating it first if it is not simplicial.
SmallVector<ConeH> unimodularDecomposition(ConeH);

// Decompose a simplicial cone into unimodular cones.
SmallVector<ConeH> unimodularDecompositionSimplicial(ConeH);

// Triangulate a non-simplicial cone into a simplicial cones.
SmallVector<ConeH> triangulate(Cone);

// Compute the generating function for a unimodular cone.
GeneratingFunction unimodularConeGeneratingFunction(ConeH);

// Compute the generating function for a polytope,
// as the sum of generating functions of its tangent cones.
GeneratingFunction polytopeGeneratingFunction(PolyhedronH);

// Substitute the generating function with the unit vector
// to find the number of terms.
MPInt substituteWithUnitVector(GeneratingFunction);

// Count the number of integer points in a polytope,
// by chaining together `polytopeGeneratingFunction`
// and `substituteWithUnitVector`.
MPInt countIntegerPoints(PolyhedronH);

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_BARVINOK_H