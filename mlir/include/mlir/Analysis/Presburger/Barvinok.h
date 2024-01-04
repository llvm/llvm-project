//===- Barvinok.h - Barvinok's Algorithm ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions relating to Barvinok's algorithm.
// These include functions to manipulate cones (define a cone object, get its
// dual, and find its index).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_BARVINOK_H
#define MLIR_ANALYSIS_PRESBURGER_BARVINOK_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include <optional>

namespace mlir {
namespace presburger {

// A polyhedron in H-representation is a set of inequalities
// in d variables with integer coefficients.
using PolyhedronH = IntegerRelation;

// A polyhedron in V-representation is a set of rays and points, i.e.,
// vectors, stored as rows of a matrix.
using PolyhedronV = IntMatrix;

// A cone in either representation is a special case of
// a polyhedron in that representation.
using ConeH = PolyhedronH;
using ConeV = PolyhedronV;

inline ConeH defineHRep(int num_vars) {
  // We don't distinguish between domain and range variables, so
  // we set the number of domain variables as 0 and the number of
  // range variables as the number of actual variables.
  // There are no symbols (we don't work with parametric cones) and no local
  // (existentially quantified) variables.
  // Once the cone is defined, we use `addInequality()` to set inequalities.
  return ConeH(PresburgerSpace::getSetSpace(/*numDims=*/num_vars,
                                            /*numSymbols=*/0,
                                            /*numLocals=*/0));
}

// Get the index of a cone, i.e., the volume of the parallelepiped
// spanned by its generators, which is equal to the number of integer
// points in its fundamental parallelepiped.
// If the index is 1, the cone is unimodular.
// Barvinok, A., and J. E. Pommersheim. "An algorithmic theory of lattice points
// in polyhedra." p. 107
// If it has more rays than the dimension, return 0.
MPInt getIndex(ConeV);

// Get the dual of a cone in H-representation, returning its V-representation.
// This assumes that the input is pointed at the origin; it assert-fails
// otherwise.
ConeV getDual(ConeH);

// Get the dual of a cone in V-representation, returning its H-representation.
// The returned cone is pointed at the origin.
ConeH getDual(ConeV);

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_BARVINOK_H
