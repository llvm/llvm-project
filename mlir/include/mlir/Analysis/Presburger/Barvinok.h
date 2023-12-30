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

// A polyhedron in H-representation is a set of relations
// in d variables with integer coefficients.
using PolyhedronH = IntegerRelation;

// A polyhedron in V-representation is a set of rays, i.e.,
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
  return ConeH(PresburgerSpace::getRelationSpace(/*numDomain=*/0,
                                                 /*numRange=*/num_vars,
                                                 /*numSymbols=*/0,
                                                 /*numLocals=*/0));
}

// Get the index of a cone.
// If it has more rays than the dimension, return 0.
MPInt getIndex(ConeV);

// Get the dual of a cone in H-representation, returning its V-representation.
ConeV getDual(ConeH);

// Get the dual of a cone in V-representation, returning its H-representation.
ConeH getDual(ConeV);

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_BARVINOK_H