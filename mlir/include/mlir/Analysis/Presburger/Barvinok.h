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

#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include <optional>

namespace mlir {
namespace presburger {

using PolyhedronH = IntegerRelation;
using PolyhedronV = IntMatrix;
using ConeH = PolyhedronH;
using ConeV = PolyhedronV;

inline ConeH defineHRep(int num_ineqs, int num_vars, int num_params = 0)
{
    // We don't distinguish between domain and range variables, so
    // we set the number of domain variables as 0 and the number of
    // range variables as the number of actual variables.
    // There are no symbols (non-parametric for now) and no local
    // (existentially quantified) variables.
    ConeH cone(PresburgerSpace::getRelationSpace(0, num_vars, num_params, 0));
    return cone;
}

// Get the index of a cone.
// If it has more rays than the dimension, return 0.
MPInt getIndex(ConeV);

// Get the dual of a cone in H-representation, returning the V-representation of it.
ConeV getDual(ConeH);

// Get the dual of a cone in V-representation, returning the H-representation of it.
ConeH getDual(ConeV);

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_BARVINOK_H