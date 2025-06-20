//===-- mlir-c/Presburger.h - C API to Presburger library ---------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to Presburger library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_PRESBURGER_H
#define MLIR_C_PRESBURGER_H
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

enum MlirPresburgerVariableKind {
  Symbol,
  Local,
  Domain,
  Range,
  SetDim = Range
};

enum MlirPresburgerBoundType { EQ, LB, UB };

enum MlirPresburgerOptimumKind { Empty, Unbounded, Bounded };

struct MlirOptionalInt64 {
  bool hasValue;
  int64_t value;
};

typedef struct MlirOptionalInt64 MlirOptionalInt64;

struct MlirOptionalVectorInt64 {
  bool hasValue;
  const int64_t *data;
  int64_t size;
};

typedef struct MlirOptionalVectorInt64 MlirOptionalVectorInt64;

struct MlirMaybeOptimum {
  enum MlirPresburgerOptimumKind kind;
  MlirOptionalVectorInt64 vector;
};

typedef struct MlirMaybeOptimum MlirMaybeOptimum;

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name
DEFINE_C_API_STRUCT(MlirPresburgerIntegerRelation, void);
#undef DEFINE_C_API_STRUCT

//===----------------------------------------------------------------------===//
// IntegerRelation creation/destruction and basic metadata operations
//===----------------------------------------------------------------------===//

/// Constructs a relation reserving memory for the specified number
/// of constraints and variables.
MLIR_CAPI_EXPORTED MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationCreate(unsigned numReservedInequalities,
                                    unsigned numReservedEqualities,
                                    unsigned numReservedCols);

/// Constructs an IntegerRelation from a packed 2D matrix of tableau
/// coefficients in row-major order. The first `numDomainVars` columns are
/// considered domain and the remaining `numRangeVars` columns are domain
/// variables.
MLIR_CAPI_EXPORTED MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationCreateFromCoefficients(
    const int64_t *inequalityCoefficients, unsigned numInequalities,
    const int64_t *equalityCoefficients, unsigned numEqualities,
    unsigned numDomainVars, unsigned numRangeVars,
    unsigned numExtraReservedInequalities = 0,
    unsigned numExtraReservedEqualities = 0, unsigned numExtraReservedCols = 0);

/// Destroys an IntegerRelation.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationDestroy(MlirPresburgerIntegerRelation relation);

//===----------------------------------------------------------------------===//
// IntegerRelation binary operations
//===----------------------------------------------------------------------===//

/// Appends constraints from `lhs` into `rhs`. This is equivalent to an
/// intersection with no simplification of any sort attempted.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationAppend(MlirPresburgerIntegerRelation lhs,
                                    MlirPresburgerIntegerRelation rhs);

/// Return the intersection of the two relations.
/// If there are locals, they will be merged.
MLIR_CAPI_EXPORTED MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationIntersect(MlirPresburgerIntegerRelation lhs,
                                       MlirPresburgerIntegerRelation rhs);

/// Return whether `lhs` and `rhs` are equal. This is integer-exact
/// and somewhat expensive, since it uses the integer emptiness check
/// (see IntegerRelation::findIntegerSample()).
MLIR_CAPI_EXPORTED bool
mlirPresburgerIntegerRelationIsEqual(MlirPresburgerIntegerRelation lhs,
                                     MlirPresburgerIntegerRelation rhs);

/// Perform a quick equality check on `lhs` and `rhs`. The relations are
/// equal if the check return true, but may or may not be equal if the check
/// returns false. The equality check is performed in a plain manner, by
/// comparing if all the equalities and inequalities in `lhs` and `rhs`
/// are the same.
MLIR_CAPI_EXPORTED bool mlirPresburgerIntegerRelationIsObviouslyEqual(
    MlirPresburgerIntegerRelation lhs, MlirPresburgerIntegerRelation rhs);

/// Return whether `lhs` is a subset of the `rhs` IntegerRelation. This is
/// integer-exact and somewhat expensive, since it uses the integer emptiness.
MLIR_CAPI_EXPORTED bool
mlirPresburgerIntegerRelationIsSubsetOf(MlirPresburgerIntegerRelation lhs,
                                        MlirPresburgerIntegerRelation rhs);

/// Merge and align symbol variables of `lhs` and `rhs` with respect to
/// identifiers. After this operation the symbol variables of both relations
/// have the same identifiers in the same order.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationMergeAndAlignSymbols(
    MlirPresburgerIntegerRelation lhs, MlirPresburgerIntegerRelation rhs);

/// Adds additional local vars to the sets such that they both have the union
/// of the local vars in each set, without changing the set of points that
/// lie in `lhs` and `rhs`.
///
/// While taking union, if a local var in `rhs` has a division
/// representation which is a duplicate of division representation, of another
/// local var, it is not added to the final union of local vars and is instead
/// merged. The new ordering of local vars is:
///
/// [Local vars of `lhs`] [Non-merged local vars of `rhs`]
///
/// The relative ordering of local vars is same as before.
///
/// After merging, if the `i^th` local variable in one set has a known
/// division representation, then the `i^th` local variable in the other set
/// either has the same division representation or no known division
/// representation.
///
/// The spaces of both relations should be compatible.
///
/// Returns the number of non-merged local vars of `rhs`, i.e. the number of
/// locals that have been added to `lhs`.
MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationMergeLocalVars(MlirPresburgerIntegerRelation lhs,
                                            MlirPresburgerIntegerRelation rhs);

/// Let the relation `lhs` be R1, and the relation `rhs` be R2. Modifies R1
/// to be the composition of R1 and R2: R1;R2.
///
/// Formally, if R1: A -> B, and R2: B -> C, then this function returns a
/// relation R3: A -> C such that a point (a, c) belongs to R3 iff there
/// exists b such that (a, b) is in R1 and, (b, c) is in R2.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationCompose(MlirPresburgerIntegerRelation lhs,
                                     MlirPresburgerIntegerRelation rhs);

/// Let the relation `lhs` be R1, and the relation `rhs` be R2. Applies the
/// relation to the domain of R2.
///
/// R1: i -> j : (0 <= i < 2, j = i)
/// R2: i -> k : (k = i floordiv 2)
/// R3: k -> j : (0 <= k < 1, 2k <=  j <= 2k + 1)
///
/// R1 = {(0, 0), (1, 1)}. R2 maps both 0 and 1 to 0.
/// So R3 = {(0, 0), (0, 1)}.
///
/// Formally, R1.applyDomain(R2) = R2.inverse().compose(R1).
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationApplyDomain(MlirPresburgerIntegerRelation lhs,
                                         MlirPresburgerIntegerRelation rhs);

/// Let the relation `lhs` be R1, and the relation `rhs` be R2. Applies the
/// relation to the range of R2.
///
/// Formally, R1.applyRange(R2) is the same as R1.compose(R2) but we provide
/// this for uniformity with `applyDomain`.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationApplyRange(MlirPresburgerIntegerRelation lhs,
                                        MlirPresburgerIntegerRelation rhs);

/// Given a relation `rhs: (A -> B)`, this operation merges the symbol and
/// local variables and then takes the composition of `rhs` on `lhs: (B ->
/// C)`. The resulting relation represents tuples of the form: `A -> C`.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationMergeAndCompose(MlirPresburgerIntegerRelation lhs,
                                             MlirPresburgerIntegerRelation rhs);

/// Updates the constraints to be the smallest bounding (enclosing) box that
/// contains the points of `lhs` set and that of `rhs`, with the symbols
/// being treated specially. For each of the dimensions, the min of the lower
/// bounds (symbolic) and the max of the upper bounds (symbolic) is computed
/// to determine such a bounding box. `rhs` is expected to have the same
/// dimensional variables as this constraint system (in the same order).
///
/// E.g.:
/// 1) this   = {0 <= d0 <= 127},
///    other  = {16 <= d0 <= 192},
///    output = {0 <= d0 <= 192}
/// 2) this   = {s0 + 5 <= d0 <= s0 + 20},
///    other  = {s0 + 1 <= d0 <= s0 + 9},
///    output = {s0 + 1 <= d0 <= s0 + 20}
/// 3) this   = {0 <= d0 <= 5, 1 <= d1 <= 9}
///    other  = {2 <= d0 <= 6, 5 <= d1 <= 15},
///    output = {0 <= d0 <= 6, 1 <= d1 <= 15}
MLIR_CAPI_EXPORTED MlirLogicalResult
mlirPresburgerIntegerRelationUnionBoundingBox(
    MlirPresburgerIntegerRelation lhs, MlirPresburgerIntegerRelation rhs);

//===----------------------------------------------------------------------===//
// IntegerRelation Tableau Inspection
//===----------------------------------------------------------------------===//

/// Returns the value at the specified equality row and column.
MLIR_CAPI_EXPORTED int64_t mlirPresburgerIntegerRelationAtEq64(
    MlirPresburgerIntegerRelation relation, unsigned row, unsigned col);

/// Returns the value at the specified inequality row and column.
MLIR_CAPI_EXPORTED int64_t mlirPresburgerIntegerRelationAtIneq64(
    MlirPresburgerIntegerRelation relation, unsigned row, unsigned col);

/// Returns the number of inequalities and equalities.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumConstraints(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of columns classified as domain variables.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumDomainVars(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of columns classified as range variables.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumRangeVars(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of columns classified as symbol variables.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumSymbolVars(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of columns classified as local variables.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumLocalVars(
    MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationNumDimVars(MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumDimAndSymbolVars(
    MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationNumVars(MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationNumCols(MlirPresburgerIntegerRelation relation);

/// Returns the number of equality constraints.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumEqualities(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of inequality constraints.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumInequalities(
    MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumReservedEqualities(
    MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationNumReservedInequalities(
    MlirPresburgerIntegerRelation relation);

/// Get the number of vars of the specified kind.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationGetNumVarKind(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind);

/// Return the index at which the specified kind of vars starts.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationGetVarKindOffset(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind);

/// Return the index at Which the specified kind of vars ends.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationGetVarKindEnd(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind);

/// Get the number of elements of the specified kind in the range
/// [varStart, varLimit).
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationGetVarKindOverLap(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind,
    unsigned varStart, unsigned varLimit);

/// Return the VarKind of the var at the specified position.
MLIR_CAPI_EXPORTED MlirPresburgerVariableKind
mlirPresburgerIntegerRelationGetVarKindAt(
    MlirPresburgerIntegerRelation relation, unsigned pos);

/// Returns the constant bound for the pos^th variable if there is one.
MLIR_CAPI_EXPORTED MlirOptionalInt64
mlirPresburgerIntegerRelationGetConstantBound64(
    MlirPresburgerIntegerRelation relation, MlirPresburgerBoundType type,
    unsigned pos);

/// Check whether all local ids have a division representation.
MLIR_CAPI_EXPORTED bool mlirPresburgerIntegerRelationHasOnlyDivLocals(
    MlirPresburgerIntegerRelation relation);

// Verify whether the relation is full-dimensional, i.e.,
// no equality holds for the relation.
//
// If there are no variables, it always returns true.
// If there is at least one variable and the relation is empty, it returns
// false.
MLIR_CAPI_EXPORTED bool
mlirPresburgerIntegerRelationIsFullDim(MlirPresburgerIntegerRelation relation);

/// Find an integer sample point satisfying the constraints using a
/// branch and bound algorithm with generalized basis reduction, with some
/// additional processing using Simplex for unbounded sets.
///
/// Returns an integer sample point if one exists, or an empty Optional
/// otherwise. The returned value also includes values of local ids.
MLIR_CAPI_EXPORTED MlirOptionalVectorInt64
mlirPresburgerIntegerRelationFindIntegerSample(
    MlirPresburgerIntegerRelation relation);

/// Compute an overapproximation of the number of integer points in the
/// relation. Symbol vars currently not supported. If the computed
/// overapproximation is infinite, an empty optional is returned.
MLIR_CAPI_EXPORTED MlirOptionalInt64 mlirPresburgerIntegerRelationComputeVolume(
    MlirPresburgerIntegerRelation relation);

/// Returns true if the given point satisfies the constraints, or false
/// otherwise. Takes the values of all vars including locals.
MLIR_CAPI_EXPORTED bool mlirPresburgerIntegerRelationContainsPoint(
    MlirPresburgerIntegerRelation relation, const int64_t *point, int64_t size);

//===----------------------------------------------------------------------===//
// IntegerRelation Tableau Manipulation
//===----------------------------------------------------------------------===//

/// Insert `num` variables of the specified kind at position `pos`.
/// Positions are relative to the kind of variable. The coefficient columns
/// corresponding to the added variables are initialized to zero. Return the
/// absolute column position (i.e., not relative to the kind of variable)
/// of the first added variable.
MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationInsertVar(MlirPresburgerIntegerRelation relation,
                                       MlirPresburgerVariableKind kind,
                                       unsigned pos, unsigned num = 1);

/// Append `num` variables of the specified kind after the last variable
/// of that kind. The coefficient columns corresponding to the added variables
/// are initialized to zero. Return the absolute column position (i.e., not
/// relative to the kind of variable) of the first appended variable.
MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationAppendVar(MlirPresburgerIntegerRelation relation,
                                       MlirPresburgerVariableKind kind,
                                       unsigned num = 1);

/// Adds an equality with the given coefficients.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationAddEquality(MlirPresburgerIntegerRelation relation,
                                         const int64_t *coeff,
                                         int64_t coeffSize);

/// Adds an inequality with the given coefficients.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationAddInequality(
    MlirPresburgerIntegerRelation relation, const int64_t *coeff,
    int64_t coeffSize);

/// Eliminate the `posB^th` local variable, replacing every instance of it
/// with the `posA^th` local variable. This should be used when the two
/// local variables are known to always take the same values.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationEliminateRedundantLocalVar(
    MlirPresburgerIntegerRelation relation, unsigned posA, unsigned posB);

/// Removes variables of the specified kind with the specified pos (or
/// within the specified range) from the system. The specified location is
/// relative to the first variable of the specified kind.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveVarKind(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind,
    unsigned pos);
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveVarRangeKind(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind,
    unsigned varStart, unsigned varLimit);

/// Removes the specified variable from the system.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationRemoveVar(MlirPresburgerIntegerRelation relation,
                                       unsigned pos);

/// Remove the (in)equalities at specified position.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveEquality(
    MlirPresburgerIntegerRelation relation, unsigned pos);
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveInequality(
    MlirPresburgerIntegerRelation relation, unsigned pos);

/// Remove the (in)equalities at positions [start, end).
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveEqualityRange(
    MlirPresburgerIntegerRelation relation, unsigned start, unsigned end);
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveInequalityRange(
    MlirPresburgerIntegerRelation relation, unsigned start, unsigned end);

/// Returns lexicographically minimal integer point.
MLIR_CAPI_EXPORTED MlirMaybeOptimum
mlirPresburgerIntegerRelationFindIntegerLexMin(
    MlirPresburgerIntegerRelation relation);

/// Swap the posA^th variable with the posB^th variable.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationSwapVar(MlirPresburgerIntegerRelation relation,
                                     unsigned posA, unsigned posB);

/// Removes all equalities and inequalities.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationClearConstraints(
    MlirPresburgerIntegerRelation relation);

/// Sets the `values.size()` variables starting at `po`s to the specified
/// values and removes them.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationSetAndEliminate(
    MlirPresburgerIntegerRelation relation, unsigned pos, const int64_t *values,
    unsigned valuesSize);

/// Removes constraints that are independent of (i.e., do not have a
/// coefficient) variables in the range [pos, pos + num).
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationRemoveIndependentConstraints(
    MlirPresburgerIntegerRelation relation, unsigned pos, unsigned num);

/// Returns true if the set can be trivially detected as being
/// hyper-rectangular on the specified contiguous set of variables.
MLIR_CAPI_EXPORTED bool mlirPresburgerIntegerRelationIsHyperRectangular(
    MlirPresburgerIntegerRelation relation, unsigned pos, unsigned num);

/// Removes duplicate constraints, trivially true constraints, and constraints
/// that can be detected as redundant as a result of differing only in their
/// constant term part. A constraint of the form <non-negative constant> >= 0
/// is considered trivially true. This method is a linear time method on the
/// constraints, does a single scan, and updates in place. It also normalizes
/// constraints by their GCD and performs GCD tightening on inequalities.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveTrivialRedundancy(
    MlirPresburgerIntegerRelation relation);

/// A more expensive check than `removeTrivialRedundancy` to detect redundant
/// inequalities.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationRemoveRedundantInequalities(
    MlirPresburgerIntegerRelation relation);

/// Removes redundant constraints using Simplex. Although the algorithm can
/// theoretically take exponential time in the worst case (rare), it is known
/// to perform much better in the average case. If V is the number of vertices
/// in the polytope and C is the number of constraints, the algorithm takes
/// O(VC) time.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveRedundantConstraints(
    MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveDuplicateDivs(
    MlirPresburgerIntegerRelation relation);

/// Simplify the constraint system by removing canonicalizing constraints and
/// removing redundant constraints.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationSimplify(MlirPresburgerIntegerRelation relation);

/// Converts variables of kind srcKind in the range [varStart, varLimit) to
/// variables of kind dstKind. If `pos` is given, the variables are placed at
/// position `pos` of dstKind, otherwise they are placed after all the other
/// variables of kind dstKind. The internal ordering among the moved variables
/// is preserved.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationConvertVarKind(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind srcKind,
    unsigned varStart, unsigned varLimit, MlirPresburgerVariableKind dstKind,
    unsigned pos);
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationConvertVarKindNoPos(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind srcKind,
    unsigned varStart, unsigned varLimit, MlirPresburgerVariableKind dstKind);
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationConvertToLocal(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind,
    unsigned varStart, unsigned varLimit);

MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveTrivialEqualities(
    MlirPresburgerIntegerRelation relation);

/// Invert the relation i.e., swap its domain and range.
///
/// Formally, let the relation `this` be R: A -> B, then this operation
/// modifies R to be B -> A.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationInverse(MlirPresburgerIntegerRelation relation);

/// Adds a constant bound for the specified variable.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationAddBound(MlirPresburgerIntegerRelation relation,
                                      MlirPresburgerBoundType type,
                                      unsigned pos, int64_t value);

/// Adds a constant bound for the specified expression.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationAddBoundExpr(
    MlirPresburgerIntegerRelation relation, MlirPresburgerBoundType type,
    const int64_t *expr, int64_t exprSize, int64_t value);

/// Tries to fold the specified variable to a constant using a trivial
/// equality detection; if successful, the constant is substituted for the
/// variable everywhere in the constraint system and then removed from the
/// system.
MLIR_CAPI_EXPORTED MlirLogicalResult
mlirPresburgerIntegerRelationConstantFoldVar(
    MlirPresburgerIntegerRelation relation, unsigned pos);

/// This method calls `constantFoldVar` for the specified range of variables,
/// `num` variables starting at position `pos`.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationConstantFoldVarRange(
    MlirPresburgerIntegerRelation relation, unsigned pos, unsigned num);

//===----------------------------------------------------------------------===//
// IntegerRelation Dump
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationDump(MlirPresburgerIntegerRelation relation);

#ifdef __cplusplus
}
#endif
#endif // MLIR_C_PRESBURGER_H