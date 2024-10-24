#ifndef MLIR_C_PRESBURGER_H
#define MLIR_C_PRESBURGER_H
#include "mlir-c/AffineExpr.h"

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

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name
DEFINE_C_API_STRUCT(MlirPresburgerIntegerRelation, void);
DEFINE_C_API_STRUCT(MlirPresburgerDynamicAPInt, const void);
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

/// Return whether `lhs` and `rhs` are equal. This is integer-exact
/// and somewhat expensive, since it uses the integer emptiness check
/// (see IntegerRelation::findIntegerSample()).
MLIR_CAPI_EXPORTED bool
mlirPresburgerIntegerRelationIsEqual(MlirPresburgerIntegerRelation lhs,
                                     MlirPresburgerIntegerRelation rhs);

/// Return the intersection of the two relations.
/// If there are locals, they will be merged.
MLIR_CAPI_EXPORTED MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationIntersect(MlirPresburgerIntegerRelation lhs,
                                       MlirPresburgerIntegerRelation rhs);

//===----------------------------------------------------------------------===//
// IntegerRelation Tableau Inspection
//===----------------------------------------------------------------------===//

/// Returns the value at the specified equality row and column.
MLIR_CAPI_EXPORTED MlirPresburgerDynamicAPInt
mlirPresburgerIntegerRelationAtEq(unsigned i, unsigned j);

/// The same, but casts to int64_t. This is unsafe and will assert-fail if the
/// value does not fit in an int64_t.
MLIR_CAPI_EXPORTED int64_t mlirPresburgerIntegerRelationAtEq64(
    MlirPresburgerIntegerRelation relation, unsigned row, unsigned col);

/// Returns the value at the specified inequality row and column.
MLIR_CAPI_EXPORTED MlirPresburgerDynamicAPInt
mlirPresburgerIntegerRelationAtIneq(MlirPresburgerIntegerRelation relation,
                                    unsigned row, unsigned col);

MLIR_CAPI_EXPORTED int64_t mlirPresburgerIntegerRelationAtIneq64(
    MlirPresburgerIntegerRelation relation, unsigned row, unsigned col);

/// Returns the number of inequalities and equalities.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumConstraints(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of inequality constraints.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumInequalities(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of equality constraints.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumEqualities(
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

/// Returns the total number of columns in the tableau.
MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationNumCols(MlirPresburgerIntegerRelation relation);

/// Return the VarKind of the var at the specified position.
MLIR_CAPI_EXPORTED MlirPresburgerVariableKind
mlirPresburgerIntegerRelationGetVarKindAt(unsigned pos);

MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationDump(MlirPresburgerIntegerRelation relation);

//===----------------------------------------------------------------------===//
// IntegerRelation Tableau Manipulation
//===----------------------------------------------------------------------===//
/// Adds an equality with the given coefficients.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationAddEquality(const int64_t *coefficients,
                                         size_t coefficientsSize);

/// Adds an inequality with the given coefficients.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationAddInequality(const int64_t *coefficients,
                                           size_t coefficientsSize);
#ifdef __cplusplus
}
#endif
#endif // MLIR_C_PRESBURGER_H