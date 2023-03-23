//===- FlatLinearValueConstraints.h - Linear Constraints --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_FLATLINEARVALUECONSTRAINTS_H
#define MLIR_ANALYSIS_FLATLINEARVALUECONSTRAINTS_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>

namespace mlir {

class AffineMap;
class IntegerSet;
class MLIRContext;
class Value;
class MemRefType;
struct MutableAffineMap;

namespace presburger {
class MultiAffineFunction;
} // namespace presburger

/// FlatLinearConstraints is an extension of IntegerPolyhedron. It provides an
/// AffineExpr-based API.
class FlatLinearConstraints : public presburger::IntegerPolyhedron {
public:
  /// Constructs a constraint system reserving memory for the specified number
  /// of constraints and variables. `valArgs` are the optional SSA values
  /// associated with each dimension/symbol. These must either be empty or match
  /// the number of dimensions and symbols.
  FlatLinearConstraints(unsigned numReservedInequalities,
                        unsigned numReservedEqualities,
                        unsigned numReservedCols, unsigned numDims,
                        unsigned numSymbols, unsigned numLocals)
      : IntegerPolyhedron(numReservedInequalities, numReservedEqualities,
                          numReservedCols,
                          presburger::PresburgerSpace::getSetSpace(
                              numDims, numSymbols, numLocals)) {
    assert(numReservedCols >= getNumVars() + 1);
  }

  /// Constructs a constraint system with the specified number of dimensions
  /// and symbols. `valArgs` are the optional SSA values associated with each
  /// dimension/symbol. These must either be empty or match the number of
  /// dimensions and symbols.
  FlatLinearConstraints(unsigned numDims = 0, unsigned numSymbols = 0,
                        unsigned numLocals = 0)
      : FlatLinearConstraints(/*numReservedInequalities=*/0,
                              /*numReservedEqualities=*/0,
                              /*numReservedCols=*/numDims + numSymbols +
                                  numLocals + 1,
                              numDims, numSymbols, numLocals) {}

  FlatLinearConstraints(const IntegerPolyhedron &fac)
      : IntegerPolyhedron(fac) {}

  /// Return the kind of this object.
  Kind getKind() const override { return Kind::FlatLinearConstraints; }

  static bool classof(const IntegerRelation *cst) {
    return cst->getKind() >= Kind::FlatLinearConstraints &&
           cst->getKind() <= Kind::FlatAffineRelation;
  }

  /// Clones this object.
  std::unique_ptr<FlatLinearConstraints> clone() const;

  /// Adds a bound for the variable at the specified position with constraints
  /// being drawn from the specified bound map. In case of an EQ bound, the
  /// bound map is expected to have exactly one result. In case of a LB/UB, the
  /// bound map may have more than one result, for each of which an inequality
  /// is added.
  ///
  /// The bound can be added as open or closed by specifying isClosedBound. In
  /// case of a LB/UB, isClosedBound = false means the bound is added internally
  /// as a closed bound by +1/-1 respectively. In case of an EQ bound, it can
  /// only be added as a closed bound.
  ///
  /// Note: The dimensions/symbols of this FlatLinearConstraints must match the
  /// dimensions/symbols of the affine map.
  LogicalResult addBound(presburger::BoundType type, unsigned pos,
                         AffineMap boundMap, bool isClosedBound);

  /// Adds a bound for the variable at the specified position with constraints
  /// being drawn from the specified bound map. In case of an EQ bound, the
  /// bound map is expected to have exactly one result. In case of a LB/UB, the
  /// bound map may have more than one result, for each of which an inequality
  /// is added.
  /// Note: The dimensions/symbols of this FlatLinearConstraints must match the
  /// dimensions/symbols of the affine map. By default the lower bound is closed
  /// and the upper bound is open.
  LogicalResult addBound(presburger::BoundType type, unsigned pos,
                         AffineMap boundMap);

  /// The `addBound` overload above hides the inherited overloads by default, so
  /// we explicitly introduce them here.
  using IntegerPolyhedron::addBound;

  /// Returns the constraint system as an integer set. Returns a null integer
  /// set if the system has no constraints, or if an integer set couldn't be
  /// constructed as a result of a local variable's explicit representation not
  /// being known and such a local variable appearing in any of the constraints.
  IntegerSet getAsIntegerSet(MLIRContext *context) const;

  /// Computes the lower and upper bounds of the first `num` dimensional
  /// variables (starting at `offset`) as an affine map of the remaining
  /// variables (dimensional and symbolic). This method is able to detect
  /// variables as floordiv's and mod's of affine expressions of other
  /// variables with respect to (positive) constants. Sets bound map to a
  /// null AffineMap if such a bound can't be found (or yet unimplemented).
  ///
  /// By default the returned lower bounds are closed and upper bounds are open.
  /// If `closedUb` is true, the upper bound is closed.
  void getSliceBounds(unsigned offset, unsigned num, MLIRContext *context,
                      SmallVectorImpl<AffineMap> *lbMaps,
                      SmallVectorImpl<AffineMap> *ubMaps,
                      bool closedUB = false);

  /// Composes an affine map whose dimensions and symbols match one to one with
  /// the dimensions and symbols of this FlatLinearConstraints. The results of
  /// the map `other` are added as the leading dimensions of this constraint
  /// system. Returns failure if `other` is a semi-affine map.
  LogicalResult composeMatchingMap(AffineMap other);

  /// Gets the lower and upper bound of the `offset` + `pos`th variable
  /// treating [0, offset) U [offset + num, symStartPos) as dimensions and
  /// [symStartPos, getNumDimAndSymbolVars) as symbols, and `pos` lies in
  /// [0, num). The multi-dimensional maps in the returned pair represent the
  /// max and min of potentially multiple affine expressions. `localExprs` holds
  /// pre-computed AffineExpr's for all local variables in the system.
  ///
  /// By default the returned lower bounds are closed and upper bounds are open.
  /// If `closedUb` is true, the upper bound is closed.
  std::pair<AffineMap, AffineMap>
  getLowerAndUpperBound(unsigned pos, unsigned offset, unsigned num,
                        unsigned symStartPos, ArrayRef<AffineExpr> localExprs,
                        MLIRContext *context, bool closedUB = false) const;

  /// Insert variables of the specified kind at position `pos`. Positions are
  /// relative to the kind of variable. The coefficient columns corresponding
  /// to the added variables are initialized to zero. `vals` are the Values
  /// corresponding to the variables. Values should not be used with
  /// VarKind::Local since values can only be attached to non-local variables.
  /// Return the absolute column position (i.e., not relative to the kind of
  /// variable) of the first added variable.
  ///
  /// Note: Empty Values are allowed in `vals`.
  unsigned insertDimVar(unsigned pos, unsigned num = 1) {
    return insertVar(VarKind::SetDim, pos, num);
  }
  unsigned insertSymbolVar(unsigned pos, unsigned num = 1) {
    return insertVar(VarKind::Symbol, pos, num);
  }
  unsigned insertLocalVar(unsigned pos, unsigned num = 1) {
    return insertVar(VarKind::Local, pos, num);
  }

  /// Append variables of the specified kind after the last variable of that
  /// kind. The coefficient columns corresponding to the added variables are
  /// initialized to zero. `vals` are the Values corresponding to the
  /// variables. Return the absolute column position (i.e., not relative to the
  /// kind of variable) of the first appended variable.
  ///
  /// Note: Empty Values are allowed in `vals`.
  unsigned appendDimVar(unsigned num = 1) {
    return appendVar(VarKind::SetDim, num);
  }
  unsigned appendSymbolVar(unsigned num = 1) {
    return appendVar(VarKind::Symbol, num);
  }
  unsigned appendLocalVar(unsigned num = 1) {
    return appendVar(VarKind::Local, num);
  }

protected:
  using VarKind = presburger::VarKind;

  /// Compute an explicit representation for local vars. For all systems coming
  /// from MLIR integer sets, maps, or expressions where local vars were
  /// introduced to model floordivs and mods, this always succeeds.
  LogicalResult computeLocalVars(SmallVectorImpl<AffineExpr> &memo,
                                 MLIRContext *context) const;

  /// Given an affine map that is aligned with this constraint system:
  /// * Flatten the map.
  /// * Add newly introduced local columns at the beginning of this constraint
  ///   system (local column pos 0).
  /// * Add equalities that define the new local columns to this constraint
  ///   system.
  /// * Return the flattened expressions via `flattenedExprs`.
  ///
  /// Note: This is a shared helper function of `addLowerOrUpperBound` and
  ///       `composeMatchingMap`.
  LogicalResult flattenAlignedMapAndMergeLocals(
      AffineMap map, std::vector<SmallVector<int64_t, 8>> *flattenedExprs);

  /// Prints the number of constraints, dimensions, symbols and locals in the
  /// FlatLinearConstraints. Also, prints for each variable whether there is
  /// an SSA Value attached to it.
  void printSpace(raw_ostream &os) const override;
};

/// FlatLinearValueConstraints represents an extension of FlatLinearConstraints
/// where each non-local variable can have an SSA Value attached to it.
class FlatLinearValueConstraints : public FlatLinearConstraints {
public:
  /// Constructs a constraint system reserving memory for the specified number
  /// of constraints and variables. `valArgs` are the optional SSA values
  /// associated with each dimension/symbol. These must either be empty or match
  /// the number of dimensions and symbols.
  FlatLinearValueConstraints(unsigned numReservedInequalities,
                             unsigned numReservedEqualities,
                             unsigned numReservedCols, unsigned numDims,
                             unsigned numSymbols, unsigned numLocals,
                             ArrayRef<std::optional<Value>> valArgs)
      : FlatLinearConstraints(numReservedInequalities, numReservedEqualities,
                              numReservedCols, numDims, numSymbols, numLocals) {
    assert(valArgs.empty() || valArgs.size() == getNumDimAndSymbolVars());
    values.reserve(numReservedCols);
    if (valArgs.empty())
      values.resize(getNumDimAndSymbolVars(), std::nullopt);
    else
      values.append(valArgs.begin(), valArgs.end());
  }

  /// Constructs a constraint system reserving memory for the specified number
  /// of constraints and variables. `valArgs` are the optional SSA values
  /// associated with each dimension/symbol. These must either be empty or match
  /// the number of dimensions and symbols.
  FlatLinearValueConstraints(unsigned numReservedInequalities,
                             unsigned numReservedEqualities,
                             unsigned numReservedCols, unsigned numDims,
                             unsigned numSymbols, unsigned numLocals,
                             ArrayRef<Value> valArgs)
      : FlatLinearConstraints(numReservedInequalities, numReservedEqualities,
                              numReservedCols, numDims, numSymbols, numLocals) {
    assert(valArgs.empty() || valArgs.size() == getNumDimAndSymbolVars());
    values.reserve(numReservedCols);
    if (valArgs.empty())
      values.resize(getNumDimAndSymbolVars(), std::nullopt);
    else
      values.append(valArgs.begin(), valArgs.end());
  }

  /// Constructs a constraint system with the specified number of dimensions
  /// and symbols. `valArgs` are the optional SSA values associated with each
  /// dimension/symbol. These must either be empty or match the number of
  /// dimensions and symbols.
  FlatLinearValueConstraints(unsigned numDims, unsigned numSymbols,
                             unsigned numLocals,
                             ArrayRef<std::optional<Value>> valArgs)
      : FlatLinearValueConstraints(/*numReservedInequalities=*/0,
                                   /*numReservedEqualities=*/0,
                                   /*numReservedCols=*/numDims + numSymbols +
                                       numLocals + 1,
                                   numDims, numSymbols, numLocals, valArgs) {}

  /// Constructs a constraint system with the specified number of dimensions
  /// and symbols. `valArgs` are the optional SSA values associated with each
  /// dimension/symbol. These must either be empty or match the number of
  /// dimensions and symbols.
  FlatLinearValueConstraints(unsigned numDims = 0, unsigned numSymbols = 0,
                             unsigned numLocals = 0,
                             ArrayRef<Value> valArgs = {})
      : FlatLinearValueConstraints(/*numReservedInequalities=*/0,
                                   /*numReservedEqualities=*/0,
                                   /*numReservedCols=*/numDims + numSymbols +
                                       numLocals + 1,
                                   numDims, numSymbols, numLocals, valArgs) {}

  FlatLinearValueConstraints(const IntegerPolyhedron &fac,
                             ArrayRef<std::optional<Value>> valArgs = {})
      : FlatLinearConstraints(fac) {
    assert(valArgs.empty() || valArgs.size() == getNumDimAndSymbolVars());
    if (valArgs.empty())
      values.resize(getNumDimAndSymbolVars(), std::nullopt);
    else
      values.append(valArgs.begin(), valArgs.end());
  }

  /// Creates an affine constraint system from an IntegerSet.
  explicit FlatLinearValueConstraints(IntegerSet set, ValueRange operands = {});

  // Construct a hyperrectangular constraint set from ValueRanges that represent
  // induction variables, lower and upper bounds. `ivs`, `lbs` and `ubs` are
  // expected to match one to one. The order of variables and constraints is:
  //
  // ivs | lbs | ubs | eq/ineq
  // ----+-----+-----+---------
  //   1   -1     0      >= 0
  // ----+-----+-----+---------
  //  -1    0     1      >= 0
  //
  // All dimensions as set as VarKind::SetDim.
  static FlatLinearValueConstraints
  getHyperrectangular(ValueRange ivs, ValueRange lbs, ValueRange ubs);

  /// Return the kind of this object.
  Kind getKind() const override { return Kind::FlatLinearValueConstraints; }

  static bool classof(const IntegerRelation *cst) {
    return cst->getKind() >= Kind::FlatLinearValueConstraints &&
           cst->getKind() <= Kind::FlatAffineRelation;
  }

  /// Replaces the contents of this FlatLinearValueConstraints with `other`.
  void clearAndCopyFrom(const IntegerRelation &other) override;

  /// Adds a constant bound for the variable associated with the given Value.
  void addBound(presburger::BoundType type, Value val, int64_t value);
  using FlatLinearConstraints::addBound;

  /// Returns the Value associated with the pos^th variable. Asserts if
  /// no Value variable was associated.
  inline Value getValue(unsigned pos) const {
    assert(pos < getNumDimAndSymbolVars() && "Invalid position");
    assert(hasValue(pos) && "variable's Value not set");
    return *values[pos];
  }

  /// Returns the Values associated with variables in range [start, end).
  /// Asserts if no Value was associated with one of these variables.
  inline void getValues(unsigned start, unsigned end,
                        SmallVectorImpl<Value> *values) const {
    assert(end <= getNumDimAndSymbolVars() && "invalid end position");
    assert(start <= end && "invalid start position");
    values->clear();
    values->reserve(end - start);
    for (unsigned i = start; i < end; i++)
      values->push_back(getValue(i));
  }
  inline void getAllValues(SmallVectorImpl<Value> *values) const {
    getValues(0, getNumDimAndSymbolVars(), values);
  }

  inline ArrayRef<std::optional<Value>> getMaybeValues() const {
    return {values.data(), values.size()};
  }

  inline ArrayRef<std::optional<Value>>
  getMaybeValues(presburger::VarKind kind) const {
    assert(kind != VarKind::Local &&
           "Local variables do not have any value attached to them.");
    return {values.data() + getVarKindOffset(kind), getNumVarKind(kind)};
  }

  /// Returns true if the pos^th variable has an associated Value.
  inline bool hasValue(unsigned pos) const {
    assert(pos < getNumDimAndSymbolVars() && "Invalid position");
    return values[pos].has_value();
  }

  /// Returns true if at least one variable has an associated Value.
  bool hasValues() const;

  unsigned appendDimVar(ValueRange vals);
  using FlatLinearConstraints::appendDimVar;

  unsigned appendSymbolVar(ValueRange vals);
  using FlatLinearConstraints::appendSymbolVar;

  unsigned insertDimVar(unsigned pos, ValueRange vals);
  using FlatLinearConstraints::insertDimVar;

  unsigned insertSymbolVar(unsigned pos, ValueRange vals);
  using FlatLinearConstraints::insertSymbolVar;

  unsigned insertVar(presburger::VarKind kind, unsigned pos,
                     unsigned num = 1) override;
  unsigned insertVar(presburger::VarKind kind, unsigned pos, ValueRange vals);

  /// Removes variables in the column range [varStart, varLimit), and copies any
  /// remaining valid data into place, updates member variables, and resizes
  /// arrays as needed.
  void removeVarRange(presburger::VarKind kind, unsigned varStart,
                      unsigned varLimit) override;
  using IntegerPolyhedron::removeVarRange;

  /// Sets the Value associated with the pos^th variable.
  inline void setValue(unsigned pos, Value val) {
    assert(pos < getNumDimAndSymbolVars() && "invalid var position");
    values[pos] = val;
  }

  /// Sets the Values associated with the variables in the range [start, end).
  /// The range must contain only dim and symbol variables.
  void setValues(unsigned start, unsigned end, ArrayRef<Value> values) {
    assert(end <= getNumVars() && "invalid end position");
    assert(start <= end && "invalid start position");
    assert(values.size() == end - start &&
           "value should be provided for each variable in the range.");
    for (unsigned i = start; i < end; ++i)
      setValue(i, values[i - start]);
  }

  /// Looks up the position of the variable with the specified Value. Returns
  /// true if found (false otherwise). `pos` is set to the (column) position of
  /// the variable.
  bool findVar(Value val, unsigned *pos) const;

  /// Returns true if a variable with the specified Value exists, false
  /// otherwise.
  bool containsVar(Value val) const;

  /// Projects out the variable that is associate with Value.
  void projectOut(Value val);
  using IntegerPolyhedron::projectOut;

  /// Swap the posA^th variable with the posB^th variable.
  void swapVar(unsigned posA, unsigned posB) override;

  /// Prints the number of constraints, dimensions, symbols and locals in the
  /// FlatAffineValueConstraints. Also, prints for each variable whether there
  /// is an SSA Value attached to it.
  void printSpace(raw_ostream &os) const override;

  /// Align `map` with this constraint system based on `operands`. Each operand
  /// must already have a corresponding dim/symbol in this constraint system.
  AffineMap computeAlignedMap(AffineMap map, ValueRange operands) const;

  /// Merge and align the variables of `this` and `other` starting at
  /// `offset`, so that both constraint systems get the union of the contained
  /// variables that is dimension-wise and symbol-wise unique; both
  /// constraint systems are updated so that they have the union of all
  /// variables, with `this`'s original variables appearing first followed
  /// by any of `other`'s variables that didn't appear in `this`. Local
  /// variables in `other` that have the same division representation as local
  /// variables in `this` are merged into one.
  //  E.g.: Input: `this`  has (%i, %j) [%M, %N]
  //               `other` has (%k, %j) [%P, %N, %M]
  //        Output: both `this`, `other` have (%i, %j, %k) [%M, %N, %P]
  //
  void mergeAndAlignVarsWithOther(unsigned offset,
                                  FlatLinearValueConstraints *other);

  /// Merge and align symbols of `this` and `other` such that both get union of
  /// of symbols that are unique. Symbols in `this` and `other` should be
  /// unique. Symbols with Value as `None` are considered to be inequal to all
  /// other symbols.
  void mergeSymbolVars(FlatLinearValueConstraints &other);

  /// Returns true if this constraint system and `other` are in the same
  /// space, i.e., if they are associated with the same set of variables,
  /// appearing in the same order. Returns false otherwise.
  bool areVarsAlignedWithOther(const FlatLinearConstraints &other);

  /// Updates the constraints to be the smallest bounding (enclosing) box that
  /// contains the points of `this` set and that of `other`, with the symbols
  /// being treated specially. For each of the dimensions, the min of the lower
  /// bounds (symbolic) and the max of the upper bounds (symbolic) is computed
  /// to determine such a bounding box. `other` is expected to have the same
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
  LogicalResult unionBoundingBox(const FlatLinearValueConstraints &other);
  using IntegerPolyhedron::unionBoundingBox;

protected:
  /// Eliminates the variable at the specified position using Fourier-Motzkin
  /// variable elimination, but uses Gaussian elimination if there is an
  /// equality involving that variable. If the result of the elimination is
  /// integer exact, `*isResultIntegerExact` is set to true. If `darkShadow` is
  /// set to true, a potential under approximation (subset) of the rational
  /// shadow / exact integer shadow is computed.
  // See implementation comments for more details.
  void fourierMotzkinEliminate(unsigned pos, bool darkShadow = false,
                               bool *isResultIntegerExact = nullptr) override;

  /// Returns false if the fields corresponding to various variable counts, or
  /// equality/inequality buffer sizes aren't consistent; true otherwise. This
  /// is meant to be used within an assert internally.
  bool hasConsistentState() const override;

  /// Values corresponding to the (column) non-local variables of this
  /// constraint system appearing in the order the variables correspond to
  /// columns. Variables that aren't associated with any Value are set to
  /// None.
  SmallVector<std::optional<Value>, 8> values;
};

/// Flattens 'expr' into 'flattenedExpr', which contains the coefficients of the
/// dimensions, symbols, and additional variables that represent floor divisions
/// of dimensions, symbols, and in turn other floor divisions.  Returns failure
/// if 'expr' could not be flattened (i.e., semi-affine is not yet handled).
/// 'cst' contains constraints that connect newly introduced local variables
/// to existing dimensional and symbolic variables. See documentation for
/// AffineExprFlattener on how mod's and div's are flattened.
LogicalResult getFlattenedAffineExpr(AffineExpr expr, unsigned numDims,
                                     unsigned numSymbols,
                                     SmallVectorImpl<int64_t> *flattenedExpr,
                                     FlatLinearConstraints *cst = nullptr);

/// Flattens the result expressions of the map to their corresponding flattened
/// forms and set in 'flattenedExprs'. Returns failure if any expression in the
/// map could not be flattened (i.e., semi-affine is not yet handled). 'cst'
/// contains constraints that connect newly introduced local variables to
/// existing dimensional and / symbolic variables. See documentation for
/// AffineExprFlattener on how mod's and div's are flattened. For all affine
/// expressions that share the same operands (like those of an affine map), this
/// method should be used instead of repeatedly calling getFlattenedAffineExpr
/// since local variables added to deal with div's and mod's will be reused
/// across expressions.
LogicalResult
getFlattenedAffineExprs(AffineMap map,
                        std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
                        FlatLinearConstraints *cst = nullptr);
LogicalResult
getFlattenedAffineExprs(IntegerSet set,
                        std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
                        FlatLinearConstraints *cst = nullptr);

LogicalResult
getMultiAffineFunctionFromMap(AffineMap map,
                              presburger::MultiAffineFunction &multiAff);

/// Re-indexes the dimensions and symbols of an affine map with given `operands`
/// values to align with `dims` and `syms` values.
///
/// Each dimension/symbol of the map, bound to an operand `o`, is replaced with
/// dimension `i`, where `i` is the position of `o` within `dims`. If `o` is not
/// in `dims`, replace it with symbol `i`, where `i` is the position of `o`
/// within `syms`. If `o` is not in `syms` either, replace it with a new symbol.
///
/// Note: If a value appears multiple times as a dimension/symbol (or both), all
/// corresponding dim/sym expressions are replaced with the first dimension
/// bound to that value (or first symbol if no such dimension exists).
///
/// The resulting affine map has `dims.size()` many dimensions and at least
/// `syms.size()` many symbols.
///
/// The SSA values of the symbols of the resulting map are optionally returned
/// via `newSyms`. This is a concatenation of `syms` with the SSA values of the
/// newly added symbols.
///
/// Note: As part of this re-indexing, dimensions may turn into symbols, or vice
/// versa.
AffineMap alignAffineMapWithValues(AffineMap map, ValueRange operands,
                                   ValueRange dims, ValueRange syms,
                                   SmallVector<Value> *newSyms = nullptr);

} // namespace mlir

#endif // MLIR_ANALYSIS_FLATLINEARVALUECONSTRAINTS_H
