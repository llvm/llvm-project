//===- AffineStructures.h - MLIR Affine Structures Class --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Structures for affine/polyhedral analysis of ML functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_ANALYSIS_AFFINESTRUCTURES_H
#define MLIR_DIALECT_AFFINE_ANALYSIS_AFFINESTRUCTURES_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

class AffineCondition;
class AffineForOp;
class AffineIfOp;
class AffineMap;
class AffineValueMap;
class IntegerSet;
class MLIRContext;
class Value;
class MemRefType;
struct MutableAffineMap;

/// FlatAffineValueConstraints represents an extension of IntegerPolyhedron
/// where each identifier can have an SSA Value attached to it.
class FlatAffineValueConstraints : public presburger::IntegerPolyhedron {
public:
  /// Constructs a constraint system reserving memory for the specified number
  /// of constraints and identifiers.
  FlatAffineValueConstraints(unsigned numReservedInequalities,
                             unsigned numReservedEqualities,
                             unsigned numReservedCols, unsigned numDims,
                             unsigned numSymbols, unsigned numLocals,
                             ArrayRef<Optional<Value>> valArgs = {})
      : IntegerPolyhedron(numReservedInequalities, numReservedEqualities,
                          numReservedCols,
                          presburger::PresburgerSpace::getSetSpace(
                              numDims, numSymbols, numLocals)) {
    assert(numReservedCols >= getNumIds() + 1);
    assert(valArgs.empty() || valArgs.size() == getNumIds());
    values.reserve(numReservedCols);
    if (valArgs.empty())
      values.resize(getNumIds(), None);
    else
      values.append(valArgs.begin(), valArgs.end());
  }

  /// Constructs a constraint system with the specified number of
  /// dimensions and symbols.
  FlatAffineValueConstraints(unsigned numDims = 0, unsigned numSymbols = 0,
                             unsigned numLocals = 0,
                             ArrayRef<Optional<Value>> valArgs = {})
      : FlatAffineValueConstraints(/*numReservedInequalities=*/0,
                                   /*numReservedEqualities=*/0,
                                   /*numReservedCols=*/numDims + numSymbols +
                                       numLocals + 1,
                                   numDims, numSymbols, numLocals, valArgs) {}

  FlatAffineValueConstraints(const IntegerPolyhedron &fac,
                             ArrayRef<Optional<Value>> valArgs = {})
      : IntegerPolyhedron(fac) {
    assert(valArgs.empty() || valArgs.size() == getNumIds());
    if (valArgs.empty())
      values.resize(getNumIds(), None);
    else
      values.append(valArgs.begin(), valArgs.end());
  }

  /// Create a flat affine constraint system from an AffineValueMap or a list of
  /// these. The constructed system will only include equalities.
  explicit FlatAffineValueConstraints(const AffineValueMap &avm);
  explicit FlatAffineValueConstraints(ArrayRef<const AffineValueMap *> avmRef);

  /// Creates an affine constraint system from an IntegerSet.
  explicit FlatAffineValueConstraints(IntegerSet set);

  FlatAffineValueConstraints(ArrayRef<const AffineValueMap *> avmRef,
                             IntegerSet set);

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
  // All dimensions as set as DimId.
  static FlatAffineValueConstraints
  getHyperrectangular(ValueRange ivs, ValueRange lbs, ValueRange ubs);

  /// Return the kind of this FlatAffineConstraints.
  Kind getKind() const override { return Kind::FlatAffineValueConstraints; }

  static bool classof(const IntegerRelation *cst) {
    return cst->getKind() == Kind::FlatAffineValueConstraints;
  }

  /// Clears any existing data and reserves memory for the specified
  /// constraints.
  void reset(unsigned numReservedInequalities, unsigned numReservedEqualities,
             unsigned numReservedCols, unsigned numDims, unsigned numSymbols,
             unsigned numLocals = 0);
  void reset(unsigned numDims = 0, unsigned numSymbols = 0,
             unsigned numLocals = 0);
  void reset(unsigned numReservedInequalities, unsigned numReservedEqualities,
             unsigned numReservedCols, unsigned numDims, unsigned numSymbols,
             unsigned numLocals, ArrayRef<Value> valArgs);
  void reset(unsigned numDims, unsigned numSymbols, unsigned numLocals,
             ArrayRef<Value> valArgs);

  /// Clones this object.
  std::unique_ptr<FlatAffineValueConstraints> clone() const;

  /// Adds constraints (lower and upper bounds) for the specified 'affine.for'
  /// operation's Value using IR information stored in its bound maps. The
  /// right identifier is first looked up using `forOp`'s Value. Asserts if the
  /// Value corresponding to the 'affine.for' operation isn't found in the
  /// constraint system. Returns failure for the yet unimplemented/unsupported
  /// cases.  Any new identifiers that are found in the bound operands of the
  /// 'affine.for' operation are added as trailing identifiers (either
  /// dimensional or symbolic depending on whether the operand is a valid
  /// symbol).
  //  TODO: add support for non-unit strides.
  LogicalResult addAffineForOpDomain(AffineForOp forOp);

  /// Adds constraints (lower and upper bounds) for each loop in the loop nest
  /// described by the bound maps `lbMaps` and `ubMaps` of a computation slice.
  /// Every pair (`lbMaps[i]`, `ubMaps[i]`) describes the bounds of a loop in
  /// the nest, sorted outer-to-inner. `operands` contains the bound operands
  /// for a single bound map. All the bound maps will use the same bound
  /// operands. Note that some loops described by a computation slice might not
  /// exist yet in the IR so the Value attached to those dimension identifiers
  /// might be empty. For that reason, this method doesn't perform Value
  /// look-ups to retrieve the dimension identifier positions. Instead, it
  /// assumes the position of the dim identifiers in the constraint system is
  /// the same as the position of the loop in the loop nest.
  LogicalResult addDomainFromSliceMaps(ArrayRef<AffineMap> lbMaps,
                                       ArrayRef<AffineMap> ubMaps,
                                       ArrayRef<Value> operands);

  /// Adds constraints imposed by the `affine.if` operation. These constraints
  /// are collected from the IntegerSet attached to the given `affine.if`
  /// instance argument (`ifOp`). It is asserted that:
  /// 1) The IntegerSet of the given `affine.if` instance should not contain
  /// semi-affine expressions,
  /// 2) The columns of the constraint system created from `ifOp` should match
  /// the columns in the current one regarding numbers and values.
  void addAffineIfOpDomain(AffineIfOp ifOp);

  /// Adds a bound for the identifier at the specified position with constraints
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
  /// Note: The dimensions/symbols of this FlatAffineConstraints must match the
  /// dimensions/symbols of the affine map.
  LogicalResult addBound(BoundType type, unsigned pos, AffineMap boundMap,
                         bool isClosedBound);

  /// Adds a bound for the identifier at the specified position with constraints
  /// being drawn from the specified bound map. In case of an EQ bound, the
  /// bound map is expected to have exactly one result. In case of a LB/UB, the
  /// bound map may have more than one result, for each of which an inequality
  /// is added.
  /// Note: The dimensions/symbols of this FlatAffineConstraints must match the
  /// dimensions/symbols of the affine map. By default the lower bound is closed
  /// and the upper bound is open.
  LogicalResult addBound(BoundType type, unsigned pos, AffineMap boundMap);

  /// Adds a bound for the identifier at the specified position with constraints
  /// being drawn from the specified bound map and operands. In case of an
  /// EQ bound, the  bound map is expected to have exactly one result. In case
  /// of a LB/UB, the bound map may have more than one result, for each of which
  /// an inequality is added.
  LogicalResult addBound(BoundType type, unsigned pos, AffineMap boundMap,
                         ValueRange operands);

  /// Adds a constant bound for the identifier associated with the given Value.
  void addBound(BoundType type, Value val, int64_t value);

  /// The `addBound` overload above hides the inherited overloads by default, so
  /// we explicitly introduce them here.
  using IntegerPolyhedron::addBound;

  /// Returns the constraint system as an integer set. Returns a null integer
  /// set if the system has no constraints, or if an integer set couldn't be
  /// constructed as a result of a local variable's explicit representation not
  /// being known and such a local variable appearing in any of the constraints.
  IntegerSet getAsIntegerSet(MLIRContext *context) const;

  /// Computes the lower and upper bounds of the first `num` dimensional
  /// identifiers (starting at `offset`) as an affine map of the remaining
  /// identifiers (dimensional and symbolic). This method is able to detect
  /// identifiers as floordiv's and mod's of affine expressions of other
  /// identifiers with respect to (positive) constants. Sets bound map to a
  /// null AffineMap if such a bound can't be found (or yet unimplemented).
  ///
  /// By default the returned lower bounds are closed and upper bounds are open.
  /// This can be changed by getClosedUB.
  void getSliceBounds(unsigned offset, unsigned num, MLIRContext *context,
                      SmallVectorImpl<AffineMap> *lbMaps,
                      SmallVectorImpl<AffineMap> *ubMaps,
                      bool getClosedUB = false);

  /// Composes an affine map whose dimensions and symbols match one to one with
  /// the dimensions and symbols of this FlatAffineConstraints. The results of
  /// the map `other` are added as the leading dimensions of this constraint
  /// system. Returns failure if `other` is a semi-affine map.
  LogicalResult composeMatchingMap(AffineMap other);

  /// Gets the lower and upper bound of the `offset` + `pos`th identifier
  /// treating [0, offset) U [offset + num, symStartPos) as dimensions and
  /// [symStartPos, getNumDimAndSymbolIds) as symbols, and `pos` lies in
  /// [0, num). The multi-dimensional maps in the returned pair represent the
  /// max and min of potentially multiple affine expressions. The upper bound is
  /// exclusive. `localExprs` holds pre-computed AffineExpr's for all local
  /// identifiers in the system.
  std::pair<AffineMap, AffineMap>
  getLowerAndUpperBound(unsigned pos, unsigned offset, unsigned num,
                        unsigned symStartPos, ArrayRef<AffineExpr> localExprs,
                        MLIRContext *context) const;

  /// Returns the bound for the identifier at `pos` from the inequality at
  /// `ineqPos` as a 1-d affine value map (affine map + operands). The returned
  /// affine value map can either be a lower bound or an upper bound depending
  /// on the sign of atIneq(ineqPos, pos). Asserts if the row at `ineqPos` does
  /// not involve the `pos`th identifier.
  void getIneqAsAffineValueMap(unsigned pos, unsigned ineqPos,
                               AffineValueMap &vmap,
                               MLIRContext *context) const;

  /// Adds slice lower bounds represented by lower bounds in `lbMaps` and upper
  /// bounds in `ubMaps` to each identifier in the constraint system which has
  /// a value in `values`. Note that both lower/upper bounds share the same
  /// operand list `operands`.
  /// This function assumes `values.size` == `lbMaps.size` == `ubMaps.size`.
  /// Note that both lower/upper bounds use operands from `operands`.
  LogicalResult addSliceBounds(ArrayRef<Value> values,
                               ArrayRef<AffineMap> lbMaps,
                               ArrayRef<AffineMap> ubMaps,
                               ArrayRef<Value> operands);

  /// Looks up the position of the identifier with the specified Value. Returns
  /// true if found (false otherwise). `pos` is set to the (column) position of
  /// the identifier.
  bool findId(Value val, unsigned *pos) const;

  /// Returns true if an identifier with the specified Value exists, false
  /// otherwise.
  bool containsId(Value val) const;

  /// Swap the posA^th identifier with the posB^th identifier.
  void swapId(unsigned posA, unsigned posB) override;

  /// Insert identifiers of the specified kind at position `pos`. Positions are
  /// relative to the kind of identifier. The coefficient columns corresponding
  /// to the added identifiers are initialized to zero. `vals` are the Values
  /// corresponding to the identifiers. Return the absolute column position
  /// (i.e., not relative to the kind of identifier) of the first added
  /// identifier.
  ///
  /// Note: Empty Values are allowed in `vals`.
  unsigned insertDimId(unsigned pos, unsigned num = 1) {
    return insertId(IdKind::SetDim, pos, num);
  }
  unsigned insertSymbolId(unsigned pos, unsigned num = 1) {
    return insertId(IdKind::Symbol, pos, num);
  }
  unsigned insertLocalId(unsigned pos, unsigned num = 1) {
    return insertId(IdKind::Local, pos, num);
  }
  unsigned insertDimId(unsigned pos, ValueRange vals);
  unsigned insertSymbolId(unsigned pos, ValueRange vals);
  unsigned insertId(presburger::IdKind kind, unsigned pos,
                    unsigned num = 1) override;
  unsigned insertId(presburger::IdKind kind, unsigned pos, ValueRange vals);

  /// Append identifiers of the specified kind after the last identifier of that
  /// kind. The coefficient columns corresponding to the added identifiers are
  /// initialized to zero. `vals` are the Values corresponding to the
  /// identifiers. Return the position of the first added column.
  ///
  /// Note: Empty Values are allowed in `vals`.
  unsigned appendDimId(ValueRange vals);
  unsigned appendSymbolId(ValueRange vals);
  unsigned appendDimId(unsigned num = 1) {
    return appendId(IdKind::SetDim, num);
  }
  unsigned appendSymbolId(unsigned num = 1) {
    return appendId(IdKind::Symbol, num);
  }
  unsigned appendLocalId(unsigned num = 1) {
    return appendId(IdKind::Local, num);
  }

  /// Removes identifiers in the column range [idStart, idLimit), and copies any
  /// remaining valid data into place, updates member variables, and resizes
  /// arrays as needed.
  void removeIdRange(presburger::IdKind kind, unsigned idStart,
                     unsigned idLimit) override;
  using IntegerPolyhedron::removeIdRange;

  /// Add the specified values as a dim or symbol id depending on its nature, if
  /// it already doesn't exist in the system. `val` has to be either a terminal
  /// symbol or a loop IV, i.e., it cannot be the result affine.apply of any
  /// symbols or loop IVs. The identifier is added to the end of the existing
  /// dims or symbols. Additional information on the identifier is extracted
  /// from the IR and added to the constraint system.
  void addInductionVarOrTerminalSymbol(Value val);

  /// Align `map` with this constraint system based on `operands`. Each operand
  /// must already have a corresponding dim/symbol in this constraint system.
  AffineMap computeAlignedMap(AffineMap map, ValueRange operands) const;

  /// Composes the affine value map with this FlatAffineValueConstrains, adding
  /// the results of the map as dimensions at the front
  /// [0, vMap->getNumResults()) and with the dimensions set to the equalities
  /// specified by the value map.
  ///
  /// Returns failure if the composition fails (when vMap is a semi-affine map).
  /// The vMap's operand Value's are used to look up the right positions in
  /// the FlatAffineConstraints with which to associate. Every operand of vMap
  /// should have a matching dim/symbol column in this constraint system (with
  /// the same associated Value).
  LogicalResult composeMap(const AffineValueMap *vMap);

  /// Projects out the identifier that is associate with Value.
  void projectOut(Value val);
  using IntegerPolyhedron::projectOut;

  /// Changes all symbol identifiers which are loop IVs to dim identifiers.
  void convertLoopIVSymbolsToDims();

  /// Updates the constraints to be the smallest bounding (enclosing) box that
  /// contains the points of `this` set and that of `other`, with the symbols
  /// being treated specially. For each of the dimensions, the min of the lower
  /// bounds (symbolic) and the max of the upper bounds (symbolic) is computed
  /// to determine such a bounding box. `other` is expected to have the same
  /// dimensional identifiers as this constraint system (in the same order).
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
  LogicalResult unionBoundingBox(const FlatAffineValueConstraints &other);
  using IntegerPolyhedron::unionBoundingBox;

  /// Merge and align the identifiers of `this` and `other` starting at
  /// `offset`, so that both constraint systems get the union of the contained
  /// identifiers that is dimension-wise and symbol-wise unique; both
  /// constraint systems are updated so that they have the union of all
  /// identifiers, with `this`'s original identifiers appearing first followed
  /// by any of `other`'s identifiers that didn't appear in `this`. Local
  /// identifiers in `other` that have the same division representation as local
  /// identifiers in `this` are merged into one.
  //  E.g.: Input: `this`  has (%i, %j) [%M, %N]
  //               `other` has (%k, %j) [%P, %N, %M]
  //        Output: both `this`, `other` have (%i, %j, %k) [%M, %N, %P]
  //
  void mergeAndAlignIdsWithOther(unsigned offset,
                                 FlatAffineValueConstraints *other);

  /// Returns true if this constraint system and `other` are in the same
  /// space, i.e., if they are associated with the same set of identifiers,
  /// appearing in the same order. Returns false otherwise.
  bool areIdsAlignedWithOther(const FlatAffineValueConstraints &other);

  /// Replaces the contents of this FlatAffineValueConstraints with `other`.
  void clearAndCopyFrom(const IntegerRelation &other) override;

  /// Returns the Value associated with the pos^th identifier. Asserts if
  /// no Value identifier was associated.
  inline Value getValue(unsigned pos) const {
    assert(hasValue(pos) && "identifier's Value not set");
    return values[pos].getValue();
  }

  /// Returns true if the pos^th identifier has an associated Value.
  inline bool hasValue(unsigned pos) const { return values[pos].hasValue(); }

  /// Returns true if at least one identifier has an associated Value.
  bool hasValues() const;

  /// Returns the Values associated with identifiers in range [start, end).
  /// Asserts if no Value was associated with one of these identifiers.
  inline void getValues(unsigned start, unsigned end,
                        SmallVectorImpl<Value> *values) const {
    assert((start < getNumIds() || start == end) && "invalid start position");
    assert(end <= getNumIds() && "invalid end position");
    values->clear();
    values->reserve(end - start);
    for (unsigned i = start; i < end; i++)
      values->push_back(getValue(i));
  }
  inline void getAllValues(SmallVectorImpl<Value> *values) const {
    getValues(0, getNumIds(), values);
  }

  inline ArrayRef<Optional<Value>> getMaybeValues() const {
    return {values.data(), values.size()};
  }

  inline ArrayRef<Optional<Value>> getMaybeDimValues() const {
    return {values.data(), getNumDimIds()};
  }

  inline ArrayRef<Optional<Value>> getMaybeSymbolValues() const {
    return {values.data() + getNumDimIds(), getNumSymbolIds()};
  }

  inline ArrayRef<Optional<Value>> getMaybeDimAndSymbolValues() const {
    return {values.data(), getNumDimIds() + getNumSymbolIds()};
  }

  /// Sets the Value associated with the pos^th identifier.
  inline void setValue(unsigned pos, Value val) {
    assert(pos < getNumIds() && "invalid id position");
    values[pos] = val;
  }

  /// Sets the Values associated with the identifiers in the range [start, end).
  void setValues(unsigned start, unsigned end, ArrayRef<Value> values) {
    assert((start < getNumIds() || end == start) && "invalid start position");
    assert(end <= getNumIds() && "invalid end position");
    assert(values.size() == end - start);
    for (unsigned i = start; i < end; ++i)
      setValue(i, values[i - start]);
  }

  /// Merge and align symbols of `this` and `other` such that both get union of
  /// of symbols that are unique. Symbols in `this` and `other` should be
  /// unique. Symbols with Value as `None` are considered to be inequal to all
  /// other symbols.
  void mergeSymbolIds(FlatAffineValueConstraints &other);

protected:
  using IdKind = presburger::IdKind;

  /// Returns false if the fields corresponding to various identifier counts, or
  /// equality/inequality buffer sizes aren't consistent; true otherwise. This
  /// is meant to be used within an assert internally.
  bool hasConsistentState() const override;

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

  /// Eliminates the identifier at the specified position using Fourier-Motzkin
  /// variable elimination, but uses Gaussian elimination if there is an
  /// equality involving that identifier. If the result of the elimination is
  /// integer exact, `*isResultIntegerExact` is set to true. If `darkShadow` is
  /// set to true, a potential under approximation (subset) of the rational
  /// shadow / exact integer shadow is computed.
  // See implementation comments for more details.
  void fourierMotzkinEliminate(unsigned pos, bool darkShadow = false,
                               bool *isResultIntegerExact = nullptr) override;

  /// Prints the number of constraints, dimensions, symbols and locals in the
  /// FlatAffineConstraints. Also, prints for each identifier whether there is
  /// an SSA Value attached to it.
  void printSpace(raw_ostream &os) const override;

  /// Values corresponding to the (column) identifiers of this constraint
  /// system appearing in the order the identifiers correspond to columns.
  /// Temporary ones or those that aren't associated with any Value are set to
  /// None.
  SmallVector<Optional<Value>, 8> values;
};

/// A FlatAffineRelation represents a set of ordered pairs (domain -> range)
/// where "domain" and "range" are tuples of identifiers. The relation is
/// represented as a FlatAffineValueConstraints with separation of dimension
/// identifiers into domain and  range. The identifiers are stored as:
/// [domainIds, rangeIds, symbolIds, localIds, constant].
class FlatAffineRelation : public FlatAffineValueConstraints {
public:
  FlatAffineRelation(unsigned numReservedInequalities,
                     unsigned numReservedEqualities, unsigned numReservedCols,
                     unsigned numDomainDims, unsigned numRangeDims,
                     unsigned numSymbols, unsigned numLocals,
                     ArrayRef<Optional<Value>> valArgs = {})
      : FlatAffineValueConstraints(
            numReservedInequalities, numReservedEqualities, numReservedCols,
            numDomainDims + numRangeDims, numSymbols, numLocals, valArgs),
        numDomainDims(numDomainDims), numRangeDims(numRangeDims) {}

  FlatAffineRelation(unsigned numDomainDims = 0, unsigned numRangeDims = 0,
                     unsigned numSymbols = 0, unsigned numLocals = 0)
      : FlatAffineValueConstraints(numDomainDims + numRangeDims, numSymbols,
                                   numLocals),
        numDomainDims(numDomainDims), numRangeDims(numRangeDims) {}

  FlatAffineRelation(unsigned numDomainDims, unsigned numRangeDims,
                     FlatAffineValueConstraints &fac)
      : FlatAffineValueConstraints(fac), numDomainDims(numDomainDims),
        numRangeDims(numRangeDims) {}

  FlatAffineRelation(unsigned numDomainDims, unsigned numRangeDims,
                     IntegerPolyhedron &fac)
      : FlatAffineValueConstraints(fac), numDomainDims(numDomainDims),
        numRangeDims(numRangeDims) {}

  /// Returns a set corresponding to the domain/range of the affine relation.
  FlatAffineValueConstraints getDomainSet() const;
  FlatAffineValueConstraints getRangeSet() const;

  /// Returns the number of identifiers corresponding to domain/range of
  /// relation.
  inline unsigned getNumDomainDims() const { return numDomainDims; }
  inline unsigned getNumRangeDims() const { return numRangeDims; }

  /// Given affine relation `other: (domainOther -> rangeOther)`, this operation
  /// takes the composition of `other` on `this: (domainThis -> rangeThis)`.
  /// The resulting relation represents tuples of the form: `domainOther ->
  /// rangeThis`.
  void compose(const FlatAffineRelation &other);

  /// Swap domain and range of the relation.
  /// `(domain -> range)` is converted to `(range -> domain)`.
  void inverse();

  /// Insert `num` identifiers of the specified kind after the `pos` identifier
  /// of that kind. The coefficient columns corresponding to the added
  /// identifiers are initialized to zero.
  void insertDomainId(unsigned pos, unsigned num = 1);
  void insertRangeId(unsigned pos, unsigned num = 1);

  /// Append `num` identifiers of the specified kind after the last identifier
  /// of that kind. The coefficient columns corresponding to the added
  /// identifiers are initialized to zero.
  void appendDomainId(unsigned num = 1);
  void appendRangeId(unsigned num = 1);

  /// Removes identifiers in the column range [idStart, idLimit), and copies any
  /// remaining valid data into place, updates member variables, and resizes
  /// arrays as needed.
  void removeIdRange(IdKind kind, unsigned idStart, unsigned idLimit) override;
  using IntegerRelation::removeIdRange;

protected:
  // Number of dimension identifers corresponding to domain identifers.
  unsigned numDomainDims;

  // Number of dimension identifers corresponding to range identifers.
  unsigned numRangeDims;
};

/// Flattens 'expr' into 'flattenedExpr', which contains the coefficients of the
/// dimensions, symbols, and additional variables that represent floor divisions
/// of dimensions, symbols, and in turn other floor divisions.  Returns failure
/// if 'expr' could not be flattened (i.e., semi-affine is not yet handled).
/// 'cst' contains constraints that connect newly introduced local identifiers
/// to existing dimensional and symbolic identifiers. See documentation for
/// AffineExprFlattener on how mod's and div's are flattened.
LogicalResult getFlattenedAffineExpr(AffineExpr expr, unsigned numDims,
                                     unsigned numSymbols,
                                     SmallVectorImpl<int64_t> *flattenedExpr,
                                     FlatAffineValueConstraints *cst = nullptr);

/// Flattens the result expressions of the map to their corresponding flattened
/// forms and set in 'flattenedExprs'. Returns failure if any expression in the
/// map could not be flattened (i.e., semi-affine is not yet handled). 'cst'
/// contains constraints that connect newly introduced local identifiers to
/// existing dimensional and / symbolic identifiers. See documentation for
/// AffineExprFlattener on how mod's and div's are flattened. For all affine
/// expressions that share the same operands (like those of an affine map), this
/// method should be used instead of repeatedly calling getFlattenedAffineExpr
/// since local variables added to deal with div's and mod's will be reused
/// across expressions.
LogicalResult
getFlattenedAffineExprs(AffineMap map,
                        std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
                        FlatAffineValueConstraints *cst = nullptr);
LogicalResult
getFlattenedAffineExprs(IntegerSet set,
                        std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
                        FlatAffineValueConstraints *cst = nullptr);

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

/// Builds a relation from the given AffineMap/AffineValueMap `map`, containing
/// all pairs of the form `operands -> result` that satisfy `map`. `rel` is set
/// to the relation built. For example, give the AffineMap:
///
///   (d0, d1)[s0] -> (d0 + s0, d0 - s0)
///
/// the resulting relation formed is:
///
///   (d0, d1) -> (r1, r2)
///   [d0  d1  r1  r2  s0  const]
///    1   0   -1   0  1     0     = 0
///    0   1    0  -1  -1    0     = 0
///
/// For AffineValueMap, the domain and symbols have Value set corresponding to
/// the Value in `map`. Returns failure if the AffineMap could not be flattened
/// (i.e., semi-affine is not yet handled).
LogicalResult getRelationFromMap(AffineMap &map, FlatAffineRelation &rel);
LogicalResult getRelationFromMap(const AffineValueMap &map,
                                 FlatAffineRelation &rel);

} // namespace mlir.

#endif // MLIR_DIALECT_AFFINE_ANALYSIS_AFFINESTRUCTURES_H
