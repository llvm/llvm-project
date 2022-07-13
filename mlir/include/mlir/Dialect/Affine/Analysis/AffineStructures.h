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
/// where each non-local variable can have an SSA Value attached to it.
class FlatAffineValueConstraints : public presburger::IntegerPolyhedron {
public:
  /// Constructs a constraint system reserving memory for the specified number
  /// of constraints and variables.
  FlatAffineValueConstraints(unsigned numReservedInequalities,
                             unsigned numReservedEqualities,
                             unsigned numReservedCols, unsigned numDims,
                             unsigned numSymbols, unsigned numLocals,
                             ArrayRef<Optional<Value>> valArgs = {})
      : IntegerPolyhedron(numReservedInequalities, numReservedEqualities,
                          numReservedCols,
                          presburger::PresburgerSpace::getSetSpace(
                              numDims, numSymbols, numLocals)) {
    assert(numReservedCols >= getNumVars() + 1);
    assert(valArgs.empty() || valArgs.size() == getNumDimAndSymbolVars());
    values.reserve(numReservedCols);
    if (valArgs.empty())
      values.resize(getNumDimAndSymbolVars(), None);
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
    assert(valArgs.empty() || valArgs.size() == getNumDimAndSymbolVars());
    if (valArgs.empty())
      values.resize(getNumDimAndSymbolVars(), None);
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
  // All dimensions as set as VarKind::SetDim.
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
  /// right variable is first looked up using `forOp`'s Value. Asserts if the
  /// Value corresponding to the 'affine.for' operation isn't found in the
  /// constraint system. Returns failure for the yet unimplemented/unsupported
  /// cases.  Any new variables that are found in the bound operands of the
  /// 'affine.for' operation are added as trailing variables (either
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
  /// exist yet in the IR so the Value attached to those dimension variables
  /// might be empty. For that reason, this method doesn't perform Value
  /// look-ups to retrieve the dimension variable positions. Instead, it
  /// assumes the position of the dim variables in the constraint system is
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
  /// Note: The dimensions/symbols of this FlatAffineConstraints must match the
  /// dimensions/symbols of the affine map.
  LogicalResult addBound(BoundType type, unsigned pos, AffineMap boundMap,
                         bool isClosedBound);

  /// Adds a bound for the variable at the specified position with constraints
  /// being drawn from the specified bound map. In case of an EQ bound, the
  /// bound map is expected to have exactly one result. In case of a LB/UB, the
  /// bound map may have more than one result, for each of which an inequality
  /// is added.
  /// Note: The dimensions/symbols of this FlatAffineConstraints must match the
  /// dimensions/symbols of the affine map. By default the lower bound is closed
  /// and the upper bound is open.
  LogicalResult addBound(BoundType type, unsigned pos, AffineMap boundMap);

  /// Adds a bound for the variable at the specified position with constraints
  /// being drawn from the specified bound map and operands. In case of an
  /// EQ bound, the  bound map is expected to have exactly one result. In case
  /// of a LB/UB, the bound map may have more than one result, for each of which
  /// an inequality is added.
  LogicalResult addBound(BoundType type, unsigned pos, AffineMap boundMap,
                         ValueRange operands);

  /// Adds a constant bound for the variable associated with the given Value.
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
  /// variables (starting at `offset`) as an affine map of the remaining
  /// variables (dimensional and symbolic). This method is able to detect
  /// variables as floordiv's and mod's of affine expressions of other
  /// variables with respect to (positive) constants. Sets bound map to a
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

  /// Gets the lower and upper bound of the `offset` + `pos`th variable
  /// treating [0, offset) U [offset + num, symStartPos) as dimensions and
  /// [symStartPos, getNumDimAndSymbolVars) as symbols, and `pos` lies in
  /// [0, num). The multi-dimensional maps in the returned pair represent the
  /// max and min of potentially multiple affine expressions. The upper bound is
  /// exclusive. `localExprs` holds pre-computed AffineExpr's for all local
  /// variables in the system.
  std::pair<AffineMap, AffineMap>
  getLowerAndUpperBound(unsigned pos, unsigned offset, unsigned num,
                        unsigned symStartPos, ArrayRef<AffineExpr> localExprs,
                        MLIRContext *context) const;

  /// Returns the bound for the variable at `pos` from the inequality at
  /// `ineqPos` as a 1-d affine value map (affine map + operands). The returned
  /// affine value map can either be a lower bound or an upper bound depending
  /// on the sign of atIneq(ineqPos, pos). Asserts if the row at `ineqPos` does
  /// not involve the `pos`th variable.
  void getIneqAsAffineValueMap(unsigned pos, unsigned ineqPos,
                               AffineValueMap &vmap,
                               MLIRContext *context) const;

  /// Adds slice lower bounds represented by lower bounds in `lbMaps` and upper
  /// bounds in `ubMaps` to each variable in the constraint system which has
  /// a value in `values`. Note that both lower/upper bounds share the same
  /// operand list `operands`.
  /// This function assumes `values.size` == `lbMaps.size` == `ubMaps.size`.
  /// Note that both lower/upper bounds use operands from `operands`.
  LogicalResult addSliceBounds(ArrayRef<Value> values,
                               ArrayRef<AffineMap> lbMaps,
                               ArrayRef<AffineMap> ubMaps,
                               ArrayRef<Value> operands);

  /// Looks up the position of the variable with the specified Value. Returns
  /// true if found (false otherwise). `pos` is set to the (column) position of
  /// the variable.
  bool findVar(Value val, unsigned *pos) const;

  /// Returns true if an variable with the specified Value exists, false
  /// otherwise.
  bool containsVar(Value mayBeVar) const;

  /// Swap the posA^th variable with the posB^th variable.
  void swapVar(unsigned posA, unsigned posB) override;

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
  unsigned insertDimVar(unsigned pos, ValueRange vals);
  unsigned insertSymbolVar(unsigned pos, ValueRange vals);
  unsigned insertVar(presburger::VarKind kind, unsigned pos,
                     unsigned num = 1) override;
  unsigned insertVar(presburger::VarKind kind, unsigned pos, ValueRange vals);

  /// Append variables of the specified kind after the last variable of that
  /// kind. The coefficient columns corresponding to the added variables are
  /// initialized to zero. `vals` are the Values corresponding to the
  /// variables. Return the position of the first added column.
  ///
  /// Note: Empty Values are allowed in `vals`.
  unsigned appendDimVar(ValueRange vals);
  unsigned appendSymbolVar(ValueRange vals);
  unsigned appendDimVar(unsigned num = 1) {
    return appendVar(VarKind::SetDim, num);
  }
  unsigned appendSymbolVar(unsigned num = 1) {
    return appendVar(VarKind::Symbol, num);
  }
  unsigned appendLocalVar(unsigned num = 1) {
    return appendVar(VarKind::Local, num);
  }

  /// Removes variables in the column range [varStart, varLimit), and copies any
  /// remaining valid data into place, updates member variables, and resizes
  /// arrays as needed.
  void removeVarRange(presburger::VarKind kind, unsigned varStart,
                      unsigned varLimit) override;
  using IntegerPolyhedron::removeVarRange;

  /// Add the specified values as a dim or symbol var depending on its nature,
  /// if it already doesn't exist in the system. `val` has to be either a
  /// terminal symbol or a loop IV, i.e., it cannot be the result affine.apply
  /// of any symbols or loop IVs. The variable is added to the end of the
  /// existing dims or symbols. Additional information on the variable is
  /// extracted from the IR and added to the constraint system.
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

  /// Projects out the variable that is associate with Value.
  void projectOut(Value val);
  using IntegerPolyhedron::projectOut;

  /// Changes all symbol variables which are loop IVs to dim variables.
  void convertLoopIVSymbolsToDims();

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
  LogicalResult unionBoundingBox(const FlatAffineValueConstraints &other);
  using IntegerPolyhedron::unionBoundingBox;

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
                                  FlatAffineValueConstraints *other);

  /// Returns true if this constraint system and `other` are in the same
  /// space, i.e., if they are associated with the same set of variables,
  /// appearing in the same order. Returns false otherwise.
  bool areVarsAlignedWithOther(const FlatAffineValueConstraints &other);

  /// Replaces the contents of this FlatAffineValueConstraints with `other`.
  void clearAndCopyFrom(const IntegerRelation &other) override;

  /// Returns the Value associated with the pos^th variable. Asserts if
  /// no Value variable was associated.
  inline Value getValue(unsigned pos) const {
    assert(pos < getNumDimAndSymbolVars() && "Invalid position");
    assert(hasValue(pos) && "variable's Value not set");
    return values[pos].getValue();
  }

  /// Returns true if the pos^th variable has an associated Value.
  inline bool hasValue(unsigned pos) const {
    assert(pos < getNumDimAndSymbolVars() && "Invalid position");
    return values[pos].has_value();
  }

  /// Returns true if at least one variable has an associated Value.
  bool hasValues() const;

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

  inline ArrayRef<Optional<Value>> getMaybeValues() const {
    return {values.data(), values.size()};
  }

  inline ArrayRef<Optional<Value>>
  getMaybeValues(presburger::VarKind kind) const {
    assert(kind != VarKind::Local &&
           "Local variables do not have any value attached to them.");
    return {values.data() + getVarKindOffset(kind), getNumVarKind(kind)};
  }

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

  /// Merge and align symbols of `this` and `other` such that both get union of
  /// of symbols that are unique. Symbols in `this` and `other` should be
  /// unique. Symbols with Value as `None` are considered to be inequal to all
  /// other symbols.
  void mergeSymbolVars(FlatAffineValueConstraints &other);

protected:
  using VarKind = presburger::VarKind;

  /// Returns false if the fields corresponding to various variable counts, or
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

  /// Eliminates the variable at the specified position using Fourier-Motzkin
  /// variable elimination, but uses Gaussian elimination if there is an
  /// equality involving that variable. If the result of the elimination is
  /// integer exact, `*isResultIntegerExact` is set to true. If `darkShadow` is
  /// set to true, a potential under approximation (subset) of the rational
  /// shadow / exact integer shadow is computed.
  // See implementation comments for more details.
  void fourierMotzkinEliminate(unsigned pos, bool darkShadow = false,
                               bool *isResultIntegerExact = nullptr) override;

  /// Prints the number of constraints, dimensions, symbols and locals in the
  /// FlatAffineConstraints. Also, prints for each variable whether there is
  /// an SSA Value attached to it.
  void printSpace(raw_ostream &os) const override;

  /// Values corresponding to the (column) non-local variables of this
  /// constraint system appearing in the order the variables correspond to
  /// columns. Variables that aren't associated with any Value are set to
  /// None.
  SmallVector<Optional<Value>, 8> values;
};

/// A FlatAffineRelation represents a set of ordered pairs (domain -> range)
/// where "domain" and "range" are tuples of variables. The relation is
/// represented as a FlatAffineValueConstraints with separation of dimension
/// variables into domain and  range. The variables are stored as:
/// [domainVars, rangeVars, symbolVars, localVars, constant].
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

  /// Returns the number of variables corresponding to domain/range of
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

  /// Insert `num` variables of the specified kind after the `pos` variable
  /// of that kind. The coefficient columns corresponding to the added
  /// variables are initialized to zero.
  void insertDomainVar(unsigned pos, unsigned num = 1);
  void insertRangeVar(unsigned pos, unsigned num = 1);

  /// Append `num` variables of the specified kind after the last variable
  /// of that kind. The coefficient columns corresponding to the added
  /// variables are initialized to zero.
  void appendDomainVar(unsigned num = 1);
  void appendRangeVar(unsigned num = 1);

  /// Removes variables in the column range [varStart, varLimit), and copies any
  /// remaining valid data into place, updates member variables, and resizes
  /// arrays as needed.
  void removeVarRange(VarKind kind, unsigned varStart,
                      unsigned varLimit) override;
  using IntegerRelation::removeVarRange;

protected:
  // Number of dimension variables corresponding to domain variables.
  unsigned numDomainDims;

  // Number of dimension variables corresponding to range variables.
  unsigned numRangeDims;
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
                                     FlatAffineValueConstraints *cst = nullptr);

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
