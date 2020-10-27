//===- AffineOps.td - Affine operation definitions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR affine operations.
//
//===----------------------------------------------------------------------===//

#ifndef AFFINE_OPS
#define AFFINE_OPS

include "mlir/Dialect/Affine/IR/AffineOpsBase.td"
include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Affine_Dialect : Dialect {
  let name = "affine";
  let cppNamespace = "";
  let hasConstantMaterializer = 1;
}

// Base class for Affine dialect ops.
class Affine_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Affine_Dialect, mnemonic, traits> {
  // For every affine op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

// Require regions to have affine.yield.
def ImplicitAffineTerminator
    : SingleBlockImplicitTerminator<"AffineYieldOp">;

def AffineApplyOp : Affine_Op<"apply", [NoSideEffect]> {
  let summary = "affine apply operation";
  let description = [{
    The affine.apply operation applies an [affine mapping](#affine-expressions)
    to a list of SSA values, yielding a single SSA value. The number of
    dimension and symbol arguments to `affine.apply` must be equal to the
    respective number of dimensional and symbolic inputs to the affine mapping;
    the affine mapping has to be one-dimensional, and so the `affine.apply`
    operation always returns one value. The input operands and result must all
    have ‘index’ type.

    Example:

    ```mlir
    #map10 = affine_map<(d0, d1) -> (d0 floordiv 8 + d1 floordiv 128)>
    ...
    %1 = affine.apply #map10 (%s, %t)

    // Inline example.
    %2 = affine.apply affine_map<(i)[s0] -> (i+s0)> (%42)[%n]
    ```
  }];
  let arguments = (ins AffineMapAttr:$map, Variadic<Index>:$mapOperands);
  let results = (outs Index);

  // TODO: The auto-generated builders should check to see if the return type
  // has a constant builder. That way we wouldn't need to explicitly specify the
  // result types here.
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "AffineMap map, ValueRange mapOperands", [{
      build(builder, result, builder.getIndexType(), map, mapOperands);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns the affine map to be applied by this operation.
    AffineMap getAffineMap() { return map(); }

    /// Returns the affine value map computed from this operation.
    AffineValueMap getAffineValueMap();

    /// Returns true if the result of this operation can be used as dimension id
    /// in the region of the closest surrounding op with trait AffineScope.
    bool isValidDim();

    /// Returns true if the result of this operation can be used as dimension id
    /// within 'region', i.e., for all its uses with `region`.
    bool isValidDim(Region *region);

    /// Returns true if the result of this operation is a symbol in the region
    /// of the closest surrounding op that has the trait AffineScope.
    bool isValidSymbol();

    /// Returns true if the result of this operation is a symbol for all its
    /// uses in `region`.
    bool isValidSymbol(Region *region);

    operand_range getMapOperands() { return getOperands(); }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def AffineForOp : Affine_Op<"for",
    [ImplicitAffineTerminator, RecursiveSideEffects,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "for operation";
  let description = [{
    Syntax:

    ```
    operation   ::= `affine.for` ssa-id `=` lower-bound `to` upper-bound
                    (`step` integer-literal)? `{` op* `}`

    lower-bound ::= `max`? affine-map-attribute dim-and-symbol-use-list | shorthand-bound
    upper-bound ::= `min`? affine-map-attribute dim-and-symbol-use-list | shorthand-bound
    shorthand-bound ::= ssa-id | `-`? integer-literal
    ```

    The `affine.for` operation represents an affine loop nest. It has one region
    containing its body. This region must contain one block that terminates with
    [`affine.yield`](#affineyield-affineyieldop). *Note:* when
    `affine.for` is printed in custom format, the terminator is omitted. The
    block has one argument of [`index`](../LangRef.md#index-type) type that
    represents the induction variable of the loop.

    The `affine.for` operation executes its body a number of times iterating
    from a lower bound to an upper bound by a stride. The stride, represented by
    `step`, is a positive constant integer which defaults to "1" if not present.
    The lower and upper bounds specify a half-open range: the range includes the
    lower bound but does not include the upper bound.

    The lower and upper bounds of a `affine.for` operation are represented as an
    application of an affine mapping to a list of SSA values passed to the map.
    The [same restrictions](#restrictions-on-dimensions-and-symbols) hold for
    these SSA values as for all bindings of SSA values to dimensions and
    symbols.

    The affine mappings for the bounds may return multiple results, in which
    case the `max`/`min` keywords are required (for the lower/upper bound
    respectively), and the bound is the maximum/minimum of the returned values.
    There is no semantic ambiguity, but MLIR syntax requires the use of these
    keywords to make things more obvious to human readers.

    Many upper and lower bounds are simple, so MLIR accepts two custom form
    syntaxes: the form that accepts a single 'ssa-id' (e.g. `%N`) is shorthand
    for applying that SSA value to a function that maps a single symbol to
    itself, e.g., `()[s]->(s)()[%N]`. The integer literal form (e.g. `-42`) is
    shorthand for a nullary mapping function that returns the constant value
    (e.g. `()->(-42)()`).

    Example showing reverse iteration of the inner loop:

    ```mlir
    #map57 = affine_map<(d0)[s0] -> (s0 - d0 - 1)>

    func @simple_example(%A: memref<?x?xf32>, %B: memref<?x?xf32>) {
      %N = dim %A, 0 : memref<?x?xf32>
      affine.for %i = 0 to %N step 1 {
        affine.for %j = 0 to %N {   // implicitly steps by 1
          %0 = affine.apply #map57(%j)[%N]
          %tmp = call @F1(%A, %i, %0) : (memref<?x?xf32>, index, index)->(f32)
          call @F2(%tmp, %B, %i, %0) : (f32, memref<?x?xf32>, index, index)->()
        }
      }
      return
    }
    ```
  }];
  let arguments = (ins Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "int64_t lowerBound, int64_t upperBound, int64_t step = 1, "
              "function_ref<void(OpBuilder &, Location, Value)> bodyBuilder "
              "    = nullptr">,
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "ValueRange lbOperands, AffineMap lbMap, "
              "ValueRange ubOperands, AffineMap ubMap, "
              "int64_t step = 1, "
              "function_ref<void(OpBuilder &, Location, Value)> bodyBuilder "
              "    = nullptr">
  ];

  let extraClassDeclaration = [{
    static StringRef getStepAttrName() { return "step"; }
    static StringRef getLowerBoundAttrName() { return "lower_bound"; }
    static StringRef getUpperBoundAttrName() { return "upper_bound"; }

    Value getInductionVar() { return getBody()->getArgument(0); }

    // TODO: provide iterators for the lower and upper bound operands
    // if the current access via getLowerBound(), getUpperBound() is too slow.

    /// Returns operands for the lower bound map.
    operand_range getLowerBoundOperands();

    /// Returns operands for the upper bound map.
    operand_range getUpperBoundOperands();

    /// Returns information about the lower bound as a single object.
    AffineBound getLowerBound();

    /// Returns information about the upper bound as a single object.
    AffineBound getUpperBound();

    /// Returns loop step.
    int64_t getStep() {
      return getAttr(getStepAttrName()).cast<IntegerAttr>().getInt();
    }

    /// Returns affine map for the lower bound.
    AffineMap getLowerBoundMap() { return getLowerBoundMapAttr().getValue(); }
    AffineMapAttr getLowerBoundMapAttr() {
      return getAttr(getLowerBoundAttrName()).cast<AffineMapAttr>();
    }
    /// Returns affine map for the upper bound. The upper bound is exclusive.
    AffineMap getUpperBoundMap() { return getUpperBoundMapAttr().getValue(); }
    AffineMapAttr getUpperBoundMapAttr() {
      return getAttr(getUpperBoundAttrName()).cast<AffineMapAttr>();
    }

    /// Set lower bound. The new bound must have the same number of operands as
    /// the current bound map. Otherwise, 'replaceForLowerBound' should be used.
    void setLowerBound(ValueRange operands, AffineMap map);
    /// Set upper bound. The new bound must not have more operands than the
    /// current bound map. Otherwise, 'replaceForUpperBound' should be used.
    void setUpperBound(ValueRange operands, AffineMap map);

    /// Set the lower bound map without changing operands.
    void setLowerBoundMap(AffineMap map);

    /// Set the upper bound map without changing operands.
    void setUpperBoundMap(AffineMap map);

    /// Set loop step.
    void setStep(int64_t step) {
      assert(step > 0 && "step has to be a positive integer constant");
      auto *context = getLowerBoundMap().getContext();
      setAttr(Identifier::get(getStepAttrName(), context),
              IntegerAttr::get(IndexType::get(context), step));
    }

    /// Returns true if the lower bound is constant.
    bool hasConstantLowerBound();
    /// Returns true if the upper bound is constant.
    bool hasConstantUpperBound();
    /// Returns true if both bounds are constant.
    bool hasConstantBounds() {
      return hasConstantLowerBound() && hasConstantUpperBound();
    }
    /// Returns the value of the constant lower bound.
    /// Fails assertion if the bound is non-constant.
    int64_t getConstantLowerBound();
    /// Returns the value of the constant upper bound. The upper bound is
    /// exclusive. Fails assertion if the bound is non-constant.
    int64_t getConstantUpperBound();
    /// Sets the lower bound to the given constant value.
    void setConstantLowerBound(int64_t value);
    /// Sets the upper bound to the given constant value.
    void setConstantUpperBound(int64_t value);

    /// Returns true if both the lower and upper bound have the same operand
    /// lists (same operands in the same order).
    bool matchingBoundOperandList();
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def AffineIfOp : Affine_Op<"if",
                           [ImplicitAffineTerminator, RecursiveSideEffects,
                            NoRegionArguments]> {
  let summary = "if-then-else operation";
  let description = [{
    Syntax:

    ```
    operation  ::= `affine.if` if-op-cond `{` op* `}` (`else` `{` op* `}`)?
    if-op-cond ::= integer-set-attr dim-and-symbol-use-list
    ```

    The `affine.if` operation restricts execution to a subset of the loop
    iteration space defined by an integer set (a conjunction of affine
    constraints). A single `affine.if` may end with an optional `else` clause.

    The condition of the `affine.if` is represented by an
    [integer set](#integer-sets) (a conjunction of affine constraints),
    and the SSA values bound to the dimensions and symbols in the integer set.
    The [same restrictions](#restrictions-on-dimensions-and-symbols) hold for
    these SSA values as for all bindings of SSA values to dimensions and
    symbols.

    The `affine.if` operation contains two regions for the "then" and "else"
    clauses.  `affine.if` may return results that are defined in its regions.
    The values defined are determined by which execution path is taken.  Each
    region of the `affine.if` must contain a single block with no arguments,
    and be terminated by `affine.yield`.  If `affine.if` defines no values,
    the `affine.yield` can be left out, and will be inserted implicitly.
    Otherwise, it must be explicit.  If no values are defined, the else block
    may be empty (i.e. contain no blocks).

    Example:

    ```mlir
    #set = affine_set<(d0, d1)[s0]: (d0 - 10 >= 0, s0 - d0 - 9 >= 0,
                                     d1 - 10 >= 0, s0 - d1 - 9 >= 0)>
    func @reduced_domain_example(%A, %X, %N) : (memref<10xi32>, i32, i32) {
      affine.for %i = 0 to %N {
         affine.for %j = 0 to %N {
           %0 = affine.apply #map42(%j)
           %tmp = call @S1(%X, %i, %0)
           affine.if #set(%i, %j)[%N] {
              %1 = affine.apply #map43(%i, %j)
              call @S2(%tmp, %A, %i, %1)
           }
        }
      }
      return
    }
    ```

    Example with an explicit yield (initialization with edge padding):

    ```mlir
    #interior = affine_set<(i, j) : (i - 1 >= 0, j - 1 >= 0,  10 - i >= 0, 10 - j >= 0)> (%i, %j)
    func @pad_edges(%I : memref<10x10xf32>) -> (memref<12x12xf32) {
      %O = alloc memref<12x12xf32>
      affine.parallel (%i, %j) = (0, 0) to (12, 12) {
        %1 = affine.if #interior (%i, %j) {
          %2 = load %I[%i - 1, %j - 1] : memref<10x10xf32>
          affine.yield %2
        } else {
          %2 = constant 0.0 : f32
          affine.yield %2 : f32
        }
        affine.store %1, %O[%i, %j] : memref<12x12xf32>
      }
      return %O
    }
    ```
  }];
  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "IntegerSet set, ValueRange args, bool withElseRegion">,
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "TypeRange resultTypes, IntegerSet set, ValueRange args,"
              "bool withElseRegion">,
  ];

  let extraClassDeclaration = [{
    static StringRef getConditionAttrName() { return "condition"; }

    IntegerSet getIntegerSet();
    void setIntegerSet(IntegerSet newSet);

    /// Sets the integer set with its operands.
    void setConditional(IntegerSet set, ValueRange operands);

    /// Returns true if an else block exists.
    bool hasElse() { return !elseRegion().empty(); }

    Block *getThenBlock() {
      assert(!thenRegion().empty() && "Unexpected empty 'then' region.");
      return &thenRegion().front();
    }

    Block *getElseBlock() {
      assert(hasElse() && "Empty 'else' region.");
      return &elseRegion().front();
    }

    OpBuilder getThenBodyBuilder() {
      assert(!thenRegion().empty() && "Unexpected empty 'then' region.");
      Block &body = thenRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
    OpBuilder getElseBodyBuilder() {
      assert(hasElse() && "No 'else' block");
      Block &body = elseRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

class AffineLoadOpBase<string mnemonic, list<OpTrait> traits = []> :
    Affine_Op<mnemonic, !listconcat(traits,
        [DeclareOpInterfaceMethods<AffineReadOpInterface>])> {
  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
      [MemRead]>:$memref,
      Variadic<Index>:$indices);

  code extraClassDeclarationBase = [{
    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 0; }

    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

    /// Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
      return getAttr(getMapAttrName()).cast<AffineMapAttr>();
    }

    static StringRef getMapAttrName() { return "map"; }
  }];
}

def AffineLoadOp : AffineLoadOpBase<"load"> {
  let summary = "affine load operation";
  let description = [{
    The "affine.load" op reads an element from a memref, where the index
    for each memref dimension is an affine expression of loop induction
    variables and symbols. The output of 'affine.load' is a new value with the
    same type as the elements of the memref. An affine expression of loop IVs
    and symbols must be specified for each dimension of the memref. The keyword
    'symbol' can be used to indicate SSA identifiers which are symbolic.

    Example 1:

    ```mlir
    %1 = affine.load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
    ```

    Example 2: Uses 'symbol' keyword for symbols '%n' and '%m'.

    ```mlir
    %1 = affine.load %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>
    ```
  }];

  let results = (outs AnyType:$result);

  let builders = [
    /// Builds an affine load op with the specified map and operands.
    OpBuilder<"OpBuilder &builder, OperationState &result, AffineMap map, "
                      "ValueRange operands">,
    /// Builds an affine load op with an identity map and operands.
    OpBuilder<"OpBuilder &builder, OperationState &result, Value memref, "
                      "ValueRange indices = {}">,
    /// Builds an affine load op with the specified map and its operands.
    OpBuilder<"OpBuilder &builder, OperationState &result, Value memref, "
                      "AffineMap map, ValueRange mapOperands">
  ];

  let extraClassDeclaration = extraClassDeclarationBase;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

class AffineMinMaxOpBase<string mnemonic, list<OpTrait> traits = []> :
    Op<Affine_Dialect, mnemonic, traits> {
  let arguments = (ins AffineMapAttr:$map, Variadic<Index>:$operands);
  let results = (outs Index);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, AffineMap affineMap, "
              "ValueRange mapOperands",
    [{
      build(builder, result, builder.getIndexType(), affineMap, mapOperands);
    }]>
  ];

  let extraClassDeclaration = [{
    static StringRef getMapAttrName() { return "map"; }
    AffineMap getAffineMap() { return map(); }
    ValueRange getMapOperands() { return operands(); }
    ValueRange getDimOperands() {
      return OperandRange{operands().begin(),
                        operands().begin() + map().getNumDims()};
    }
    ValueRange getSymbolOperands() {
      return OperandRange{operands().begin() + map().getNumDims(),
                        operands().end()};
    }
  }];
  let verifier = [{ return ::verifyAffineMinMaxOp(*this); }];
  let printer = [{ return ::printAffineMinMaxOp(p, *this); }];
  let parser = [{ return ::parseAffineMinMaxOp<$cppClass>(parser, result); }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def AffineMinOp : AffineMinMaxOpBase<"min", [NoSideEffect]> {
  let summary = "min operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `affine.min` affine-map-attribute dim-and-symbol-use-list
    ```

    The `affine.min` operation applies an [affine mapping](#affine-expressions)
    to a list of SSA values, and returns the minimum value of all result
    expressions. The number of dimension and symbol arguments to `affine.min`
    must be equal to the respective number of dimensional and symbolic inputs to
    the affine mapping; the `affine.min` operation always returns one value. The
    input operands and result must all have 'index' type.

    Example:

    ```mlir
    %0 = affine.min affine_map<(d0)[s0] -> (1000, d0 + 512, s0)> (%arg0)[%arg1]
    ```
  }];
}

def AffineMaxOp : AffineMinMaxOpBase<"max", [NoSideEffect]> {
  let summary = "max operation";
  let description = [{
    The "max" operation computes the maximum value result from a multi-result
    affine map.

    Example:

    ```mlir
    %0 = affine.max (d0) -> (1000, d0 + 512) (%i0) : index
    ```
  }];
}

def AffineParallelOp : Affine_Op<"parallel", 
    [ImplicitAffineTerminator, RecursiveSideEffects,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "multi-index parallel band operation";
  let description = [{
    The "affine.parallel" operation represents a hyper-rectangular affine
    parallel band, defining multiple SSA values for its induction variables. It
    has one region capturing the parallel band body. The induction variables are
    represented as arguments of this region. These SSA values always have type
    index, which is the size of the machine word. The strides, represented by
    steps, are positive constant integers which defaults to "1" if not present.
    The lower and upper bounds specify a half-open range: the range includes the
    lower bound but does not include the upper bound. The body region must
    contain exactly one block that terminates with "affine.yield".

    The lower and upper bounds of a parallel operation are represented as an
    application of an affine mapping to a list of SSA values passed to the map.
    The same restrictions hold for these SSA values as for all bindings of SSA
    values to dimensions and symbols.

    Each value yielded by affine.yield will be accumulated/reduced via one of
    the reduction methods defined in the AtomicRMWKind enum.  The order of
    reduction is unspecified, and lowering may produce any valid ordering.
    Loops with a 0 trip count will produce as a result the identity value
    associated with each reduction (i.e. 0.0 for addf, 1.0 for mulf).  Assign
    reductions for loops with a trip count != 1 produces undefined results.

    Note: Calling AffineParallelOp::build will create the required region and
    block, and insert the required terminator if it is trivial (i.e. no values
    are yielded).  Parsing will also create the required region, block, and 
    terminator, even when they are missing from the textual representation.

    Example (3x3 valid convolution):

    ```mlir
    fuction @conv_2d(%D : memref<100x100xf32>, %K : memref<3x3xf32>) -> (memref<98x98xf32>) {
      %O = alloc memref<98x98xf32>
      affine.parallel (%x, %y) = (0, 0) to (98, 98) {
        %0 = affine.parallel (%kx, %ky) = (0, 0) to (2, 2) reduce ("addf") {
          %1 = affine.load %D[%x + %kx, %y + %ky] : memref<100x100xf32>
          %2 = affine.load %K[%kx, %ky] : memref<3x3xf32>
          %3 = mulf %1, %2 : f32
          affine.yield %3 : f32
        }
        affine.store %0, O[%x, %y] : memref<98x98xf32>
      }
      return %O
    }
    ```
  }];

  let arguments = (ins
     TypedArrayAttrBase<AtomicRMWKindAttr, "Reduction ops">:$reductions,
     AffineMapAttr:$lowerBoundsMap,
     AffineMapAttr:$upperBoundsMap,
     I64ArrayAttr:$steps,
     Variadic<Index>:$mapOperands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "ArrayRef<Type> resultTypes, "
              "ArrayRef<AtomicRMWKind> reductions, "
              "ArrayRef<int64_t> ranges">,
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "ArrayRef<Type> resultTypes, "
              "ArrayRef<AtomicRMWKind> reductions, "
              "AffineMap lbMap, ValueRange lbArgs, "
              "AffineMap ubMap, ValueRange ubArgs">,
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "ArrayRef<Type> resultTypes, "
              "ArrayRef<AtomicRMWKind> reductions, "
              "AffineMap lbMap, ValueRange lbArgs, "
              "AffineMap ubMap, ValueRange ubArgs, "
              "ArrayRef<int64_t> steps">
  ];

  let extraClassDeclaration = [{
    /// Get the number of dimensions.
    unsigned getNumDims();

    operand_range getLowerBoundsOperands();
    operand_range getUpperBoundsOperands();

    AffineValueMap getLowerBoundsValueMap();
    AffineValueMap getUpperBoundsValueMap();
    AffineValueMap getRangesValueMap();

    /// Get ranges as constants, may fail in dynamic case.
    Optional<SmallVector<int64_t, 8>> getConstantRanges();

    Block *getBody();
    OpBuilder getBodyBuilder();
    MutableArrayRef<BlockArgument> getIVs() {
      return getBody()->getArguments();
    }
    void setSteps(ArrayRef<int64_t> newSteps);

    static StringRef getReductionsAttrName() { return "reductions"; }
    static StringRef getLowerBoundsMapAttrName() { return "lowerBoundsMap"; }
    static StringRef getUpperBoundsMapAttrName() { return "upperBoundsMap"; }
    static StringRef getStepsAttrName() { return "steps"; }
  }];
}

def AffinePrefetchOp : Affine_Op<"prefetch"> {
  let summary = "affine prefetch operation";
  let description = [{
    The "affine.prefetch" op prefetches data from a memref location described
    with an affine subscript similar to affine.load, and has three attributes:
    a read/write specifier, a locality hint, and a cache type specifier as shown
    below:

    ```mlir
    affine.prefetch %0[%i, %j + 5], read, locality<3>, data : memref<400x400xi32>
    ```

    The read/write specifier is either 'read' or 'write', the locality hint
    specifier ranges from locality<0> (no locality) to locality<3> (extremely
    local keep in cache). The cache type specifier is either 'data' or 'instr'
    and specifies whether the prefetch is performed on data cache or on
    instruction cache.
  }];

  let arguments = (ins AnyMemRef:$memref, Variadic<Index>:$indices,
                   BoolAttr:$isWrite,
                   Confined<I32Attr, [IntMinValue<0>,
                     IntMaxValue<3>]>:$localityHint,
                   BoolAttr:$isDataCache);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value memref,"
    "AffineMap map, ArrayRef<Value> mapOperands, bool isWrite,"
    "unsigned localityHint, bool isDataCache",
    [{
      assert(map.getNumInputs() == mapOperands.size()
             && "inconsistent index info");
      auto localityHintAttr = builder.getI32IntegerAttr(localityHint);
      auto isWriteAttr = builder.getBoolAttr(isWrite);
      auto isDataCacheAttr = builder.getBoolAttr(isDataCache);
      result.addOperands(memref);
      result.addAttribute(getMapAttrName(), AffineMapAttr::get(map));
      result.addOperands(mapOperands);
      result.addAttribute(getLocalityHintAttrName(), localityHintAttr);
      result.addAttribute(getIsWriteAttrName(), isWriteAttr);
      result.addAttribute(getIsDataCacheAttrName(), isDataCacheAttr);
    }]>];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return memref().getType().cast<MemRefType>();
    }

    /// Returns the affine map used to index the memref for this operation.
    AffineMap getAffineMap() { return getAffineMapAttr().getValue(); }
    AffineMapAttr getAffineMapAttr() {
      return getAttr(getMapAttrName()).cast<AffineMapAttr>();
    }

    /// Returns the AffineMapAttr associated with 'memref'.
    NamedAttribute getAffineMapAttrForMemRef(Value mref) {
      assert(mref == memref());
      return {Identifier::get(getMapAttrName(), getContext()),
        getAffineMapAttr()};
    }

    /// Get affine map operands.
    operand_range getMapOperands() {
      return {operand_begin() + 1, operand_end()};
    }

    static StringRef getMapAttrName() { return "map"; }
    static StringRef getLocalityHintAttrName() { return "localityHint"; }
    static StringRef getIsWriteAttrName() { return "isWrite"; }
    static StringRef getIsDataCacheAttrName() { return "isDataCache"; }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

class AffineStoreOpBase<string mnemonic, list<OpTrait> traits = []> :
    Affine_Op<mnemonic, !listconcat(traits,
        [DeclareOpInterfaceMethods<AffineWriteOpInterface>])> {
  code extraClassDeclarationBase = [{
    /// Returns the operand index of the value to be stored.
    unsigned getStoredValOperandIndex() { return 0; }

    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 1; }

    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

    /// Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
      return getAttr(getMapAttrName()).cast<AffineMapAttr>();
    }

    /// Returns the AffineMapAttr associated with 'memref'.
    NamedAttribute getAffineMapAttrForMemRef(Value memref) {
      assert(memref == getMemRef());
      return {Identifier::get(getMapAttrName(), getContext()),
              getAffineMapAttr()};
    }

    static StringRef getMapAttrName() { return "map"; }
  }];
}

def AffineStoreOp : AffineStoreOpBase<"store"> {
  let summary = "affine store operation";
  let description = [{
    The "affine.store" op writes an element to a memref, where the index
    for each memref dimension is an affine expression of loop induction
    variables and symbols. The 'affine.store' op stores a new value which is the
    same type as the elements of the memref. An affine expression of loop IVs
    and symbols must be specified for each dimension of the memref. The keyword
    'symbol' can be used to indicate SSA identifiers which are symbolic.

    Example 1:

    ```mlir
    affine.store %v0, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
    ```

    Example 2: Uses 'symbol' keyword for symbols '%n' and '%m'.

    ```mlir
    affine.store %v0, %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>
    ```
  }];
  let arguments = (ins AnyType:$value,
      Arg<AnyMemRef, "the reference to store to",
      [MemWrite]>:$memref,
      Variadic<Index>:$indices);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
                      "Value valueToStore, Value memref, ValueRange indices">,
    OpBuilder<"OpBuilder &builder, OperationState &result, "
                      "Value valueToStore, Value memref, AffineMap map, "
                      "ValueRange mapOperands">
  ];

  let extraClassDeclaration = extraClassDeclarationBase;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def AffineYieldOp : Affine_Op<"yield", [NoSideEffect, Terminator, ReturnLike]> {
  let summary = "Yield values to parent operation";
  let description = [{
    "affine.yield" yields zero or more SSA values from an affine op region and
    terminates the region. The semantics of how the values yielded are used
    is defined by the parent operation.
    If "affine.yield" has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the "affine.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def AffineVectorLoadOp : AffineLoadOpBase<"vector_load"> {
  let summary = "affine vector load operation";
  let description = [{
    The "affine.vector_load" is the vector counterpart of
    [affine.load](#affineload-operation). It reads a slice from a
    [MemRef](../LangRef.md#memref-type), supplied as its first operand,
    into a [vector](../LangRef.md#vector-type) of the same base elemental type.
    The index for each memref dimension is an affine expression of loop induction
    variables and symbols. These indices determine the start position of the read
    within the memref. The shape of the return vector type determines the shape of
    the slice read from the memref. This slice is contiguous along the respective
    dimensions of the shape. Strided vector loads will be supported in the future.
    An affine expression of loop IVs and symbols must be specified for each
    dimension of the memref. The keyword 'symbol' can be used to indicate SSA
    identifiers which are symbolic.

    Example 1: 8-wide f32 vector load.

    ```mlir
    %1 = affine.vector_load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>, vector<8xf32>
    ```

    Example 2: 4-wide f32 vector load. Uses 'symbol' keyword for symbols '%n' and '%m'.

    ```mlir
    %1 = affine.vector_load %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>, vector<4xf32>
    ```

    Example 3: 2-dim f32 vector load.

    ```mlir
    %1 = affine.vector_load %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
    ```

    TODOs:
    * Add support for strided vector loads.
    * Consider adding a permutation map to permute the slice that is read from memory
    (see [vector.transfer_read](../Vector/#vectortransfer_read-vectortransferreadop)).
  }];

  let results = (outs AnyVector:$result);

  let extraClassDeclaration = extraClassDeclarationBase # [{
    VectorType getVectorType() {
      return result().getType().cast<VectorType>();
    }
  }];
}

def AffineVectorStoreOp : AffineStoreOpBase<"vector_store"> {
  let summary = "affine vector store operation";
  let description = [{
    The "affine.vector_store" is the vector counterpart of
    [affine.store](#affinestore-affinestoreop). It writes a
    [vector](../LangRef.md#vector-type), supplied as its first operand,
    into a slice within a [MemRef](../LangRef.md#memref-type) of the same base
    elemental type, supplied as its second operand.
    The index for each memref dimension is an affine expression of loop
    induction variables and symbols. These indices determine the start position
    of the write within the memref. The shape of th input vector determines the
    shape of the slice written to the memref. This slice is contiguous along the
    respective dimensions of the shape. Strided vector stores will be supported
    in the future.
    An affine expression of loop IVs and symbols must be specified for each
    dimension of the memref. The keyword 'symbol' can be used to indicate SSA
    identifiers which are symbolic.

    Example 1: 8-wide f32 vector store.

    ```mlir
    affine.vector_store %v0, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>, vector<8xf32>
    ```

    Example 2: 4-wide f32 vector store. Uses 'symbol' keyword for symbols '%n' and '%m'.

    ```mlir
    affine.vector_store %v0, %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>, vector<4xf32>
    ```

    Example 3: 2-dim f32 vector store.

    ```mlir
    affine.vector_store %v0, %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
    ```

    TODOs:
    * Add support for strided vector stores.
    * Consider adding a permutation map to permute the slice that is written to memory
    (see [vector.transfer_write](../Vector/#vectortransfer_write-vectortransferwriteop)).
  }];

  let arguments = (ins AnyVector:$value,
      Arg<AnyMemRef, "the reference to store to",
      [MemWrite]>:$memref,
      Variadic<Index>:$indices);

  let extraClassDeclaration = extraClassDeclarationBase # [{
    VectorType getVectorType() {
      return value().getType().cast<VectorType>();
    }
  }];
}

#endif // AFFINE_OPS
