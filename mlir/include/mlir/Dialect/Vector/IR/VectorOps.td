//===- VectorOps.td - Vector op definitions ---------------*- tablegen -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR vector operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_IR_VECTOR_OPS
#define MLIR_DIALECT_VECTOR_IR_VECTOR_OPS

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "mlir/Dialect/Vector/Interfaces/MaskableOpInterface.td"
include "mlir/Dialect/Vector/Interfaces/MaskingOpInterface.td"
include "mlir/Dialect/Vector/IR/Vector.td"
include "mlir/Dialect/Vector/IR/VectorAttributes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/IndexingMapOpInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/MemOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"

// TODO: Add an attribute to specify a different algebra with operators other
// than the current set: {*, +}.
def Vector_ContractionOp :
  Vector_Op<"contract", [
      IndexingMapOpInterface,
      Pure,
      PredOpTrait<"lhs and rhs have same element type", TCopVTEtIsSameAs<0, 1>>,
      PredOpTrait<"third operand acc and result have same element type",
                  TCresVTEtIsSameAsOpBase<0, 2>>,
      DeclareOpInterfaceMethods<MaskableOpInterface>,
      DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>
    ]>,
    Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs, AnyType:$acc,
               ArrayAttr:$indexing_maps,
               Vector_IteratorTypeArrayAttr:$iterator_types,
               DefaultValuedAttr<Vector_CombiningKindAttr,
                                 "CombiningKind::ADD">:$kind)>,
    Results<(outs AnyType)> {
  let summary = "vector contraction operation";
  let description = [{
    Computes the sum of products of vector elements along contracting
    dimension pairs from 2 vectors of rank M and N respectively, adds this
    intermediate result to the accumulator argument of rank K, and returns a
    vector result of rank K (where K = num_lhs_free_dims + num_rhs_free_dims +
    num_batch_dims (see dimension type descriptions below)). For K = 0 (no
    free or batch dimensions), the accumulator and output are a scalar.

    If operands and the result have types of different bitwidths, operands are
    promoted to have the same bitwidth as the result before performing the
    contraction. For integer types, only signless integer types are supported,
    and the promotion happens via sign extension.

    An iterator type attribute list must be specified, where each element of
    the list represents an iterator with one of the following types:

    *   "reduction": reduction dimensions are present in the lhs and rhs
        arguments but not in the output (and accumulator
        argument). These are the dimensions along which the vector
        contraction op computes the sum of products, and
        contracting dimension pair dimension sizes must match
        between lhs/rhs.

    *   "parallel": Batch dimensions are iterator type "parallel", and
        are non-contracting dimensions present in the lhs, rhs and
        output. The lhs/rhs co-iterate along the batch dimensions,
        which should be expressed in their indexing maps.

        Free dimensions are iterator type "parallel", and are
        non-contraction, non-batch dimensions accessed by either the
        lhs or rhs (but not both). The lhs and rhs free dimensions
        are unrelated to each other and do not co-iterate, which
        should be expressed in their indexing maps.

    An indexing map attribute list must be specified with an entry for lhs, rhs
    and acc arguments. An indexing map attribute specifies a mapping from each
    iterator in the iterator type list, to each dimension of an N-D vector.

    An optional kind attribute may be used to specify the combining function
    between the intermediate result and accumulator argument of rank K. This
    attribute can take the values `add`/`mul`/`minsi`/`minui`/`maxsi`/`maxui`
    /`and`/`or`/`xor` for integers, and `add`/`mul`/`minnumf`/`maxnumf`
    /`minimumf`/`maximumf` for floats. The default is `add`.

    Example:

    ```mlir
    // Simple DOT product (K = 0).
    #contraction_accesses = [
     affine_map<(i) -> (i)>,
     affine_map<(i) -> (i)>,
     affine_map<(i) -> ()>
    ]
    #contraction_trait = {
      indexing_maps = #contraction_accesses,
      iterator_types = ["reduction"]
    }
    %3 = vector.contract #contraction_trait %0, %1, %2
      : vector<10xf32>, vector<10xf32> into f32

    // 2D vector contraction with one contracting dimension (matmul, K = 2).
    #contraction_accesses = [
      affine_map<(i, j, k) -> (i, k)>,
      affine_map<(i, j, k) -> (k, j)>,
      affine_map<(i, j, k) -> (i, j)>
    ]
    #contraction_trait = {
      indexing_maps = #contraction_accesses,
      iterator_types = ["parallel", "parallel", "reduction"]
    }

    %3 = vector.contract #contraction_trait %0, %1, %2
      : vector<4x3xf32>, vector<3x7xf32> into vector<4x7xf32>

    // 4D to 3D vector contraction with two contracting dimensions and
    // one batch dimension (K = 3).
    #contraction_accesses = [
      affine_map<(b0, f0, f1, c0, c1) -> (c0, b0, c1, f0)>,
      affine_map<(b0, f0, f1, c0, c1) -> (b0, c1, c0, f1)>,
      affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>
    ]
    #contraction_trait = {
      indexing_maps = #contraction_accesses,
      iterator_types = ["parallel", "parallel", "parallel",
                        "reduction", "reduction"]
    }

    %4 = vector.contract #contraction_trait %0, %1, %2
        : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>

    // Vector contraction with mixed typed. lhs/rhs have different element
    // types than accumulator/result.
    %5 = vector.contract #contraction_trait %0, %1, %2
      : vector<10xf16>, vector<10xf16> into f32

    // Contract with max (K = 0).
    #contraction_accesses = [
     affine_map<(i) -> (i)>,
     affine_map<(i) -> (i)>,
     affine_map<(i) -> ()>
    ]
    #contraction_trait = {
      indexing_maps = #contraction_accesses,
      iterator_types = ["reduction"],
      kind = #vector.kind<maxnumf>
    }
    %6 = vector.contract #contraction_trait %0, %1, %2
      : vector<10xf32>, vector<10xf32> into f32
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$acc,
      "ArrayAttr":$indexingMaps, "ArrayAttr":$iteratorTypes)>,
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$acc,
      "ArrayRef<ArrayRef<AffineExpr>>":$indexingExprs,
      "ArrayRef<IteratorType>":$iteratorTypes)>,
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$acc,
      "ArrayAttr":$indexingMaps, "ArrayAttr":$iteratorTypes,
      "CombiningKind":$kind)>
  ];
  let extraClassDeclaration = [{
    VectorType getLhsType() {
      return ::llvm::cast<VectorType>(getLhs().getType());
    }
    VectorType getRhsType() {
      return ::llvm::cast<VectorType>(getRhs().getType());
    }
    Type getAccType() { return getAcc().getType(); }
    Type getResultType() { return getResult().getType(); }
    SmallVector<StringRef> getTraitAttrNames();
    static unsigned getAccOperandIndex() { return 2; }

    llvm::SmallVector<::mlir::AffineMap, 4> getIndexingMapsArray() {
      return llvm::to_vector<4>(getIndexingMaps().getAsValueRange<::mlir::AffineMapAttr>());
    }

    // Returns the bounds of each dimension in the iteration space spanned
    // by the iterator types of this operation.
    void getIterationBounds(SmallVectorImpl<int64_t> &iterationBounds);

    // Returns a list of index maps, where there is a list entry for each
    // op indexing map attribute (i.e. one for each input and output, with
    // the output listed last). Each index map, maps from this operations
    // iteration space, to vector dimensions of the maps input/output.
    void getIterationIndexMap(
      std::vector<DenseMap<int64_t, int64_t>> &iterationIndexMap);

    std::vector<std::pair<int64_t, int64_t>> getContractingDimMap();
    std::vector<std::pair<int64_t, int64_t>> getBatchDimMap();

    static CombiningKind getDefaultKind() {
      return CombiningKind::ADD;
    }

    SmallVector<IteratorType> getIteratorTypesArray() {
      auto range =
          getIteratorTypes()
              .template getAsValueRange<IteratorTypeAttr, IteratorType>();
      return {range.begin(), range.end()};
    }

    //===------------------------------------------------------------------===//
    // IndexingMapOpInterface interface methods implementation.
    //===------------------------------------------------------------------===//
    ArrayRef<int64_t> getShape(OpOperand * opOperand) {
      Type t = opOperand->get().getType();
      if (auto vt = dyn_cast<VectorType>(t))
        return vt.getShape();
      return {};
    }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Vector_ReductionOp :
  Vector_Op<"reduction", [Pure,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>,
     OptionalTypesMatchWith<"dest and acc have the same type",
                            "dest", "acc", "::llvm::cast<Type>($_self)">,
     DeclareOpInterfaceMethods<ArithFastMathInterface>,
     DeclareOpInterfaceMethods<MaskableOpInterface>,
     DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>
    ]>,
    Arguments<(ins Vector_CombiningKindAttr:$kind,
               AnyVectorOfAnyRank:$vector,
               Optional<AnyType>:$acc,
               DefaultValuedAttr<
                 Arith_FastMathAttr,
                 "::mlir::arith::FastMathFlags::none">:$fastmath)>,
    Results<(outs AnyType:$dest)> {
  let summary = "reduction operation";
  let description = [{
    Reduces an 1-D vector "horizontally" into a scalar using the given
    operation: `add`/`mul`/`minsi`/`minui`/`maxsi`/`maxui`/`and`/`or`/`xor` for
    integers, and `add`/`mul`/`minnumf`/`maxnumf`/`minimumf`/`maximumf` for
    floats. Reductions also allow an optional fused accumulator.

    Note that these operations are restricted to 1-D vectors to remain
    close to the corresponding LLVM intrinsics:

    http://llvm.org/docs/LangRef.html#vector-reduction-intrinsics

    Example:

    ```mlir
    %1 = vector.reduction <add>, %0 : vector<16xf32> into f32

    %3 = vector.reduction <xor>, %2 : vector<4xi32> into i32

    %4 = vector.reduction <mul>, %0, %1 : vector<16xf32> into f32
    ```
  }];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getVector().getType());
    }
  }];
  let builders = [
    // Builder that infers the type of `dest`.
    OpBuilder<(ins "CombiningKind":$kind, "Value":$vector, "Value":$acc,
                    CArg<"::mlir::arith::FastMathFlags",
                         "::mlir::arith::FastMathFlags::none">:$fastMathFlags)>,
    // Builder that infers the type of `dest` and has no accumulator.
    OpBuilder<(ins "CombiningKind":$kind, "Value":$vector,
                    CArg<"::mlir::arith::FastMathFlags",
                         "::mlir::arith::FastMathFlags::none">:$fastMathFlags)>
  ];

  let assemblyFormat = "$kind `,` $vector (`,` $acc^)? (`fastmath` `` $fastmath^)?"
                       " attr-dict `:` type($vector) `into` type($dest)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Vector_MultiDimReductionOp :
  Vector_Op<"multi_reduction", [Pure,
     AllTypesMatch<["dest", "acc"]>,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>,
     DeclareOpInterfaceMethods<InferTypeOpInterface>,
     DeclareOpInterfaceMethods<MaskableOpInterface>,
     DeclareOpInterfaceMethods<VectorUnrollOpInterface,
                               ["getShapeForUnroll"]>]>,
    Arguments<(ins Vector_CombiningKindAttr:$kind,
                   AnyVectorOfNonZeroRank:$source,
                   AnyType:$acc,
                   DenseI64ArrayAttr:$reduction_dims)>,
    Results<(outs AnyType:$dest)> {
  let summary = "Multi-dimensional reduction operation";
  let description = [{
    Reduces an n-D vector into an (n-k)-D vector (or a scalar when k == n)
    using the given operation: `add`/`mul`/`minsi`/`minui`/`maxsi`/`maxui`
    /`and`/`or`/`xor` for integers, and `add`/`mul`/`minnumf`/`maxnumf`/`minimumf`
    /`maximumf` for floats.
    Takes an initial accumulator operand.

    Example:

    ```mlir
    %1 = vector.multi_reduction <add>, %0, %acc0 [1, 3] :
      vector<4x8x16x32xf32> to vector<4x16xf32>
    %2 = vector.multi_reduction <add>, %1, %acc1 [0, 1] :
      vector<4x16xf32> to f32
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$source, "Value":$acc,
                   "ArrayRef<bool>":$reductionMask, "CombiningKind":$kind)>
  ];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getSource().getType());
    }
    Type getDestType() {
      return getDest().getType();
    }

    bool isReducedDim(int64_t d) {
      assert(d >= 0 && d < static_cast<int64_t>(getReductionMask().size()) &&
        "d overflows the number of dims");
      return getReductionMask()[d];
    }

    SmallVector<bool> getReductionMask() {
      SmallVector<bool> res(getSourceVectorType().getRank(), false);
      for (int64_t dim : getReductionDims())
        res[dim] = true;
      return res;
    }
    static SmallVector<bool> getReductionMask(
        ArrayRef<int64_t> reductionDims, unsigned sourceRank) {
      SmallVector<bool> res(sourceRank, false);
      for (auto idx : reductionDims)
        res[idx] = true;
      return res;
    }
  }];
  let assemblyFormat =
    "$kind `,` $source `,` $acc attr-dict $reduction_dims `:` type($source) `to` type($dest)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Vector_BroadcastOp :
  Vector_Op<"broadcast", [Pure,
     DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>,
     DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins AnyType:$source)>,
    Results<(outs AnyVectorOfAnyRank:$vector)> {
  let summary = "broadcast operation";
  let description = [{
    Broadcasts the scalar or k-D vector value in the source operand
    to a n-D result vector such that the broadcast makes sense, i.e.,
    the source operand is duplicated to match the given rank and sizes
    in the result vector. The legality rules are:
    * the source operand must have the same element type as the result type
    * a k-D vector <s_1 x .. x s_k x type> can be broadcast to
      a n-D vector <t_1 x .. x t_n x type> if
       * k <= n, and
       * the sizes in the trailing dimensions n-k < i <= n with j=i+k-n
          match exactly as s_j = t_i or s_j = 1:
       ```
           t_1 x   ..  t_n-k x t_n-k+1 x .. x t_i x .. x t_n
                               s_1     x .. x s_j x .. x s_k
               <duplication>         <potential stretch>
       ```
       * in addition, any scalable unit dimension, `[1]`, must match exactly.

    The source operand is duplicated over all the missing leading dimensions
    and stretched over the trailing dimensions where the source has a non-equal
    dimension of 1 (stretching a trailing dimension is also referred to as
    "dim-1" broadcasting). These rules imply that any scalar broadcast (k=0) to
    any shaped vector with the same element type is always legal.

    Example:

    ```mlir
    %0 = arith.constant 0.0 : f32
    %1 = vector.broadcast %0 : f32 to vector<16xf32>
    %2 = vector.broadcast %1 : vector<16xf32> to vector<4x16xf32>
    ```
  }];
  let extraClassDeclaration = [{
    Type getSourceType() { return getSource().getType(); }
    VectorType getResultVectorType() {
      return ::llvm::cast<VectorType>(getVector().getType());
    }

    /// Return the dimensions of the result vector that were formerly ones in the
    /// source tensor and thus correspond to "dim-1" broadcasting.
    llvm::SetVector<int64_t> computeBroadcastedUnitDims();

    /// Broadcast `value` to a vector of `dstShape`, knowing that exactly the
    /// `broadcastedDims` dimensions in the dstShape are broadcasted.
    /// This requires (and asserts) that the broadcast is free of "dim-1"
    /// broadcasting.
    /// Since vector.broadcast only allows expanding leading dimensions, an extra
    /// vector.transpose may be inserted to make the broadcast possible.
    /// `value`, `dstShape` and `broadcastedDims` must be properly specified or
    /// the helper will assert. This means:
    ///   1. `dstShape` must not be empty.
    ///   2. `broadcastedDims` must be confined to [0 .. rank(value.getResultVectorType)]
    ///   2. `dstShape` trimmed of the dimensions specified in `broadcastedDims`
    //       must match the `value` shape.
    static Value createOrFoldBroadcastOp(
      OpBuilder &b, Value value,
      ArrayRef<int64_t> dstShape,
      const llvm::SetVector<int64_t> &broadcastedDims);
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($vector)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Vector_ShuffleOp
    : Vector_Op<
          "shuffle",
          [Pure,
           PredOpTrait<"first operand v1 and result have same element type",
                       TCresVTEtIsSameAsOpBase<0, 0>>,
           PredOpTrait<"second operand v2 and result have same element type",
                       TCresVTEtIsSameAsOpBase<0, 1>>,
           InferTypeOpAdaptor]>,
      Arguments<(ins AnyFixedVectorOfAnyRank:$v1, AnyFixedVectorOfAnyRank:$v2,
          DenseI64ArrayAttr:$mask)>,
      Results<(outs AnyVectorOfNonZeroRank:$vector)> {
  let summary = "shuffle operation";
  let description = [{
    The shuffle operation constructs a permutation (or duplication) of elements
    from two input vectors, returning a vector with the same element type as
    the input and a length that is the same as the shuffle mask. The two input
    vectors must have the same element type, same rank, and trailing dimension
    sizes and shuffles their values in the leading dimension (which may differ
    in size) according to the given mask. The legality rules are:
    * the two operands must have the same element type as the result
      - Either, the two operands and the result must have the same
        rank and trailing dimension sizes, viz. given two k-D operands
                v1 : <s_1 x s_2 x .. x s_k x type> and
                v2 : <t_1 x t_2 x .. x t_k x type>
        we have s_i = t_i for all 1 < i <= k
      - Or, the two operands must be 0-D vectors and the result is a 1-D vector.
    * the mask length equals the leading dimension size of the result
    * numbering the input vector indices left to right across the operands, all
      mask values must be within range, viz. given two k-D operands v1 and v2
      above, all mask values are in the range [0,s_1+t_1). The value `-1`
      represents a poison mask value, which specifies that the selected element
      is poison.

    Note, scalable vectors are not supported.

    Example:

    ```mlir
    %0 = vector.shuffle %a, %b[0, 3]
               : vector<2xf32>, vector<2xf32>       ; yields vector<2xf32>
    %1 = vector.shuffle %c, %b[0, 1, 2]
               : vector<2x16xf32>, vector<1x16xf32> ; yields vector<3x16xf32>
    %2 = vector.shuffle %a, %b[3, 2, 1, 0]
               : vector<2xf32>, vector<2xf32>       ; yields vector<4xf32>
    %3 = vector.shuffle %a, %b[0, 1]
               : vector<f32>, vector<f32>           ; yields vector<2xf32>
    %4 = vector.shuffle %a, %b[0, 4, -1, -1, -1, -1]
               : vector<4xf32>, vector<4xf32>       ; yields vector<6xf32>
    ```
  }];

  let extraClassDeclaration = extraPoisonClassDeclaration # [{
    VectorType getV1VectorType() {
      return ::llvm::cast<VectorType>(getV1().getType());
    }
    VectorType getV2VectorType() {
      return ::llvm::cast<VectorType>(getV2().getType());
    }
    VectorType getResultVectorType() {
      return ::llvm::cast<VectorType>(getVector().getType());
    }
  }];

  let assemblyFormat = "operands $mask attr-dict `:` type(operands)";

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def ResultIsDoubleSourceVectorType : TypesMatchWith<
    "type of 'result' is double the width of the inputs",
    "lhs", "result",
    [{
      [&]() -> ::mlir::VectorType {
        auto vectorType = ::llvm::cast<::mlir::VectorType>($_self);
        ::mlir::VectorType::Builder builder(vectorType);
        if (vectorType.getRank() == 0) {
          static constexpr int64_t v2xTyShape[] = {2};
          return builder.setShape(v2xTyShape);
        }
        auto lastDim = vectorType.getRank() - 1;
        return builder.setDim(lastDim, vectorType.getDimSize(lastDim) * 2);
      }()
    }]>;

def Vector_InterleaveOp :
  Vector_Op<"interleave", [Pure, AllTypesMatch<["lhs", "rhs"]>,
    ResultIsDoubleSourceVectorType]> {
  let summary = "constructs a vector by interleaving two input vectors";
  let description = [{
    The interleave operation constructs a new vector by interleaving the
    elements from the trailing (or final) dimension of two input vectors,
    returning a new vector where the trailing dimension is twice the size.

    Note that for the n-D case this differs from the interleaving possible with
    `vector.shuffle`, which would only operate on the leading dimension.

    Another key difference is this operation supports scalable vectors, though
    currently a general LLVM lowering is limited to the case where only the
    trailing dimension is scalable.

    Example:
    ```mlir
    %a = arith.constant dense<[0, 1]> : vector<2xi32>
    %b = arith.constant dense<[2, 3]> : vector<2xi32>
    // The value of `%0` is `[0, 2, 1, 3]`.
    %0 = vector.interleave %a, %b : vector<2xi32> -> vector<4xi32>

    // Examples showing allowed input and result types.
    %1 = vector.interleave %c, %d : vector<f16> -> vector<2xf16>
    %2 = vector.interleave %e, %f : vector<6x3xf32> -> vector<6x6xf32>
    %3 = vector.interleave %g, %h : vector<[4]xi32> -> vector<[8]xi32>
    %4 = vector.interleave %i, %j : vector<2x4x[2]xf64> -> vector<2x4x[4]xf64>
    ```
  }];

  let arguments = (ins AnyVectorOfAnyRank:$lhs, AnyVectorOfAnyRank:$rhs);
  let results = (outs AnyVectorOfNonZeroRank:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs  attr-dict `:` type($lhs) `->` type($result)
  }];

  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getLhs().getType());
    }
    VectorType getResultVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];
}

class ResultIsHalfSourceVectorType<string result> : TypesMatchWith<
  "the trailing dimension of the results is half the width of source trailing dimension",
  "source", result,
  [{
    [&]() -> ::mlir::VectorType {
      auto vectorType = ::llvm::cast<mlir::VectorType>($_self);
      ::mlir::VectorType::Builder builder(vectorType);
      auto lastDim = vectorType.getRank() - 1;
      auto newDimSize = vectorType.getDimSize(lastDim) / 2;;
      if (newDimSize <= 0)
         return vectorType; // (invalid input type)
      return builder.setDim(lastDim, newDimSize);
    }()
  }]
>;

def SourceVectorEvenElementCount : PredOpTrait<
  "the trailing dimension of the source vector has an even number of elements",
  CPred<[{
    [&](){
      auto srcVec = getSourceVectorType();
      return srcVec.getDimSize(srcVec.getRank() - 1) % 2 == 0;
    }()
  }]>
>;

def Vector_DeinterleaveOp :
  Vector_Op<"deinterleave", [Pure,
    SourceVectorEvenElementCount,
    ResultIsHalfSourceVectorType<"res1">,
    AllTypesMatch<["res1", "res2"]>
    ]> {
      let summary = "constructs two vectors by deinterleaving an input vector";
      let description = [{
        The deinterleave operation constructs two vectors from a single input
        vector. The first result vector contains the elements from even indexes
        of the input, and the second contains elements from odd indexes. This is
        the inverse of a `vector.interleave` operation.

        Each output's trailing dimension is half of the size of the input
        vector's trailing dimension. This operation requires the input vector
        to have a rank > 0 and an even number of elements in its trailing
        dimension.

        The operation supports scalable vectors.

        Example:
        ```mlir
        %0, %1 = vector.deinterleave %a
                   : vector<8xi8> -> vector<4xi8>
        %2, %3 = vector.deinterleave %b
                   : vector<2x8xi8> -> vector<2x4xi8>
        %4, %5 = vector.deinterleave %c
                   : vector<2x8x4xi8> -> vector<2x8x2xi8>
        %6, %7 = vector.deinterleave %d
                   : vector<[8]xf32> -> vector<[4]xf32>
        %8, %9 = vector.deinterleave %e
                   : vector<2x[6]xf64> -> vector<2x[3]xf64>
        %10, %11 = vector.deinterleave %f
                   : vector<2x4x[6]xf64> -> vector<2x4x[3]xf64>
        ```
      }];

      let arguments = (ins AnyVectorOfNonZeroRank:$source);
      let results = (outs AnyVectorOfNonZeroRank:$res1, AnyVectorOfNonZeroRank:$res2);

      let assemblyFormat = [{
        $source attr-dict `:` type($source) `->` type($res1)
      }];

      let extraClassDeclaration = [{
        VectorType getSourceVectorType() {
          return ::llvm::cast<VectorType>(getSource().getType());
        }
        VectorType getResultVectorType() {
          return ::llvm::cast<VectorType>(getRes1().getType());
        }
      }];
    }

def Vector_ExtractOp :
  Vector_Op<"extract", [Pure,
     DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>,
     PredOpTrait<"operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>,
     InferTypeOpAdaptorWithIsCompatible]> {
  let summary = "extract operation";
  let description = [{
    Extracts an (n − k)-D result sub-vector from an n-D source vector at a
    specified k-D position. When n = k, the result degenerates to a scalar
    element.

    Static and dynamic indices must be greater or equal to zero and less than
    the size of the corresponding dimension. The result is undefined if any
    index is out-of-bounds. The value `-1` represents a poison index, which
    specifies that the extracted element is poison.

    Example:

    ```mlir
    %1 = vector.extract %0[3]: vector<8x16xf32> from vector<4x8x16xf32>
    %2 = vector.extract %0[2, 1, 3]: f32 from vector<4x8x16xf32>
    %4 = vector.extract %0[%a, %b, %c]: f32 from vector<4x8x16xf32>
    %5 = vector.extract %0[2, %b]: vector<16xf32> from vector<4x8x16xf32>
    %6 = vector.extract %10[-1, %c]: f32 from vector<4x16xf32>
    ```
  }];

  let arguments = (ins
    AnyVectorOfAnyRank:$source,
    Variadic<Index>:$dynamic_position,
    DenseI64ArrayAttr:$static_position
  );
  let results = (outs AnyType:$result);

  let builders = [
    // Builder to extract a scalar from a rank-0 vector.
    OpBuilder<(ins "Value":$source)>,
    OpBuilder<(ins "Value":$source, "int64_t":$position)>,
    OpBuilder<(ins "Value":$source, "OpFoldResult":$position)>,
    OpBuilder<(ins "Value":$source, "ArrayRef<int64_t>":$position)>,
    OpBuilder<(ins "Value":$source, "ArrayRef<OpFoldResult>":$position)>,
  ];

  let extraClassDeclaration = extraPoisonClassDeclaration # [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getSource().getType());
    }

    /// Return a vector with all the static and dynamic position indices.
    SmallVector<OpFoldResult> getMixedPosition() {
      OpBuilder builder(getContext());
      return getMixedValues(getStaticPosition(), getDynamicPosition(), builder);
    }

    unsigned getNumIndices() {
      return getStaticPosition().size();
    }

    /// Return "true" if the op has at least one dynamic position.
    bool hasDynamicPosition() {
      return !getDynamicPosition().empty();
    }

    /// Wrapper for getSource, which replaced getVector.
    [[deprecated("Use getSource instead!")]] ::mlir::Value getVector() {
      return getSource();
    }
  }];

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($dynamic_position, $static_position)
    attr-dict `:` type($result) `from` type($source)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Vector_FMAOp :
  Op<Vector_Dialect, "fma", [
       Pure, AllTypesMatch<["lhs", "rhs", "acc", "result"]>,
       DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>
     ] # ElementwiseMappable.traits>,
    Arguments<(ins VectorOfAnyRankOf<[AnyFloat]>:$lhs,
                   VectorOfAnyRankOf<[AnyFloat]>:$rhs,
                   VectorOfAnyRankOf<[AnyFloat]>:$acc)>,
    Results<(outs VectorOfAnyRankOf<[AnyFloat]>:$result)> {
  let summary = "vector fused multiply-add";
  let description = [{
    Multiply-add expressions operate on n-D vectors and compute a fused
    pointwise multiply-and-accumulate: `$result = $lhs * $rhs + $acc`.
    All operands and result have the same vector type. The semantics
    of the operation correspond to those of the `llvm.fma`
    [intrinsic](https://llvm.org/docs/LangRef.html#int-fma). In the
    particular case of lowering to LLVM, this is guaranteed to lower
    to the `llvm.fma.*` intrinsic.

    Example:

    ```mlir
    %3 = vector.fma %0, %1, %2: vector<8x16xf32>
    ```
  }];
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs)";
  let extraClassDeclaration = [{
    VectorType getVectorType() { return ::llvm::cast<VectorType>(getLhs().getType()); }
  }];
}

def Vector_ToElementsOp : Vector_Op<"to_elements", [
    InferTypeOpAdaptor, Pure,
    ShapedTypeMatchesElementCountAndTypes<"source", "elements">]> {
  let summary = "operation that decomposes a vector into all its scalar elements";
  let description = [{
    This operation decomposes all the scalar elements from a vector. The
    decomposed scalar elements are returned in row-major order. The number of
    scalar results must match the number of elements in the input vector type.
    All the result elements have the same result type, which must match the
    element type of the input vector. Scalable vectors are not supported.

    Examples:

    ```mlir
    // Decompose a 0-D vector.
    %0 = vector.to_elements %v0 : vector<f32>
    // %0 = %v0[0]

    // Decompose a 1-D vector.
    %0:2 = vector.to_elements %v1 : vector<2xf32>
    // %0#0 = %v1[0]
    // %0#1 = %v1[1]

    // Decompose a 2-D.
    %0:6 = vector.to_elements %v2 : vector<2x3xf32>
    // %0#0 = %v2[0, 0]
    // %0#1 = %v2[0, 1]
    // %0#2 = %v2[0, 2]
    // %0#3 = %v2[1, 0]
    // %0#4 = %v2[1, 1]
    // %0#5 = %v2[1, 2]

    // Decompose a 3-D vector.
    %0:6 = vector.to_elements %v3 : vector<3x1x2xf32>
    // %0#0 = %v3[0, 0, 0]
    // %0#1 = %v3[0, 0, 1]
    // %0#2 = %v3[1, 0, 0]
    // %0#3 = %v3[1, 0, 1]
    // %0#4 = %v3[2, 0, 0]
    // %0#5 = %v3[2, 0, 1]
    ```
  }];

  let arguments = (ins AnyVectorOfAnyRank:$source);
  let results = (outs Variadic<AnyType>:$elements);
  let assemblyFormat = "$source attr-dict `:` type($source)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Vector_FromElementsOp : Vector_Op<"from_elements", [
    Pure,
    ShapedTypeMatchesElementCountAndTypes<"dest", "elements">]> {
  let summary = "operation that defines a vector from scalar elements";
  let description = [{
    This operation defines a vector from one or multiple scalar elements. The
    scalar elements are arranged in row-major within the vector. The number of
    elements must match the number of elements in the result type. All elements
    must have the same type, which must match the element type of the result
    vector type. Scalable vectors are not supported.

    Examples:

    ```mlir
    // Define a 0-D vector.
    %0 = vector.from_elements %f1 : vector<f32>
    // [%f1]

    // Define a 1-D vector.
    %1 = vector.from_elements %f1, %f2 : vector<2xf32>
    // [%f1, %f2]

    // Define a 2-D vector.
    %2 = vector.from_elements %f1, %f2, %f3, %f4, %f5, %f6 : vector<2x3xf32>
    // [[%f1, %f2, %f3], [%f4, %f5, %f6]]

    // Define a 3-D vector.
    %3 = vector.from_elements %f1, %f2, %f3, %f4, %f5, %f6 : vector<3x1x2xf32>
    // [[[%f1, %f2]], [[%f3, %f4]], [[%f5, %f6]]]
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs AnyFixedVectorOfAnyRank:$dest);
  let assemblyFormat = "$elements attr-dict `:` type($dest)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Vector_InsertOp :
  Vector_Op<"insert", [Pure,
     DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>,
     AllTypesMatch<["dest", "result"]>]> {
  let summary = "insert operation";
  let description = [{
    Inserts an (n - k)-D sub-vector (value-to-store) into an n-D destination
    vector at a specified k-D position. When n = 0, value-to-store degenerates
    to a scalar element inserted into the n-D destination vector.

    Static and dynamic indices must be greater or equal to zero and less than
    the size of the corresponding dimension. The result is undefined if any
    index is out-of-bounds. The value `-1` represents a poison index, which
    specifies that the resulting vector is poison.

    Example:

    ```mlir
    %2 = vector.insert %0, %1[3] : vector<8x16xf32> into vector<4x8x16xf32>
    %5 = vector.insert %3, %4[2, 1, 3] : f32 into vector<4x8x16xf32>
    %11 = vector.insert %9, %10[%a, %b, %c] : f32 into vector<4x8x16xf32>
    %12 = vector.insert %4, %10[2, %b] : vector<16xf32> into vector<4x8x16xf32>
    %13 = vector.insert %20, %1[-1, %c] : f32 into vector<4x16xf32>
    ```
  }];

  let arguments = (ins
    AnyType:$valueToStore,
    AnyVectorOfAnyRank:$dest,
    Variadic<Index>:$dynamic_position,
    DenseI64ArrayAttr:$static_position
  );
  let results = (outs AnyVectorOfAnyRank:$result);

  let builders = [
    // Builder to insert a scalar/rank-0 vector into a rank-0 vector.
    OpBuilder<(ins "Value":$valueToStore, "Value":$dest)>,
    OpBuilder<(ins "Value":$valueToStore, "Value":$dest, "int64_t":$position)>,
    OpBuilder<(ins "Value":$valueToStore, "Value":$dest, "OpFoldResult":$position)>,
    OpBuilder<(ins "Value":$valueToStore, "Value":$dest, "ArrayRef<int64_t>":$position)>,
    OpBuilder<(ins "Value":$valueToStore, "Value":$dest, "ArrayRef<OpFoldResult>":$position)>,
  ];

  let extraClassDeclaration = extraPoisonClassDeclaration # [{
    Type getValueToStoreType() { return getValueToStore().getType(); }
    VectorType getDestVectorType() {
      return ::llvm::cast<VectorType>(getDest().getType());
    }

    /// Return a vector with all the static and dynamic position indices.
    SmallVector<OpFoldResult> getMixedPosition() {
      OpBuilder builder(getContext());
      return getMixedValues(getStaticPosition(), getDynamicPosition(), builder);
    }

    unsigned getNumIndices() {
      return getStaticPosition().size();
    }

    bool hasDynamicPosition() {
      return llvm::any_of(getDynamicPosition(),
                          [](Value operand) { return operand != nullptr; });
    }
  }];

  let assemblyFormat = [{
    $valueToStore `,` $dest custom<DynamicIndexList>($dynamic_position, $static_position)
    attr-dict `:` type($valueToStore) `into` type($dest)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Vector_ScalableInsertOp :
  Vector_Op<"scalable.insert", [Pure,
       AllElementTypesMatch<["valueToStore", "dest"]>,
       AllTypesMatch<["dest", "result"]>,
       PredOpTrait<"position is a multiple of the source length.",
        CPred<
          "(getPos() % getSourceVectorType().getNumElements()) == 0"
        >>]>,
     Arguments<(ins VectorOfRank<[1]>:$valueToStore,
                    ScalableVectorOfRank<[1]>:$dest,
                    I64Attr:$pos)>,
     Results<(outs ScalableVectorOfRank<[1]>:$result)> {
  let summary = "insert subvector into scalable vector operation";
  // NOTE: This operation is designed to map to `llvm.vector.insert`, and its
  //       documentation should be kept aligned with LLVM IR:
  //       https://llvm.org/docs/LangRef.html#llvm-vector-insert-intrinsic
  let description = [{
    This operations takes a rank-1 fixed-length or scalable subvector and
    inserts it within the destination scalable vector starting from the
    position specificed by `pos`. If the source vector is scalable, the
    insertion position will be scaled by the runtime scaling factor of the
    source subvector.

    The insertion position must be a multiple of the minimum size of the source
    vector. For the operation to be well defined, the source vector must fit in
    the destination vector from the specified position. Since the destination
    vector is scalable and its runtime length is unknown, the validity of the
    operation can't be verified nor guaranteed at compile time.

    Example:

    ```mlir
    %2 = vector.scalable.insert %0, %1[8] : vector<4xf32> into vector<[16]xf32>
    %5 = vector.scalable.insert %3, %4[0] : vector<8xf32> into vector<[4]xf32>
    %8 = vector.scalable.insert %6, %7[0] : vector<[4]xf32> into vector<[8]xf32>
    ```

    Invalid example:
    ```mlir
    %2 = vector.scalable.insert %0, %1[5] : vector<4xf32> into vector<[16]xf32>
    ```
  }];

  let assemblyFormat = [{
    $valueToStore `,` $dest `[` $pos `]` attr-dict `:` type($valueToStore) `into` type($dest)
  }];

  let extraClassDeclaration = extraPoisonClassDeclaration # [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getValueToStore().getType());
    }
    VectorType getDestVectorType() {
      return ::llvm::cast<VectorType>(getDest().getType());
    }
  }];
}

def Vector_ScalableExtractOp :
  Vector_Op<"scalable.extract", [Pure,
       AllElementTypesMatch<["source", "result"]>,
       PredOpTrait<"position is a multiple of the result length.",
        CPred<
          "(getPos() % getResultVectorType().getNumElements()) == 0"
        >>]>,
     Arguments<(ins ScalableVectorOfRank<[1]>:$source,
                    I64Attr:$pos)>,
     Results<(outs VectorOfRank<[1]>:$result)> {
  let summary = "extract subvector from scalable vector operation";
  // NOTE: This operation is designed to map to `llvm.vector.extract`, and its
  //       documentation should be kept aligned with LLVM IR:
  //       https://llvm.org/docs/LangRef.html#llvm-vector-extract-intrinsic
  let description = [{
    Takes rank-1 source vector and a position `pos` within the source
    vector, and extracts a subvector starting from that position.

    The extraction position must be a multiple of the minimum size of the result
    vector. For the operation to be well defined, the destination vector must
    fit within the source vector from the specified position. Since the source
    vector is scalable and its runtime length is unknown, the validity of the
    operation can't be verified nor guaranteed at compile time.

    Example:

    ```mlir
    %1 = vector.scalable.extract %0[8] : vector<4xf32> from vector<[8]xf32>
    %3 = vector.scalable.extract %2[0] : vector<[4]xf32> from vector<[8]xf32>
    ```

    Invalid example:
    ```mlir
    %1 = vector.scalable.extract %0[5] : vector<4xf32> from vector<[16]xf32>
    ```
  }];

  let assemblyFormat = [{
    $source `[` $pos `]` attr-dict `:` type($result) `from` type($source)
  }];

  let extraClassDeclaration = extraPoisonClassDeclaration # [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getSource().getType());
    }
    VectorType getResultVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
    /// Wrapper for getSource, which replaced getVector.
    [[deprecated("Use getSource instead!")]] ::mlir::Value getVector() {
      return getSource();
    }
  }];
}

def Vector_InsertStridedSliceOp :
  Vector_Op<"insert_strided_slice", [Pure,
    PredOpTrait<"operand #0 and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>,
    AllTypesMatch<["dest", "result"]>]>,
    Arguments<(ins AnyVectorOfNonZeroRank:$valueToStore, AnyVectorOfNonZeroRank:$dest, I64ArrayAttr:$offsets,
               I64ArrayAttr:$strides)>,
    Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "strided_slice operation";
  let description = [{
    Takes a k-D valueToStore vector, an n-D destination vector (n >= k), n-sized
    `offsets` integer array attribute, a k-sized `strides` integer array attribute
    and inserts the k-D valueToStore vector as a strided subvector at the proper offset
    into the n-D destination vector.

    At the moment strides must contain only 1s.

    Returns an n-D vector that is a copy of the n-D destination vector in which
    the last k-D dimensions contain the k-D valueToStore vector elements strided at
    the proper location as specified by the offsets.

    Example:

    ```mlir
    %2 = vector.insert_strided_slice %0, %1
        {offsets = [0, 0, 2], strides = [1, 1]}:
      vector<2x4xf32> into vector<16x4x8xf32>
    ```
  }];

  let assemblyFormat = [{
    $valueToStore `,` $dest attr-dict `:` type($valueToStore) `into` type($dest)
  }];

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$dest,
      "ArrayRef<int64_t>":$offsets, "ArrayRef<int64_t>":$strides)>
  ];
  let extraClassDeclaration = [{
    // TODO: Rename
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getValueToStore().getType());
    }
    VectorType getDestVectorType() {
      return ::llvm::cast<VectorType>(getDest().getType());
    }
    bool hasNonUnitStrides() {
      return llvm::any_of(getStrides(), [](Attribute attr) {
        return ::llvm::cast<IntegerAttr>(attr).getInt() != 1;
      });
    }
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Vector_OuterProductOp :
  Vector_Op<"outerproduct", [Pure,
    PredOpTrait<"lhs operand and result have same element type",
                TCresVTEtIsSameAsOpBase<0, 0>>,
    PredOpTrait<"rhs operand and result have same element type",
                TCresVTEtIsSameAsOpBase<0, 1>>,
    DeclareOpInterfaceMethods<MaskableOpInterface>]>,
    Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyType:$rhs,
               Optional<AnyVectorOfNonZeroRank>:$acc,
               DefaultValuedAttr<Vector_CombiningKindAttr, "CombiningKind::ADD">:$kind)>,
    Results<(outs AnyVectorOfNonZeroRank)> {
  let summary = "vector outerproduct with optional fused add";
  let description = [{
    Takes 2 1-D vectors and returns the 2-D vector containing the outer-product,
    as illustrated below:
    ```
     outer |   [c, d]
     ------+------------
       [a, | [ [a*c, a*d],
        b] |   [b*c, b*d] ]
    ```
    This operation also accepts a 1-D vector lhs and a scalar rhs. In this
    case a simple AXPY operation is performed, which returns a 1-D vector.
    ```
        [a, b] * c = [a*c, b*c]
    ```

    An optional extra vector argument with the same shape as the output
    vector may be specified in which case the operation returns the sum of
    the outer-product and the extra vector. In this multiply-accumulate
    scenario for floating-point arguments, the rounding mode is enforced
    by guaranteeing that a fused-multiply add operation is emitted. When
    lowered to the LLVMIR dialect, this form emits `llvm.intr.fma`, which
    is guaranteed to lower to actual `fma` instructions on x86.

    An optional kind attribute may be specified to be: `add`/`mul`/`minsi`
    /`minui`/`maxsi`/`maxui`/`and`/`or`/`xor` for integers, and `add`/`mul`
    /`minnumf`/`maxnumf`/`minimumf`/`maximumf` for floats. The default is
    `add`.

    Example:

    ```
    %2 = vector.outerproduct %0, %1: vector<4xf32>, vector<8xf32>
    return %2: vector<4x8xf32>

    %3 = vector.outerproduct %0, %1, %2:
      vector<4xf32>, vector<8xf32>, vector<4x8xf32>
    return %3: vector<4x8xf32>

    %4 = vector.outerproduct %0, %1, %2 {kind = #vector.kind<maxnumf>}:
      vector<4xf32>, vector<8xf32>, vector<4x8xf32>
    return %3: vector<4x8xf32>

    %6 = vector.outerproduct %4, %5: vector<10xf32>, f32
    return %6: vector<10xf32>

    ```
  }];
  let builders = [
    // Build an op without mask, use the type of `acc` as the return type.
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$acc)>
  ];
  let extraClassDeclaration = [{
    VectorType getOperandVectorTypeLHS() {
      return ::llvm::cast<VectorType>(getLhs().getType());
    }
    Type getOperandTypeRHS() {
      return getRhs().getType();
    }
    VectorType getOperandVectorTypeACC() {
      return getAcc()
        ? ::llvm::cast<VectorType>(getAcc().getType())
        : VectorType();
    }
    VectorType getResultVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
    static CombiningKind getDefaultKind() {
      return CombiningKind::ADD;
    }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Vector_ExtractStridedSliceOp :
  Vector_Op<"extract_strided_slice", [Pure,
    PredOpTrait<"operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins AnyVectorOfNonZeroRank:$source, I64ArrayAttr:$offsets,
               I64ArrayAttr:$sizes, I64ArrayAttr:$strides)>,
    Results<(outs AnyVectorOfNonZeroRank)> {
  let summary = "extract_strided_slice operation";
  let description = [{
    Takes an n-D vector, k-D `offsets` integer array attribute, a k-sized
    `sizes` integer array attribute, a k-sized `strides` integer array
    attribute and extracts the n-D subvector at the proper offset.

    At the moment strides must contain only 1s.

    Returns an n-D vector where the first k-D dimensions match the `sizes`
    attribute. The returned subvector contains the elements starting at offset
    `offsets` and ending at `offsets + sizes`.

    Example:

    ```mlir
    %1 = vector.extract_strided_slice %0
        {offsets = [0, 2], sizes = [2, 4], strides = [1, 1]}:
      vector<4x8x16xf32> to vector<2x4x16xf32>

    // TODO: Evolve to a range form syntax similar to:
    %1 = vector.extract_strided_slice %0[0:2:1][2:4:1]
      vector<4x8x16xf32> to vector<2x4x16xf32>
    ```

    TODO: Implement support for poison indices.
  }];
  let builders = [
    OpBuilder<(ins "Value":$source, "ArrayRef<int64_t>":$offsets,
      "ArrayRef<int64_t>":$sizes, "ArrayRef<int64_t>":$strides)>
  ];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getSource().getType());
    }
    void getOffsets(SmallVectorImpl<int64_t> &results);
    bool hasNonUnitStrides() {
      return llvm::any_of(getStrides(), [](Attribute attr) {
        return ::llvm::cast<IntegerAttr>(attr).getInt() != 1;
      });
    }
    /// Wrapper for getSource, which replaced getVector.
    [[deprecated("Use getSource instead!")]] ::mlir::Value getVector() {
      return getSource();
    }
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type(results)";
}

// TODO: Tighten semantics so that masks and inbounds can't be used
// simultaneously within the same transfer op.
def Vector_TransferReadOp :
  Vector_Op<"transfer_read", [
      DeclareOpInterfaceMethods<VectorTransferOpInterface>,
      DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>,
      DeclareOpInterfaceMethods<MaskableOpInterface>,
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      DeclareOpInterfaceMethods<ConditionallySpeculatable>,
      DeclareOpInterfaceMethods<MemorySpaceCastConsumerOpInterface>,
      AttrSizedOperandSegments,
      DestinationStyleOpInterface
    ]>,
    Arguments<(ins AnyShaped:$base,
                   Variadic<Index>:$indices,
                   AffineMapAttr:$permutation_map,
                   AnyType:$padding,
                   Optional<VectorOfNonZeroRankOf<[I1]>>:$mask,
                   BoolArrayAttr:$in_bounds)>,
    Results<(outs AnyVectorOfAnyRank:$vector)> {

  let summary = "Reads a supervector from memory into an SSA vector value.";

  let description = [{
    The `vector.transfer_read` op performs a read from a slice within a
    [MemRef](../LangRef.md#memref-type) or a Ranked
    [Tensor](../LangRef.md#tensor-type) supplied as its first operand
    into a [vector](../LangRef.md#vector-type) of the same base elemental type.

    A memref/tensor operand with vector element type, must have its vector
    element type match a suffix (shape and element type) of the vector (e.g.
    memref<3x2x6x4x3xf32>, vector<1x1x4x3xf32>).

    The slice is further defined by a full-rank index within the MemRef/Tensor,
    supplied as the operands `[1 .. 1 + rank(memref/tensor))` that defines the
    starting point of the transfer (e.g. `%A[%i0, %i1, %i2]`).

    The permutation_map [attribute](../LangRef.md#attributes) is an
    [affine-map](Affine.md#affine-maps) which specifies the transposition on the
    slice to match the vector shape. The permutation map may be implicit and
    omitted from parsing and printing if it is the canonical minor identity map
    (i.e. if it does not permute or broadcast any dimension).

    The size of the slice is specified by the size of the vector, given as the
    return type.

    An SSA value `padding` of the same elemental type as the MemRef/Tensor is
    provided to specify a fallback value in the case of out-of-bounds accesses
    and/or masking.

    An optional SSA value `mask` may be specified to mask out elements read from
    the MemRef/Tensor. The `mask` type is an `i1` vector with a shape that
    matches how elements are read from the MemRef/Tensor, *before* any
    permutation or broadcasting. Elements whose corresponding mask element is
    `0` are masked out and replaced with `padding`.

    For every vector dimension, the boolean array attribute `in_bounds`
    specifies if the transfer is guaranteed to be within the source bounds. If
    set to "false", accesses (including the starting point) may run
    out-of-bounds along the respective vector dimension as the index increases.
    Non-vector dimensions *must* always be in-bounds. The `in_bounds` array
    length has to be equal to the vector rank. This attribute has a default
    value: `false` (i.e. "out-of-bounds"). When skipped in the textual IR, the
    default value is assumed. Similarly, the OP printer will omit this
    attribute when all dimensions are out-of-bounds (i.e. the default value is
    used).

    A `vector.transfer_read` can be lowered to a simple load if all dimensions
    are specified to be within bounds and no `mask` was specified.

    This operation is called 'read' by opposition to 'load' because the
    super-vector granularity is generally not representable with a single
    hardware register. A `vector.transfer_read` is thus a mid-level abstraction
    that supports super-vectorization with non-effecting padding for full-tile
    only operations.

    More precisely, let's dive deeper into the permutation_map for the following
    MLIR:

    ```mlir
    vector.transfer_read %A[%expr1, %expr2, %expr3, %expr4]
      { permutation_map : (d0,d1,d2,d3) -> (d2,0,d0) } :
      memref<?x?x?x?xf32>, vector<3x4x5xf32>
    ```

    This operation always reads a slice starting at `%A[%expr1, %expr2, %expr3,
    %expr4]`. The size of the slice can be inferred from the resulting vector
    shape and walking back through the permutation map: 3 along d2 and 5 along
    d0, so the slice is: `%A[%expr1 : %expr1 + 5, %expr2, %expr3:%expr3 + 3, %expr4]`

    That slice needs to be read into a `vector<3x4x5xf32>`. Since the
    permutation map is not full rank, there must be a broadcast along vector
    dimension `1`.

    A notional lowering of vector.transfer_read could generate code resembling:

    ```mlir
    // %expr1, %expr2, %expr3, %expr4 defined before this point
    // alloc a temporary buffer for performing the "gather" of the slice.
    %tmp = memref.alloc() : memref<vector<3x4x5xf32>>
    for %i = 0 to 3 {
      affine.for %j = 0 to 4 {
        affine.for %k = 0 to 5 {
          // Note that this load does not involve %j.
          %a = load %A[%expr1 + %k, %expr2, %expr3 + %i, %expr4] : memref<?x?x?x?xf32>
          // Update the temporary gathered slice with the individual element
          %slice = memref.load %tmp : memref<vector<3x4x5xf32>> -> vector<3x4x5xf32>
          %updated = vector.insert %a, %slice[%i, %j, %k] : f32 into vector<3x4x5xf32>
          memref.store %updated, %tmp : memref<vector<3x4x5xf32>>
    }}}
    // At this point we gathered the elements from the original
    // memref into the desired vector layout, stored in the `%tmp` allocation.
    %vec = memref.load %tmp : memref<vector<3x4x5xf32>> -> vector<3x4x5xf32>
    ```

    On a GPU one could then map `i`, `j`, `k` to blocks and threads. Notice that
    the temporary storage footprint could conceptually be only `3 * 5` values but
    `3 * 4 * 5` values are actually transferred between `%A` and `%tmp`.

    Alternatively, if a notional vector broadcast operation were available, we
    could avoid the loop on `%j` and the lowered code would resemble:

    ```mlir
    // %expr1, %expr2, %expr3, %expr4 defined before this point
    %tmp = memref.alloc() : memref<vector<3x4x5xf32>>
    for %i = 0 to 3 {
      affine.for %k = 0 to 5 {
        %a = load %A[%expr1 + %k, %expr2, %expr3 + %i, %expr4] : memref<?x?x?x?xf32>
        %slice = memref.load %tmp : memref<vector<3x4x5xf32>> -> vector<3x4x5xf32>
        // Here we only store to the first element in dimension one
        %updated = vector.insert %a, %slice[%i, 0, %k] : f32 into vector<3x4x5xf32>
        memref.store %updated, %tmp : memref<vector<3x4x5xf32>>
    }}
    // At this point we gathered the elements from the original
    // memref into the desired vector layout, stored in the `%tmp` allocation.
    // However we haven't replicated them alongside the first dimension, we need
    // to broadcast now.
    %partialVec = load %tmp : memref<vector<3x4x5xf32>> -> vector<3x4x5xf32>
    %vec = broadcast %tmpvec, 1 : vector<3x4x5xf32>
    ```

    where `broadcast` broadcasts from element 0 to all others along the
    specified dimension. This time, the number of loaded element is `3 * 5`
    values.
    An additional `1` broadcast is required. On a GPU this broadcast could be
    implemented using a warp-shuffle if loop `j` were mapped to `threadIdx.x`.

    Syntax
    ```
    operation ::= ssa-id `=` `vector.transfer_read` ssa-use-list
      `{` attribute-entry `} :` memref-type `,` vector-type
    ```

    Example:

    ```mlir
    // Read the slice `%A[%i0, %i1:%i1+256, %i2:%i2+32]` into vector<32x256xf32>
    // and pad with %f0 to handle the boundary case:
    %f0 = arith.constant 0.0f : f32
    affine.for %i0 = 0 to %0 {
      affine.for %i1 = 0 to %1 step 256 {
        affine.for %i2 = 0 to %2 step 32 {
          %v = vector.transfer_read %A[%i0, %i1, %i2], (%f0)
               {permutation_map: (d0, d1, d2) -> (d2, d1)} :
               memref<?x?x?xf32>, vector<32x256xf32>
    }}}

    // or equivalently (rewrite with vector.transpose)
    %f0 = arith.constant 0.0f : f32
    affine.for %i0 = 0 to %0 {
      affine.for %i1 = 0 to %1 step 256 {
        affine.for %i2 = 0 to %2 step 32 {
          %v0 = vector.transfer_read %A[%i0, %i1, %i2], (%f0)
               {permutation_map: (d0, d1, d2) -> (d1, d2)} :
               memref<?x?x?xf32>, vector<256x32xf32>
          %v = vector.transpose %v0, [1, 0] :
              vector<256x32xf32> to vector<32x256f32>
    }}}

    // Read the slice `%A[%i0, %i1]` (i.e. the element `%A[%i0, %i1]`) into
    // vector<128xf32>. The underlying implementation will require a 1-D vector
    // broadcast:
    affine.for %i0 = 0 to %0 {
      affine.for %i1 = 0 to %1 {
        %3 = vector.transfer_read %A[%i0, %i1]
             {permutation_map: (d0, d1) -> (0)} :
             memref<?x?xf32>, vector<128xf32>
      }
    }

    // Read from a memref with vector element type.
    %4 = vector.transfer_read %arg1[%c3, %c3], %vf0
      {permutation_map = (d0, d1)->(d0, d1)}
        : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>

    // Read from a tensor with vector element type.
    %4 = vector.transfer_read %arg1[%c3, %c3], %vf0
      {permutation_map = (d0, d1)->(d0, d1)}
        : tensor<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>

    // Special encoding for 0-d transfer with 0-d tensor/memref, vector shape
    // {1} and permutation_map () -> (0).
    %0 = vector.transfer_read %arg0[], %f0 {permutation_map = affine_map<()->(0)>} :
      tensor<f32>, vector<1xf32>
    ```
  }];

  let builders = [
    /// 1. Builder that sets padding to `padding` or poison if not provided and
    /// an empty mask (variant with attrs).
    OpBuilder<(ins "VectorType":$vectorType,
                   "Value":$source,
                   "ValueRange":$indices,
                   "std::optional<Value>":$padding,
                   "AffineMapAttr":$permutationMapAttr,
                   "ArrayAttr":$inBoundsAttr)>,
    /// 2. Builder that sets padding to `padding` or poison if not provided and
    /// an empty mask (variant without attrs).
    OpBuilder<(ins "VectorType":$vectorType,
                   "Value":$source,
                   "ValueRange":$indices,
                   "std::optional<Value>":$padding,
                   "AffineMap":$permutationMap,
                   CArg<"std::optional<ArrayRef<bool>>", "::std::nullopt">:$inBounds)>,
    /// 3. Builder that sets padding to `padding` or poison if not provided and
    /// permutation map to 'getMinorIdentityMap'.
    OpBuilder<(ins "VectorType":$vectorType,
                   "Value":$source,
                   "ValueRange":$indices,
                   "std::optional<Value>":$padding,
                   CArg<"std::optional<ArrayRef<bool>>", "::std::nullopt">:$inBounds)>
  ];

  let extraClassDeclaration = [{
    // MaskableOpInterface methods.
    bool supportsPassthru() { return true; }

    MutableOperandRange getDpsInitsMutable() {
      return MutableOperandRange(getOperation(), /*start=*/0, /*length=*/0);
    }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

// TODO: Tighten semantics so that masks and inbounds can't be used
// simultaneously within the same transfer op.
def Vector_TransferWriteOp :
  Vector_Op<"transfer_write", [
      DeclareOpInterfaceMethods<VectorTransferOpInterface>,
      DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>,
      DeclareOpInterfaceMethods<MaskableOpInterface>,
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      DeclareOpInterfaceMethods<ConditionallySpeculatable>,
      DeclareOpInterfaceMethods<MemorySpaceCastConsumerOpInterface>,
      AttrSizedOperandSegments,
      DestinationStyleOpInterface
  ]>,
    Arguments<(ins AnyVectorOfAnyRank:$valueToStore,
                   AnyShaped:$base,
                   Variadic<Index>:$indices,
                   AffineMapAttr:$permutation_map,
                   Optional<VectorOfNonZeroRankOf<[I1]>>:$mask,
                   BoolArrayAttr:$in_bounds)>,
    Results<(outs Optional<AnyRankedTensor>:$result)> {

  let summary = "The vector.transfer_write op writes a supervector to memory.";

  let description = [{
    The `vector.transfer_write` op performs a write from a
    [vector](../LangRef.md#vector-type), supplied as its first operand, into a
    slice within a [MemRef](../LangRef.md#memref-type) or a Ranked
    [Tensor](../LangRef.md#tensor-type) of the same base elemental type,
    supplied as its second operand.

    A vector memref/tensor operand must have its vector element type match a
    suffix (shape and element type) of the vector (e.g. memref<3x2x6x4x3xf32>,
    vector<1x1x4x3xf32>). If the operand is a tensor, the operation returns a
    new tensor of the same type.

    The slice is further defined by a full-rank index within the MemRef/Tensor,
    supplied as the operands `[2 .. 2 + rank(memref/tensor))` that defines the
    starting point of the transfer (e.g. `%A[%i0, %i1, %i2, %i3]`).

    The permutation_map [attribute](../LangRef.md#attributes) is an
    [affine-map](Affine.md#affine-maps) which specifies the transposition on the
    slice to match the vector shape. The permutation map may be implicit and
    omitted from parsing and printing if it is the canonical minor identity map
    (i.e. if it does not permute any dimension). In contrast to `transfer_read`,
    write ops cannot have broadcast dimensions.

    The size of the slice is specified by the size of the vector.

    An optional SSA value `mask` may be specified to mask out elements written
    to the MemRef/Tensor. The `mask` type is an `i1` vector with a shape that
    matches how elements are written into the MemRef/Tensor, *after* applying
    any permutation. Elements whose corresponding mask element is `0` are
    masked out.

    For every vector dimension, the boolean array attribute `in_bounds`
    specifies if the transfer is guaranteed to be within the source bounds. If
    set to "false", accesses (including the starting point) may run
    out-of-bounds along the respective vector dimension as the index increases.
    Non-vector dimensions *must* always be in-bounds. The `in_bounds` array
    length has to be equal to the vector rank. This attribute has a default
    value: `false` (i.e. "out-of-bounds"). When skipped in the textual IR, the
    default value is assumed. Similarly, the OP printer will omit this
    attribute when all dimensions are out-of-bounds (i.e. the default value is
    used).

     A `vector.transfer_write` can be lowered to a simple store if all
     dimensions are specified to be within bounds and no `mask` was specified.

    This operation is called 'write' by opposition to 'store' because the
    super-vector granularity is generally not representable with a single
    hardware register. A `vector.transfer_write` is thus a
    mid-level abstraction that supports super-vectorization with non-effecting
    padding for full-tile-only code. It is the responsibility of
    `vector.transfer_write`'s implementation to ensure the memory writes are
    valid. Different lowerings may be pertinent depending on the hardware
    support.

    Example:

    ```mlir
    // write vector<16x32x64xf32> into the slice
    //   `%A[%i0, %i1:%i1+32, %i2:%i2+64, %i3:%i3+16]`:
    for %i0 = 0 to %0 {
      affine.for %i1 = 0 to %1 step 32 {
        affine.for %i2 = 0 to %2 step 64 {
          affine.for %i3 = 0 to %3 step 16 {
            %val = `ssa-value` : vector<16x32x64xf32>
            vector.transfer_write %val, %A[%i0, %i1, %i2, %i3]
              {permutation_map: (d0, d1, d2, d3) -> (d3, d1, d2)} :
              vector<16x32x64xf32>, memref<?x?x?x?xf32>
    }}}}

    // or equivalently (rewrite with vector.transpose)
    for %i0 = 0 to %0 {
      affine.for %i1 = 0 to %1 step 32 {
        affine.for %i2 = 0 to %2 step 64 {
          affine.for %i3 = 0 to %3 step 16 {
            %val = `ssa-value` : vector<16x32x64xf32>
            %valt = vector.transpose %val, [1, 2, 0] :
                  vector<16x32x64xf32> -> vector<32x64x16xf32>
            vector.transfer_write %valt, %A[%i0, %i1, %i2, %i3]
              {permutation_map: (d0, d1, d2, d3) -> (d1, d2, d3)} :
              vector<32x64x16xf32>, memref<?x?x?x?xf32>
    }}}}

    // write to a memref with vector element type.
    vector.transfer_write %4, %arg1[%c3, %c3]
      {permutation_map = (d0, d1)->(d0, d1)}
        : vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>

    // return a tensor where the vector is inserted into the source tensor.
    %5 = vector.transfer_write %4, %arg1[%c3, %c3]
      {permutation_map = (d0, d1)->(d0, d1)}
        : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>

    // Special encoding for 0-d transfer with 0-d tensor/memref, vector shape
    // {1} and permutation_map () -> (0).
    %1 = vector.transfer_write %0, %arg0[] {permutation_map = affine_map<()->(0)>} :
      vector<1xf32>, tensor<f32>
    ```
  }];

  let builders = [
    /// 1. Builder with type inference.
    OpBuilder<(ins "Value":$vector,
                   "Value":$dest,
                   "ValueRange":$indices,
                   "AffineMapAttr":$permutationMapAttr,
                   "Value":$mask,
                   "ArrayAttr":$inBoundsAttr)>,
    /// 2. Builder with type inference that sets an empty mask (variant with attrs).
    OpBuilder<(ins "Value":$vector,
                   "Value":$dest,
                   "ValueRange":$indices,
                   "AffineMapAttr":$permutationMapAttr,
                   "ArrayAttr":$inBoundsAttr)>,
    /// 3. Builder with type inference that sets an empty mask (variant without attrs).
    OpBuilder<(ins "Value":$vector,
                   "Value":$dest,
                   "ValueRange":$indices,
                   "AffineMap":$permutationMap,
                   CArg<"std::optional<ArrayRef<bool>>", "::std::nullopt">:$inBounds)>,
    /// 4. Builder with type inference that sets an empty mask and sets permutation
    /// map to 'getMinorIdentityMap'.
    OpBuilder<(ins "Value":$vector,
                   "Value":$dest,
                   "ValueRange":$indices,
                   CArg<"std::optional<ArrayRef<bool>>", "::std::nullopt">:$inBounds)>,
  ];

  let extraClassDeclaration = [{
    /// This method is added to maintain uniformity with load/store
    ///  ops of other dialects.
    Value getValue() { return getVector(); }

    MutableOperandRange getDpsInitsMutable() { return getBaseMutable(); }
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Vector_LoadOp : Vector_Op<"load", [
    DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>,
    DeclareOpInterfaceMethods<MemorySpaceCastConsumerOpInterface>
  ]> {
  let summary = "reads an n-D slice of memory into an n-D vector";
  let description = [{
    The 'vector.load' operation reads an n-D slice of memory into an n-D
    vector. It takes a 'base' memref, an index for each memref dimension and a
    result vector type as arguments. It returns a value of the result vector
    type. The 'base' memref and indices determine the start memory address from
    which to read. Each index provides an offset for each memref dimension
    based on the element type of the memref. The shape of the result vector
    type determines the shape of the slice read from the start memory address.
    The elements along each dimension of the slice are strided by the memref
    strides. When loading more than 1 element, only unit strides are allowed
    along the most minor memref dimension. These constraints guarantee that
    elements read along the first dimension of the slice are contiguous in
    memory.

    The memref element type can be a scalar or a vector type. If the memref
    element type is a scalar, it should match the element type of the result
    vector. If the memref element type is vector, it should match the result
    vector type.

    Example: 0-D vector load on a scalar memref.
    ```mlir
    %result = vector.load %base[%i, %j] : memref<100x100xf32>, vector<f32>
    ```

    Example: 1-D vector load on a scalar memref.
    ```mlir
    %result = vector.load %base[%i, %j] : memref<100x100xf32>, vector<8xf32>
    ```

    Example: 1-D vector load on a vector memref.
    ```mlir
    %result = vector.load %memref[%i, %j] : memref<200x100xvector<8xf32>>, vector<8xf32>
    ```

    Example:  2-D vector load on a scalar memref.
    ```mlir
    %result = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<4x8xf32>
    ```

    Example:  2-D vector load on a vector memref.
    ```mlir
    %result = vector.load %memref[%i, %j] : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
    ```

    Representation-wise, the 'vector.load' operation permits out-of-bounds
    reads. Support and implementation of out-of-bounds vector loads is
    target-specific. No assumptions should be made on the value of elements
    loaded out of bounds. Not all targets may support out-of-bounds vector
    loads.

    Example:  Potential out-of-bound vector load.
    ```mlir
    %result = vector.load %memref[%index] : memref<?xf32>, vector<8xf32>
    ```

    Example:  Explicit out-of-bound vector load.
    ```mlir
    %result = vector.load %memref[%c0] : memref<7xf32>, vector<8xf32>
    ```

    An optional `alignment` attribute allows to specify the byte alignment of the
    load operation. It must be a positive power of 2. The operation must access
    memory at an address aligned to this boundary. Violating this requirement
    triggers immediate undefined behavior.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
      [MemRead]>:$base,
      Variadic<Index>:$indices,
      DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal,
      OptionalAttr<IntValidAlignment<I64Attr>>: $alignment);

  let builders = [
    OpBuilder<(ins "VectorType":$resultType,
                   "Value":$base,
                   "ValueRange":$indices,
                   CArg<"bool", "false">:$nontemporal,
                   CArg<"llvm::MaybeAlign", "llvm::MaybeAlign()">:$alignment), [{
      return build($_builder, $_state, resultType, base, indices, nontemporal,
                   alignment.has_value() ? $_builder.getI64IntegerAttr(alignment->value()) :
                                    nullptr);
    }]>,
    OpBuilder<(ins "TypeRange":$resultTypes,
                   "Value":$base,
                   "ValueRange":$indices,
                   CArg<"bool", "false">:$nontemporal,
                   CArg<"llvm::MaybeAlign", "llvm::MaybeAlign()">:$alignment), [{
      return build($_builder, $_state, resultTypes, base, indices, nontemporal,
                   alignment.has_value() ? $_builder.getI64IntegerAttr(alignment->value()) :
                                    nullptr);
    }]>
  ];

  let results = (outs AnyVectorOfAnyRank:$result);

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }

    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];

  let hasFolder = 1;
  let hasVerifier = 1;

  let assemblyFormat =
      "$base `[` $indices `]` attr-dict `:` type($base) `,` type($result)";
}

def Vector_StoreOp : Vector_Op<"store", [
    DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>,
    DeclareOpInterfaceMethods<MemorySpaceCastConsumerOpInterface>
  ]> {
  let summary = "writes an n-D vector to an n-D slice of memory";
  let description = [{
    The 'vector.store' operation writes an n-D vector to an n-D slice of memory.
    It takes the vector value to be stored, a 'base' memref and an index for
    each memref dimension. The 'base' memref and indices determine the start
    memory address from which to write. Each index provides an offset for each
    memref dimension based on the element type of the memref. The shape of the
    vector value to store determines the shape of the slice written from the
    start memory address. The elements along each dimension of the slice are
    strided by the memref strides. When storing more than 1 element, only unit
    strides are allowed along the most minor memref dimension. These constraints
    guarantee that elements written along the first dimension of the slice are
    contiguous in memory.

    The memref element type can be a scalar or a vector type. If the memref
    element type is a scalar, it should match the element type of the value
    to store. If the memref element type is vector, it should match the type
    of the value to store.

    Example: 0-D vector store on a scalar memref.
    ```mlir
    vector.store %valueToStore, %memref[%i, %j] : memref<200x100xf32>, vector<f32>
    ```

    Example: 1-D vector store on a scalar memref.
    ```mlir
    vector.store %valueToStore, %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
    ```

    Example: 1-D vector store on a vector memref.
    ```mlir
    vector.store %valueToStore, %memref[%i, %j] : memref<200x100xvector<8xf32>>, vector<8xf32>
    ```

    Example:  2-D vector store on a scalar memref.
    ```mlir
    vector.store %valueToStore, %memref[%i, %j] : memref<200x100xf32>, vector<4x8xf32>
    ```

    Example:  2-D vector store on a vector memref.
    ```mlir
    vector.store %valueToStore, %memref[%i, %j] : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
    ```

    Representation-wise, the 'vector.store' operation permits out-of-bounds
    writes. Support and implementation of out-of-bounds vector stores are
    target-specific. No assumptions should be made on the memory written out of
    bounds. Not all targets may support out-of-bounds vector stores.

    Example:  Potential out-of-bounds vector store.
    ```mlir
    vector.store %valueToStore, %memref[%index] : memref<?xf32>, vector<8xf32>
    ```

    Example:  Explicit out-of-bounds vector store.
    ```mlir
    vector.store %valueToStore, %memref[%c0] : memref<7xf32>, vector<8xf32>
    ```

    An optional `alignment` attribute allows to specify the byte alignment of the
    store operation. It must be a positive power of 2. The operation must access
    memory at an address aligned to this boundary. Violating this requirement
    triggers immediate undefined behavior.
  }];

  let arguments = (ins
      AnyVectorOfAnyRank:$valueToStore,
      Arg<AnyMemRef, "the reference to store to",
      [MemWrite]>:$base,
      Variadic<Index>:$indices,
      DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal,
      OptionalAttr<IntValidAlignment<I64Attr>>: $alignment);

  let builders = [
    OpBuilder<(ins "Value":$valueToStore,
                   "Value":$base,
                   "ValueRange":$indices,
                   CArg<"bool", "false">:$nontemporal,
                   CArg<"llvm::MaybeAlign", "llvm::MaybeAlign()">:$alignment), [{
      return build($_builder, $_state, valueToStore, base, indices, nontemporal,
                   alignment.has_value() ? $_builder.getI64IntegerAttr(alignment->value()) :
                                    nullptr);
    }]>
  ];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }

    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getValueToStore().getType());
    }
  }];

  let hasFolder = 1;
  let hasVerifier = 1;

  let assemblyFormat = "$valueToStore `,` $base `[` $indices `]` attr-dict "
                       "`:` type($base) `,` type($valueToStore)";
}

def Vector_MaskedLoadOp :
  Vector_Op<"maskedload", [DeclareOpInterfaceMethods<MemorySpaceCastConsumerOpInterface>]>,
    Arguments<(ins Arg<AnyMemRef, "", [MemRead]>:$base,
               Variadic<Index>:$indices,
               VectorOfNonZeroRankOf<[I1]>:$mask,
               AnyVectorOfNonZeroRank:$pass_thru,
               OptionalAttr<IntValidAlignment<I64Attr>>: $alignment)>,
    Results<(outs AnyVectorOfNonZeroRank:$result)> {

  let summary = "loads elements from memory into a vector as defined by a mask vector";

  let description = [{
    The masked load reads elements from memory into a vector as defined
    by a base with indices and a mask vector. When the mask is set, the
    element is read from memory. Otherwise, the corresponding element is taken
    from a pass-through vector. Informally the semantics are:
    ```
    result[0] := if mask[0] then base[i + 0] else pass_thru[0]
    result[1] := if mask[1] then base[i + 1] else pass_thru[1]
    etc.
    ```

    If a mask bit is set and the corresponding index is out-of-bounds for the
    given base, the behavior is undefined. If a mask bit is not set, the value
    comes from the pass-through vector regardless of the index, and the index is
    allowed to be out-of-bounds.

    The masked load can be used directly where applicable, or can be used
    during progressively lowering to bring other memory operations closer to
    hardware ISA support for a masked load. The semantics of the operation
    closely correspond to those of the `llvm.masked.load`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-load-intrinsics).

    Examples:

    ```mlir
    %0 = vector.maskedload %base[%i], %mask, %pass_thru
       : memref<?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>

    %1 = vector.maskedload %base[%i, %j], %mask, %pass_thru
       : memref<?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
    ```

    An optional `alignment` attribute allows to specify the byte alignment of the
    load operation. It must be a positive power of 2. The operation must access
    memory at an address aligned to this boundary. Violating this requirement
    triggers immediate undefined behavior.
  }];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getMaskVectorType() {
      return ::llvm::cast<VectorType>(getMask().getType());
    }
    VectorType getPassThruVectorType() {
      return ::llvm::cast<VectorType>(getPassThru().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];
  let assemblyFormat = "$base `[` $indices `]` `,` $mask `,` $pass_thru attr-dict `:` "
    "type($base) `,` type($mask) `,` type($pass_thru) `into` type($result)";
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "VectorType":$resultType,
                   "Value":$base,
                   "ValueRange":$indices,
                   "Value":$mask,
                   "Value":$passthrough,
                   CArg<"llvm::MaybeAlign", "llvm::MaybeAlign()">:$alignment), [{
      return build($_builder, $_state, resultType, base, indices, mask, passthrough,
                   alignment.has_value() ? $_builder.getI64IntegerAttr(alignment->value()) :
                                    nullptr);
    }]>,
    OpBuilder<(ins "TypeRange":$resultTypes,
                   "Value":$base,
                   "ValueRange":$indices,
                   "Value":$mask,
                   "Value":$passthrough,
                   CArg<"llvm::MaybeAlign", "llvm::MaybeAlign()">:$alignment), [{
      return build($_builder, $_state, resultTypes, base, indices, mask, passthrough,
                   alignment.has_value() ? $_builder.getI64IntegerAttr(alignment->value()) :
                                    nullptr);
    }]>
  ];
}

def Vector_MaskedStoreOp :
  Vector_Op<"maskedstore", [DeclareOpInterfaceMethods<MemorySpaceCastConsumerOpInterface>]>,
    Arguments<(ins Arg<AnyMemRef, "", [MemWrite]>:$base,
               Variadic<Index>:$indices,
               VectorOfNonZeroRankOf<[I1]>:$mask,
               AnyVectorOfNonZeroRank:$valueToStore,
               OptionalAttr<IntValidAlignment<I64Attr>>: $alignment)> {

  let summary = "stores elements from a vector into memory as defined by a mask vector";

  let description = [{
    The masked store operation writes elements from a vector into memory
    as defined by a base with indices and a mask vector. When the mask is
    set, the corresponding element from the vector is written to memory. Otherwise,
    no action is taken for the element. Informally the semantics are:
    ```
    if (mask[0]) base[i+0] = value[0]
    if (mask[1]) base[i+1] = value[1]
    etc.
    ```

    If a mask bit is set and the corresponding index is out-of-bounds for the
    given base, the behavior is undefined. If a mask bit is not set, no value
    is stored regardless of the index, and the index is allowed to be
    out-of-bounds.

    The masked store can be used directly where applicable, or can be used
    during progressively lowering to bring other memory operations closer to
    hardware ISA support for a masked store. The semantics of the operation
    closely correspond to those of the `llvm.masked.store`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-store-intrinsics).

    Examples:

    ```mlir
    vector.maskedstore %base[%i], %mask, %value
      : memref<?xf32>, vector<8xi1>, vector<8xf32>

    vector.maskedstore %base[%i, %j], %mask, %value
      : memref<?x?xf32>, vector<16xi1>, vector<16xf32>
    ```

    An optional `alignment` attribute allows to specify the byte alignment of the
    store operation. It must be a positive power of 2. The operation must access
    memory at an address aligned to this boundary. Violating this requirement
    triggers immediate undefined behavior.
  }];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getMaskVectorType() {
      return ::llvm::cast<VectorType>(getMask().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getValueToStore().getType());
    }
  }];
  let assemblyFormat =
      "$base `[` $indices `]` `,` $mask `,` $valueToStore "
      "attr-dict `:` type($base) `,` type($mask) `,` type($valueToStore)";
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$base,
                   "ValueRange":$indices,
                   "Value":$mask,
                   "Value":$valueToStore,
                   CArg<"llvm::MaybeAlign", "llvm::MaybeAlign()">:$alignment), [{
      return build($_builder, $_state, base, indices, mask, valueToStore,
                   alignment.has_value() ? $_builder.getI64IntegerAttr(alignment->value()) :
                                    nullptr);
    }]>
  ];
}

def Vector_GatherOp :
  Vector_Op<"gather", [
    DeclareOpInterfaceMethods<MaskableOpInterface>,
    DeclareOpInterfaceMethods<MemorySpaceCastConsumerOpInterface>,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>
  ]>,
    Arguments<(ins Arg<TensorOrMemRef<[AnyType]>, "", [MemRead]>:$base,
               Variadic<Index>:$offsets,
               VectorOfNonZeroRankOf<[AnyInteger, Index]>:$indices,
               VectorOfNonZeroRankOf<[I1]>:$mask,
               AnyVectorOfNonZeroRank:$pass_thru,
               OptionalAttr<IntValidAlignment<I64Attr>>: $alignment)>,
    Results<(outs AnyVectorOfNonZeroRank:$result)> {

  let summary = [{
    Gathers elements from memory or ranked tensor into a vector as defined by an
    index vector and a mask vector.
  }];

  let description = [{
    The gather operation returns an n-D vector whose elements are either loaded
    from a k-D memref or tensor, or taken from an n-D pass-through vector, depending
    on the values of an n-D mask vector.

    If a mask bit is set, the corresponding result element is taken from `base`
    at an index defined by k indices and n-D `index_vec`. Otherwise, the element
    is taken from the pass-through vector. As an example, suppose that `base` is
    3-D and the result is 2-D:

    ```mlir
    func.func @gather_3D_to_2D(
        %base: memref<?x10x?xf32>, %ofs_0: index, %ofs_1: index, %ofs_2: index,
        %indices: vector<2x3xi32>, %mask: vector<2x3xi1>,
        %fall_thru: vector<2x3xf32>) -> vector<2x3xf32> {
            %result = vector.gather %base[%ofs_0, %ofs_1, %ofs_2]
                                   [%indices], %mask, %fall_thru : [...]
            return %result : vector<2x3xf32>
    }
    ```

    The indexing semantics are then,

    ```
    result[i,j] := if mask[i,j] then base[i0, i1, i2 + indices[i,j]]
                   else pass_thru[i,j]
    ```
    The index into `base` only varies in the innermost ((k-1)-th) dimension.

    If a mask bit is set and the corresponding index is out-of-bounds for the
    given base, the behavior is undefined. If a mask bit is not set, the value
    comes from the pass-through vector regardless of the index, and the index is
    allowed to be out-of-bounds.

    The gather operation can be used directly where applicable, or can be used
    during progressively lowering to bring other memory operations closer to
    hardware ISA support for a gather.

    An optional `alignment` attribute allows to specify the byte alignment of the
    gather operation. It must be a positive power of 2. The operation must access
    memory at an address aligned to this boundary. Violating this requirement
    triggers immediate undefined behavior.

    Examples:

    ```mlir
    // 1-D memref gathered to 2-D vector.
    %0 = vector.gather %base[%c0][%v], %mask, %pass_thru
       : memref<?xf32>, vector<2x16xi32>, vector<2x16xi1>, vector<2x16xf32> into vector<2x16xf32>

    // 2-D memref gathered to 1-D vector.
    %1 = vector.gather %base[%i, %j][%v], %mask, %pass_thru
       : memref<16x16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
    ```
  }];

  let extraClassDeclaration = [{
    ShapedType getBaseType() { return getBase().getType(); }
    VectorType getIndexVectorType() { return getIndices().getType(); }
    VectorType getMaskVectorType() { return getMask().getType(); }
    VectorType getPassThruVectorType() { return getPassThru().getType(); }
    VectorType getVectorType() { return getResult().getType(); }
  }];

  let assemblyFormat =
    "$base `[` $offsets `]` `[` $indices `]` `,` "
    "$mask `,` $pass_thru attr-dict `:` type($base) `,` "
    "type($indices)  `,` type($mask) `,` type($pass_thru) "
    "`into` type($result)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "VectorType":$resultType,
                   "Value":$base,
                   "ValueRange":$indices,
                   "Value":$index_vec,
                   "Value":$mask,
                   "Value":$passthrough,
                   CArg<"llvm::MaybeAlign", "llvm::MaybeAlign()">:$alignment), [{
      return build($_builder, $_state, resultType, base, indices, index_vec, mask, passthrough,
                   alignment.has_value() ? $_builder.getI64IntegerAttr(alignment->value()) :
                                    nullptr);
    }]>
  ];
}

def Vector_ScatterOp :
  Vector_Op<"scatter", [DeclareOpInterfaceMethods<MemorySpaceCastConsumerOpInterface>]>,
    Arguments<(ins Arg<AnyMemRef, "", [MemWrite]>:$base,
               Variadic<Index>:$offsets,
               VectorOfNonZeroRankOf<[AnyInteger, Index]>:$indices,
               VectorOfNonZeroRankOf<[I1]>:$mask,
               AnyVectorOfNonZeroRank:$valueToStore,
               OptionalAttr<IntValidAlignment<I64Attr>>: $alignment)> {

  let summary = [{
    scatters elements from a vector into memory as defined by an index vector
    and a mask vector
  }];

  let description = [{
    The scatter operation stores elements from a n-D vector into memory as
    defined by a base with indices and an additional n-D index vector, but
    only if the corresponding bit in a n-D mask vector is set. Otherwise, no
    action is taken for that element. Informally the semantics are:
    ```
    if (mask[0]) base[index[0]] = value[0]
    if (mask[1]) base[index[1]] = value[1]
    etc.
    ```

    If a mask bit is set and the corresponding index is out-of-bounds for the
    given base, the behavior is undefined. If a mask bit is not set, no value
    is stored regardless of the index, and the index is allowed to be
    out-of-bounds.

    If the index vector contains two or more duplicate indices, the behavior is
    undefined. Underlying implementation may enforce strict sequential
    semantics.
    TODO: always enforce strict sequential semantics?

    The scatter operation can be used directly where applicable, or can be used
    during progressively lowering to bring other memory operations closer to
    hardware ISA support for a scatter. The semantics of the operation closely
    correspond to those of the `llvm.masked.scatter`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-scatter-intrinsics).

    An optional `alignment` attribute allows to specify the byte alignment of the
    scatter operation. It must be a positive power of 2. The operation must access
    memory at an address aligned to this boundary. Violating this requirement
    triggers immediate undefined behavior.

    Examples:

    ```mlir
    vector.scatter %base[%c0][%v], %mask, %value
        : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>

    vector.scatter %base[%i, %j][%v], %mask, %value
        : memref<16x16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
    ```
  }];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() { return getBase().getType(); }
    VectorType getIndexVectorType() { return getIndices().getType(); }
    VectorType getMaskVectorType() { return getMask().getType(); }
    VectorType getVectorType() { return getValueToStore().getType(); }
  }];

  let assemblyFormat =
      "$base `[` $offsets `]` `[` $indices `]` `,` "
      "$mask `,` $valueToStore attr-dict `:` type($base) `,` "
      "type($indices)  `,` type($mask) `,` type($valueToStore)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$base,
                   "ValueRange":$indices,
                   "Value":$index_vec,
                   "Value":$mask,
                   "Value":$valueToStore,
                   CArg<"llvm::MaybeAlign", "llvm::MaybeAlign()">: $alignment), [{
      return build($_builder, $_state, base, indices, index_vec, mask, valueToStore,
                   alignment.has_value() ? $_builder.getI64IntegerAttr(alignment->value()) :
                                    nullptr);
    }]>
  ];
}

def Vector_ExpandLoadOp :
  Vector_Op<"expandload", [DeclareOpInterfaceMethods<MemorySpaceCastConsumerOpInterface>]>,
    Arguments<(ins Arg<AnyMemRef, "", [MemRead]>:$base,
               Variadic<Index>:$indices,
               FixedVectorOfNonZeroRankOf<[I1]>:$mask,
               AnyVectorOfNonZeroRank:$pass_thru,
               OptionalAttr<IntValidAlignment<I64Attr>>: $alignment)>,
    Results<(outs AnyVectorOfNonZeroRank:$result)> {

  let summary = "reads elements from memory and spreads them into a vector as defined by a mask";

  let description = [{
    The expand load reads elements from memory into a vector as defined by a
    base with indices and a mask vector. Expansion only applies to the innermost
    dimension. When the mask is set, the next element is read from memory.
    Otherwise, the corresponding element is taken from a pass-through vector.
    Informally the semantics are:

    ```
    index = i
    result[0] := if mask[0] then base[index++] else pass_thru[0]
    result[1] := if mask[1] then base[index++] else pass_thru[1]
    etc.
    ```

    Note that the index increment is done conditionally.

    If a mask bit is set and the corresponding index is out-of-bounds for the
    given base, the behavior is undefined. If a mask bit is not set, the value
    comes from the pass-through vector regardless of the index, and the index is
    allowed to be out-of-bounds.

    The expand load can be used directly where applicable, or can be used
    during progressively lowering to bring other memory operations closer to
    hardware ISA support for an expand. The semantics of the operation closely
    correspond to those of the `llvm.masked.expandload`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-expandload-intrinsics).

    An optional `alignment` attribute allows to specify the byte alignment of the
    load operation. It must be a positive power of 2. The operation must access
    memory at an address aligned to this boundary. Violating this requirement
    triggers immediate undefined behavior.

    Note, at the moment this Op is only available for fixed-width vectors.

    Examples:

    ```mlir
    %0 = vector.expandload %base[%i], %mask, %pass_thru
       : memref<?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>

    %1 = vector.expandload %base[%i, %j], %mask, %pass_thru
       : memref<?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
    ```
  }];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getMaskVectorType() {
      return ::llvm::cast<VectorType>(getMask().getType());
    }
    VectorType getPassThruVectorType() {
      return ::llvm::cast<VectorType>(getPassThru().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];
  let assemblyFormat = "$base `[` $indices `]` `,` $mask `,` $pass_thru attr-dict `:` "
    "type($base) `,` type($mask) `,` type($pass_thru) `into` type($result)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "VectorType":$resultType,
                   "Value":$base,
                   "ValueRange":$indices,
                   "Value":$mask,
                   "Value":$passthrough,
                   CArg<"llvm::MaybeAlign", "llvm::MaybeAlign()">:$alignment), [{
      return build($_builder, $_state, resultType, base, indices, mask, passthrough,
                   alignment.has_value() ? $_builder.getI64IntegerAttr(alignment->value()) :
                                    nullptr);
    }]>
  ];
}

def Vector_CompressStoreOp :
  Vector_Op<"compressstore", [DeclareOpInterfaceMethods<MemorySpaceCastConsumerOpInterface>]>,
    Arguments<(ins Arg<AnyMemRef, "", [MemWrite]>:$base,
               Variadic<Index>:$indices,
               FixedVectorOfNonZeroRankOf<[I1]>:$mask,
               AnyVectorOfNonZeroRank:$valueToStore,
               OptionalAttr<IntValidAlignment<I64Attr>>: $alignment)> {

  let summary = "writes elements selectively from a vector as defined by a mask";

  let description = [{
    The compress store operation writes elements from a vector into memory as
    defined by a base with indices and a mask vector. Compression only applies
    to the innermost dimension. When the mask is set, the corresponding element
    from the vector is written next to memory.  Otherwise, no action is taken
    for the element. Informally the semantics are:

    ```
    index = i
    if (mask[0]) base[index++] = value[0]
    if (mask[1]) base[index++] = value[1]
    etc.
    ```

    Note that the index increment is done conditionally.

    If a mask bit is set and the corresponding index is out-of-bounds for the
    given base, the behavior is undefined. If a mask bit is not set, no value
    is stored regardless of the index, and the index is allowed to be
    out-of-bounds.

    The compress store can be used directly where applicable, or can be used
    during progressively lowering to bring other memory operations closer to
    hardware ISA support for a compress. The semantics of the operation closely
    correspond to those of the `llvm.masked.compressstore`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-compressstore-intrinsics).

    An optional `alignment` attribute allows to specify the byte alignment of the
    store operation. It must be a positive power of 2. The operation must access
    memory at an address aligned to this boundary. Violating this requirement
    triggers immediate undefined behavior.

    Note, at the moment this Op is only available for fixed-width vectors.

    Examples:

    ```mlir
    vector.compressstore %base[%i], %mask, %value
      : memref<?xf32>, vector<8xi1>, vector<8xf32>

    vector.compressstore %base[%i, %j], %mask, %value
      : memref<?x?xf32>, vector<16xi1>, vector<16xf32>
    ```
  }];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getMaskVectorType() {
      return ::llvm::cast<VectorType>(getMask().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getValueToStore().getType());
    }
  }];
  let assemblyFormat =
      "$base `[` $indices `]` `,` $mask `,` $valueToStore attr-dict `:` "
      "type($base) `,` type($mask) `,` type($valueToStore)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "Value":$base,
                   "ValueRange":$indices,
                   "Value":$mask,
                   "Value":$valueToStore,
                   CArg<"llvm::MaybeAlign", "llvm::MaybeAlign()">:$alignment), [{
      return build($_builder, $_state, base, indices, valueToStore, mask,
                   alignment.has_value() ? $_builder.getI64IntegerAttr(alignment->value()) :
                                    nullptr);
    }]>
  ];
}

def Vector_ShapeCastOp :
  Vector_Op<"shape_cast", [Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]>,
    Arguments<(ins AnyVectorOfAnyRank:$source)>,
    Results<(outs AnyVectorOfAnyRank:$result)> {
  let summary = "shape_cast casts between vector shapes";
  let description = [{
    Casts to a vector with the same number of elements, element type, and
    number of scalable dimensions.

    It is currently assumed that this operation does not require moving data,
    and that it will be folded away before lowering vector operations.

    There is an exception to the folding expectation when targeting
    llvm.intr.matrix operations. We need a type conversion back and forth from a
    2-D MLIR vector to a 1-D flattened LLVM vector.shape_cast lowering to LLVM
    is supported in that particular case, for now.

    Examples:

    ```mlir
    %1 = vector.shape_cast %0 : vector<4x3xf32> to vector<3x2x2xf32>

    // with 2 scalable dimensions (number of which must be preserved).
    %3 = vector.shape_cast %2 : vector<[2]x3x[4]xi8> to vector<3x[1]x[8]xi8>
    ```
  }];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getSource().getType());
    }
    VectorType getResultVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Vector_BitCastOp :
  Vector_Op<"bitcast", [Pure, AllRanksMatch<["source", "result"]>]>,
    Arguments<(ins AnyVectorOfAnyRank:$source)>,
    Results<(outs AnyVectorOfAnyRank:$result)>{
  let summary = "bitcast casts between vectors";
  let description = [{
    The bitcast operation casts between vectors of the same rank, the minor 1-D
    vector size is casted to a vector with a different element type but same
    bitwidth. In case of 0-D vectors, the bitwidth of element types must be
    equal.

    Example:

    ```mlir
    // Example casting to a smaller element type.
    %1 = vector.bitcast %0 : vector<5x1x4x3xf32> to vector<5x1x4x6xi16>

    // Example casting to a bigger element type.
    %3 = vector.bitcast %2 : vector<10x12x8xi8> to vector<10x12x2xi32>

    // Example casting to an element type of the same size.
    %5 = vector.bitcast %4 : vector<5x1x4x3xf32> to vector<5x1x4x3xi32>

    // Example casting of 0-D vectors.
    %7 = vector.bitcast %6 : vector<f32> to vector<i32>
    ```
  }];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getSource().getType());
    }
    VectorType getResultVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Vector_TypeCastOp :
  Vector_Op<"type_cast", [Pure, ViewLikeOpInterface]>,
    Arguments<(ins StaticShapeMemRefOf<[AnyType]>:$memref)>,
    Results<(outs AnyMemRef:$result)> {
  let summary = "type_cast op converts a scalar memref to a vector memref";
  let description = [{
    Performs a conversion from a memref with scalar element to a memref with a
    *single* vector element, copying the shape of the memref to the vector. This
    is the minimal viable operation that is required to makeke
    super-vectorization operational. It can be seen as a special case of the
    `view` operation but scoped in the super-vectorization context.

    Example:

    ```mlir
    %A  = memref.alloc() : memref<5x4x3xf32>
    %VA = vector.type_cast %A : memref<5x4x3xf32> to memref<vector<5x4x3xf32>>
    ```
  }];

  /// Build the canonical memRefType with a single vector.
  /// E.g. memref<4 x 5 x vector<6 x f32>> -> memref<vector<4 x 5 x 6 x f32>>.
  let builders = [OpBuilder<(ins "Value":$source)>];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getMemref().getType());
    }
    MemRefType getResultMemRefType() {
      return ::llvm::cast<MemRefType>(getResult().getType());
    }
    // Implement ViewLikeOpInterface.
    Value getViewSource() { return getMemref(); }
  }];

  let assemblyFormat = [{
    $memref attr-dict `:` type($memref) `to` type($result)
  }];
  let hasVerifier = 1;
}

def Vector_ConstantMaskOp :
  Vector_Op<"constant_mask", [Pure]>,
    Arguments<(ins DenseI64ArrayAttr:$mask_dim_sizes)>,
    Results<(outs VectorOfAnyRankOf<[I1]>)> {
  let summary = "creates a constant vector mask";
  let description = [{
    Creates and returns a vector mask where elements of the result vector
    are set to '0' or '1', based on whether the element indices are contained
    within a hyper-rectangular region specified by the 'mask_dim_sizes'
    array attribute argument. Each element of the 'mask_dim_sizes' array,
    specifies an exclusive upper bound [0, mask-dim-size-element-value)
    for a unique dimension in the vector result. The conjunction of the ranges
    define a hyper-rectangular region within which elements values are set to 1
    (otherwise element values are set to 0). Each value of 'mask_dim_sizes' must
    be non-negative and not greater than the size of the corresponding vector
    dimension (as opposed to vector.create_mask which allows this). Sizes that
    correspond to scalable dimensions are implicitly multiplied by vscale,
    though currently only zero (none set) or the size of the dim/vscale
    (all set) are supported.

    Example:

    ```mlir
    // create a constant vector mask of size 4x3xi1 with elements in range
    // 0 <= row <= 2 and 0 <= col <= 1 are set to 1 (others to 0).
    %1 = vector.constant_mask [3, 2] : vector<4x3xi1>

    print %1
                  columns
                0    1    2
              |------------
            0 | 1    1    0
      rows  1 | 1    1    0
            2 | 1    1    0
            3 | 0    0    0
    ```
  }];

  let builders = [
    // Build with mixed static/dynamic operands.
    OpBuilder<(ins "VectorType":$type, "ConstantMaskKind":$kind)>
  ];

  let extraClassDeclaration = [{
    /// Return the result type of this op.
    VectorType getVectorType() {
      return cast<VectorType>(getOperation()->getResultTypes()[0]);
    }

    /// Return whether the mask is a uniform vector of `1`s.
    bool isAllOnesMask();
  }];

  let assemblyFormat = "$mask_dim_sizes attr-dict `:` type(results)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Vector_CreateMaskOp :
  Vector_Op<"create_mask", [Pure]>,
    Arguments<(ins Variadic<Index>:$operands)>,
    Results<(outs VectorOfAnyRankOf<[I1]>)> {
  let summary = "creates a vector mask";
  let description = [{
    Creates and returns a vector mask where elements of the result vector
    are set to '0' or '1', based on whether the element indices are contained
    within a hyper-rectangular region specified by the operands. Specifically,
    each operand specifies a range [0, operand-value) for a unique dimension in
    the vector result. The conjunction of the operand ranges define a
    hyper-rectangular region within which elements values are set to 1
    (otherwise element values are set to 0). If operand-value is negative, it is
    treated as if it were zero, and if it is greater than the corresponding
    dimension size, it is treated as if it were equal to the dimension size.

    Example:

    ```mlir
    // create a vector mask of size 4x3xi1 where elements in range
    // 0 <= row <= 2 and 0 <= col <= 1 are set to 1 (others to 0).
    %1 = vector.create_mask %c3, %c2 : vector<4x3xi1>

    print %1
                  columns
                0    1    2
              |------------
            0 | 1    1    0
      rows  1 | 1    1    0
            2 | 1    1    0
            3 | 0    0    0
    ```
  }];

  let builders = [
    // Build with mixed static/dynamic operands.
    OpBuilder<(ins "VectorType":$type, "ArrayRef<OpFoldResult>":$mixedOperands)>
  ];

  let extraClassDeclaration = [{
    /// Return the result type of this op.
    VectorType getVectorType() {
      return cast<VectorType>(getOperation()->getResultTypes()[0]);
    }
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  let assemblyFormat = "$operands attr-dict `:` type(results)";
}

def Vector_MaskOp : Vector_Op<"mask", [
  SingleBlockImplicitTerminator<"vector::YieldOp">,
  DeclareOpInterfaceMethods<MaskingOpInterface>,
  RecursiveMemoryEffects, NoRegionArguments
]> {
  let summary = "Predicates a maskable vector operation";
  let description = [{
    The `vector.mask` is a `MaskingOpInterface` operation that predicates the
    execution of another operation. It takes an `i1` vector mask and an
    optional passthru vector as arguments.

    A implicitly `vector.yield`-terminated region encloses the operation to be
    masked. Values used within the region are captured from above. Only one
    *maskable* operation can be masked with a `vector.mask` operation at a time.
    An operation is *maskable* if it implements the `MaskableOpInterface`. The
    terminator yields all results from the maskable operation to the result of
    this operation. No other values are allowed to be yielded.

    An empty `vector.mask` operation is currently legal to enable optimizations
    across the `vector.mask` region. However, this might change in the future
    once vector transformations gain better support for `vector.mask`.
    TODO: Consider making empty `vector.mask` illegal.

    The vector mask argument holds a bit for each vector lane and determines
    which vector lanes should execute the maskable operation and which ones
    should not. The `vector.mask` operation returns the value produced by the
    masked execution of the nested operation, if any. The masked-off lanes in
    the result vector are taken from the corresponding lanes of the pass-thru
    argument, if provided, or left unmodified, otherwise. At this point, 0-D
    vectors are not supported by `vector.mask`. They may be supported in the
    future.

    The `vector.mask` operation does not prescribe how a maskable operation
    should be masked or how a masked operation should be lowered. Masking
    constraints and some semantic details are provided by each maskable
    operation through the `MaskableOpInterface`. Lowering of masked operations
    is implementation defined. For instance, scalarizing the masked operation
    or executing the operation for the masked-off lanes are valid lowerings as
    long as the execution of masked-off lanes does not change the observable
    behavior of the program.

    Examples:

    ```
      %0 = vector.mask %mask { vector.reduction <add>, %a : vector<8xi32> into i32 } : vector<8xi1> -> i32
    ```

    ```
      %0 = vector.mask %mask, %passthru { arith.divsi %a, %b : vector<8xi32> } : vector<8xi1> -> vector<8xi32>
    ```

    ```
      vector.mask %mask { vector.transfer_write %val, %t0[%idx] : vector<16xf32>, memref<?xf32> } : vector<16xi1>
    ```

    ```
      vector.mask %mask { vector.transfer_write %val, %t0[%idx] : vector<16xf32>, tensor<?xf32> } : vector<16xi1> -> tensor<?xf32>
    ```
  }];

  // TODO: Support multiple passthru values.
  let arguments = (ins VectorOfNonZeroRankOf<[I1]>:$mask,
                   Optional<AnyType>:$passthru);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$maskRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$mask, "Operation *":$maskableOp,
                   CArg<"function_ref<void(OpBuilder &, Operation *)>">:$maskRegion)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$mask, "Operation *":$maskableOp,
                   CArg<"function_ref<void(OpBuilder &, Operation *)>">:$maskRegion)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$mask, "Value":$passthru,
                   "Operation *":$maskableOp,
                   CArg<"function_ref<void(OpBuilder &, Operation *)>">:$maskRegion)>
  ];

  let extraClassDeclaration = [{
    Block *getMaskBlock() { return &getMaskRegion().front(); }

    /// Returns true if mask op is not masking any operation.
    bool isEmpty() {
      Block *block = getMaskBlock();
      if (block->getOperations().size() > 1)
        return false;
      return true;
    }

    static void ensureTerminator(Region &region, Builder &builder,
                                 Location loc);
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Vector_TransposeOp :
  Vector_Op<"transpose", [Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>,
    PredOpTrait<"operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]> {
  let summary = "vector transpose operation";
  let description = [{
    Takes a n-D vector and returns the transposed n-D vector defined by
    the permutation of ranks in the n-sized integer array attribute (in case
    of 0-D vectors the array attribute must be empty).

    In the operation

    ```mlir
    %1 = vector.transpose %0, [i_1, .., i_n]
      : vector<d_1 x .. x d_n x f32>
      to vector<d_trans[0] x .. x d_trans[n-1] x f32>
    ```

    the `permutation` array [i_1, .., i_n] must be a permutation of [0, .., n-1].

    Example:

    ```mlir
    %1 = vector.transpose %0, [1, 0] : vector<2x3xf32> to vector<3x2xf32>

     [ [a, b, c],       [ [a, d],
       [d, e, f] ]  ->    [b, e],
                          [c, f] ]
    ```
  }];

  let arguments = (ins AnyVectorOfAnyRank:$vector,
                       DenseI64ArrayAttr:$permutation);
  let results = (outs AnyVectorOfAnyRank:$result);

  let builders = [
    OpBuilder<(ins "Value":$vector, "ArrayRef<int64_t>":$permutation)>
  ];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getVector().getType());
    }
    VectorType getResultVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];
  let assemblyFormat = [{
    $vector `,` $permutation attr-dict `:` type($vector) `to` type($result)
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Vector_PrintOp :
  Vector_Op<"print", [
    MemoryEffects<[MemWrite]>,
    PredOpTrait<
      "`source` or `punctuation` are not set when printing strings",
      CPred<"!getStringLiteral() || (!getSource() && getPunctuation() == PrintPunctuation::NewLine)">
    >,
  ]>,
  Arguments<(ins Optional<Type<Or<[
    AnyVectorOfAnyRank.predicate,
    AnyInteger.predicate, Index.predicate, AnyFloat.predicate
  ]>>>:$source, DefaultValuedAttr<Vector_PrintPunctuation,
                      "::mlir::vector::PrintPunctuation::NewLine">:$punctuation,
                OptionalAttr<Builtin_StringAttr>:$stringLiteral)
  > {
  let summary = "print operation (for testing and debugging)";
  let description = [{
    Prints the source vector (or scalar) to stdout in a human-readable format
    (for testing and debugging). No return value.

    Example:

    ```mlir
    %v = arith.constant dense<0.0> : vector<4xf32>
    vector.print %v : vector<4xf32>
    ```

    When lowered to LLVM, the vector print is decomposed into elementary
    printing method calls that at runtime will yield:

    ```
    ( 0.0, 0.0, 0.0, 0.0 )
    ```

    This is printed to stdout via a small runtime support library, which only
    needs to provide a few printing methods (single value for all data
    types, opening/closing bracket, comma, newline).

    By default `vector.print` adds a newline after the vector, but this can be
    controlled by the `punctuation` attribute. For example, to print a comma
    after instead do:

    ```mlir
    vector.print %v : vector<4xf32> punctuation <comma>
    ```

    Note that it is possible to use the punctuation attribute alone. The
    following will print a single newline:

    ```mlir
    vector.print punctuation <newline>
    ```

    Additionally, to aid with debugging and testing `vector.print` can also
    print constant strings:

    ```mlir
    vector.print str "Hello, World!"
    ```
  }];
  let extraClassDeclaration = [{
    Type getPrintType() {
      return getSource().getType();
    }
  }];
  let builders = [
    OpBuilder<(ins "PrintPunctuation":$punctuation), [{
      build($_builder, $_state, {}, punctuation, {});
    }]>,
    OpBuilder<(ins "::mlir::Value":$source), [{
      build($_builder, $_state, source, PrintPunctuation::NewLine);
    }]>,
    OpBuilder<(ins "::mlir::Value":$source, "PrintPunctuation":$punctuation), [{
      build($_builder, $_state, source, punctuation, {});
    }]>,
    OpBuilder<(ins "::llvm::StringRef":$string), [{
      build($_builder, $_state, {}, PrintPunctuation::NewLine, $_builder.getStringAttr(string));
    }]>,
  ];

  let assemblyFormat = [{
      ($source^ `:` type($source))?
        oilist(
            `str` $stringLiteral
          | `punctuation` $punctuation)
        attr-dict
    }];
}


//===----------------------------------------------------------------------===//
// VectorScaleOp
//===----------------------------------------------------------------------===//

// TODO: In the future, we might want to have scalable vectors with different
//       scales for different dimensions. E.g.: vector<[16]x[16]xf32>, in
//       which case we might need to add an index to 'vscale' to select one
//       of them. In order to support GPUs, we might also want to differentiate
//       between a 'global' scale, a scale that's fixed throughout the
//       execution, and a 'local' scale that is fixed but might vary with each
//       call to the function. For that, it might be useful to have a
//       'vector.scale.global' and a 'vector.scale.local' operation.
def VectorScaleOp : Vector_Op<"vscale",
  [Pure, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]
> {
  let summary = "Load vector scale size";
  let description = [{
    The `vscale` op returns the scale of the scalable vectors, a positive
    integer value that is constant at runtime but unknown at compile-time.
    The scale of the vector indicates the multiplicity of the vectors and
    vector operations. For example, a `vector<[4]xi32>` is equivalent to
    `vscale` consecutive `vector<4xi32>`; and an operation on a
    `vector<[4]xi32>` is equivalent to performing that operation `vscale`
    times, once on each `<4xi32>` segment of the scalable vector. The `vscale`
    op can be used to calculate the step in vector-length agnostic (VLA) loops.
    Right now we only support one contiguous set of scalable dimensions, all of
    them grouped and scaled with the value returned by 'vscale'.
  }];
  let results = (outs Index:$res);
  let assemblyFormat = "attr-dict";

  let extraClassDefinition = [{
    void $cppClass::getAsmResultNames(
        ::llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
      setNameFn(getResult(), "vscale");
    }
  }];
}

//===----------------------------------------------------------------------===//
// VectorScanOp
//===----------------------------------------------------------------------===//

def Vector_ScanOp :
  Vector_Op<"scan", [Pure,
    AllTypesMatch<["source", "dest"]>,
    AllTypesMatch<["initial_value", "accumulated_value"]> ]>,
    Arguments<(ins Vector_CombiningKindAttr:$kind,
                   AnyVectorOfNonZeroRank:$source,
                   AnyVectorOfAnyRank:$initial_value,
                   I64Attr:$reduction_dim,
                   BoolAttr:$inclusive)>,
    Results<(outs AnyVectorOfNonZeroRank:$dest,
                  AnyVectorOfAnyRank:$accumulated_value)> {
  let summary = "Scan operation";
  let description = [{
    Performs an inclusive/exclusive scan on an n-D vector along a single
    dimension returning an n-D result vector using the given
    operation (`add`/`mul`/`minsi`/`minui`/`maxsi`/`maxui`/`and`/`or`/`xor` for
    integers, and `add`/`mul`/`minnumf`/`maxnumf`/`minimumf`/`maximumf` for
    floats), and a specified value for the initial value. The operator returns
    the result of scan as well as the result of the last reduction in the scan.

    Example:

    ```mlir
    %1:2 = vector.scan <add>, %0, %acc {inclusive = false, reduction_dim = 1 : i64} :
      vector<4x8x16x32xf32>, vector<4x16x32xf32>
    ```
  }];

  let extraClassDeclaration = [{
    VectorType getSourceType() {
      return ::llvm::cast<VectorType>(getSource().getType());
    }
    VectorType getDestType() {
      return ::llvm::cast<VectorType>(getDest().getType());
    }
    VectorType getAccumulatorType() {
      return ::llvm::cast<VectorType>(getAccumulatedValue().getType());
    }
    VectorType getInitialValueType() {
      return ::llvm::cast<VectorType>(getInitialValue().getType());
    }
  }];
  let assemblyFormat =
    "$kind `,` $source `,` $initial_value attr-dict `:` "
    "type($source) `,` type($initial_value) ";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VectorStepOp
//===----------------------------------------------------------------------===//

def Vector_StepOp : Vector_Op<"step", [
    Pure,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "A linear sequence of values from 0 to N";
  let description = [{
    A `step` operation produces an index vector, i.e. a 1-D vector of values of
    index type that represents a linear sequence from 0 to N-1, where N is the
    number of elements in the `result` vector.

    Supports fixed-width and scalable vectors.

    Examples:

    ```mlir
    %0 = vector.step : vector<4xindex> ; [0, 1, 2, 3]
    %1 = vector.step : vector<[4]xindex> ; [0, 1, .., <vscale * 4 - 1>]
    ```
  }];
  let results = (outs VectorOfRankAndType<[1], [Index]>:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasCanonicalizer = 1;
}

def Vector_YieldOp : Vector_Op<"yield", [
    Pure, ReturnLike, Terminator]> {
  let summary = "Terminates and yields values from vector regions.";
  let description = [{
    "vector.yield" yields an SSA value from the Vector dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If "vector.yield" has any operands, the operands must correspond to the
    parent operation's results.
    If the parent operation defines no value the vector.yield may be omitted
    when printing the region.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}


#endif // MLIR_DIALECT_VECTOR_IR_VECTOR_OPS
