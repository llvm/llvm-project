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

#ifndef VECTOR_OPS
#define VECTOR_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def Vector_Dialect : Dialect {
  let name = "vector";
  let cppNamespace = "::mlir::vector";
  let hasConstantMaterializer = 1;
}

// Base class for Vector dialect ops.
class Vector_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Vector_Dialect, mnemonic, traits> {
  // For every vector op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

// The "kind" of combining function for contractions and reductions.
def COMBINING_KIND_ADD : BitEnumAttrCase<"ADD", 0x1, "add">;
def COMBINING_KIND_MUL : BitEnumAttrCase<"MUL", 0x2, "mul">;
def COMBINING_KIND_MINUI : BitEnumAttrCase<"MINUI", 0x4, "minui">;
def COMBINING_KIND_MINSI : BitEnumAttrCase<"MINSI", 0x8, "minsi">;
def COMBINING_KIND_MINF : BitEnumAttrCase<"MINF", 0x10, "minf">;
def COMBINING_KIND_MAXUI : BitEnumAttrCase<"MAXUI", 0x20, "maxui">;
def COMBINING_KIND_MAXSI : BitEnumAttrCase<"MAXSI", 0x40, "maxsi">;
def COMBINING_KIND_MAXF : BitEnumAttrCase<"MAXF", 0x80, "maxf">;
def COMBINING_KIND_AND : BitEnumAttrCase<"AND", 0x100, "and">;
def COMBINING_KIND_OR  : BitEnumAttrCase<"OR", 0x200, "or">;
def COMBINING_KIND_XOR : BitEnumAttrCase<"XOR", 0x400, "xor">;

def CombiningKind : BitEnumAttr<
    "CombiningKind",
    "Kind of combining function for contractions and reductions",
    [COMBINING_KIND_ADD, COMBINING_KIND_MUL, COMBINING_KIND_MINUI,
     COMBINING_KIND_MINSI, COMBINING_KIND_MINF, COMBINING_KIND_MAXUI,
     COMBINING_KIND_MAXSI, COMBINING_KIND_MAXF, COMBINING_KIND_AND,
     COMBINING_KIND_OR, COMBINING_KIND_XOR]> {
  let cppNamespace = "::mlir::vector";
  let genSpecializedAttr = 0;
}

def Vector_CombiningKindAttr : DialectAttr<
    Vector_Dialect,
    CPred<"$_self.isa<::mlir::vector::CombiningKindAttr>()">,
    "Kind of combining function for contractions and reductions"> {
  let storageType = "::mlir::vector::CombiningKindAttr";
  let returnType = "::mlir::vector::CombiningKind";
  let convertFromStorage = "$_self.getKind()";
  let constBuilderCall =
          "::mlir::vector::CombiningKindAttr::get($0, $_builder.getContext())";
}

// TODO: Add an attribute to specify a different algebra with operators other
// than the current set: {*, +}.
def Vector_ContractionOp :
  Vector_Op<"contract", [
      NoSideEffect,
      PredOpTrait<"lhs and rhs have same element type", TCopVTEtIsSameAs<0, 1>>,
      PredOpTrait<"third operand acc and result have same element type",
                  TCresVTEtIsSameAsOpBase<0, 2>>,
      DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>
    ]>,
    Arguments<(ins AnyVector:$lhs, AnyVector:$rhs, AnyType:$acc,
               Variadic<VectorOf<[I1]>>:$masks,
               AffineMapArrayAttr:$indexing_maps, ArrayAttr:$iterator_types,
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

    Optional vector mask arguments (produced by CreateMaskOp or ConstantMaskOp)
    specify the dynamic dimension sizes of valid data within the lhs/rhs vector
    arguments.

    An iterator type attribute list must be specified, where each element of
    the list represents an iterator with one of the following types:

    *) "reduction": reduction dimensions are present in the lhs and rhs
                    arguments but not in the output (and accumulator
                    argument). These are the dimensions along which the vector
                    contraction op computes the sum of products, and
                    contracting dimension pair dimension sizes must match
                    between lhs/rhs.
    *) "parallel": Batch dimensions are iterator type "parallel", and
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
    attribute can take the values add/mul/min/max for int/fp, and/or/xor for
    int only. The default is "add".

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

    // 4D vector contraction with two contracting dimensions and optional
    // vector mask arguments.
    %lhs_mask = vector.constant_mask [7, 8, 16, 15] : vector<7x8x16x15xi1>
    %rhs_mask = vector.constant_mask [8, 16, 7, 5] : vector<8x16x7x5xi1>

    %5 = vector.contract #contraction_trait %0, %1, %2, %lhs_mask, %rhs_mask
       : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x8x5xf32>

    // Vector contraction with mixed typed. lhs/rhs have different element
    // types than accumulator/result.
    %6 = vector.contract #contraction_trait %0, %1, %2
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
      kind = #vector.kind<max>
    }
    %7 = vector.contract #contraction_trait %0, %1, %2
      : vector<10xf32>, vector<10xf32> into f32
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$acc,
      "ArrayAttr":$indexingMaps, "ArrayAttr":$iteratorTypes)>,
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$acc,
      "ArrayRef<ArrayRef<AffineExpr>>":$indexingExprs,
      "ArrayRef<StringRef>":$iteratorTypes)>
  ];
  let extraClassDeclaration = [{
    VectorType getLhsType() {
      return lhs().getType().cast<VectorType>();
    }
    VectorType getRhsType() {
      return rhs().getType().cast<VectorType>();
    }
    Type getAccType() { return acc().getType(); }
    VectorType getLHSVectorMaskType() {
      if (llvm::size(masks()) != 2) return VectorType();
      return getOperand(3).getType().cast<VectorType>();
    }
    VectorType getRHSVectorMaskType() {
      if (llvm::size(masks()) != 2) return VectorType();
      return getOperand(4).getType().cast<VectorType>();
    }
    Type getResultType() { return getResult().getType(); }
    ArrayRef<StringRef> getTraitAttrNames();
    SmallVector<AffineMap, 4> getIndexingMaps();
    static unsigned getAccOperandIndex() { return 2; }

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

    static constexpr StringRef getKindAttrName() { return "kind"; }

    static CombiningKind getDefaultKind() {
      return CombiningKind::ADD;
    }
  }];

  let hasCanonicalizer = 1;
}

def Vector_ReductionOp :
  Vector_Op<"reduction", [NoSideEffect,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins StrAttr:$kind, AnyVector:$vector, Variadic<AnyType>:$acc)>,
    Results<(outs AnyType:$dest)> {
  let summary = "reduction operation";
  let description = [{
    Reduces an 1-D vector "horizontally" into a scalar using the given
    operation (add/mul/min/max for int/fp and and/or/xor for int only).
    Some reductions (add/mul for fp) also allow an optional fused
    accumulator.

    Note that these operations are restricted to 1-D vectors to remain
    close to the corresponding LLVM intrinsics:

    http://llvm.org/docs/LangRef.html#vector-reduction-intrinsics

    Example:

    ```mlir
    %1 = vector.reduction "add", %0 : vector<16xf32> into f32

    %3 = vector.reduction "xor", %2 : vector<4xi32> into i32

    %4 = vector.reduction "mul", %0, %1 : vector<16xf32> into f32
    ```
  }];
  let extraClassDeclaration = [{
    VectorType getVectorType() {
      return vector().getType().cast<VectorType>();
    }
  }];
}

def Vector_MultiDimReductionOp :
  Vector_Op<"multi_reduction", [NoSideEffect,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins Vector_CombiningKindAttr:$kind,
                   AnyVector:$source,
                   I64ArrayAttr:$reduction_dims)>,
    Results<(outs AnyType:$dest)> {
  let summary = "Multi-dimensional reduction operation";
  let description = [{
    Reduces an n-D vector into an (n-k)-D vector (or a scalar when k == n) 
    using the given operation (add/mul/min/max for int/fp and and/or/xor for
    int only).

    Example:

    ```mlir
    %1 = vector.multi_reduction "add", %0 [1, 3] :
      vector<4x8x16x32xf32> into vector<4x16xf32>
    %2 = vector.multi_reduction "add", %1 [0, 1] :
      vector<4x16xf32> into f32
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$source, "ArrayRef<bool>":$reductionMask,
                   "CombiningKind":$kind)>
  ];
  let extraClassDeclaration = [{
    static StringRef getKindAttrName() { return "kind"; }
    static StringRef getReductionDimsAttrName() { return "reduction_dims"; }

    VectorType getSourceVectorType() {
      return source().getType().cast<VectorType>();
    }
    Type getDestType() {
      return dest().getType();
    }

    bool isReducedDim(int64_t d) {
      assert(d >= 0 && d < static_cast<int64_t>(getReductionMask().size()) &&
        "d overflows the number of dims");
      return getReductionMask()[d];
    }

    SmallVector<bool> getReductionMask() {
      SmallVector<bool> res(getSourceVectorType().getRank(), false);
      for (auto ia : reduction_dims().getAsRange<IntegerAttr>())
        res[ia.getInt()] = true;
      return res;
    }
    static SmallVector<bool> getReductionMask(
        ArrayRef<int64_t> reductionDims, unsigned sourceRank) {
      SmallVector<bool> res(sourceRank, false);
      for (auto idx : reductionDims)
        res[idx] = true;
      return res;
    }

    static SmallVector<int64_t> inferDestShape(
      ArrayRef<int64_t> sourceShape, ArrayRef<bool> reducedDimsMask) {
      assert(sourceShape.size() == reducedDimsMask.size() &&
             "sourceShape and maks of different sizes");
      SmallVector<int64_t> res;
      for (auto it : llvm::zip(reducedDimsMask, sourceShape))
        if (!std::get<0>(it))
          res.push_back(std::get<1>(it));
      return res;
    }

    static Type inferDestType(
      ArrayRef<int64_t> sourceShape, ArrayRef<bool> reducedDimsMask, Type elementType) {
      auto targetShape = inferDestShape(sourceShape, reducedDimsMask);
      // TODO: update to also allow 0-d vectors when available.
      if (targetShape.empty())
        return elementType;
      return VectorType::get(targetShape, elementType);
    }
  }];
  let assemblyFormat =
    "$kind `,` $source attr-dict $reduction_dims `:` type($source) `to` type($dest)";
  let hasFolder = 1;
}

def Vector_BroadcastOp :
  Vector_Op<"broadcast", [NoSideEffect,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins AnyType:$source)>,
    Results<(outs AnyVector:$vector)> {
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
    The source operand is duplicated over all the missing leading dimensions
    and stretched over the trailing dimensions where the source has a non-equal
    dimension of 1. These rules imply that any scalar broadcast (k=0) to any
    shaped vector with the same element type is always legal.

    Example:

    ```mlir
    %0 = constant 0.0 : f32
    %1 = vector.broadcast %0 : f32 to vector<16xf32>
    %2 = vector.broadcast %1 : vector<16xf32> to vector<4x16xf32>
    ```
  }];
  let extraClassDeclaration = [{
    Type getSourceType() { return source().getType(); }
    VectorType getVectorType() {
      return vector().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($vector)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Vector_ShuffleOp :
  Vector_Op<"shuffle", [NoSideEffect,
     PredOpTrait<"first operand v1 and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>,
     PredOpTrait<"second operand v2 and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 1>>]>,
     Arguments<(ins AnyVector:$v1, AnyVector:$v2, I64ArrayAttr:$mask)>,
     Results<(outs AnyVector:$vector)> {
  let summary = "shuffle operation";
  let description = [{
    The shuffle operation constructs a permutation (or duplication) of elements
    from two input vectors, returning a vector with the same element type as
    the input and a length that is the same as the shuffle mask. The two input
    vectors must have the same element type, rank, and trailing dimension sizes
    and shuffles their values in the leading dimension (which may differ in size)
    according to the given mask. The legality rules are:
    * the two operands must have the same element type as the result
    * the two operands and the result must have the same rank and trailing
      dimension sizes, viz. given two k-D operands
              v1 : <s_1 x s_2 x .. x s_k x type> and
              v2 : <t_1 x t_2 x .. x t_k x type>
      we have s_i = t_i for all 1 < i <= k
    * the mask length equals the leading dimension size of the result
    * numbering the input vector indices left to right across the operands, all
      mask values must be within range, viz. given two k-D operands v1 and v2
      above, all mask values are in the range [0,s_1+t_1)

    Example:

    ```mlir
    %0 = vector.shuffle %a, %b[0, 3]
               : vector<2xf32>, vector<2xf32>       ; yields vector<2xf32>
    %1 = vector.shuffle %c, %b[0, 1, 2]
               : vector<2x16xf32>, vector<1x16xf32> ; yields vector<3x16xf32>
    %2 = vector.shuffle %a, %b[3, 2, 1, 0]
               : vector<2xf32>, vector<2xf32>       ; yields vector<4xf32>
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$v1, "Value":$v2, "ArrayRef<int64_t>")>
  ];
  let extraClassDeclaration = [{
    static StringRef getMaskAttrName() { return "mask"; }
    VectorType getV1VectorType() {
      return v1().getType().cast<VectorType>();
    }
    VectorType getV2VectorType() {
      return v2().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return vector().getType().cast<VectorType>();
    }
  }];
}

def Vector_ExtractElementOp :
  Vector_Op<"extractelement", [NoSideEffect,
     TypesMatchWith<"result type matches element type of vector operand",
                    "vector", "result",
                    "$_self.cast<ShapedType>().getElementType()">]>,
    Arguments<(ins AnyVector:$vector, AnySignlessInteger:$position)>,
    Results<(outs AnyType:$result)> {
  let summary = "extractelement operation";
  let description = [{
    Takes an 1-D vector and a dynamic index position and extracts the
    scalar at that position. Note that this instruction resembles
    vector.extract, but is restricted to 1-D vectors and relaxed
    to dynamic indices. It is meant to be closer to LLVM's version:
    https://llvm.org/docs/LangRef.html#extractelement-instruction

    Example:

    ```mlir
    %c = constant 15 : i32
    %1 = vector.extractelement %0[%c : i32]: vector<16xf32>
    ```
  }];
  let assemblyFormat = [{
    $vector `[` $position `:` type($position) `]` attr-dict `:` type($vector)
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "int64_t":$position)>,
    OpBuilder<(ins "Value":$source, "Value":$position)>
  ];
  let extraClassDeclaration = [{
    VectorType getVectorType() {
      return vector().getType().cast<VectorType>();
    }
  }];
}

def Vector_ExtractOp :
  Vector_Op<"extract", [NoSideEffect,
     PredOpTrait<"operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins AnyVector:$vector, I64ArrayAttr:$position)>,
    Results<(outs AnyType)> {
  let summary = "extract operation";
  let description = [{
    Takes an n-D vector and a k-D position and extracts the (n-k)-D vector at
    the proper position. Degenerates to an element type in the 0-D case.

    Example:

    ```mlir
    %1 = vector.extract %0[3]: vector<4x8x16xf32>
    %2 = vector.extract %0[3, 3, 3]: vector<4x8x16xf32>
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$source, "ArrayRef<int64_t>":$position)>,
    // Convenience builder which assumes the values in `position` are defined by
    // ConstantIndexOp.
    OpBuilder<(ins "Value":$source, "ValueRange":$position)>
  ];
  let extraClassDeclaration = [{
    static StringRef getPositionAttrName() { return "position"; }
    VectorType getVectorType() {
      return vector().getType().cast<VectorType>();
    }
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Vector_ExtractMapOp :
  Vector_Op<"extract_map", [NoSideEffect]>,
    Arguments<(ins AnyVector:$vector, Variadic<Index>:$ids)>,
    Results<(outs AnyVector)> {
  let summary = "vector extract map operation";
  let description = [{
    Takes an N-D vector and extracts a sub-part of the vector starting at id
    along each dimension.

    The dimension associated to each element of `ids` used to extract are
    implicitly deduced from the destination type. For each dimension the
    multiplicity is the destination dimension size divided by the source
    dimension size, each dimension with a multiplicity greater than 1 is
    associated to the next id, following ids order.
    For example if the source type is `vector<64x4x32xf32>` and the destination
    type is `vector<4x4x2xf32>`, the first id maps to dimension 0 and the second
    id to dimension 2.

    Similarly to vector.tuple_get, this operation is used for progressive
    lowering and should be folded away before converting to LLVM.

    It is different than `vector.extract_slice` and
    `vector.extract_strided_slice` as it takes a Value as index instead of an
    attribute. Also in the future it is meant to support extracting along any
    dimensions and not only the most major ones.

    For instance:
    ```
    // dynamic computation producing the value 0 of index type
    %idx0 = ... : index
    // dynamic computation producing the value 1 of index type
    %idx1 = ... : index
    %0 = constant dense<0, 1, 2, 3>: vector<4xi32>
    // extracts values [0, 1]
    %1 = vector.extract_map %0[%idx0] : vector<4xi32> to vector<2xi32>
    // extracts values [1, 2]
    %2 = vector.extract_map %0[%idx1] : vector<4xi32> to vector<2xi32>
    ```

    Example:

    ```mlir
    %ev = vector.extract_map %v[%id] : vector<32xf32> to vector<1xf32>
    %ev1 = vector.extract_map %v1[%id1, %id2] : vector<64x4x32xf32>
      to vector<4x4x2xf32>
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$vector, "ValueRange":$ids,
                  "ArrayRef<int64_t>":$multiplicity,
                  "AffineMap":$map)>];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return vector().getType().cast<VectorType>();
    }
    VectorType getResultType() {
      return getResult().getType().cast<VectorType>();
    }
    void getMultiplicity(SmallVectorImpl<int64_t> &multiplicity);
    AffineMap map();
  }];
  let assemblyFormat = [{
    $vector `[` $ids `]` attr-dict `:` type($vector) `to` type(results)
  }];

  let hasFolder = 1;
}

def Vector_FMAOp :
  Op<Vector_Dialect, "fma", [
       NoSideEffect, AllTypesMatch<["lhs", "rhs", "acc", "result"]>,
       DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>
     ] # ElementwiseMappable.traits>,
    Arguments<(ins AnyVector:$lhs, AnyVector:$rhs, AnyVector:$acc)>,
    Results<(outs AnyVector:$result)> {
  let summary = "vector fused multiply-add";
  let description = [{
    Multiply-add expressions operate on n-D vectors and compute a fused
    pointwise multiply-and-accumulate: `$result = `$lhs * $rhs + $acc`.
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
  // Fully specified by traits.
  let verifier = ?;
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs)";
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$acc),
    [{build($_builder, $_state, lhs.getType(), lhs, rhs, acc);}]>
  ];
  let extraClassDeclaration = [{
    VectorType getVectorType() { return lhs().getType().cast<VectorType>(); }
  }];
}

def Vector_InsertElementOp :
  Vector_Op<"insertelement", [NoSideEffect,
     TypesMatchWith<"source operand type matches element type of result",
                    "result", "source",
                    "$_self.cast<ShapedType>().getElementType()">,
     AllTypesMatch<["dest", "result"]>]>,
     Arguments<(ins AnyType:$source, AnyVector:$dest,
                    AnySignlessInteger:$position)>,
     Results<(outs AnyVector:$result)> {
  let summary = "insertelement operation";
  let description = [{
    Takes a scalar source, an 1-D destination vector and a dynamic index
    position and inserts the source into the destination at the proper
    position.  Note that this instruction resembles vector.insert, but
    is restricted to 1-D vectors and relaxed to dynamic indices. It is
    meant to be closer to LLVM's version:
    https://llvm.org/docs/LangRef.html#insertelement-instruction

    Example:

    ```mlir
    %c = constant 15 : i32
    %f = constant 0.0f : f32
    %1 = vector.insertelement %f, %0[%c : i32]: vector<16xf32>
    ```
  }];
  let assemblyFormat = [{
    $source `,` $dest `[` $position `:` type($position) `]` attr-dict `:`
    type($result)
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "Value":$dest, "int64_t":$position)>,
    OpBuilder<(ins "Value":$source, "Value":$dest, "Value":$position)>
  ];
  let extraClassDeclaration = [{
    Type getSourceType() { return source().getType(); }
    VectorType getDestVectorType() {
      return dest().getType().cast<VectorType>();
    }
  }];

}

def Vector_InsertOp :
  Vector_Op<"insert", [NoSideEffect,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>,
     AllTypesMatch<["dest", "res"]>]>,
     Arguments<(ins AnyType:$source, AnyVector:$dest, I64ArrayAttr:$position)>,
     Results<(outs AnyVector:$res)> {
  let summary = "insert operation";
  let description = [{
    Takes an n-D source vector, an (n+k)-D destination vector and a k-D position
    and inserts the n-D source into the (n+k)-D destination at the proper
    position. Degenerates to a scalar source type when n = 0.

    Example:

    ```mlir
    %2 = vector.insert %0, %1[3] : vector<8x16xf32> into vector<4x8x16xf32>
    %5 = vector.insert %3, %4[3, 3, 3] : f32 into vector<4x8x16xf32>
    ```
  }];
  let assemblyFormat = [{
    $source `,` $dest $position attr-dict `:` type($source) `into` type($dest)
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<int64_t>":$position)>,
    // Convenience builder which assumes all values are constant indices.
    OpBuilder<(ins "Value":$source, "Value":$dest, "ValueRange":$position)>
  ];
  let extraClassDeclaration = [{
    static StringRef getPositionAttrName() { return "position"; }
    Type getSourceType() { return source().getType(); }
    VectorType getDestVectorType() {
      return dest().getType().cast<VectorType>();
    }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Vector_InsertMapOp :
  Vector_Op<"insert_map", [NoSideEffect, AllTypesMatch<["dest", "result"]>]>,
    Arguments<(ins AnyVector:$vector, AnyVector:$dest, Variadic<Index>:$ids)>,
    Results<(outs AnyVector:$result)> {
  let summary = "vector insert map operation";
  let description = [{
    Inserts a N-D vector and within a larger vector starting at id. The new
    vector created will have the same size as the destination operand vector.

    The dimension associated to each element of `ids` used to insert is
    implicitly deduced from the source type (see `ExtractMapOp` for details).
    For example if source type is `vector<4x4x2xf32>` and the destination type
    is `vector<64x4x32xf32>`, the first id maps to dimension 0 and the second id
    to dimension 2.

    Similarly to vector.tuple_get, this operation is used for progressive
    lowering and should be folded away before converting to LLVM.

    It is different than `vector.insert` and `vector.insert_strided_slice` as it
    takes a Value as index instead of an attribute. Also in the future it is
    meant to support inserting along any dimensions and not only the most major
    ones.

    This operations is meant to be used in combination with vector.extract_map.

    For instance:
    ```
    // dynamic computation producing the value 0 of index type
    %idx0 = ... : index
    // dynamic computation producing the value 1 of index type
    %idx1 = ... : index /
    %0 = constant dense<0, 1, 2, 3>: vector<4xi32>
    // extracts values [0, 1]
    %1 = vector.extract_map %0[%idx0] : vector<4xi32> to vector<2xi32>
    // extracts values [1, 2]
    %2 = vector.extract_map %0[%idx1] : vector<4xi32> to vector<2xi32>
    // insert [0, 1] into [x, x, x, x] and produce [0, 1, x, x]
    %3 = vector.insert_map %1, %0[%idx0] : vector<2xi32> into vector<4xi32>
    // insert [1, 2] into [x, x, x, x] and produce [x, 1, 2, x]
    %4 = vector.insert_map %2, %0[%idx1] : vector<2xi32> into vector<4xi32>
    ```
    Example:

    ```mlir
    %v = vector.insert_map %ev %v[%id] : vector<1xf32> into vector<32xf32>
    %v1 = vector.insert_map %ev1, %v1[%arg0, %arg1] : vector<2x4x1xf32>
      into vector<64x4x32xf32>
    ```
  }];
  let builders = [OpBuilder<(ins "Value":$vector, "Value":$dest,
                                "ValueRange":$ids)>];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return vector().getType().cast<VectorType>();
    }
    VectorType getResultType() {
      return getResult().getType().cast<VectorType>();
    }
    // Return a map indicating the dimension mapping to the given ids.
    AffineMap map();
  }];
  let assemblyFormat = [{
    $vector `,` $dest `[` $ids `]` attr-dict
      `:` type($vector) `into` type($result)
  }];
}

def Vector_InsertStridedSliceOp :
  Vector_Op<"insert_strided_slice", [NoSideEffect,
    PredOpTrait<"operand #0 and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>,
    AllTypesMatch<["dest", "res"]>]>,
    Arguments<(ins AnyVector:$source, AnyVector:$dest, I64ArrayAttr:$offsets,
               I64ArrayAttr:$strides)>,
    Results<(outs AnyVector:$res)> {
  let summary = "strided_slice operation";
  let description = [{
    Takes a k-D source vector, an n-D destination vector (n >= k), n-sized
    `offsets` integer array attribute, a k-sized `strides` integer array attribute
    and inserts the k-D source vector as a strided subvector at the proper offset
    into the n-D destination vector.

    At the moment strides must contain only 1s.

    Returns an n-D vector that is a copy of the n-D destination vector in which
    the last k-D dimensions contain the k-D source vector elements strided at
    the proper location as specified by the offsets.

    Example:

    ```mlir
    %2 = vector.insert_strided_slice %0, %1
        {offsets = [0, 0, 2], strides = [1, 1]}:
      vector<2x4xf32> into vector<16x4x8xf32>
    ```
  }];

  let assemblyFormat = [{
    $source `,` $dest attr-dict `:` type($source) `into` type($dest)
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<int64_t>":$offsets, "ArrayRef<int64_t>":$strides)>
  ];
  let extraClassDeclaration = [{
    static StringRef getOffsetsAttrName() { return "offsets"; }
    static StringRef getStridesAttrName() { return "strides"; }
    VectorType getSourceVectorType() {
      return source().getType().cast<VectorType>();
    }
    VectorType getDestVectorType() {
      return dest().getType().cast<VectorType>();
    }
  }];

  let hasFolder = 1;
}

def Vector_OuterProductOp :
  Vector_Op<"outerproduct", [NoSideEffect,
    PredOpTrait<"lhs operand and result have same element type",
                TCresVTEtIsSameAsOpBase<0, 0>>,
    PredOpTrait<"rhs operand and result have same element type",
                TCresVTEtIsSameAsOpBase<0, 1>>]>,
    Arguments<(ins AnyVector:$lhs, AnyType:$rhs,
               Variadic<AnyVector>:$acc,
               DefaultValuedAttr<Vector_CombiningKindAttr, "CombiningKind::ADD">:$kind)>,
    Results<(outs AnyVector)> {
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

    An optional kind attribute may be specified to be add/mul/min/max
    for int/fp, and and/or/xor for int only. The default is "add", in which
    case the operation returns a fused multiply-add. In other cases it returns
    a multiply followed by the appropriate operation (for example, a compare and
    select for "max").

    Example:

    ```
    %2 = vector.outerproduct %0, %1: vector<4xf32>, vector<8xf32>
    return %2: vector<4x8xf32>

    %3 = vector.outerproduct %0, %1, %2:
      vector<4xf32>, vector<8xf32>, vector<4x8xf32>
    return %3: vector<4x8xf32>

    %4 = vector.outerproduct %0, %1, %2 {kind = #vector.kind<max>}:
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
      return lhs().getType().cast<VectorType>();
    }
    Type getOperandTypeRHS() {
      return rhs().getType();
    }
    VectorType getOperandVectorTypeACC() {
      return (llvm::size(acc()) == 0)
        ? VectorType()
        : (*acc().begin()).getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return getResult().getType().cast<VectorType>();
    }
    static constexpr StringRef getKindAttrName() {
      return "kind";
    }
    static CombiningKind getDefaultKind() {
      return CombiningKind::ADD;
    }
  }];
}

// TODO: Add transformation which decomposes ReshapeOp into an optimized
// sequence of vector rotate/shuffle/select operations.
def Vector_ReshapeOp :
  Vector_Op<"reshape", [AttrSizedOperandSegments, NoSideEffect]>,
    Arguments<(ins AnyVector:$vector, Variadic<Index>:$input_shape,
               Variadic<Index>:$output_shape,
               I64ArrayAttr:$fixed_vector_sizes)>,
    Results<(outs AnyVector:$result)> {
  let summary = "vector reshape operation";
  let description = [{
    Reshapes its vector operand from 'input_shape' to 'output_shape' maintaining
    fixed vector dimension 'fixed_vector_sizes' on the innermost vector
    dimensions.

    The parameters 'input_shape' and 'output_shape' represent valid data shapes
    across fixed vector shapes. For example, if a vector has a valid data
    shape [6] with fixed vector size [8], then the valid data elements are
    assumed to be stored at the beginning of the vector with the remaining
    vector elements undefined.

    In the examples below, valid data elements are represented by an alphabetic
    character, and undefined data elements are represented by '-'.

    Example

      vector<1x8xf32> with valid data shape [6], fixed vector sizes [8]

                input: [a, b, c, d, e, f]

           layout map: (d0) -> (d0 floordiv 8, d0 mod 8)

        vector layout: [a, b, c, d, e, f, -, -]

    Example

      vector<2x8xf32> with valid data shape [10], fixed vector sizes [8]

                input: [a, b, c, d, e, f, g, h, i, j]

           layout map: (d0) -> (d0 floordiv 8, d0 mod 8)

        vector layout: [[a, b, c, d, e, f, g, h],
                        [i, j, -, -, -, -, -, -]]

    Example

      vector<2x2x2x3xf32> with valid data shape [3, 5], fixed vector sizes
      [2, 3]

                input: [[a, b, c, d, e],
                        [f, g, h, i, j],
                        [k, l, m, n, o]]

           layout map: (d0, d1) -> (d0 floordiv 3, d1 floordiv 5,
                                    d0 mod 3, d1 mod 5)

        vector layout: [[[[a, b, c],
                          [f, g, h]]
                         [[d, e, -],
                          [i, j, -]]],
                        [[[k, l, m],
                          [-, -, -]]
                         [[n, o, -],
                          [-, -, -]]]]

    Example

      %1 = vector.reshape %0, [%c3, %c6], [%c2, %c9], [4]
        : vector<3x2x4xf32> to vector<2x3x4xf32>

             input: [[a, b, c, d, e, f],
                     [g, h, i, j, k, l],
                     [m, n, o, p, q, r]]

        layout map: (d0, d1) -> (d0, d1 floordiv 4, d1 mod 4)


      Input vector:  [[[a, b, c, d],
                       [e, f, -, -]],
                      [[g, h, i, j],
                       [k, l, -, -]],
                      [[m, n, o, p],
                       [q, r, -, -]]]

      Output vector:  [[[a, b, c, d],
                        [e, f, g, h],
                        [i, -, -, -]],
                       [[j, k, l, m],
                        [n, o, p, q],
                        [r, -, -, -]]]
  }];

  let extraClassDeclaration = [{
    VectorType getInputVectorType() {
      return vector().getType().cast<VectorType>();
    }
    VectorType getOutputVectorType() {
      return getResult().getType().cast<VectorType>();
    }

    /// Returns as integer value the number of input shape operands.
    int64_t getNumInputShapeSizes() { return input_shape().size(); }

    /// Returns as integer value the number of output shape operands.
    int64_t getNumOutputShapeSizes() { return output_shape().size(); }

    void getFixedVectorSizes(SmallVectorImpl<int64_t> &results);

    static StringRef getFixedVectorSizesAttrName() {
      return "fixed_vector_sizes";
    }
    static StringRef getInputShapeAttrName() { return "input_shape"; }
    static StringRef getOutputShapeAttrName() { return "output_shape"; }
  }];

  let assemblyFormat = [{
    $vector `,` `[` $input_shape `]` `,` `[` $output_shape `]` `,`
    $fixed_vector_sizes attr-dict `:` type($vector) `to` type($result)
  }];
}

def Vector_ExtractStridedSliceOp :
  Vector_Op<"extract_strided_slice", [NoSideEffect,
    PredOpTrait<"operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins AnyVector:$vector, I64ArrayAttr:$offsets,
               I64ArrayAttr:$sizes, I64ArrayAttr:$strides)>,
    Results<(outs AnyVector)> {
  let summary = "extract_strided_slice operation";
  let description = [{
    Takes an n-D vector, k-D `offsets` integer array attribute, a k-sized
    `sizes` integer array attribute, a k-sized `strides` integer array
    attribute and extracts the n-D subvector at the proper offset.

    At the moment strides must contain only 1s.
    // TODO: support non-1 strides.

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
  }];
  let builders = [
    OpBuilder<(ins "Value":$source, "ArrayRef<int64_t>":$offsets,
      "ArrayRef<int64_t>":$sizes, "ArrayRef<int64_t>":$strides)>
  ];
  let extraClassDeclaration = [{
    static StringRef getOffsetsAttrName() { return "offsets"; }
    static StringRef getSizesAttrName() { return "sizes"; }
    static StringRef getStridesAttrName() { return "strides"; }
    VectorType getVectorType(){ return vector().getType().cast<VectorType>(); }
    void getOffsets(SmallVectorImpl<int64_t> &results);
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let assemblyFormat = "$vector attr-dict `:` type($vector) `to` type(results)";
}

def Vector_TransferReadOp :
  Vector_Op<"transfer_read", [
      DeclareOpInterfaceMethods<VectorTransferOpInterface>,
      DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>,
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      AttrSizedOperandSegments
    ]>,
    Arguments<(ins AnyShaped:$source, Variadic<Index>:$indices,
               AffineMapAttr:$permutation_map, AnyType:$padding,
               Optional<VectorOf<[I1]>>:$mask,
               OptionalAttr<BoolArrayAttr>:$in_bounds)>,
    Results<(outs AnyVector:$vector)> {

  let summary = "Reads a supervector from memory into an SSA vector value.";

  let description = [{
    The `vector.transfer_read` op performs a read from a slice within a
    [MemRef](../LangRef.md#memref-type) or a Ranked
    [Tensor](../LangRef.md#tensor-type) supplied as its first operand into a
    [vector](../LangRef.md#vector-type) of the same base elemental type.

    A memref/tensor operand with vector element type, must have its vector
    element type match a suffix (shape and element type) of the vector (e.g.
    memref<3x2x6x4x3xf32>, vector<1x1x4x3xf32>).

    The slice is further defined by a full-rank index within the MemRef/Tensor,
    supplied as the operands `2 .. 1 + rank(memref/tensor)`.

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

    An optional SSA value `mask` of the same shape as the vector type may be
    specified to mask out elements. Such elements will be replaces with
    `padding`. Elements whose corresponding mask element is `0` are masked out.

    An optional boolean array attribute is provided to specify which dimensions
    of the transfer are guaranteed to be within bounds. The length of the array
    must equal the rank of the vector type. Broadcast dimensions must always be
    in-bounds. The absence of this optional `in_bounds` attribute signifies that
    any dimension of the transfer (except for broadcasts) may be out-of-bounds.
    A `vector.transfer_read` can be lowered to a simple load if all dimensions
    are specified to be within bounds and no `mask` was specified.

    Note that `in_bounds` is specified for result dimensions and not input
    dimensions. The starting point of the transfer, i.e.,
    `%A[%expr1, %expr2, %expr3, %expr4]` in the example below, is expected to
    be in-bounds and as indices are increasing, accesses may run out-of-bounds.

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
    %expr4]`. The size of the slice is 3 along d2 and 5 along d0, so the slice
    is: `%A[%expr1 : %expr1 + 5, %expr2, %expr3:%expr3 + 3, %expr4]`

    That slice needs to be read into a `vector<3x4x5xf32>`. Since the
    permutation map is not full rank, there must be a broadcast along vector
    dimension `1`.

    A notional lowering of vector.transfer_read could generate code resembling:

    ```mlir
    // %expr1, %expr2, %expr3, %expr4 defined before this point
    %tmp = alloc() : vector<3x4x5xf32>
    %view_in_tmp = "element_type_cast"(%tmp) : memref<1xvector<3x4x5xf32>>
    for %i = 0 to 3 {
      affine.for %j = 0 to 4 {
        affine.for %k = 0 to 5 {
          %a = load %A[%expr1 + %k, %expr2, %expr3 + %i, %expr4] :
            memref<?x?x?x?xf32>
          store %tmp[%i, %j, %k] : vector<3x4x5xf32>
    }}}
    %c0 = constant 0 : index
    %vec = load %view_in_tmp[%c0] : vector<3x4x5xf32>
    ```

    On a GPU one could then map `i`, `j`, `k` to blocks and threads. Notice that
    the temporary storage footprint is `3 * 5` values but `3 * 4 * 5` values are
    actually transferred between `%A` and `%tmp`.

    Alternatively, if a notional vector broadcast operation were available, the
    lowered code would resemble:

    ```mlir
    // %expr1, %expr2, %expr3, %expr4 defined before this point
    %tmp = alloc() : vector<3x4x5xf32>
    %view_in_tmp = "element_type_cast"(%tmp) : memref<1xvector<3x4x5xf32>>
    for %i = 0 to 3 {
      affine.for %k = 0 to 5 {
        %a = load %A[%expr1 + %k, %expr2, %expr3 + %i, %expr4] :
          memref<?x?x?x?xf32>
        store %tmp[%i, 0, %k] : vector<3x4x5xf32>
    }}
    %c0 = constant 0 : index
    %tmpvec = load %view_in_tmp[%c0] : vector<3x4x5xf32>
    %vec = broadcast %tmpvec, 1 : vector<3x4x5xf32>
    ```

    where `broadcast` broadcasts from element 0 to all others along the
    specified dimension. This time, the temporary storage footprint is `3 * 5`
    values which is the same amount of data as the `3 * 5` values transferred.
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
    %f0 = constant 0.0f : f32
    for %i0 = 0 to %0 {
      affine.for %i1 = 0 to %1 step 256 {
        affine.for %i2 = 0 to %2 step 32 {
          %v = vector.transfer_read %A[%i0, %i1, %i2], (%f0)
               {permutation_map: (d0, d1, d2) -> (d2, d1)} :
               memref<?x?x?xf32>, vector<32x256xf32>
    }}}

    // Read the slice `%A[%i0, %i1]` (i.e. the element `%A[%i0, %i1]`) into
    // vector<128xf32>. The underlying implementation will require a 1-D vector
    // broadcast:
    for %i0 = 0 to %0 {
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
    // Builder that sets padding to zero.
    OpBuilder<(ins "VectorType":$vector, "Value":$source,
      "ValueRange":$indices, "AffineMap":$permutationMap,
      CArg<"ArrayRef<bool>", "{}">:$inBounds)>,
    // Builder that sets permutation map to 'getMinorIdentityMap'.
    OpBuilder<(ins "VectorType":$vector, "Value":$source,
      "ValueRange":$indices, "Value":$padding,
      CArg<"ArrayRef<bool>", "{}">:$inBounds)>,
    // Builder that sets permutation map (resp. padding) to
    // 'getMinorIdentityMap' (resp. zero).
    OpBuilder<(ins "VectorType":$vector, "Value":$source,
      "ValueRange":$indices, CArg<"ArrayRef<bool>", "{}">:$inBounds)>,
    // Builder that does not set mask.
    OpBuilder<(ins "Type":$vector, "Value":$source,
      "ValueRange":$indices, "AffineMapAttr":$permutationMap, "Value":$padding,
      "ArrayAttr":$inBounds)>,
    // Builder that does not set mask.
    OpBuilder<(ins "Type":$vector, "Value":$source,
      "ValueRange":$indices, "AffineMap":$permutationMap, "Value":$padding,
      "ArrayAttr":$inBounds)>
  ];

  let extraClassDeclaration = [{
    /// Temporary convenience builders to account for the fact that we do not
    /// have 0-d vectors atm. These create a constant `vector<1xt>` and
    /// insert/extract into it.
    // Builder that sets permutation map (resp. padding) to
    // 'getMinorIdentityMap' (resp. zero).
    static Value createScalarOp(OpBuilder &builder, Location loc, Value source,
                                ValueRange indices,
                                ArrayRef<bool> inBounds = ArrayRef<bool>{});
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Vector_TransferWriteOp :
  Vector_Op<"transfer_write", [
      DeclareOpInterfaceMethods<VectorTransferOpInterface>,
      DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>,
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      AttrSizedOperandSegments
  ]>,
    Arguments<(ins AnyVector:$vector, AnyShaped:$source,
               Variadic<Index>:$indices,
               AffineMapAttr:$permutation_map,
               Optional<VectorOf<[I1]>>:$mask,
               OptionalAttr<BoolArrayAttr>:$in_bounds)>,
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
    supplied as the operands `3 .. 2 + rank(memref/tensor)`.

    The permutation_map [attribute](../LangRef.md#attributes) is an
    [affine-map](Affine.md#affine-maps) which specifies the transposition on the
    slice to match the vector shape. The permutation map may be implicit and
    omitted from parsing and printing if it is the canonical minor identity map
    (i.e. if it does not permute any dimension). In contrast to `transfer_read`,
    write ops cannot have broadcast dimensions.

    The size of the slice is specified by the size of the vector.

    An optional SSA value `mask` of the same shape as the vector type may be
    specified to mask out elements. Elements whose corresponding mask element
    is `0` are masked out.

    An optional boolean array attribute is provided to specify which dimensions
    of the transfer are guaranteed to be within bounds. The absence of this
    `in_bounds` attribute signifies that any dimension of the transfer may be
    out-of-bounds. A `vector.transfer_write` can be lowered to a simple store
    if all dimensions are specified to be within bounds and no `mask` was
    specified.

    An optional boolean array attribute is provided to specify which dimensions
    of the transfer are guaranteed to be within bounds. The length of the array
    must equal the rank of the vector type. The absence of this optional
    `in_bounds` attribute signifies that any dimension of the transfer
    may be out-of-bounds. A `vector.transfer_write` can be lowered to a simple
    store if all dimensions are specified to be within bounds and no `mask` was
    specified.

    Note that `in_bounds` is specified for result dimensions and not input
    dimensions. The starting point of the transfer, i.e.,
    `%A[%expr1, %expr2, %expr3, %expr4]` in the example below, is expected to
    be in-bounds and as indices are increasing, accesses may run out-of-bounds.

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
    // Builder that sets an empty mask.
    OpBuilder<(ins "Value":$vector, "Value":$source, "ValueRange":$indices,
      "AffineMap":$permutationMap, CArg<"ArrayRef<bool>", "{}">:$inBounds)>,
    // Builder that sets permutation map to 'getMinorIdentityMap'.
    OpBuilder<(ins "Value":$vector, "Value":$source, "ValueRange":$indices,
      CArg<"ArrayRef<bool>", "{}">:$inBounds)>,
    OpBuilder<(ins "Value":$vector, "Value":$source, "ValueRange":$indices,
      "AffineMapAttr":$permutationMap, "ArrayAttr":$inBounds)>,
    OpBuilder<(ins "Value":$vector, "Value":$source, "ValueRange":$indices,
      "AffineMap":$permutationMap, "Value":$mask, "ArrayAttr":$inBounds)>,
    OpBuilder<(ins "Value":$vector, "Value":$source, "ValueRange":$indices,
      "AffineMap":$permutationMap, "ArrayAttr":$inBounds)>,
  ];

  let extraClassDeclaration = [{
    /// Temporary convenience builders to account for the fact that we do not
    /// have 0-d vectors atm. These create a constant `vector<1xt>` and
    /// insert/extract into it.
    // Builder that sets permutation map (resp. padding) to
    // 'getMinorIdentityMap' (resp. zero).
    static Operation *createScalarOp(
      OpBuilder &builder, Location loc, Value value,
      Value dest, ValueRange indices,
      ArrayRef<bool> inBounds = ArrayRef<bool>{});
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Vector_LoadOp : Vector_Op<"load"> {
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
    strides. Only unit strides are allowed along the most minor memref
    dimension. These constraints guarantee that elements read along the first
    dimension of the slice are contiguous in memory.

    The memref element type can be a scalar or a vector type. If the memref
    element type is a scalar, it should match the element type of the result
    vector. If the memref element type is vector, it should match the result
    vector type.

    Example 1: 1-D vector load on a scalar memref.
    ```mlir
    %result = vector.load %base[%i, %j] : memref<100x100xf32>, vector<8xf32>
    ```

    Example 2: 1-D vector load on a vector memref.
    ```mlir
    %result = vector.load %memref[%i, %j] : memref<200x100xvector<8xf32>>, vector<8xf32>
    ```

    Example 3:  2-D vector load on a scalar memref.
    ```mlir
    %result = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<4x8xf32>
    ```

    Example 4:  2-D vector load on a vector memref.
    ```mlir
    %result = vector.load %memref[%i, %j] : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
    ```

    Representation-wise, the 'vector.load' operation permits out-of-bounds
    reads. Support and implementation of out-of-bounds vector loads is
    target-specific. No assumptions should be made on the value of elements
    loaded out of bounds. Not all targets may support out-of-bounds vector
    loads.

    Example 5:  Potential out-of-bound vector load.
    ```mlir
    %result = vector.load %memref[%index] : memref<?xf32>, vector<8xf32>
    ```

    Example 6:  Explicit out-of-bound vector load.
    ```mlir
    %result = vector.load %memref[%c0] : memref<7xf32>, vector<8xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
      [MemRead]>:$base,
      Variadic<Index>:$indices);
  let results = (outs AnyVector:$result);

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }

    VectorType getVectorType() {
      return result().getType().cast<VectorType>();
    }
  }];

  let hasFolder = 1;

  let assemblyFormat =
      "$base `[` $indices `]` attr-dict `:` type($base) `,` type($result)";
}

def Vector_StoreOp : Vector_Op<"store"> {
  let summary = "writes an n-D vector to an n-D slice of memory";
  let description = [{
    The 'vector.store' operation writes an n-D vector to an n-D slice of memory.
    It takes the vector value to be stored, a 'base' memref and an index for
    each memref dimension. The 'base' memref and indices determine the start
    memory address from which to write. Each index provides an offset for each
    memref dimension based on the element type of the memref. The shape of the
    vector value to store determines the shape of the slice written from the
    start memory address. The elements along each dimension of the slice are
    strided by the memref strides. Only unit strides are allowed along the most
    minor memref dimension. These constraints guarantee that elements written
    along the first dimension of the slice are contiguous in memory.

    The memref element type can be a scalar or a vector type. If the memref
    element type is a scalar, it should match the element type of the value
    to store. If the memref element type is vector, it should match the type
    of the value to store.

    Example 1: 1-D vector store on a scalar memref.
    ```mlir
    vector.store %valueToStore, %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
    ```

    Example 2: 1-D vector store on a vector memref.
    ```mlir
    vector.store %valueToStore, %memref[%i, %j] : memref<200x100xvector<8xf32>>, vector<8xf32>
    ```

    Example 3:  2-D vector store on a scalar memref.
    ```mlir
    vector.store %valueToStore, %memref[%i, %j] : memref<200x100xf32>, vector<4x8xf32>
    ```

    Example 4:  2-D vector store on a vector memref.
    ```mlir
    vector.store %valueToStore, %memref[%i, %j] : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
    ```

    Representation-wise, the 'vector.store' operation permits out-of-bounds
    writes. Support and implementation of out-of-bounds vector stores are
    target-specific. No assumptions should be made on the memory written out of
    bounds. Not all targets may support out-of-bounds vector stores.

    Example 5:  Potential out-of-bounds vector store.
    ```mlir
    vector.store %valueToStore, %memref[%index] : memref<?xf32>, vector<8xf32>
    ```

    Example 6:  Explicit out-of-bounds vector store.
    ```mlir
    vector.store %valueToStore, %memref[%c0] : memref<7xf32>, vector<8xf32>
    ```
  }];

  let arguments = (ins AnyVector:$valueToStore,
      Arg<AnyMemRef, "the reference to store to",
      [MemWrite]>:$base,
      Variadic<Index>:$indices);

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }

    VectorType getVectorType() {
      return valueToStore().getType().cast<VectorType>();
    }
  }];

  let hasFolder = 1;

  let assemblyFormat = "$valueToStore `,` $base `[` $indices `]` attr-dict "
                       "`:` type($base) `,` type($valueToStore)";
}

def Vector_MaskedLoadOp :
  Vector_Op<"maskedload">,
    Arguments<(ins Arg<AnyMemRef, "", [MemRead]>:$base,
               Variadic<Index>:$indices,
               VectorOfRankAndType<[1], [I1]>:$mask,
               VectorOfRank<[1]>:$pass_thru)>,
    Results<(outs VectorOfRank<[1]>:$result)> {

  let summary = "loads elements from memory into a vector as defined by a mask vector";

  let description = [{
    The masked load reads elements from memory into a 1-D vector as defined
    by a base with indices and a 1-D mask vector. When the mask is set, the
    element is read from memory. Otherwise, the corresponding element is taken
    from a 1-D pass-through vector. Informally the semantics are:
    ```
    result[0] := mask[0] ? base[i+0] : pass_thru[0]
    result[1] := mask[1] ? base[i+1] : pass_thru[1]
    etc.
    ```
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
  }];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
    VectorType getMaskVectorType() {
      return mask().getType().cast<VectorType>();
    }
    VectorType getPassThruVectorType() {
      return pass_thru().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return result().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat = "$base `[` $indices `]` `,` $mask `,` $pass_thru attr-dict `:` "
    "type($base) `,` type($mask) `,` type($pass_thru) `into` type($result)";
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Vector_MaskedStoreOp :
  Vector_Op<"maskedstore">,
    Arguments<(ins Arg<AnyMemRef, "", [MemWrite]>:$base,
               Variadic<Index>:$indices,
               VectorOfRankAndType<[1], [I1]>:$mask,
               VectorOfRank<[1]>:$valueToStore)> {

  let summary = "stores elements from a vector into memory as defined by a mask vector";

  let description = [{
    The masked store operation writes elements from a 1-D vector into memory
    as defined by a base with indices and a 1-D mask vector. When the mask is
    set, the corresponding element from the vector is written to memory. Otherwise,
    no action is taken for the element. Informally the semantics are:
    ```
    if (mask[0]) base[i+0] = value[0]
    if (mask[1]) base[i+1] = value[1]
    etc.
    ```
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
  }];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
    VectorType getMaskVectorType() {
      return mask().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return valueToStore().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat =
      "$base `[` $indices `]` `,` $mask `,` $valueToStore "
      "attr-dict `:` type($base) `,` type($mask) `,` type($valueToStore)";
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Vector_GatherOp :
  Vector_Op<"gather">,
    Arguments<(ins Arg<AnyMemRef, "", [MemRead]>:$base,
               Variadic<Index>:$indices,
	       VectorOfRankAndType<[1], [AnyInteger, Index]>:$index_vec,
               VectorOfRankAndType<[1], [I1]>:$mask,
               VectorOfRank<[1]>:$pass_thru)>,
    Results<(outs VectorOfRank<[1]>:$result)> {

  let summary = "gathers elements from memory into a vector as defined by an index vector and mask";

  let description = [{
    The gather operation gathers elements from memory into a 1-D vector as
    defined by a base with indices and an additional 1-D index vector, but
    only if the corresponding bit is set in a 1-D mask vector. Otherwise, the
    element is taken from a 1-D pass-through vector. Informally the semantics
    are:
    ```
    result[0] := mask[0] ? base[index[0]] : pass_thru[0]
    result[1] := mask[1] ? base[index[1]] : pass_thru[1]
    etc.
    ```
    The vector dialect leaves out-of-bounds behavior undefined.

    The gather operation can be used directly where applicable, or can be used
    during progressively lowering to bring other memory operations closer to
    hardware ISA support for a gather. The semantics of the operation closely
    correspond to those of the `llvm.masked.gather`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-gather-intrinsics).

    Examples:

    ```mlir
    %0 = vector.gather %base[%c0][%v], %mask, %pass_thru
       : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>

    %1 = vector.gather %base[%i, %j][%v], %mask, %pass_thru
       : memref<16x16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
    ```
  }];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
    VectorType getIndexVectorType() {
      return index_vec().getType().cast<VectorType>();
    }
    VectorType getMaskVectorType() {
      return mask().getType().cast<VectorType>();
    }
    VectorType getPassThruVectorType() {
      return pass_thru().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return result().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat =
    "$base `[` $indices `]` `[` $index_vec `]` `,` "
    "$mask `,` $pass_thru attr-dict `:` type($base) `,` "
    "type($index_vec)  `,` type($mask) `,` type($pass_thru) "
    "`into` type($result)";
  let hasCanonicalizer = 1;
}

def Vector_ScatterOp :
  Vector_Op<"scatter">,
    Arguments<(ins Arg<AnyMemRef, "", [MemWrite]>:$base,
               Variadic<Index>:$indices,
	       VectorOfRankAndType<[1], [AnyInteger, Index]>:$index_vec,
               VectorOfRankAndType<[1], [I1]>:$mask,
               VectorOfRank<[1]>:$valueToStore)> {

  let summary = "scatters elements from a vector into memory as defined by an index vector and mask";

  let description = [{
    The scatter operation scatters elements from a 1-D vector into memory as
    defined by a base with indices and an additional 1-D index vector, but
    only if the corresponding bit in a 1-D mask vector is set. Otherwise, no
    action is taken for that element. Informally the semantics are:
    ```
    if (mask[0]) base[index[0]] = value[0]
    if (mask[1]) base[index[1]] = value[1]
    etc.
    ```
    The vector dialect leaves out-of-bounds and repeated index behavior
    undefined. Underlying implementations may enforce strict sequential
    semantics for the latter, though.
    TODO: enforce the latter always?

    The scatter operation can be used directly where applicable, or can be used
    during progressively lowering to bring other memory operations closer to
    hardware ISA support for a scatter. The semantics of the operation closely
    correspond to those of the `llvm.masked.scatter`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-scatter-intrinsics).

    Examples:

    ```mlir
    vector.scatter %base[%c0][%v], %mask, %value
        : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>

    vector.scatter %base[%i, %j][%v], %mask, %value
        : memref<16x16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
    ```
  }];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
    VectorType getIndexVectorType() {
      return index_vec().getType().cast<VectorType>();
    }
    VectorType getMaskVectorType() {
      return mask().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return valueToStore().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat =
      "$base `[` $indices `]` `[` $index_vec `]` `,` "
      "$mask `,` $valueToStore attr-dict `:` type($base) `,` "
      "type($index_vec)  `,` type($mask) `,` type($valueToStore)";
  let hasCanonicalizer = 1;
}

def Vector_ExpandLoadOp :
  Vector_Op<"expandload">,
    Arguments<(ins Arg<AnyMemRef, "", [MemRead]>:$base,
               Variadic<Index>:$indices,
               VectorOfRankAndType<[1], [I1]>:$mask,
               VectorOfRank<[1]>:$pass_thru)>,
    Results<(outs VectorOfRank<[1]>:$result)> {

  let summary = "reads elements from memory and spreads them into a vector as defined by a mask";

  let description = [{
    The expand load reads elements from memory into a 1-D vector as defined
    by a base with indices and a 1-D mask vector. When the mask is set, the
    next element is read from memory. Otherwise, the corresponding element
    is taken from a 1-D pass-through vector. Informally the semantics are:
    ```
    index = i
    result[0] := mask[0] ? base[index++] : pass_thru[0]
    result[1] := mask[1] ? base[index++] : pass_thru[1]
    etc.
    ```
    Note that the index increment is done conditionally.

    The expand load can be used directly where applicable, or can be used
    during progressively lowering to bring other memory operations closer to
    hardware ISA support for an expand. The semantics of the operation closely
    correspond to those of the `llvm.masked.expandload`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-expandload-intrinsics).

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
      return base().getType().cast<MemRefType>();
    }
    VectorType getMaskVectorType() {
      return mask().getType().cast<VectorType>();
    }
    VectorType getPassThruVectorType() {
      return pass_thru().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return result().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat = "$base `[` $indices `]` `,` $mask `,` $pass_thru attr-dict `:` "
    "type($base) `,` type($mask) `,` type($pass_thru) `into` type($result)";
  let hasCanonicalizer = 1;
}

def Vector_CompressStoreOp :
  Vector_Op<"compressstore">,
    Arguments<(ins Arg<AnyMemRef, "", [MemWrite]>:$base,
               Variadic<Index>:$indices,
               VectorOfRankAndType<[1], [I1]>:$mask,
               VectorOfRank<[1]>:$valueToStore)> {

  let summary = "writes elements selectively from a vector as defined by a mask";

  let description = [{
    The compress store operation writes elements from a 1-D vector into memory
    as defined by a base with indices and a 1-D mask vector. When the mask is
    set, the corresponding element from the vector is written next to memory.
    Otherwise, no action is taken for the element. Informally the semantics are:
    ```
    index = i
    if (mask[0]) base[index++] = value[0]
    if (mask[1]) base[index++] = value[1]
    etc.
    ```
    Note that the index increment is done conditionally.

    The compress store can be used directly where applicable, or can be used
    during progressively lowering to bring other memory operations closer to
    hardware ISA support for a compress. The semantics of the operation closely
    correspond to those of the `llvm.masked.compressstore`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-compressstore-intrinsics).

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
      return base().getType().cast<MemRefType>();
    }
    VectorType getMaskVectorType() {
      return mask().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return valueToStore().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat =
      "$base `[` $indices `]` `,` $mask `,` $valueToStore attr-dict `:` "
      "type($base) `,` type($mask) `,` type($valueToStore)";
  let hasCanonicalizer = 1;
}

def Vector_ShapeCastOp :
  Vector_Op<"shape_cast", [NoSideEffect]>,
    Arguments<(ins AnyVector:$source)>,
    Results<(outs AnyVector:$result)> {
  let summary = "shape_cast casts between vector shapes";
  let description = [{
    The shape_cast operation casts between an n-D source vector shape and
    a k-D result vector shape (the element type remains the same).

    If reducing rank (n > k), result dimension sizes must be a product
    of contiguous source dimension sizes.
    If expanding rank (n < k), source dimensions must factor into a
    contiguous sequence of destination dimension sizes.
    Each source dim is expanded (or contiguous sequence of source dims combined)
    in source dimension list order (i.e. 0 <= i < n), to produce a contiguous
    sequence of result dims (or a single result dim), in result dimension list
    order (i.e. 0 <= j < k). The product of all source dimension sizes and all
    result dimension sizes must match.

    It is currently assumed that this operation does not require moving data,
    and that it will be folded away before lowering vector operations.

    There is an exception to the folding expectation when targeting
    llvm.intr.matrix operations. We need a type conversion back and forth from a
    2-D MLIR vector to a 1-D flattened LLVM vector.shape_cast lowering to LLVM
    is supported in that particular case, for now.

    Example:

    ```mlir
    // Example casting to a lower vector rank.
    %1 = vector.shape_cast %0 : vector<5x1x4x3xf32> to vector<20x3xf32>

    // Example casting to a higher vector rank.
    %3 = vector.shape_cast %2 : vector<10x12x8xf32> to vector<5x2x3x4x8xf32>

    ```
  }];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return source().getType().cast<VectorType>();
    }
    VectorType getResultVectorType() {
      return getResult().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Vector_BitCastOp :
  Vector_Op<"bitcast", [NoSideEffect, AllRanksMatch<["source", "result"]>]>,
    Arguments<(ins AnyVector:$source)>,
    Results<(outs AnyVector:$result)>{
  let summary = "bitcast casts between vectors";
  let description = [{
    The bitcast operation casts between vectors of the same rank, the minor 1-D
    vector size is casted to a vector with a different element type but same
    bitwidth.

    Example:

    ```mlir
    // Example casting to a smaller element type.
    %1 = vector.bitcast %0 : vector<5x1x4x3xf32> to vector<5x1x4x6xi16>

    // Example casting to a bigger element type.
    %3 = vector.bitcast %2 : vector<10x12x8xi8> to vector<10x12x2xi32>

    // Example casting to an element type of the same size.
    %5 = vector.bitcast %4 : vector<5x1x4x3xf32> to vector<5x1x4x3xi32>
    ```
  }];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return source().getType().cast<VectorType>();
    }
    VectorType getResultVectorType() {
      return getResult().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
  let hasFolder = 1;
}

def Vector_TypeCastOp :
  Vector_Op<"type_cast", [NoSideEffect, ViewLikeOpInterface]>,
    Arguments<(ins StaticShapeMemRefOf<[AnyType]>:$memref)>,
    Results<(outs AnyMemRef:$result)> {
  let summary = "type_cast op converts a scalar memref to a vector memref";
  let description = [{
    Performs a conversion from a memref with scalar element to a memref with a
    *single* vector element, copying the shape of the memref to the vector. This
    is the minimal viable operation that is required to makeke
    super-vectorization operational. It can be seen as a special case of the
    `view` operation but scoped in the super-vectorization context.

    Syntax:

    ```
    operation ::= `vector.type_cast` ssa-use : memref-type to memref-type
    ```

    Example:

    ```mlir
    %A  = alloc() : memref<5x4x3xf32>
    %VA = vector.type_cast %A : memref<5x4x3xf32> to memref<vector<5x4x3xf32>>
    ```
  }];

  /// Build the canonical memRefType with a single vector.
  /// E.g. memref<4 x 5 x vector<6 x f32>> -> memref<vector<4 x 5 x 6 x f32>>.
  let builders = [OpBuilder<(ins "Value":$source)>];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return memref().getType().cast<MemRefType>();
    }
    MemRefType getResultMemRefType() {
      return getResult().getType().cast<MemRefType>();
    }
    // Implement ViewLikeOpInterface.
    Value getViewSource() { return memref(); }
  }];

  let assemblyFormat = [{
    $memref attr-dict `:` type($memref) `to` type($result)
  }];
}

def Vector_ConstantMaskOp :
  Vector_Op<"constant_mask", [NoSideEffect]>,
    Arguments<(ins I64ArrayAttr:$mask_dim_sizes)>,
    Results<(outs VectorOf<[I1]>)> {
  let summary = "creates a constant vector mask";
  let description = [{
    Creates and returns a vector mask where elements of the result vector
    are set to '0' or '1', based on whether the element indices are contained
    within a hyper-rectangular region specified by the 'mask_dim_sizes'
    array attribute argument. Each element of the 'mask_dim_sizes' array,
    specifies an exclusive upper bound [0, mask-dim-size-element-value)
    for a unique dimension in the vector result. The conjunction of the ranges
    define a hyper-rectangular region within which elements values are set to 1
    (otherwise element values are set to 0).

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

  let extraClassDeclaration = [{
    static StringRef getMaskDimSizesAttrName() { return "mask_dim_sizes"; }
  }];
  let assemblyFormat = "$mask_dim_sizes attr-dict `:` type(results)";
}

def Vector_CreateMaskOp :
  Vector_Op<"create_mask", [NoSideEffect]>,
    Arguments<(ins Variadic<Index>:$operands)>, Results<(outs VectorOf<[I1]>)> {
  let summary = "creates a vector mask";
  let description = [{
    Creates and returns a vector mask where elements of the result vector
    are set to '0' or '1', based on whether the element indices are contained
    within a hyper-rectangular region specified by the operands. Specifically,
    each operand specifies a range [0, operand-value) for a unique dimension in
    the vector result. The conjunction of the operand ranges define a
    hyper-rectangular region within which elements values are set to 1
    (otherwise element values are set to 0).

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

  let hasCanonicalizer = 1;
  let assemblyFormat = "$operands attr-dict `:` type(results)";
}

def Vector_TransposeOp :
  Vector_Op<"transpose", [NoSideEffect,
    PredOpTrait<"operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins AnyVector:$vector, I64ArrayAttr:$transp)>,
    Results<(outs AnyVector:$result)> {
  let summary = "vector transpose operation";
  let description = [{
    Takes a n-D vector and returns the transposed n-D vector defined by
    the permutation of ranks in the n-sized integer array attribute.
    In the operation

    ```mlir
    %1 = vector.transpose %0, [i_1, .., i_n]
      : vector<d_1 x .. x d_n x f32>
      to vector<d_trans[0] x .. x d_trans[n-1] x f32>
    ```

    the transp array [i_1, .., i_n] must be a permutation of [0, .., n-1].

    Example:

    ```mlir
    %1 = vector.transpose %0, [1, 0] : vector<2x3xf32> to vector<3x2xf32>

     [ [a, b, c],       [ [a, d],
       [d, e, f] ]  ->    [b, e],
                          [c, f] ]
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$vector, "ArrayRef<int64_t>":$transp)>
  ];
  let extraClassDeclaration = [{
    VectorType getVectorType() {
      return vector().getType().cast<VectorType>();
    }
    VectorType getResultType() {
      return result().getType().cast<VectorType>();
    }
    void getTransp(SmallVectorImpl<int64_t> &results);
    static StringRef getTranspAttrName() { return "transp"; }
  }];
  let assemblyFormat = [{
    $vector `,` $transp attr-dict `:` type($vector) `to` type($result)
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Vector_PrintOp :
  Vector_Op<"print", []>, Arguments<(ins AnyType:$source)> {
  let summary = "print operation (for testing and debugging)";
  let description = [{
    Prints the source vector (or scalar) to stdout in human readable
    format (for testing and debugging). No return value.

    Example:

    ```mlir
    %0 = constant 0.0 : f32
    %1 = vector.broadcast %0 : f32 to vector<4xf32>
    vector.print %1 : vector<4xf32>

    when lowered to LLVM, the vector print is unrolled into
    elementary printing method calls that at runtime will yield

    ( 0.0, 0.0, 0.0, 0.0 )

    on stdout when linked with a small runtime support library,
    which only needs to provide a few printing methods (single
    value for all data types, opening/closing bracket, comma,
    newline).
    ```
  }];
  let verifier = ?;
  let extraClassDeclaration = [{
    Type getPrintType() {
      return source().getType();
    }
  }];
  let assemblyFormat = "$source attr-dict `:` type($source)";
}

//===----------------------------------------------------------------------===//
// Ops used for supporting progressive lowering and conversion type changes.
// The Ops are typically not used directly by higher level dialects, but are
// used by intra-dialect rewriting rules to bring vector operations closer
// to the hardware ISA.
//===----------------------------------------------------------------------===//

/// Vector dialect matrix multiplication op that operates on flattened 1-D
/// MLIR vectors. This is the counterpart of llvm.matrix.multiply in MLIR.
/// This may seem redundant with vector.contract but it serves the purposes of
/// more progressive lowering and localized type conversion on the path:
///   `vector<...x...xf32> -> vector<...xf32> -> !llvm<... x float>`.
def Vector_MatmulOp : Vector_Op<"matrix_multiply", [NoSideEffect,
        PredOpTrait<"lhs operand and result have same element type",
                    TCresVTEtIsSameAsOpBase<0, 0>>,
        PredOpTrait<"rhs operand and result have same element type",
                    TCresVTEtIsSameAsOpBase<0, 1>>]>,
      Arguments<(
        // TODO: tighten vector element types that make sense.
        ins VectorOfRankAndType<[1],
              [AnySignlessInteger, AnySignedInteger, Index, AnyFloat]>:$lhs,
            VectorOfRankAndType<[1],
              [AnySignlessInteger, AnySignedInteger, Index, AnyFloat]>:$rhs,
            I32Attr:$lhs_rows, I32Attr:$lhs_columns, I32Attr:$rhs_columns)>,
      Results<(
        outs VectorOfRankAndType<[1],
               [AnySignlessInteger, AnySignedInteger, Index, AnyFloat]>:$res)>
{
  let summary = "Vector matrix multiplication op that operates on flattened 1-D"
    " MLIR vectors";
  let description = [{
    This is the counterpart of llvm.matrix.multiply in MLIR. It serves the
    purposes of more progressive lowering and localized type conversion.
    Higher levels typically lower matrix multiplications into 'vector.contract'
    operations. Subsequent rewriting rule progressively lower these operations
    into 'vector.matrix_multiply' operations to bring the operations closer
    to the hardware ISA.

    The ‘vector.matrix_multiply’ op treats `lhs` as matrix with <lhs_rows> rows
    and <lhs_columns> columns, `rhs` as matrix with <lhs_columns> rows and
    <rhs_columns> and multiplies them. The result matrix is returned embedded in
    the result vector.

    Also see:

    http://llvm.org/docs/LangRef.html#llvm-matrix-multiply-intrinsic

    Example:

    ```mlir
    %C = vector.matrix_multiply %A, %B
      { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32 } :
      (vector<64xf64>, vector<48xf64>) -> vector<12xf64>
    ```
  }];
  let builders = [
   OpBuilder<(ins "Value":$lhs, "Value":$rhs, "unsigned":$lhsRows,
     "unsigned":$lhsColumns, "unsigned":$rhsColumns),
   [{
     $_state.addOperands({lhs, rhs});
     $_state.addAttribute("lhs_rows",$_builder.getI32IntegerAttr(lhsRows));
     $_state.addAttribute("lhs_columns",$_builder.getI32IntegerAttr(lhsColumns));
     $_state.addAttribute("rhs_columns",$_builder.getI32IntegerAttr(rhsColumns));
     $_state.addTypes(VectorType::get(lhsRows * rhsColumns,
       lhs.getType().cast<VectorType>().getElementType()));
   }]>,
  ];
  let verifier = ?;
  let assemblyFormat = "$lhs `,` $rhs attr-dict "
    "`:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)";
}

/// Vector dialect matrix tranposition op that operates on flattened 1-D
/// MLIR vectors. This is the counterpart of llvm.matrix.transpose in MLIR.
/// This may seem redundant with vector.transpose but it serves the purposes of
/// more progressive lowering and localized type conversion on the path:
///   `vector<...x...xf32> -> vector<...xf32> -> !llvm<... x float>`.
def Vector_FlatTransposeOp : Vector_Op<"flat_transpose", [NoSideEffect,
  PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(
      // TODO: tighten vector element types that make sense.
      ins VectorOfRankAndType<[1],
            [AnySignlessInteger, AnySignedInteger, Index, AnyFloat]>:$matrix,
          I32Attr:$rows, I32Attr:$columns)>,
    Results<(
      outs VectorOfRankAndType<[1],
             [AnySignlessInteger, AnySignedInteger, Index, AnyFloat]>:$res)> {
  let summary = "Vector matrix transposition on flattened 1-D MLIR vectors";
  let description = [{
    This is the counterpart of llvm.matrix.transpose in MLIR. It serves
    the purposes of more progressive lowering and localized type conversion.
    Higher levels typically lower matrix tranpositions into 'vector.transpose'
    operations. Subsequent rewriting rule progressively lower these operations
    into 'vector.flat_transpose' operations to bring the operations closer
    to the hardware ISA.

    The ‘vector.flat_transpose’ op treats the 1-D input `matrix` as
    a 2-D matrix with <rows> rows and <columns> columns, and returns the
    transposed matrix in flattened form in 'res'.

    Also see:

    http://llvm.org/docs/LangRef.html#llvm-matrix-transpose-intrinsic

    Example:

    ```mlir
    %1 = vector.flat_transpose %0 { rows = 4: i32, columns = 4: i32 }
       : (vector<16xf32>) -> vector<16xf32>
    ```
  }];
  let verifier = ?;
  let assemblyFormat = "$matrix attr-dict `:` type($matrix) `->` type($res)";
}

#endif // VECTOR_OPS
