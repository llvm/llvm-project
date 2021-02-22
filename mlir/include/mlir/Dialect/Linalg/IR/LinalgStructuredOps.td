//===- LinalgStructuredOps.td - Linalg dialect library ops -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for structured operations on buffers
// that correspond to underlying library calls (e.g. BLAS).
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_STRUCTURED_OPS
#define LINALG_STRUCTURED_OPS

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base Tablegen class for Linalg ops.
// Linalg ops that correspond to library calls operate on ShapedType as their
// first operands. These may be optionally followed by non-view operands
// depending on the specific Linalg op.
class LinalgStructuredBase_Op<string mnemonic, list<OpTrait> props>
  : Op<Linalg_Dialect, mnemonic, !listconcat(props, [
       LinalgStructuredInterface])> {
  code structuredOpsBaseDecls = [{
    // Return the number of induction variables in the basic block. This should
    // always be 0 for index-free linalg ops. For IndexedGeneric, this must be
    // equal to numLoops.
    unsigned getNumPayloadInductionVariables() {
      return isa<IndexedGenericOp>(this->getOperation()) ? getNumLoops() : 0;
    }
  }];
}

class LinalgStructured_Op<string mnemonic, list<OpTrait> props>
  : LinalgStructuredBase_Op<mnemonic,
       !listconcat(props, [
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>])> {
  code structuredOpsDecls = structuredOpsBaseDecls # [{
    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  }];
  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// Named Linalg ops, implemented as special configurations of generic ops.
//===----------------------------------------------------------------------===//
// At the moment these are not declarative and require a bunch of C++ code.
// In the future, these should be migrated to a declarative specification.
def CopyOp : LinalgStructured_Op<"copy", [CopyOpInterface]> {
  let description = [{
    Copies the data in the input view into the output view.

    Usage:

    ```mlir
    linalg.copy(%arg0, %arg1) : memref<?xf32, stride_specification>,
                                memref<?xf32, stride_specification>
    ```

    One possible lowering to loop form is:

    ```mlir
    %0 = linalg.dim %arg0, 0 : index
    scf.for %i0 = %c0 to %0 step %c1 {
      %1 = load %arg0[%i0] : memref<?xf32, stride_specification>
      store %1, %arg1[%i0] : memref<?xf32, stride_specification>
    }
    ```

    Optionally, can take `input_permutation` and `output_permutation` attributes
    to reorder the dimensions of the input and output views.

    Usage:

    ```mlir
    linalg.copy(%arg0, %arg1) {inputPermutation : (i, j, k) -> (i, k, j),
                               outputPermutation : (i, j, k) -> (k, j, i)} :
      memref<?x?x?xf32, stride_specification>,
      memref<?x?x?xf32, stride_specification>
    ```

    One possible lowering to loop form is:

    ```mlir
    %0 = linalg.dim %arg0, 0
    %1 = linalg.dim %arg0, 1
    %2 = linalg.dim %arg0, 2
    scf.for %i0 = %c0 to %{{.*}} step %c1 {
      scf.for %i1 = %c0 to %{{.*}} step %c1 {
        scf.for %i2 = %c0 to %{{.*}} step %c1 {
          %3 = load %arg0[%i0, %i2, %i1] :
                  memref<?x?x?xf32, stride_specification>
          store %3, %arg1[%i2, %i1, %i0] :
                  memref<?x?x?xf32, stride_specification>
    ```

    The views are expected to be compatible for correctness but this is not
    enforced at the moment.
  }];

  let arguments = (ins
    AnyStridedMemRef:$input,
    AnyStridedMemRef:$output,
    OptionalAttr<AffineMapAttr>:$inputPermutation,
    OptionalAttr<AffineMapAttr>:$outputPermutation);
  let regions = (region AnyRegion:$region);

  let builders = [
    OpBuilderDAG<(ins "Value":$input, "Value":$output,
      CArg<"AffineMap", "AffineMap()">:$inputPermutation,
      CArg<"AffineMap", "AffineMap()">:$outputPermutation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

  let extraClassDeclaration = structuredOpsDecls # [{
    ValueRange inputs() { return getOperands().take_front(); }
    ValueRange outputs() { return getOperands().take_back(); }

    // Rank-polymorphic.
    //   filling_value -> O(ivs) with parallel iterators.
    ArrayAttr iterator_types() {
      unsigned nPar = getInputShapedType(0).getRank();
      return Builder(getContext()).getStrArrayAttr(
        SmallVector<StringRef, 8>(nPar, getParallelIteratorTypeName()));
    }

    // I(input_perm(ivs)) -> O(output_perm(ivs))
    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto maybeInputMap = inputPermutation();
      auto maybeOutputMap = outputPermutation();
      unsigned inputRank = getInputShapedType(0).getRank();
      unsigned outputRank = getOutputShapedType(0).getRank();
      return Builder(getContext()).getAffineMapArrayAttr({
          extractOrIdentityMap(maybeInputMap, inputRank, context),
          extractOrIdentityMap(maybeOutputMap, outputRank, context)});
    }

    Value getSource() { return input();}
    Value getTarget() { return output(); }

    static void regionBuilder(Block &block, ValueRange captures);
    static std::function<void(Block &block, ValueRange captures)>
    getRegionBuilder() {
      return &regionBuilder;
    }
    static unsigned getNumRegionArgs() { return 2; }
  }];
  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = [{
    `(` $input `,` $output `)` attr-dict `:`
        type($input) `,` type($output)
      custom<CopyOpRegion>($region, ref(type($input)), ref(type($input)))
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let skipDefaultBuilders = 1;
}

def FillOp : LinalgStructured_Op<"fill", []> {
  let arguments = (ins AnyShaped:$output,
                   AnyTypeOf<[AnyFloat, AnySignlessInteger, AnyVector]>:$value);
  let results = (outs Optional<AnyRankedTensor>:$result);
  let regions = (region AnyRegion:$region);
  let extraClassDeclaration = structuredOpsDecls # [{
    ValueRange inputs() { return {}; }
    ValueRange outputs() { return getOperands().take_front(); }

    // Rank-polymorphic.
    //   filling_value -> O(ivs) with parallel iterators.
    ArrayAttr iterator_types() {
      unsigned nPar = getOutputShapedType(0).getRank();
      return Builder(getContext()).getStrArrayAttr(
        SmallVector<StringRef, 8>(nPar, getParallelIteratorTypeName()));
    }

    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      // filling_value -> O(ivs)
      return Builder(getContext()).getAffineMapArrayAttr({
          extractOrIdentityMap(llvm::None, getNumParallelLoops(), context)});
    }

    static void regionBuilder(Block &block, ValueRange captures);
    static std::function<void(Block &block, ValueRange captures)>
    getRegionBuilder() {
      return &regionBuilder;
    }
    static unsigned getNumRegionArgs() { return 1; }
  }];

  let assemblyFormat = [{
    `(` $output `,` $value `)` attr-dict `:`
        type($output) `,` type($value) (`->` type($result)^)?
      custom<FillOpRegion>($region, ref(type($output)), ref($value))
  }];

  let builders = [
    OpBuilderDAG<(ins "Value":$output, "Value":$value)>
  ];

  let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

/// A base class for pooling operation such as conv. The arguments must contain
/// optional arguments `strides`, `dilations` and `padding` with following type:
///   OptionalAttr<I64ArrayAttr>:$strides
///   OptionalAttr<I64ArrayAttr>:$dilations
///   OptionalAttr<I64ElementsAttr>:$padding
/// `strides` denotes the step of each window along the dimension.
class PoolingBase_Op<string mnemonic, list<OpTrait> props>
  : LinalgStructured_Op<mnemonic, props> {
  let description = [{
    Performs an N-D pooling operation similarly to the description in the TF
    documentation:
    https://www.tensorflow.org/api_docs/python/tf/nn/pool

    Different from the description, this operation doesn't perform on batch and
    channel. It only takes tensors of rank `N`.

    ```
      output[x[0], ..., x[N-1]] =
        REDUCE_{z[0], ..., z[N-1]}
          input[
                x[0] * strides[0] - pad_before[0] + dilation_rate[0]*z[0],
                ...
                x[N-1]*strides[N-1] - pad_before[N-1] + dilation_rate[N-1]*z[N-1]
                ],
    ```

    The required optional arguments are:
      - strides: an i64 array specifying the stride (i.e. step) for window
        loops.
      - dilations: an i64 array specifying the filter upsampling/input
        downsampling rate
      - padding: an i64 array of pairs (low, high) specifying the number of
        elements to pad along a dimension.

    If strides or dilations attributes are missing then the default value is
    one for each of the input dimensions. Similarly, padding values are zero
    for both low and high in each of the dimensions, if not specified.
  }];

  code commonUtils = structuredOpsDecls # [{
    int64_t getStride(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!strides().hasValue()) return 1;
      return strides()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getDilation(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!dilations().hasValue()) return 1;
      return dilations()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getLowPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 0});
    }

    int64_t getHighPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 1});
    }

    static std::function<void(Block &, ValueRange captures)> getRegionBuilder()
    {
      return nullptr;
    }
  }];
}

// Only support buffer semantics.
def ConvOp : PoolingBase_Op<"conv", []> {
  let description = [{
    Generic n-D convolution as described in the TF documentation:
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/convolution

    ```
      output[b, x[0], ..., x[N-1], k] =
      sum_{z[0], ..., z[N-1], q}
          filter[z[0], ..., z[N-1], q, k] *
          padded_input[b,
                       x[0] * strides[0] + dilation_rate[0] * z[0],
                       ...,
                       x[N-1] * strides[N-1] + dilation_rate[N-1] * z[N-1],
                       q]
    ```
  }];

  // Following the TF source of truth above, strides, dilations and padding are
  // integer attributes of the same rank as the number of window dimensions.
  // The padding attribute specifies the amount of zero padding to be applied to
  // the base area, which is a n-d array of (low, high) padding. Each pair has
  // the low padding as the first element and the high padding as the second
  // element. Using padding is equivalent to inserting those same zero values
  // into the input before doing the convolution.
  let arguments = (ins AnyStridedMemRef:$filter, AnyStridedMemRef:$input,
                   AnyStridedMemRef:$output,
                   OptionalAttr<I64ArrayAttr>:$strides,
                   OptionalAttr<I64ArrayAttr>:$dilations,
                   OptionalAttr<I64ElementsAttr>:$padding);

  let extraClassDeclaration = commonUtils # [{
    ValueRange inputs() { return getOperands().slice(0, 2); }
    ValueRange outputs() { return getOperands().take_back(); }

    // TODO: extend to support more than 1 dimensions and potentially grouping
    // too.
    unsigned getNumBatchDimensions() { return 1; }

    unsigned getNumInputFeatureDimensions() { return 1; }

    unsigned getNumOutputFeatureDimensions() { return 1; }

    unsigned getNumSpatialDimensions() {
      return getOutputShapedType(0).getRank() - getNumBatchDimensions() -
             getNumOutputFeatureDimensions();
    }

    ArrayAttr iterator_types() {
      // Outer parallel loops are always the number of output dimensions; i.e.
      // [b, xs, q] in the TF notation above.
      unsigned nPar = getOutputShapedType(0).getRank();
      unsigned nRed = getNumInputFeatureDimensions();
      // Window loops are a special kind of reduction that is never tiled or
      // parallelized across; i.e. [zs] in the TF notation above whose number
      // match `xs` (i.e. 1 window loop per "image" dimension).
      // This may evolve in the future.
      // Conditionally check nPar is large enough for cases of ill-formed op:
      // this avoids overflows before hitting the verifier.
      assert(nPar > getNumBatchDimensions() + getNumInputFeatureDimensions() &&
             "expected at least one window dimension (i.e. memref ranks greater "
             "than 2). See 'func @conv_rank_limit' in "
             "mlir/test/Dialect/Linalg/invalid.mlir");
      unsigned nWin =
        nPar - getNumBatchDimensions() - getNumInputFeatureDimensions();
      SmallVector<StringRef, 8> iters(nPar, getParallelIteratorTypeName());
      iters.reserve(nPar + nRed + nWin);
      iters.append(nRed, getReductionIteratorTypeName());
      iters.append(nWin, getWindowIteratorTypeName());
      return Builder(getContext()).getStrArrayAttr(iters);
    }

    //   F(z0, ..., zN-1, q, k) *
    //     I(b, x0 + z0 - pad_low_0, ..., xN-1 + zN-1 - pad_low_N-1, q)
    //   ->  O(b, x0, ..., xN-1, k)
    // for N equal to `nWindow`. If there is no padding attribute, it will be
    // ignored.
    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto nWin = getNumWindowLoops();
      assert(nWin > 0 && "expected at least one window dimension (i.e. memref "
                         "ranks greater than 2)");
      unsigned idx = 0;
      // In the following, AffineDimExprs are indexed in loop order:
      //   [ b, xs, k,           q,                     zs]
      //    parallels     non-window reductions     windows
      //
      // Parallel dims are exactly the dimensions indexing `output`:
      //     output[b, x[0], ..., x[N-1], k]; i.e.
      //  * batch dimensions (bs with #bs = 1 for now)
      //  * "image" dimensions (xs with #xs = #zs = output_rank - #bs - #ks)
      //  * output filter dimensions (ks with #ks = 1 for now)
      auto bs = makeAffineDimExprs(getNumBatchDimensions(), idx, context);
      auto xs = makeAffineDimExprs(nWin, idx, context);
      auto ks = makeAffineDimExprs(
        getNumOutputFeatureDimensions(), idx, context);
      // Non-window reduction dim: sum_{z[0], ..., z[N-1], q}
      auto qs = makeAffineDimExprs(
        getNumInputFeatureDimensions(), idx, context);
      // Window reduction dims: sum_{z[0], ..., z[N-1], q}
      auto zs = makeAffineDimExprs(nWin, idx, context);
      // Construct the weighedSum expression.
      auto ws = weightedPoolingInputIndex(*this, xs, zs);
      return Builder(getContext()).getAffineMapArrayAttr({
        // filter[z[0], ..., z[N-1], q, k]
        AffineMap::get(idx, 0, concat(concat(zs, qs), ks), context),
        // input[b,
        //       x[0]*s[0] + d[0]*z[0] - pad_low[0],
        //       ...
        //       x[N-1]*s[N-1] + d[N-1]*z[N-1] - pad_low[N-1],
        //       q]
        AffineMap::get(idx, 0, concat(concat(bs, ws), qs), context),
        // output[b, x[0], ..., x[N-1], k]
        AffineMap::get(idx, 0, concat(concat(bs, xs), ks), context)});
    }
  }];

  let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

// Only support buffer semantics.
class SingleInputPoolingBase_Op<string mnemonic>
    : PoolingBase_Op<mnemonic, []> {
  let description = [{
    A base class for single input pooling function.

    TODO: Figure out a better way to handle window dimensions, i.e., eliminate
    the fake memref.
    The window dimensions are specified by argument `windowDims`. The i-th
    dimension in the shape of `windowDims` denotes the size of the window along
    dimension i. For example, if the window size is 2x3, then a memref<2x3>
    should be passed to the operation as `windowDims`.
  }];

  let arguments = (ins AnyStridedMemRef:$input,
                   AnyStridedMemRef:$windowDims,
                   AnyStridedMemRef:$output,
                   OptionalAttr<I64ArrayAttr>:$strides,
                   OptionalAttr<I64ArrayAttr>:$dilations,
                   OptionalAttr<I64ElementsAttr>:$padding);

  let extraClassDeclaration = commonUtils# [{
    ValueRange inputs() { return getOperands().slice(0, 2); }
    ValueRange outputs() { return getOperands().take_back(); }

    ArrayAttr iterator_types() {
      // Outer parallel loops are always the number of output dimensions.
      unsigned nPar = getOutputShapedType(0).getRank();
      // The window loops has the same number loops with output dimensions.
      unsigned nWin = nPar;
      SmallVector<StringRef, 8> iters(nPar, getParallelIteratorTypeName());
      iters.reserve(nPar + nWin);
      iters.append(nWin, getWindowIteratorTypeName());
      return Builder(getContext()).getStrArrayAttr(iters);
    }

    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto nPar = getNumParallelLoops();
      auto nWin = getNumWindowLoops();
      assert(nWin > 0 && "expected at least one window dimension");
      unsigned idx = 0;
      auto outputDims = makeAffineDimExprs(nPar, idx, context);
      auto windowDims = makeAffineDimExprs(nWin, idx, context);
      // Construct the weighedSum expression.
      auto inputDims =
          weightedPoolingInputIndex(*this, outputDims, windowDims);
      return Builder(getContext()).getAffineMapArrayAttr({
        // input
        AffineMap::get(idx, 0, inputDims, context),
        // windowDims
        AffineMap::get(idx, 0, windowDims, context),
        // output
        AffineMap::get(idx, 0, outputDims, context)});
    }
  }];

  let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def PoolingMaxOp: SingleInputPoolingBase_Op<"pooling_max"> {
  let description = [{
    Takes max op as pooling operation, i.e., it samples the maximum value in the
    window.
  }];
}

def PoolingMinOp: SingleInputPoolingBase_Op<"pooling_min"> {
  let description = [{
    Takes min op as pooling operation, i.e., it samples the minimum value in the
    window.
  }];
}

def PoolingSumOp: SingleInputPoolingBase_Op<"pooling_sum"> {
  let description = [{
    Takes add op as pooling operation, i.e., it accumulates the values in the
    window.
  }];
}

//===----------------------------------------------------------------------===//
// Generic Linalg ops.
//===----------------------------------------------------------------------===//
def LinalgOperand: AnyTypeOf<[AnyRankedTensor, AnyStridedMemRef]>;

class LinalgOperandOfRank<int rank>: Type<
  And<[
    LinalgOperand.predicate,
    CPred<"$_self.cast<ShapedType>().getRank() == " # rank>]
  >>;

class GenericOpBase<string mnemonic> : LinalgStructuredBase_Op<mnemonic, [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       AffineMapArrayAttr:$indexing_maps,
                       ArrayAttr:$iterator_types,
                       OptionalAttr<StrAttr>:$doc,
                       OptionalAttr<StrAttr>:$library_call,
                       // ArrayAttr of StrArrayAttr:
                       OptionalAttr<ArrayAttr>:$sparse);
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
  let regions = (region AnyRegion:$region);
  let extraClassDeclaration = structuredOpsBaseDecls # [{
    SmallVector<StringRef, 8> linalgTraitAttrNames() {
      return SmallVector<StringRef, 8>{
        getDocAttrName(),
        getIndexingMapsAttrName(), getLibraryCallAttrName(),
        getIteratorTypesAttrName(),
      };
    }
    std::string getLibraryCallName() {
      return library_call().hasValue() ?
        library_call()->str() : "op_has_no_registered_library_name";
    }

    static std::function<void(Block &, ValueRange)> getRegionBuilder() {
      return nullptr;
    }
  }];
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseGenericOp(parser, result); }];
}

/// Index-free GenericOp.
def GenericOp : GenericOpBase<"generic"> {
  let description = [{
    Generic Linalg op form where the key properties of the computation are
    specified as attributes. In pretty form, a `linalg.generic` op is written
    as:

      ```mlir
      linalg.generic #trait_attribute
          ins(%A, %B : memref<?x?xf32, stride_specification>,
                       memref<?x?xf32, stride_specification>)
          outs(%C : memref<?x?xf32, stride_specification>)
          attrs = {other-optional-attributes}
          {region}
      ```

    Where #trait_attributes is an alias of a dictionary attribute containing:
      - doc [optional]: a documentation string
      - indexing_maps: a list of AffineMapAttr, one AffineMapAttr per each input
        and output view. Such AffineMapAttr specifies the mapping between the
        loops and the indexing within each view.
      - library_call [optional]: a StringAttr containing the name of an
        external library function that the linalg.generic operation maps to.
        The external library is assumed to be dynamically linked and no strong
        compile-time guarantees are provided. In the absence of such a library
        call, linalg.generic will always lower to loops.
      - iterator_types: an ArrayAttr specifying the type of the enclosing loops.
        Each element of the list represents and iterator of one of the following
        types:
          parallel, reduction, window
      - sparse: an optional list with per-dimension sparsity annotations (either
        "D" for dense or "S" for sparse) for each input and output view.

    Example:
    Defining a #matmul_trait attribute in MLIR can be done as follows:
      ```mlir
      #matmul_accesses = [
        (m, n, k) -> (m, k),
        (m, n, k) -> (k, n),
        (m, n, k) -> (m, n)
      ]
      #matmul_trait = {
        doc = "C(m, n) += A(m, k) * B(k, n)",
        indexing_maps = #matmul_accesses,
        library_call = "linalg_matmul",
        iterator_types = ["parallel", "parallel", "reduction"]
      }
      ```

    And can be reused in multiple places as:
      ```mlir
      linalg.generic #matmul_trait
        ins(%A, %B : memref<?x?xf32, stride_specification>,
                     memref<?x?xf32, stride_specification>)
        outs(%C : memref<?x?xf32, stride_specification>)
        {other-optional-attributes} {
        ^bb0(%a: f32, %b: f32, %c: f32) :
          %d = mulf %a, %b: f32
          %e = addf %c, %d: f32
          linalg.yield %e : f32
      }
      ```

    This may lower to either:
      ```mlir
      call @linalg_matmul(%A, %B, %C) :
        (memref<?x?xf32, stride_specification>,
         memref<?x?xf32, stride_specification>,
         memref<?x?xf32, stride_specification>)
        -> ()
      ```

    or IR resembling:
    ```mlir
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        scf.for %k = %c0 to %K step %c1 {
          %a = load %A[%m, %k] : memref<?x?xf32, stride_specification>
          %b = load %B[%k, %n] : memref<?x?xf32, stride_specification>
          %c = load %C[%m, %n] : memref<?x?xf32, stride_specification>
          %d = mulf %a, %b: f32
          %e = addf %c, %d: f32
          store %e, %C[%m, %n] : memref<?x?x?xf32, stride_specification>
        }
      }
    }
    ```

    To allow progressive lowering from the value world (a.k.a tensor values) to
    the buffer world (a.k.a memref values), a `linalg.generic` op allows mixing
    tensors and buffers operands and tensor results.

    ```mlir
    %C = linalg.generic #trait_attribute
      ins(%A, %B : tensor<?x?xf32>, memref<?x?xf32, stride_specification>)
      outs(%C : tensor<?x?xf32>)
      {other-optional-attributes}
      {region}
      -> (tensor<?x?xf32>)
    ```
  }];

  let builders = [
    OpBuilderDAG<(ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
      "ValueRange":$outputs, "ArrayRef<AffineMap>":$indexingMaps,
      "ArrayRef<StringRef>":$iteratorTypes, "StringRef":$doc,
      "StringRef":$libraryCall,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">)>,
    OpBuilderDAG<(ins "ValueRange":$inputs, "ValueRange":$outputBuffers,
      "ArrayRef<AffineMap>":$indexingMaps, "ArrayRef<StringRef>":$iteratorTypes,
      "StringRef":$doc, "StringRef":$libraryCall,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">)>,
    OpBuilderDAG<(ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
      "ValueRange":$outputs, "ArrayRef<AffineMap>":$indexingMaps,
      "ArrayRef<StringRef>":$iteratorTypes,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">)>,
    OpBuilderDAG<(ins "ValueRange":$inputs, "ValueRange":$outputBuffers,
      "ArrayRef<AffineMap>":$indexingMaps, "ArrayRef<StringRef>":$iteratorTypes,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">)>
  ];
  let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

/// GenericOp with Indexing (i.e. multi-for style in which the region is passed
/// the enclosing loop induction variables)
def IndexedGenericOp : GenericOpBase<"indexed_generic"> {
  let description = [{
    Indexed Generic Linalg op form where the key properties of the computation
    are specified as attributes. In pretty form, a `linalg.indexed_generic` op
    is written as:

      ```mlir
      linalg.indexed_generic #trait_attribute
          ins(%A, %B : memref<?x?xf32, stride_specification>,
                       memref<?x?xf32, stride_specification>)
          outs(%C : memref<?x?xf32, stride_specification>)
          attrs = {other-optional-attributes}
          {region}
      ```

    Where #trait_attributes is an alias of a dictionary attribute containing:
      - doc [optional]: a documentation string
      - indexing_maps: a list of AffineMapAttr, one AffineMapAttr per each input
        and output view. Such AffineMapAttr specifies the mapping between the
        loops and the indexing within each view.
      - library_call [optional]: a StringAttr containing the name of an
        external library function that the linalg.indexed_generic operation
        maps to.  The external library is assumed to be dynamically linked and
        no strong compile-time guarantees are provided. In the absence of such
        a library call, linalg.indexed_generic will always lower to loops.
      - iterator_types: an ArrayAttr they type of the enclosing loops; Each
        element of the list represents and iterator of one of the following
        types:
          parallel, reduction, window

    Example:
    Defining a #matmul_trait attribute in MLIR can be done as follows:

    ```mlir
    #matmul_accesses = [
      (m, n, k) -> (m, k),
      (m, n, k) -> (k, n),
      (m, n, k) -> (m, n)
    ]
    #matmul_trait = {
      doc = "C(m, n) += A(m, k) * B(k, n)",
      indexing_maps = #matmul_accesses,
      library_call = "linalg_matmul",
      iterator_types = ["parallel", "parallel", "reduction"]
    }
    ```

    And can be reused in multiple places as:

    ```mlir
      linalg.indexed_generic #matmul_trait
         ins(%A, %B : memref<?x?xf32, stride_specification>,
                      memref<?x?xf32, stride_specification>)
        outs(%C : memref<?x?xf32, stride_specification>) {
      (%offset_m: index, %offset_n: index, %offset_k: index,
       %a: f32, %b: f32, %c: f32) :
        "some_optional_computation"(%offset_m, %offset_n, %offset_k)
        %d = mulf %a, %b: f32
        %e = addf %c, %d: f32
        linalg_yield %e : f32
    }
    ```

    This may lower to either:

    ```mlir
    call @linalg_matmul(%offset_m, %offset_n, %offset_k, %A, %B, %C) :
      (index, index, index,
       memref<?x?xf32, stride_specification>,
       memref<?x?xf32, stride_specification>,
       memref<?x?xf32, stride_specification>)
      -> ()
    ```

    or IR resembling:

    ```mlir
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        scf.for %k = %c0 to %K step %c1 {
          %a = load %A[%m, %k] : memref<?x?xf32, stride_specification>
          %b = load %B[%k, %n] : memref<?x?xf32, stride_specification>
          %c = load %C[%m, %n] : memref<?x?xf32, stride_specification>
          "some_optional_computation"(%m, %n, %k)
          %d = mulf %a, %b: f32
          %e = addf %c, %d: f32
          store %d, %C[%m, %n] : memref<?x?x?xf32, stride_specification>
        }
      }
    }
    ```

    To allow progressive lowering from the value world (a.k.a tensor values) to
    the buffer world (a.k.a memref values), a `linalg.indexed_generic` op
    allows mixing tensors and buffers operands and tensor results.

    ```mlir
    %C = linalg.indexed_generic #trait_attribute
       ins(%A, %B : tensor<?x?xf32>, memref<?x?xf32, stride_specification>)
      outs(%C : tensor<?x?xf32>)
      {other-optional-attributes}
      {region_with_index_arguments}
      -> (tensor<?x?xf32>)
    ```
  }];

  let builders = [
    OpBuilderDAG<(ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
      "ValueRange":$outputs, "ArrayRef<AffineMap>":$indexingMaps,
      "ArrayRef<StringRef>":$iteratorTypes, "StringRef":$doc,
      "StringRef":$libraryCall,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>",
           "nullptr">)>,
    OpBuilderDAG<(ins "ValueRange":$inputs, "ValueRange":$outputBuffers,
      "ArrayRef<AffineMap>":$indexingMaps, "ArrayRef<StringRef>":$iteratorTypes,
      "StringRef":$doc, "StringRef":$libraryCall,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>",
           "nullptr">)>,
    OpBuilderDAG<(ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
      "ValueRange":$outputs, "ArrayRef<AffineMap>":$indexingMaps,
      "ArrayRef<StringRef>":$iteratorTypes,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>",
           "nullptr">)>,
    OpBuilderDAG<(ins "ValueRange":$inputs, "ValueRange":$outputBuffers,
      "ArrayRef<AffineMap>":$indexingMaps, "ArrayRef<StringRef>":$iteratorTypes,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>",
           "nullptr">)>
  ];
  let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Named Linalg ops, implemented as a declarative configurations of generic ops.
//===----------------------------------------------------------------------===//

// This file is auto-generated from a TC def specification.
include "mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.tcgen.td"
include "mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yamlgen.td"

#endif // LINALG_STRUCTURED_OPS
