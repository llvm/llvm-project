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

include "mlir/Dialect/AffineOps/AffineOpsBase.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"

// The Linalg `NInputs` trait provides the API for ops that are known
// to have a specified number of inputs, all passed as operands.
// See Linalg/LinalgTraits.h for implementation details an usage.
class NInputs<int args_in> :
  NativeOpTrait<"linalg::NInputs<" # !cast<string>(args_in) # ">::Impl"> {}

// The Linalg `NOutputs` trait provides the API for ops that are known
// to have a specified number of outputs, all passed as operands.
// See Linalg/LinalgTraits.h for implementation details an usage.
class NOutputs<int args_out> :
  NativeOpTrait<"linalg::NOutputs<" # !cast<string>(args_out) # ">::Impl"> {}

def StructuredOpTraits : NativeOpTrait<"linalg::StructuredOpTraits">;

// The linalg 'LinalgStructuredInterface' provides access to the 'LinalgOp'
// interface.
def LinalgStructuredInterface : OpInterface<"LinalgOp"> {
  let methods = [
    //========================================================================//
    // Loop types handling.
    //========================================================================//
    InterfaceMethod<
      "Return the number of parallel loops within the current operation.",
      "unsigned", "getNumParallelLoops"
    >,
    InterfaceMethod<
      "Return the number of reduction loops within the current operation.",
      "unsigned", "getNumReductionLoops"
    >,
    InterfaceMethod<
      "Return the number of window loops within the current operation.",
      "unsigned", "getNumWindowLoops"
    >,
    InterfaceMethod<
      "Return the number of loops within the current operation.",
      "unsigned", "getNumLoops">,

    InterfaceMethod<
      [{Returns true if the current operation has only one loop and it's a
        reduction loop}],
      "unsigned", "hasSingleReductionLoop">,

    //========================================================================//
    // Input arguments handling.
    //========================================================================//
    InterfaceMethod<
      "Return the number of inputs from the current operation.",
      "unsigned", "getNumInputs"
    >,
    InterfaceMethod<"Return the input view at the given index.",
      "Value ", "getInput", (ins "unsigned":$i)
    >,
    InterfaceMethod<[{
        Return the index of the given input value `v`, or `None` if the value is
        not an input.
      }],
      "llvm::Optional<unsigned>", "getIndexOfInput", (ins "Value ":$v)
    >,
    InterfaceMethod<
      "Return the input operands from the current operation.",
      "Operation::operand_range", "getInputs"
    >,
    InterfaceMethod<[{
        Return the type of the input shape at the given index.
      }], "ShapedType", "getInputShapedType", (ins "unsigned":$i)>,
    InterfaceMethod<[{
        Return the subset of input operands that are of ranked tensor type.
      }], "SmallVector<RankedTensorType, 4>", "getInputTensorTypes">,

    //========================================================================//
    // Output arguments handling.
    //========================================================================//
    InterfaceMethod<
      "Return the number of outputs from the current operation.",
      "unsigned", "getNumOutputs"
    >,
    InterfaceMethod<"Return the output buffer at the given index.",
      "Value ", "getOutputBuffer", (ins "unsigned":$i)
    >,
    InterfaceMethod<[{
        Return the index of the given buffer value, or `None` if the value is
        not part of the output buffers.
      }],
      "llvm::Optional<unsigned>", "getIndexOfOutputBuffer", (ins "Value ":$view)
    >,
    InterfaceMethod<[{
        Return the type of the output buffer at the given index.
      }], "MemRefType", "getOutputBufferType", (ins "unsigned":$i)>,
    InterfaceMethod<[{
        Return the results that are of ranked tensor type.
      }], "SmallVector<RankedTensorType, 4>", "getOutputTensorTypes">,
    InterfaceMethod<
      "Return the output buffers (operands) from the current operation.",
      "Operation::operand_range", "getOutputBuffers"
    >,

    //========================================================================//
    // Input and Output arguments handling.
    //========================================================================//
    InterfaceMethod<
      "Return the number of inputs and outputs, irrespective of their buffer "
      "or tensor type.",
      "unsigned", "getNumInputsAndOutputs"
    >,
    InterfaceMethod<
      "Return the number of inputs, irrespective of their buffer or tensor "
      "type, and output buffers",
      "unsigned", "getNumInputsAndOutputBuffers"
    >,
    InterfaceMethod<
      "Return the range over inputs (irrespective of type) and output buffers.",
      "Operation::operand_range", "getInputsAndOutputBuffers"
    >,

    //========================================================================//
    // Other interface methods.
    //========================================================================//
    InterfaceMethod<
      "Return the reference iterators for this named op (if any are specied). "
      "These reference iterators are used to specify the default behavior of "
      "the op. Typically this would be a static method but in order to allow "
      "rank-polymorphic ops, this needs to be per object instance. Named ops "
      "must define referenceIterators, even if empty for the 0-D case. "
      "Generic ops on the other hand have a None `referenceIterators`",
      "llvm::Optional<SmallVector<StringRef, 8>>", "referenceIterators"
    >,
    InterfaceMethod<
      "Return the reference indexing maps for this named op (if any are "
      "specified). Typically this would be a static method but in order to "
      "allow rank-polymorphic ops, this needs to be per object instance. Named "
      "ops must define referenceIterators, even if empty for the 0-D case. "
      "Generic ops on the other hand have a None `referenceIndexingMaps`",
      "llvm::Optional<SmallVector<AffineMap, 8>>", "referenceIndexingMaps"
    >,
    InterfaceMethod<
      "Return the iterator types attribute within the current operation.",
      "ArrayAttr", "iterator_types"
    >,
    InterfaceMethod<
      "Return the indexing maps attribute within the current operation.",
      "ArrayAttr", "indexing_maps"
    >,
    InterfaceMethod<"Return the input or output indexing map at index `i`.",
      "AffineMap", "getIndexingMap", (ins "unsigned":$i)
    >,
    InterfaceMethod<"Return the input indexing map at index `i`.",
      "AffineMap", "getInputIndexingMap", (ins "unsigned":$i)
    >,
    InterfaceMethod<"Return the output indexing map at index `i`.",
      "AffineMap", "getOutputIndexingMap", (ins "unsigned":$i)
    >,
    InterfaceMethod<[{
        Return whether the op has only MemRef input and outputs.
      }], "bool", "hasBufferSemantics">,
    InterfaceMethod<[{
        Return whether the op has only RankedTensor input and outputs.
      }], "bool", "hasTensorSemantics">,

    //========================================================================//
    // Other static interface methods.
    //========================================================================//
    StaticInterfaceMethod<[{
        Create an operation of the current type with the given location,
        operands, and attributes.
      }],
      "Operation *", "create",
      (ins "OpBuilder &":$builder, "Location":$loc,
           "ValueRange":$operands,
           "ArrayRef<NamedAttribute>":$attributes), [{
        return builder.create<ConcreteOp>(loc, ArrayRef<Type>{}, operands,
                                          attributes);
      }]
    >,
    InterfaceMethod<[{
        Clone the current operation with the given location and operands. This
        is used to abstract away the optional underlying region creation.
      }],
      "Operation *", "clone",
      (ins "OpBuilder &":$b, "Location":$loc, "ValueRange":$operands), [{
        BlockAndValueMapping map;
        unsigned numRegions = op.getOperation()->getNumRegions();
        Operation *res = create(b, loc, operands, op.getAttrs());
        assert(res->getNumRegions() == numRegions && "inconsistent # regions");
        for (unsigned ridx = 0; ridx < numRegions; ++ridx)
          op.getOperation()->getRegion(ridx).cloneInto(
            &res->getRegion(ridx), map);
        return res;
      }]
    >
  ];
}

// Base Tablegen class for Linalg ops.
// Linalg ops that correspond to library calls operate on linalg::View as their
// first operands. These may be optionally followed by non-view operands
// depending on the specific Linalg op.
class LinalgStructuredBase_Op<string mnemonic, list<OpTrait> props>
  : Op<Linalg_Dialect, mnemonic,
       !listconcat(props, [StructuredOpTraits, LinalgStructuredInterface])> {
}

class LinalgStructured_Op<string mnemonic, list<OpTrait> props>
  : LinalgStructuredBase_Op<mnemonic, props> {
  code libraryCallName = [{
    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  }];
  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

////////////////////////////////////////////////////////////////////////////////
// Named Linalg ops, implemented as special configurations of a generic op.
////////////////////////////////////////////////////////////////////////////////
def CopyOp : LinalgStructured_Op<"copy", [NInputs<1>, NOutputs<1>]> {
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
      loop.for %i0 = %c0 to %0 step %c1 {
        %1 = linalg.load %arg0[%i0] : memref<?xf32, stride_specification>
        linalg.store %1, %arg1[%i0] : memref<?xf32, stride_specification>
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
      loop.for %i0 = %c0 to %{{.*}} step %c1 {
        loop.for %i1 = %c0 to %{{.*}} step %c1 {
          loop.for %i2 = %c0 to %{{.*}} step %c1 {
            %3 = linalg.load %arg0[%i0, %i2, %i1] :
                    memref<?x?x?xf32, stride_specification>
            linalg.store %3, %arg1[%i2, %i1, %i0] :
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
  // TODO(ntv) this should go away once the usage of OptionalAttr triggers
  // emission of builders with default arguments left unspecified.
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value input, Value output", [{
    return build(
      builder, result, input, output, AffineMapAttr(), AffineMapAttr());
  }]>];
  let extraClassDeclaration = libraryCallName # [{
    // Defined in C++ for now.
    // TODO(ntv): auto-generate.
    ArrayAttr indexing_maps();

    // Rank-polymorphic.
    //   filling_value -> O(ivs) with parallel iterators.
    llvm::Optional<SmallVector<StringRef, 8>> referenceIterators() {
      unsigned nPar = input().getType().cast<ShapedType>().getRank();
      return SmallVector<StringRef, 8>(nPar, getParallelIteratorTypeName());
    }

    llvm::Optional<SmallVector<AffineMap, 8>> referenceIndexingMaps() {
      llvm_unreachable("NYI referenceIndexingMaps for CopyOp");
    }
  }];
  let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
}

def FillOp : LinalgStructured_Op<"fill", [NInputs<0>, NOutputs<1>]> {
  let arguments = (ins AnyStridedMemRef:$output,
                   AnyTypeOf<[AnyFloat, AnyInteger, AnyVector]>:$value);
  let extraClassDeclaration = libraryCallName # [{
    // Defined in C++ for now.
    // TODO(ntv): auto-generate.
    ArrayAttr indexing_maps();

    // Rank-polymorphic.
    //   filling_value -> O(ivs) with parallel iterators.
    llvm::Optional<SmallVector<StringRef, 8>> referenceIterators() {
      unsigned nPar = output().getType().cast<ShapedType>().getRank();
      return SmallVector<StringRef, 8>(nPar, getParallelIteratorTypeName());
    }

    llvm::Optional<SmallVector<AffineMap, 8>> referenceIndexingMaps() {
      llvm_unreachable("NYI referenceIndexingMaps for CopyOp");
    }
  }];
  let hasFolder = 1;
}

def DotOp : LinalgStructured_Op<"dot", [NInputs<2>, NOutputs<1>]> {
  let arguments = (ins AnyStridedMemRefOfRank<1>,
                       AnyStridedMemRefOfRank<1>,
                       AnyStridedMemRefOfRank<0>);
  let extraClassDeclaration = libraryCallName # [{
    // Defined in C++ for now.
    // TODO(ntv): auto-generate.
    ArrayAttr indexing_maps();

    llvm::Optional<SmallVector<StringRef, 8>> referenceIterators() {
      return SmallVector<StringRef, 8>{getReductionIteratorTypeName()};
    }

    llvm::Optional<SmallVector<AffineMap, 8>> referenceIndexingMaps() {
      llvm_unreachable("NYI referenceIndexingMaps for DotOp");
    }
  }];

  let hasFolder = 1;
}

def MatvecOp : LinalgStructured_Op<"matvec", [NInputs<2>, NOutputs<1>]> {
  let arguments = (ins AnyStridedMemRefOfRank<2>,
                       AnyStridedMemRefOfRank<1>,
                       AnyStridedMemRefOfRank<1>);
  let extraClassDeclaration = libraryCallName # [{
    // Defined in C++ for now.
    // TODO(ntv): auto-generate.
    ArrayAttr indexing_maps();

    llvm::Optional<SmallVector<StringRef, 8>> referenceIterators() {
      return SmallVector<StringRef, 8>{
        getParallelIteratorTypeName(),
        getReductionIteratorTypeName()};
    }

    llvm::Optional<SmallVector<AffineMap, 8>> referenceIndexingMaps() {
      llvm_unreachable("NYI referenceIndexingMaps for MatvecOp");
    }
  }];

  let hasFolder = 1;
}

def MatmulOp : LinalgStructured_Op<"matmul", [NInputs<2>, NOutputs<1>]> {
  let arguments = (ins AnyStridedMemRefOfRank<2>,
                       AnyStridedMemRefOfRank<2>,
                       AnyStridedMemRefOfRank<2>);
  let extraClassDeclaration = libraryCallName # [{
    // Defined in C++ for now.
    // TODO(ntv): auto-generate.
    ArrayAttr indexing_maps();

    llvm::Optional<SmallVector<StringRef, 8>> referenceIterators() {
      return SmallVector<StringRef, 8>{
        getParallelIteratorTypeName(),
        getParallelIteratorTypeName(),
        getReductionIteratorTypeName()};
    }

    llvm::Optional<SmallVector<AffineMap, 8>> referenceIndexingMaps() {
      llvm_unreachable("NYI referenceIndexingMaps for MatmulOp");
    }
  }];

  let hasFolder = 1;
}

def ConvOp : LinalgStructured_Op<"conv", [NInputs<2>, NOutputs<1>]> {
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

  // TODO(ntv) padding.
  // Following the TF source of truth above, strides and dilations are integer
  // attributes of the same rank as the number of window dimensions.
  let arguments = (ins AnyStridedMemRef:$filter, AnyStridedMemRef:$input,
                   AnyStridedMemRef:$output,
                   OptionalAttr<I64ArrayAttr>:$strides,
                   OptionalAttr<I64ArrayAttr>:$dilations);
  let extraClassDeclaration = libraryCallName # [{
    // TODO(ntv) extend to support more than 1 dimensions and potentially
    // grouping too.
    unsigned getNumBatchDimensions() { return 1; }
    unsigned getNumInputFeatureDimensions() { return 1; }
    unsigned getNumOutputFeatureDimensions() { return 1; }

    // Defined in C++ for now.
    // TODO(ntv): auto-generate.
    ArrayAttr indexing_maps();

    llvm::Optional<SmallVector<StringRef, 8>> referenceIterators() {
      // Outer parallel loops are always the number of output dimensions; i.e.
      // [b, xs, q] in the TF notation above.
      unsigned nPar = getOutputShapedType(0).getRank();
      unsigned nRed = getNumInputFeatureDimensions();
      // Window loops are a special kind of reduction that is never tiled or
      // parallelized across; i.e. [zs] in the TF notation above whose number
      // match `xs` (i.e. 1 window loop per "image" dimension).
      // This may evolve in the future.
      unsigned nWin =
        nPar - getNumBatchDimensions() - getNumInputFeatureDimensions();
      SmallVector<StringRef, 8> iters(nPar, getParallelIteratorTypeName());
      iters.reserve(nPar + nRed + nWin);
      iters.append(nRed, getReductionIteratorTypeName());
      iters.append(nWin, getWindowIteratorTypeName());
      return iters;
    }

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

    llvm::Optional<SmallVector<AffineMap, 8>> referenceIndexingMaps() {
      llvm_unreachable("NYI referenceIndexingMaps for MatmulOp");
    }
  }];

  let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
}

def LinalgOperand: Type<
  Or<[AnyRankedTensor.predicate, AnyStridedMemRef.predicate]>>;

class LinalgOperandOfRank<int rank>: Type<
  And<[
    LinalgOperand.predicate,
    CPred<"$_self.cast<ShapedType>().getRank() == " # rank>]
  >>;

////////////////////////////////////////////////////////////////////////////////
// Generic Linalg ops.
////////////////////////////////////////////////////////////////////////////////
class GenericOpBase<string mnemonic> : LinalgStructuredBase_Op<mnemonic, []> {
  let arguments = (ins Variadic<LinalgOperand>:$views,
                   I64Attr:$args_in,
                   I64Attr:$args_out,
                   AffineMapArrayAttr:$indexing_maps,
                   ArrayAttr:$iterator_types,
                   OptionalAttr<StrAttr>:$doc,
                   OptionalAttr<FlatSymbolRefAttr>:$fun,
                   OptionalAttr<StrAttr>:$library_call);
  let results = (outs Variadic<AnyRankedTensor>:$output_tensors);
  let regions = (region AnyRegion:$region);
  let extraClassDeclaration = [{
    SmallVector<StringRef, 8> linalgTraitAttrNames() {
      return SmallVector<StringRef, 8>{
        getArgsInAttrName(), getArgsOutAttrName(), getDocAttrName(),
        getFunAttrName(), getIndexingMapsAttrName(), getLibraryCallAttrName(),
        getIteratorTypesAttrName()
      };
    }

    unsigned getNumInputs() { return args_in().getSExtValue(); }

    unsigned getNumOutputs() { return args_out().getSExtValue(); }

    FuncOp getFunction() {
      auto moduleOp = getParentOfType<ModuleOp>();
      return fun().hasValue() ?
        moduleOp.lookupSymbol<FuncOp>(fun().getValue()) : FuncOp();
    }

    StringRef getLibraryCallName() {
      return library_call().hasValue() ? library_call().getValue() : "";
    }

    llvm::Optional<SmallVector<StringRef, 8>> referenceIterators() {
      llvm_unreachable(
        "No such thing as reference iterator types for a generic op.");
     }

    llvm::Optional<SmallVector<AffineMap, 8>> referenceIndexingMaps() {
      llvm_unreachable(
        "No such thing as reference indexing maps for a generic op.");
     }
  }];
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseGenericOp(parser, result); }];
}

/// Index-free GenericOp.
def GenericOp : GenericOpBase<"generic"> {
  let description = [{
    Generic Linalg op form where the key properties of the computation are
    specified as attributes. In pretty form, a linalg.generic op is written as:

      ```mlir
        linalg.generic #trait_attribute %A, %B, %C {other-attributes} :
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>
      ```

    Where #trait_attributes is an alias of a dictionary attribute containing:
      - args_in: an I64Attr representing the number of input (readonly) views
      - args_out: an I64Attr representing the number of output (readwrite) views
      - doc [optional]: a documentation string
      - fun: a FlatSymbolRefAttr that must resolve to an existing function
        symbol. To support inplace updates in a generic fashion, the signature
        of the function must be:
        ```
          fun([input views element types], [output views element types])
            -> ([output views element types])
        ```
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

    Example:
    Defining a #matmul_trait attribute in MLIR can be done as follows:
      ```mlir
        func @fma(%a: f32, %b: f32, %c: f32) -> f32 {
          %d = mulf %a, %b: f32
          %e = addf %c, %d: f32
          return %e: f32
        }
        #matmul_accesses = [
          (m, n, k) -> (m, k),
          (m, n, k) -> (k, n),
          (m, n, k) -> (m, n)
        ]
        #matmul_trait = {
          doc = "C(m, n) += A(m, k) * B(k, n)",
          fun = @fma,
          indexing_maps = #matmul_accesses,
          library_call = "linalg_matmul",
          n_views = [2, 1],
          iterator_types = ["parallel", "parallel", "reduction"]
        }
      ```

    And can be reused in multiple places as:
      ```mlir
        linalg.generic #matmul_trait %A, %B, %C [other-attributes] :
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>
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
    loop.for %m = %c0 to %M step %c1 {
      loop.for %n = %c0 to %N step %c1 {
        loop.for %k = %c0 to %K step %c1 {
          %a = linalg.load %A[%m, %k] : memref<?x?xf32, stride_specification>
          %b = linalg.load %B[%k, %n] : memref<?x?xf32, stride_specification>
          %c = linalg.load %C[%m, %n] : memref<?x?xf32, stride_specification>
          %d = call @func_of_elements(%a, %b, %c)
                 : (f32, f32, f32) -> (f32)
          linalg.store %d, %C[%m, %n] : memref<?x?x?xf32, stride_specification>
        }
      }
    }
    ```

    To allow progressive lowering from the value world (a.k.a tensor values) to
    the buffer world (a.k.a memref values), a `linalg.generic` op accepts
    mixing input and output ranked tensor values with input and output memrefs.

    ```mlir
      %C = linalg.generic #trait_attribute %A, %B {other-attributes} :
        tensor<?x?xf32>,
        memref<?x?xf32, stride_specification>
        -> (tensor<?x?xf32>)
    ```

    In this case, the number of outputs (args_out) must match the sum of (1) the
    number of output buffer operands and (2) the number of tensor return values.
    The semantics is that the `linalg.indexed_generic` op produces (i.e.
    allocates and fills) its tensor return values.

    Tensor values must be legalized by a buffer allocation pass before most
    transformations can be applied. Such legalization moves tensor return values
    into output buffer operands and updates the region arguments accordingly.

    Transformations that create control-flow around linalg.indexed_generic
    operations are not expected to work with tensors because SSA values do not
    escape naturally. Still, transformations and rewrites that take advantage of
    tensor SSA values are expected to be useful and will be added in the near
    future.
  }];

  let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
}

/// GenericOp with Indexing (i.e. multi-for style in which the region is passed
/// the enclosing loop induction variables)
def IndexedGenericOp : GenericOpBase<"indexed_generic"> {
  let description = [{
    Indexed Generic Linalg op form where the key properties of the computation
    are specified as attributes. In pretty form, a linalg.indexed_generic op is
    written as:

      ```mlir
        linalg.indexed_generic #trait_attribute %A, %B, %C {other-attributes} :
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>
      ```

    Where #trait_attributes is an alias of a dictionary attribute containing:
      - args_in: an I64Attr representing the number of input (readonly) views
      - args_out: an I64Attr representing the number of output (readwrite) views
      - doc [optional]: a documentation string
      - fun: a FlatSymbolRefAttr that must resolve to an existing function
        symbol. To support inplace updates in a generic fashion, the signature
        of the function must be:
        ```
          fun([index types of induction variables], [input views element types],
              [output views element types]) -> ([output views element types])
        ```
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
        func @fma(%offset_m: index, %offset_n: index, %offset_k: index,
                  %a: f32, %b: f32, %c: f32)
          -> f32
        {
          "some_optional_condition"(%offset_m, %offset_n, %offset_k)
          %d = mulf %a, %b: f32
          %e = addf %c, %d: f32
          return %e: f32
        }
        #matmul_accesses = [
          (m, n, k) -> (m, k),
          (m, n, k) -> (k, n),
          (m, n, k) -> (m, n)
        ]
        #matmul_trait = {
          doc = "C(m, n) += A(m, k) * B(k, n)",
          fun = @fma,
          indexing_maps = #matmul_accesses,
          library_call = "linalg_matmul",
          n_views = [2, 1],
          iterator_types = ["parallel", "parallel", "reduction"]
        }
      ```

    And can be reused in multiple places as:
      ```mlir
        linalg.indexed_generic #matmul_trait %A, %B, %C [other-attributes] :
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>
      ```

    This may lower to either:
      ```mlir
        call @linalg_matmul(%offset_m, %offset_n, %offset_k, %A, %B, %C) :
          (memref<?x?xf32, stride_specification>,
           memref<?x?xf32, stride_specification>,
           memref<?x?xf32, stride_specification>)
          -> ()
      ```

    or IR resembling:
    ```mlir
    loop.for %m = %c0 to %M step %c1 {
      loop.for %n = %c0 to %N step %c1 {
        loop.for %k = %c0 to %K step %c1 {
          %a = linalg.load %A[%m, %k] : memref<?x?xf32, stride_specification>
          %b = linalg.load %B[%k, %n] : memref<?x?xf32, stride_specification>
          %c = linalg.load %C[%m, %n] : memref<?x?xf32, stride_specification>
          %d = call @func_of_elements_and_indices(%m, %n, %k, %a, %b, %c)
                 : (index, index, index, f32, f32, f32) -> (f32)
          linalg.store %d, %C[%m, %n] : memref<?x?x?xf32, stride_specification>
        }
      }
    }
    ```

    To allow progressive lowering from the value world (a.k.a tensor values) to
    the buffer world (a.k.a memref values), a `linalg.indexed_generic` op
    accepts mixing input and output ranked tensor values with input and output
    memrefs.

    ```mlir
      %C = linalg.indexed_generic #trait_attribute %A, %B {other-attributes}
      : tensor<?x?xf32>,
        memref<?x?xf32, stride_specification>
        -> (tensor<?x?xf32>)
    ```

    In this case, the number of outputs (args_out) must match the sum of (1) the
    number of output buffer operands and (2) the number of tensor return values.
    The semantics is that the `linalg.indexed_generic` op produces (i.e.
    allocates and fills) its return values.

    Tensor values must be legalized by a buffer allocation pass before most
    transformations can be applied. Such legalization moves tensor return values
    into output buffer operands and updates the region argument accordingly.

    Transformations that create control-flow around linalg.indexed_generic
    operations are not expected to work with tensors because SSA values do not
    escape naturally. Still, transformations and rewrites that take advantage of
    tensor SSA values are expected to be useful and will be added in the near
    future.
  }];

  let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
}

#endif // LINALG_STRUCTURED_OPS
