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
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

// Base Tablegen class for Linalg ops.
// Linalg ops that correspond to library calls operate on ShapedType as their
// first operands. These may be optionally followed by non-view operands
// depending on the specific Linalg op.
class LinalgStructuredBase_Op<string mnemonic, list<Trait> props>
  : Op<Linalg_Dialect, mnemonic, !listconcat([
       SingleBlockImplicitTerminator<"YieldOp">,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DeclareOpInterfaceMethods<ConditionallySpeculatable>,
       RecursiveMemoryEffects,
       DestinationStyleOpInterface,
       LinalgStructuredInterface,
       ReifyRankedShapedTypeOpInterface], props)> {
  code structuredOpsBaseDecls = [{
    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      return !this->getBody()->getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  }];
}

//===----------------------------------------------------------------------===//
// Generic Linalg ops.
//===----------------------------------------------------------------------===//

def GenericOp : LinalgStructuredBase_Op<"generic", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    AttrSizedOperandSegments]> {
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
          %d = arith.mulf %a, %b: f32
          %e = arith.addf %c, %d: f32
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
          %d = arith.mulf %a, %b: f32
          %e = arith.addf %c, %d: f32
          store %e, %C[%m, %n] : memref<?x?x?xf32, stride_specification>
        }
      }
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       AffineMapArrayAttr:$indexing_maps,
                       IteratorTypeArrayAttr:$iterator_types,
                       OptionalAttr<StrAttr>:$doc,
                       OptionalAttr<StrAttr>:$library_call);
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
  let regions = (region AnyRegion:$region);

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
      "ValueRange":$outputs, "ArrayAttr":$indexingMaps,
      "ArrayAttr":$iteratorTypes, "StringAttr":$doc,
      "StringAttr":$libraryCall,
      "function_ref<void(OpBuilder &, Location, ValueRange)>",
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
      "ValueRange":$outputs, "ArrayRef<AffineMap>":$indexingMaps,
      "ArrayRef<utils::IteratorType>":$iteratorTypes, "StringRef":$doc,
      "StringRef":$libraryCall,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputBuffers,
      "ArrayRef<AffineMap>":$indexingMaps, "ArrayRef<utils::IteratorType>":$iteratorTypes,
      "StringRef":$doc, "StringRef":$libraryCall,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
      "ValueRange":$outputs, "ArrayRef<AffineMap>":$indexingMaps,
      "ArrayRef<utils::IteratorType>":$iteratorTypes,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputBuffers,
      "ArrayRef<AffineMap>":$indexingMaps, "ArrayRef<utils::IteratorType>":$iteratorTypes,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    SmallVector<StringRef, 8> linalgTraitAttrNames() {
      return SmallVector<StringRef, 8>{
        getDocAttrName(),
        getIndexingMapsAttrName(), getLibraryCallAttrName(),
        getIteratorTypesAttrName(),
      };
    }
    std::string getLibraryCallName() {
      return getLibraryCall() ?
        getLibraryCall()->str() : "op_has_no_registered_library_name";
    }

    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>,
                              function_ref<InFlightDiagnostic()>)>
    getRegionBuilder() {
      return nullptr;
    }

    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    // Return true only if GenericOp has a single input and single
    // output, and the body is a single yieldOp that yields the input.
    // This check is useful when trying to determine if the op is
    // essentially a transpose, broadcast, copy or something like that.
    bool isSingleYieldOp() {
      if (!isSingleInputOutput())
        return false;
     Block *body = getBody();
     if (body->getOperations().size() != 1)
       return false;

     auto yieldOp = dyn_cast<linalg::YieldOp>(body->back());
       if (!yieldOp || yieldOp.getNumOperands() != 1 ||
           yieldOp->getOperand(0) != body->getArgument(0))
         return false;
     return true;
   }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Map op.
//===----------------------------------------------------------------------===//

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "", "::mlir::ShapedType">;

def MapOp : LinalgStructuredBase_Op<"map", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Elementwise operations";
  let description = [{
    Models elementwise operations on tensors in terms of arithmetic operations
    on the corresponding elements.

    Example:
    ```mlir
      %add = linalg.map
          ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
          outs(%init: tensor<64xf32>)
          (%lhs_elem: f32, %rhs_elem: f32) {
            %0 = arith.addf %lhs_elem, %rhs_elem: f32
            linalg.yield %0: f32
          }
    ```

    Shortened print form is available. Applies to simple maps with one
    non-yield operation inside the body.

    The example above will be printed as:
    ```mlir
      %add = linalg.map { arith.addf }
          ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
          outs(%init: tensor<64xf32>)
    ```
  }];

  let arguments = (ins
    // Input args
    Variadic<TensorOrMemref>:$inputs,

    // Output arg
    TensorOrMemref:$init
  );
  let results = (outs Variadic<AnyTensor>:$result);
  let regions = (region SizedRegion<1>:$mapper);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$init,
      "function_ref<void(OpBuilder &, Location, ValueRange)>",
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Implement functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Implement functions necessary for DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }

    SmallVector<OpOperand *> getOpOperandsMatchingBBargs() {
      return getDpsInputOperands();
    }

    bool payloadUsesValueFromOperand(OpOperand * opOperand) {
      if (isDpsInit(opOperand)) return false;
      return !getMatchingBlockArgument(opOperand).use_empty();
    }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>,
                              function_ref<InFlightDiagnostic()>)>
    getRegionBuilder() {
      return nullptr;
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Reduce op.
//===----------------------------------------------------------------------===//

def ReduceOp : LinalgStructuredBase_Op<"reduce", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    SameVariadicOperandSize,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Reduce operator";
  let description = [{
    Executes `combiner` on the `dimensions` of `inputs` and returns the
    reduced result. The `dimensions` attribute needs to list the reduction
    dimensions in increasing order.

    Example:
    ```mlir
      %reduce = linalg.reduce
          ins(%input:tensor<16x32x64xf32>)
          outs(%init:tensor<16x64xf32>)
          dimensions = [1]
          (%in: f32, %out: f32) {
            %0 = arith.addf %out, %in: f32
            linalg.yield %0: f32
          }
    ```

    Shortened print form is available. Applies to simple (not variadic) reduces
    with one non-yield operation inside the body. Applies only if the operation
    takes `%out` as the first argument.

    The example above will be printed as:
    ```mlir
          %reduce = linalg.reduce { arith.addf }
          ins(%input:tensor<16x32x64xf32>)
          outs(%init:tensor<16x64xf32>)
          dimensions = [1]
    ```
  }];

  let arguments = (ins
    // Input arg
    Variadic<TensorOrMemref>:$inputs,
    // Output arg
    Variadic<TensorOrMemref>:$inits,

    ConfinedAttr<DenseI64ArrayAttr,
                 [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$dimensions
  );
  let results = (outs Variadic<AnyTensor>);
  let regions = (region SizedRegion<1>:$combiner);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$inits,
      "ArrayRef<int64_t>":$dimensions,
      "function_ref<void(OpBuilder &, Location, ValueRange)>",
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Implement functions necessary for DestinationStyleOpInterface.
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>,
                              function_ref<InFlightDiagnostic()>)>
    getRegionBuilder() {
      return nullptr;
    }
    MutableOperandRange getDpsInitsMutable() { return getInitsMutable(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Transpose op.
//===----------------------------------------------------------------------===//

def TransposeOp : LinalgStructuredBase_Op<"transpose", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Transpose operator";
  let description = [{
    Permutes the dimensions of `input` according to the given `permutation`.
      `dim(result, i) = dim(input, permutation[i])`

    This op actually moves data, unlike `memref.transpose` which is a metadata
    operation only that produces a transposed "view".

    Example:
    ```mlir
      %transpose = linalg.transpose
          ins(%input:tensor<16x64xf32>)
          outs(%init:tensor<64x16xf32>)
          permutation = [1, 0]
    ```
  }];

  let arguments = (ins
    // Input arg
    TensorOrMemref:$input,
    // Output arg
    TensorOrMemref:$init,

    DenseI64ArrayAttr:$permutation
  );
  let results = (outs Variadic<AnyTensor>:$result);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$init,
        "DenseI64ArrayAttr":$permutation, CArg<"ArrayRef<NamedAttribute>",
        "{}">:$attributes)>,
    OpBuilder<(ins "Value":$input, "Value":$init,
        "ArrayRef<int64_t>":$permutation, CArg<"ArrayRef<NamedAttribute>",
        "{}">:$attributes)>,
  ];

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Implement functions necessary for DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }

    static void regionBuilder(mlir::ImplicitLocOpBuilder &b, mlir::Block &block,
        mlir::ArrayRef<mlir::NamedAttribute>, function_ref<InFlightDiagnostic()> emitError) {
      OpBuilder::InsertionGuard guard(b);
      linalg::YieldOp::create(b, b.getLoc(), block.getArgument(0));
    }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>,
                              function_ref<InFlightDiagnostic()>)>
      getRegionBuilder() {
      return regionBuilder;
    }
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Broadcast op.
//===----------------------------------------------------------------------===//

def BroadcastOp : LinalgStructuredBase_Op<"broadcast", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Static broadcast operator";
  let description = [{
    Broadcast the input into the given shape by adding `dimensions`.

    Example:
    ```mlir
      %bcast = linalg.broadcast
          ins(%input:tensor<16xf32>)
          outs(%init:tensor<16x64xf32>)
          dimensions = [1]
    ```
  }];

  let arguments = (ins
    // Input arg
    TensorOrMemref:$input,
    // Output arg
    TensorOrMemref:$init,

    DenseI64ArrayAttr:$dimensions
  );
  let results = (outs Variadic<AnyTensor>:$result);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$init,
        "DenseI64ArrayAttr":$dimensions, CArg<"ArrayRef<NamedAttribute>",
        "{}">:$attributes)>,
    OpBuilder<(ins "Value":$input, "Value":$init,
        "ArrayRef<int64_t>":$dimensions, CArg<"ArrayRef<NamedAttribute>",
        "{}">:$attributes)>,
  ];

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Implement functions necessary for DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }

    static void regionBuilder(mlir::ImplicitLocOpBuilder &b, mlir::Block &block,
                              mlir::ArrayRef<mlir::NamedAttribute>, 
                              function_ref<InFlightDiagnostic()> emitError) {
      OpBuilder::InsertionGuard guard(b);
      linalg::YieldOp::create(b, b.getLoc(), block.getArgument(0));
    }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>,
                              function_ref<InFlightDiagnostic()>)>
      getRegionBuilder() {
      return regionBuilder;
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Op definition for ElementwiseOp
//===----------------------------------------------------------------------===//
def ElementwiseOp : LinalgStructuredBase_Op<"elementwise", [
                   AttrSizedOperandSegments]> {
  let summary = [{ Performs element-wise operation }];
  let description = [{
    The attribute `kind` describes arithmetic operation to perform. The
    operation kind can be unary (e.g. max), binary (e.g. add) or ternary
    (e.g. select).

    By default, all indexing maps are identities. In the case of default
    indexing map, all input and output shapes must match. The number of dims in
    each of the identity maps is equal to the rank of the output type.

    Affine-maps for operands and result are required to be provided by the user
    when a transpose and/or broadcast is needed on any operand. When a map is not
    provided, default identity maps are inferred for each operand.

    Iterator-types are always all `parallel`.
    Iterator-types are needed for constructing the underlying structured op.

    The number of dims of the iterator-types are inferred from the rank of
    the result type.

    Example:

    Defining a unary linalg.elementwise with default indexing-map:
    ```mlir
    %exp = linalg.elementwise
        kind=#linalg.elementwise_kind<exp>
        ins(%x : tensor<4x16x8xf32>)
        outs(%y: tensor<4x16x8xf32>) -> tensor<4x16x8xf32>
    ```

    Defining a binary linalg.elementwise with user-defined indexing-map:
    ```mlir
    %add = linalg.elementwise
        kind=#linalg.elementwise_kind<add>
        indexing_maps = [#transpose, #broadcast, #identity]
        ins(%exp, %arg1 : tensor<4x16x8xf32>, tensor<4x16xf32>)
        outs(%arg2: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    ```
  }];

  let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
      ElementwiseKindAttr:$kind,
      DefaultValuedOptionalAttr<AffineMapArrayAttr, "{}">:$indexing_maps
    );

  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
  let regions = (region AnyRegion:$region);
  let skipDefaultBuilders = 1;

  let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, ElementwiseOp::getRegionBuilder());
      }]>,

     OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
          "ElementwiseKindAttr":$kind,
          "ArrayAttr":$indexingMaps,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addAttribute("kind", kind);
        $_state.addAttribute("indexing_maps", indexingMaps);
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
                          attributes, ElementwiseOp::getRegionBuilder());
       }]>
    ];

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = structuredOpsBaseDecls # [{
      /// Get the arity enum corresponding to the kind of op, e.g. if arg is
      /// `ElementwiseKind::add`, return `ElementwiseArityGroup::Binary`.
      static ElementwiseArityGroup getArityGroup(ElementwiseKind n);

      /// Both user-specified and default indexing map will always depend on
      /// the current Op instance.
      static bool hasDynamicIndexingMaps() { return true; }

      /// Implements the block region builder for the elementwiseOp. This is
      /// called by the 'fillStructuredOpRegion'.
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs,
                                function_ref<InFlightDiagnostic()> emitError);

      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>,
                                function_ref<InFlightDiagnostic()>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      /// Returns rank of the result tensor/memref. Useful for knowing
      /// the dimensionality of the iteration space when others means
      /// are not possible e.g. absence of user-provided indexing map.
      unsigned getResultRank() {
        Value output = getDpsInitOperand(0)->get();
        ShapedType shapedType = llvm::cast<ShapedType>(output.getType());
        return shapedType.getRank();
      }

      /// Returns N 'parallel' iterator types where N is rank of result.
      SmallVector<utils::IteratorType> getIteratorTypesArray();

      /// The default indexing maps are identities.
      /// There will be N+1 such maps, where N is the arity of the Op.
      static SmallVector<AffineMap>
      getDefaultIndexingMaps(unsigned NumMaps, unsigned numDims,
                             MLIRContext *context);

      /// Destination passing style interface method.
      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      std::string getLibraryCallName() {
        return generateLibraryCallName(getOperation());
      }
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MatmulOp
//===----------------------------------------------------------------------===//

def MatmulOp : LinalgStructuredBase_Op<"matmul", [
               AttrSizedOperandSegments,
               LinalgContractionOpInterface]> {

  let summary = [{
    Performs a matrix multiplication of two 2D inputs without broadcast or transpose.
    }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply,
    promoting them to the same data type as the accumulator/output.

    Broadcast and Transpose semantics can be appiled by specifying the explicit attribute
    'indexing_maps' as shown below.This is a list attribute, so the list must include all
    the maps if specified.

    Example Transpose:
    ```mlir
    linalg.matmul
        indexing_maps = [affine_map<(m, n, k) -> (k, m)>, // transpose
                         affine_map<(m, n, k) -> (k, n)>,
                         affine_map<(m, n, k) -> (m, n)>]
        ins(%arg0, %arg1 : memref<5x3xf32>,memref<5x7xf32>)
        outs(%arg2: memref<3x7xf32>)
     ```

    Example Broadcast:
     ```mlir
    linalg.matmul
        indexing_maps = [affine_map<(m, n, k) -> (k)>,     // broadcast
                         affine_map<(m, n, k) -> (k, n)>,
                         affine_map<(m, n, k) -> (m, n)>]
        ins(%arg0, %arg1 : memref<3xf32>, memref<5x7xf32>)
        outs(%arg2: memref<3x7xf32>)
    ```

    Example Broadcast and transpose:
    ```mlir
    linalg.matmul
        indexing_maps = [affine_map<(m, n, k) -> (k, m)>, // transpose
                         affine_map<(m, n, k) -> (k)>,    // broadcast
                         affine_map<(m, n, k) -> (m, n)>]
        ins(%arg0, %arg1 : memref<5x3xf32>, memref<7xf32>)
        outs(%arg2: memref<3x7xf32>)
    ```
    }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
      DefaultValuedOptionalAttr<
        AffineMapArrayAttr,
        "MatmulOp::getDefaultIndexingMaps($_builder.getContext())"
      >:$indexing_maps,
      DefaultValuedOptionalAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildMatmulOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, MatmulOp::getRegionBuilder(),
          MatmulOp::getDefaultIndexingMaps($_builder.getContext()));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildMatmulOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, MatmulOp::getRegionBuilder(),
          MatmulOp::getDefaultIndexingMaps($_builder.getContext()));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs,
       "Attribute":$cast, CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addAttribute("cast", cast);
        buildMatmulOp($_builder, $_state, resultTensorTypes, inputs, outputs,
          attributes, MatmulOp::getRegionBuilder(),
          MatmulOp::getDefaultIndexingMaps($_builder.getContext()));
      }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      SmallVector<utils::IteratorType> getIteratorTypesArray();

      /// Implements the block region builder.
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs,
                                function_ref<InFlightDiagnostic()> emitError);

      /// Returns a list of AffineMap with the default matmul indexing charactristic.
      static SmallVector<AffineMap> getDefaultIndexingMaps(MLIRContext *context);

      /// Returns true if the given broadcast map \p bcastMap is valid for this op.
      bool isValidLhsRhsBroadcastMap(AffineMap bcastMap);

      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>,
                                function_ref<InFlightDiagnostic()>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      bool hasDynamicIndexingMaps();
      /// Returns true if the user defined indexing maps are not equal to default maps.
      bool hasUserDefinedMaps();
    }];
}

//===----------------------------------------------------------------------===//
// Contract op.
//===----------------------------------------------------------------------===//

def ContractOp : LinalgStructuredBase_Op<"contract", [
               AttrSizedOperandSegments,
               LinalgContractionOpInterface]> {
  let summary = [{
    Perform a contraction on two inputs, accumulating into the third.
  }];
  let description = [{
    The semantics of contracting inputs `A` and `B` on top of `C` to produce
    output `D` is given by

      `D[H] = (SUM_{(I ∪ J) \ H} A[I] * B[J]) + C[H]`

    where `I`, `J`, and `H` are tuples of (pairwise distinct) dimension
    identifiers - meant to range over valid indices - corresponding to the
    results of the mandatory (projected permutation) `indexing_maps` for `A`,
    `B` and `C`. `SUM_{dims}` means reduce over all valid indices for the
    dimensions in the set `dims` (with `I`, `J`, and `K` treated as _sets_ of
    dim identifiers).

    The iteration space consists of all dimensions in `I`, `J` and `H`, i.e. the
    domain of each of the `affine_map`s. Like for einsums, the iteration type of
    each dim is inferred and is either:

    - reduction: the dim is used to index into `A` and `B` but not `C`. Per the
      above semantics, these dims will be contracted, i.e. reduced over.

    - parallel: the dim is used to index into `C` and at least one of `A` and
      `B`, and - deriving from matmul terminology - is either an "M-like" dim
      (if used on `A` and `C`), an "N-like" dim (if used on `B` and `C`) or a
      "batch"-dim (if used to index into `A`, `B`, and `C`).

    For example, batch-matmul is given by `I = ⟨ b, m, k ⟩`, `J = ⟨ b, k, n ⟩`,
    `H = ⟨ b, m, n ⟩` (with `k` as a contracting reduction-dimension while `m`,
    `n` and `b` have parallel iteration-type) and gets represented as:

    ```mlir
    %D = linalg.contract
        indexing_maps = [affine_map<(batch, m, n, k) -> (batch, m, k)>,
                         affine_map<(batch, m, n, k) -> (batch, k, n)>,
                         affine_map<(batch, m, n, k) -> (batch, m, n)>]
        ins(%A, %B: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%C: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    ```

    Note that by permuting dims in the `affine_map`s' results, accesses to
    to the inputs and output can be arbitrarily transposed. Similarly, arbitrary
    broadcasts can be achieved through leaving out dims on either input operand.
    For example, the following is a variant of batch-matmul with a transposition
    applied to `A` while `B`'s 2D-matrix gets broadcasted along the batch dim:

    ```mlir
    linalg.contract
        indexing_maps = [affine_map<(batch, m, n, k) -> (batch, k, m)>,
                         affine_map<(batch, m, n, k) -> (k, n)>,
                         affine_map<(batch, m, n, k) -> (batch, m, n)>]
        ins(%A, %B: memref<?x?x?xf32>, memref<?x?xf32>)
        outs(%C: memref<?x?x?xf32>)
    ```

    Numeric casting is performed on the operands to the inner multiplication,
    promoting/truncating them to the same data type as the accumulator/output.

    TODO: Allow control over the combining/accumulating op and possibly the
          multiplication op.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs,
    AffineMapArrayAttr:$indexing_maps,
    DefaultValuedOptionalAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
  );
  let results = (outs Variadic<AnyShaped>:$result_tensors);
  // NB: The only reason this op has a region - and it get populated at op build
  //     time - is that currently the LinalgOp interface exposes methods that
  //     assume a relevant region is available to be queried at any time.
  let regions = (region SizedRegion<1>:$combiner);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
      "ValueRange":$outputs, "ArrayAttr":$indexingMaps,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addAttribute("indexing_maps", indexingMaps);
        buildStructuredOp($_builder, $_state, resultTensorTypes, inputs,
                          outputs, attributes, regionBuilder);
      }]>,
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
      "ArrayAttr":$indexingMaps,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addAttribute("indexing_maps", indexingMaps);
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
                          attributes, regionBuilder);
      }]>
  ];
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare/implement functions necessary for LinalgStructuredInterface.

    /// Infer iterator types for each dim in the domain of IndexingMaps.
    SmallVector<utils::IteratorType> getIteratorTypesArray();

    /// IndexingMaps always depends on attr associated to current Op instance.
    bool hasDynamicIndexingMaps() { return true; };
    bool hasUserDefinedMaps() { return true; };

    static unsigned getNumRegionArgs();

    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs,
                              function_ref<InFlightDiagnostic()> emitError);

    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>,
                              function_ref<InFlightDiagnostic()>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Implement function necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchMatmulOp
//===----------------------------------------------------------------------===//

def BatchMatmulOp : LinalgStructuredBase_Op<"batch_matmul", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a batched matrix multiplication of two 3D inputs.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.

    Broadcast and Transpose semantics can be appiled by specifying the explicit attribute
    'indexing_maps' as shown below. This is a list attribute, so must include maps for all
    arguments if specified.

    Example Transpose:
    ```mlir
    linalg.batch_matmul
        indexing_maps = [affine_map<(batch, m, n, k) -> (batch, k, m)>, // transpose
                         affine_map<(batch, m, n, k) -> (batch, k, n)>,
                         affine_map<(batch, m, n, k) -> (batch, m, n)>]
        ins(%arg0, %arg1 : memref<2x5x3xf32>,memref<2x5x7xf32>)
        outs(%arg2: memref<2x3x7xf32>)
    ```

    Example Broadcast:
    ```mlir
    linalg.batch_matmul
        indexing_maps = [affine_map<(batch, m, n, k) -> (k)>,           // broadcast
                         affine_map<(batch, m, n, k) -> (batch, k, n)>,
                         affine_map<(batch, m, n, k) -> (batch, m, n)>]
        ins(%arg0, %arg1 : memref<5xf32>, memref<2x5x7xf32>)
        outs(%arg2: memref<2x3x7xf32>)
    ```

    Example Broadcast and Transpose:
    ```mlir
    linalg.batch_matmul
        indexing_maps = [affine_map<(batch, m, n, k) -> (m, k)>,        // broadcast
                         affine_map<(batch, m, n, k) -> (batch, n, k)>, // transpose
                         affine_map<(batch, m, n, k) -> (batch, m, n)>]
        ins(%arg0, %arg1 : memref<3x5xf32>, memref<2x7x5xf32>)
        outs(%arg2: memref<2x3x7xf32>)
    ```
}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
      DefaultValuedOptionalAttr<
        AffineMapArrayAttr,
        "BatchMatmulOp::getDefaultIndexingMaps($_builder.getContext())"
      >:$indexing_maps,
      DefaultValuedOptionalAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildBatchMatmulOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, BatchMatmulOp::getRegionBuilder(),
          BatchMatmulOp::getDefaultIndexingMaps($_builder.getContext()));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildBatchMatmulOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, BatchMatmulOp::getRegionBuilder(),
          BatchMatmulOp::getDefaultIndexingMaps($_builder.getContext()));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
          "Attribute":$cast, CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttribute("cast", cast);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion(),
        BatchMatmulOp::getDefaultIndexingMaps($_builder.getContext());
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{

      SmallVector<utils::IteratorType> getIteratorTypesArray();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs,
                                function_ref<InFlightDiagnostic()> emitError);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>,
                                function_ref<InFlightDiagnostic()>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      /// Returns a list with default AffineMap(s), i.e. without broadcasts and transpositions.
      static SmallVector<AffineMap> getDefaultIndexingMaps(MLIRContext *context);

      /// Returns true if the given broadcast map \p bcastMap is valid for this op.
      bool isValidLhsRhsBroadcastMap(AffineMap bcastMap, bool isLHS = true);

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      bool hasDynamicIndexingMaps() { return true; }
      std::string getLibraryCallName();
      /// Returns true if the user defined indexing maps are not equal to default maps.
      bool hasUserDefinedMaps();
    }];
}


//===----------------------------------------------------------------------===//
// Op definition for BatchReduceMatmulOp
//===----------------------------------------------------------------------===//

def BatchReduceMatmulOp : LinalgStructuredBase_Op<"batch_reduce_matmul", [
                          AttrSizedOperandSegments,
                          LinalgContractionOpInterface]> {
    
  let summary = [{Performs a batch-reduce matrix multiplication on two inputs.
    The partial multiplication results are reduced into a 2D output.}];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply,
    promoting them to the same data type as the accumulator/output.

    Broadcast and Transpose semantics can be applied by specifying the explicit attribute
    'indexing_maps' as shown below. This is a list attribute, so must include maps for all
    arguments if specified.

    Example Transpose:
    ```mlir
    linalg.batch_reduce_matmul
        indexing_maps = [affine_map<(batch, m, n, k) -> (batch, k, m)>, // transpose
                         affine_map<(batch, m, n, k) -> (batch, k, n)>,
                         affine_map<(batch, m, n, k) -> (m, n)>]
        ins(%arg0, %arg1 : memref<2x5x3xf32>,memref<2x5x7xf32>)
        outs(%arg2: memref<3x7xf32>)
    ```

    Example Broadcast:
    ```mlir
    linalg.batch_reduce_matmul
        indexing_maps = [affine_map<(batch, m, n, k) -> (k)>,         // broadcast
                         affine_map<(batch, m, n, k) -> (batch, k, n)>,
                         affine_map<(batch, m, n, k) -> (m, n)>]
        ins(%arg0, %arg1 : memref<5xf32>, memref<2x5x7xf32>)
        outs(%arg2: memref<3x7xf32>)
    ```

    Example Broadcast and Transpose:
    ```mlir
    linalg.batch_reduce_matmul
        indexing_maps = [affine_map<(batch, m, n, k) -> (m, k)>,        // broadcast
                         affine_map<(batch, m, n, k) -> (batch, n, k)>, // transpose
                         affine_map<(batch, m, n, k) -> (m, n)>]
        ins(%arg0, %arg1 : memref<3x5xf32>, memref<2x7x5xf32>)
        outs(%arg2: memref<3x7xf32>)
    ```
    }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
      DefaultValuedOptionalAttr<
        AffineMapArrayAttr,
        "BatchReduceMatmulOp::getDefaultIndexingMaps($_builder.getContext())"
      >:$indexing_maps,
      DefaultValuedOptionalAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildBatchReduceMatmulOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, BatchReduceMatmulOp::getRegionBuilder(),
          BatchReduceMatmulOp::getDefaultIndexingMaps($_builder.getContext()));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildBatchReduceMatmulOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, BatchReduceMatmulOp::getRegionBuilder(),
          BatchReduceMatmulOp::getDefaultIndexingMaps($_builder.getContext()));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs,
       "Attribute":$cast, CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addAttribute("cast", cast);
        buildBatchReduceMatmulOp($_builder, $_state, resultTensorTypes, inputs, outputs,
          attributes, BatchReduceMatmulOp::getRegionBuilder(),
          BatchReduceMatmulOp::getDefaultIndexingMaps($_builder.getContext()));
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      SmallVector<utils::IteratorType> getIteratorTypesArray();

      /// Implements the block region builder.
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs,
                                function_ref<InFlightDiagnostic()> emitError);

      /// Returns a list of AffineMap with the default batch_reduce_matmul indexing charactristic.
      static SmallVector<AffineMap> getDefaultIndexingMaps(MLIRContext *context);

      /// Returns true if the given broadcast map \p bcastMap is valid for this op.
      bool isValidLhsRhsBroadcastMap(AffineMap bcastMap, bool isLHS = true);

      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>,
                              function_ref<InFlightDiagnostic()>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      bool hasDynamicIndexingMaps() { return true; };
      /// Returns true if the user defined indexing maps are not equal to default maps.
      bool hasUserDefinedMaps();
    }];
}

//===----------------------------------------------------------------------===//
// Named Linalg ops, implemented as a declarative configurations of generic ops.
//===----------------------------------------------------------------------===//

include "mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yamlgen.td"

#endif // LINALG_STRUCTURED_OPS
