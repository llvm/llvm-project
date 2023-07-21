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
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return nullptr;
    }
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - getOutputs().size(), getNumOperands};
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
    ```
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
    ```
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
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - 1, getNumOperands};
    }
    OpOperandVector getOpOperandsMatchingBBargs() {
      return getDpsInputOperands();
    }

    bool payloadUsesValueFromOperand(OpOperand * opOperand) {
      if (isDpsInit(opOperand)) return false;
      return !getMatchingBlockArgument(opOperand).use_empty();
    }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
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
    ```
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
    ```
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
                              mlir::ArrayRef<mlir::NamedAttribute>)>
    getRegionBuilder() {
      return nullptr;
    }
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      return {getInits().size(), getNumOperands()};
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Transpose op.
//===----------------------------------------------------------------------===//

def TransposeOp : LinalgStructuredBase_Op<"transpose", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    SameVariadicOperandSize,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Transpose operator";
  let description = [{
    Permutes the dimensions of `input` according to the given `permutation`.
      `dim(result, i) = dim(input, permutation[i])`

    This op actually moves data, unlike `memref.transpose` which is a metadata
    operation only that produces a transposed "view".

    Example:
    ```
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
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - 1, getNumOperands};
    }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
        mlir::ArrayRef<mlir::NamedAttribute>)>
      getRegionBuilder() {
      return nullptr;
    }

    static void createRegion(::mlir::OpBuilder &opBuilder,
                             ::mlir::OperationState & odsState);
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Broadcast op.
//===----------------------------------------------------------------------===//

def BroadcastOp : LinalgStructuredBase_Op<"broadcast", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    SameVariadicOperandSize,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Static broadcast operator";
  let description = [{
    Broadcast the input into the given shape by adding `dimensions`.

    Example:
    ```
      %bcast = linalg.broadcast
          ins(%input:tensor<16xf32>)
          inits(%init:tensor<16x64xf32>)
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
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - 1, getNumOperands};
    }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
        mlir::ArrayRef<mlir::NamedAttribute>)>
      getRegionBuilder() {
      return nullptr;
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Named Linalg ops, implemented as a declarative configurations of generic ops.
//===----------------------------------------------------------------------===//

include "mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yamlgen.td"

#endif // LINALG_STRUCTURED_OPS
