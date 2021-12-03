//===- BufferizationOps.td - Bufferization op definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUFFERIZATION_OPS
#define BUFFERIZATION_OPS

include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizationBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"

class Bufferization_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Bufferization_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CloneOp
//===----------------------------------------------------------------------===//

def Bufferization_CloneOp : Bufferization_Op<"clone", [
    CopyOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<AllocationOpInterface, ["buildDealloc", "buildClone"]>
  ]> {
  let builders = [
    OpBuilder<(ins "Value":$value), [{
      return build($_builder, $_state, value.getType(), value);
    }]>];

  let description = [{
    Clones the data in the input view into an implicitly defined output view.

    Usage:

    ```mlir
    %arg1 = bufferization.clone %arg0 : memref<?xf32> to memref<?xf32>
    ```

    Valid implementations of this operation may alias the input and output
    views or create an actual copy. Mutating the source or result
    of the clone operation after the clone operation thus leads to undefined
    behavior.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "", []>:$input);
  let results = (outs Arg<AnyRankedOrUnrankedMemRef, "", []>:$output);

  let extraClassDeclaration = [{
    Value getSource() { return input(); }
    Value getTarget() { return output(); }
  }];

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  let hasFolder = 1;
  let verifier = ?;
  let hasCanonicalizer = 1;
}
//===----------------------------------------------------------------------===//
// ToTensorOp
//===----------------------------------------------------------------------===//

def Bufferization_ToTensorOp : Bufferization_Op<"to_tensor",
    [SameOperandsAndResultShape, SameOperandsAndResultElementType,
     TypesMatchWith<"result type matches tensor equivalent of 'memref'",
                    "memref", "result",
                    "memref::getTensorTypeFromMemRefType($_self)">]> {
  let summary = "memref to tensor operation";
  let description = [{
    Create a tensor from a memref, making an independent copy of the element
    data. The result value is a tensor whose shape and element type match the
    memref operand.

    The opposite of this op is to_memref. Together, these two ops are
    useful for source/target materializations when doing type conversions
    involving tensors and memrefs.

    Example:

    ```mlir
    // Produces a value of tensor<4x?xf32> type.
    %12 = bufferization.to_tensor %10 : memref<4x?xf32, #layout, memspace0>
    ```

    If tensor load is used in the bufferization steps, mutating the source
    buffer after loading leads to undefined behavior.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef,
                       "the reference to load from", [MemRead]>:$memref);
  let results = (outs AnyTensor:$result);
  // MemrefToTensor is fully verified by traits.
  let verifier = ?;

  let builders = [
    OpBuilder<(ins "Value":$memref), [{
      $_state.addOperands(memref);
      $_state.addTypes(memref::getTensorTypeFromMemRefType(memref.getType()));
    }]>];

  let extraClassDeclaration = [{
    /// The result of a to_tensor is always a tensor.
    TensorType getType() {
      Type resultType = getResult().getType();
      if (resultType.isa<TensorType>())
        return resultType.cast<TensorType>();
      return {};
    }
  }];

  let assemblyFormat = "$memref attr-dict `:` type($memref)";

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// ToMemrefOp
//===----------------------------------------------------------------------===//

def Bufferization_ToMemrefOp : Bufferization_Op<"to_memref",
    [SameOperandsAndResultShape, SameOperandsAndResultElementType, NoSideEffect,
     TypesMatchWith<"type of 'tensor' is the tensor equivalent of 'memref'",
                    "memref", "tensor",
                    "memref::getTensorTypeFromMemRefType($_self)">]> {
  let summary = "tensor to memref cast operation";
  let description = [{
    Casts a tensor to a memref.

    ```mlir
    // Result type is tensor<4x?xf32>
    %12 = bufferization.to_memref %10 : memref<4x?xf32, #map0, 42>
    ```

    Note, that mutating the result of the to_memref operation leads to
    undefined behavior.

    This operation is a specialized variant of the built-in
    unrealized_conversion_cast and is intended for use in the context of
    gradual bufferization.
  }];

  let arguments = (ins AnyTensor:$tensor);
  let results = (outs AnyRankedOrUnrankedMemRef:$memref);
  // This op is fully verified by traits.
  let verifier = ?;

  let assemblyFormat = "$tensor attr-dict `:` type($memref)";

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

#endif // BUFFERIZATION_OPS
