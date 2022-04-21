//===- TransformOps.td - Transform dialect operations ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS

include "mlir/IR/OpAsmInterface.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"

def SequenceOp : TransformDialectOp<"sequence",
    [DeclareOpInterfaceMethods<TransformOpInterface>, OpAsmOpInterface,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">]> {
  let summary = "Contains a sequence of other transform ops to apply";
  let description = [{
    The transformations indicated by the sequence are applied in order of their
    appearance. Each value produced by a transformation within the sequence
    corresponds to an operation or a group of operations in the payload IR.
    Each value may be used at most once by another transformation operation as
    the transformation is likely to replace the transformed operation with
    another operation or a group thereof. In such cases, the transformation
    operation is expected to produce a new value to denote the newly produced
    operations that can be transformed further. During application, if any
    transformation in the sequence fails, the entire sequence fails immediately
    leaving the payload IR in potentially invalid state, i.e., this operation
    offers no transformation rollback capabilities.

    The entry block of this operation has a single argument that maps to either
    the operand if provided or the top-level container operation of the payload
    IR, typically the root operation of the pass interpreting the transform
    dialect. Operand omission is only allowed for sequences not contained in
    another sequence.
  }];

  let arguments = (ins Optional<PDL_Operation>:$root);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat =
    "($root^)? attr-dict-with-keyword regions (`:` type($results)^)?";

  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "transform"; }

    Block *getBodyBlock() {
      return &getBody().front();
    }
  }];

  let hasVerifier = 1;
}

def YieldOp : TransformDialectOp<"yield", [Terminator]> {
  let summary = "Yields operation handles from a transform IR region";
  let description = [{
    This terminator operation yields operation handles from regions of the
    transform IR ops back to the containing op. It is not itself associated with
    any transformation on the payload IR and is used for flow purposes only.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "operands attr-dict (`:` type($operands)^)?";

  let builders = [
    OpBuilder<(ins), [{
      return build($_builder, $_state, ::mlir::ValueRange());
    }]>
  ];
}

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS
