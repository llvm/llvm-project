//===- TensorEncoding.td - Tensor encoding interfaces ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interfaces associated with tensor encoding attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_TENSORINTERFACES
#define MLIR_IR_TENSORINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Attribute interface to verify a tensor encoding.
//===----------------------------------------------------------------------===//

def VerifiableTensorEncoding : AttrInterface<"VerifiableTensorEncoding"> {
  let cppNamespace = "::mlir";
  let description = [{
    Verifies an encoding attribute for a tensor.

    This interface also doubles as the contract some `tensor.*`
    canonicalization patterns rely on when refining a tensor's shape to be
    more static (e.g. folding a `tensor.cast` into a consumer, or turning a
    constant `Value` operand into a static dimension). When such a pattern
    needs to decide whether an existing encoding still applies to the
    refined shape:

    - if the encoding implements this interface, the pattern re-runs
      `verifyEncoding` against the refined shape and keeps the encoding
      only if it still holds; this is how a rank- or shape-dependent
      encoding (e.g. a sparse tensor encoding, which encodes a per-dimension
      layout) gets dropped instead of ending up attached to a
      `RankedTensorType` it is not valid for.
    - if the encoding does not implement this interface, it is treated as
      opaque and shape-agnostic, and is propagated onto the refined type
      unconditionally. Not implementing the interface is thus an implicit
      opt-out: the attribute's author is asserting that its validity does
      not depend on the tensor's shape, and takes on responsibility for
      that invariant.

    Patterns that instead merge or otherwise combine multiple tensors with
    possibly-differing encodings (e.g. `tensor.concat`, or collapsing
    dimensions via `tensor.collapse_shape`) do not attempt to verify or
    propagate an encoding at all, since there is no way to statically prove
    an arbitrary encoding remains valid once tensors are combined or a
    dynamic dimension folds two static ones together — the encoding is
    dropped from the result in that case.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Verifies the encoding is valid for a tensor type with the
        given shape and element type. Generates a diagnostic using
        the supplied callback on failure.
      }],
      /*retTy=*/"::llvm::LogicalResult",
      /*methodName=*/"verifyEncoding",
      /*args=*/(ins
        "::mlir::ArrayRef<int64_t>":$shape,
        "::mlir::Type":$elementType,
        "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
    >,
  ];
}

#endif // MLIR_IR_TENSORINTERFACES
