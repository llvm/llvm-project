//===- TuneExtensionOps.td - Transform dialect operations --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS
#define MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"

def KnobOp : Op<Transform_Dialect, "tune.knob", [
  DeclareOpInterfaceMethods<TransformOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Represents a tunable parameter with a set of options";

  let description = [{
    Provides a representation for "tunables" within schedules.

    Each op represents a single tunable, which has a `name` and a set
    of valid `options` described by an attribute. Without a specified
    `selected` option, this op represents a non-deterministic choice
    that has yet to be resolved -- as such, the interpreter runtime
    semantics is to raise a failure.

    The non-deterministic choice is resolved through providing a
    `selected` attribute. When provided, the interpreter runtime
    semantics are to return the `selected` attribute as a param through
    the op's result.

    -----

    In case the `options` attribute is an `ArrayAttr`, the verifier
    checks that the provided `selected` attribute occurs in `options`.
  }];
  let cppNamespace = [{ mlir::transform::tune }];
  let hasVerifier = 1;

  let arguments = (ins Builtin_StringAttr:$name,
                       AnyAttr:$options,
                       OptionalAttr<AnyAttr>:$selected);
  let results = (outs TransformParamTypeInterface:$result);

  let assemblyFormat =
      "`<` $name `>` (`=` $selected^ `from`)? `options` `=` $options attr-dict `->` type(results)";
}

#endif // MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS
