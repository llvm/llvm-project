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
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"

//===----------------------------------------------------------------------===//
// KnobOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// AlternativesOp
//===----------------------------------------------------------------------===//

def AlternativesOp : Op<Transform_Dialect, "tune.alternatives", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands",
         "getRegionInvocationBounds",
         "getSuccessorInputs"]>,
  DeclareOpInterfaceMethods<TransformOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">,
  NoRegionArguments
]> {
  let summary = "Represents a choice among its regions, i.e. sub-schedules";

  let description = [{
    This op represents a choice over which of its regions is to be used.

    When `selected_region` is provided, the semantics are that this op is to be
    substituted for by the selected region, meaning the region's results become
    the results of this op. Without a provided `selected_region`, the semantics
    are that this non-deterministic choice is yet to be resolved -- which in
    terms of the op's interpreted semantics is a failure.

    The `selected_region` argument is either an `IntegerAttr` or a param holding
    an `IntegerAttr`, which should provide a valid zero-based index with respect
    to the number of alternatives, i.e. regions.
  }];
  let cppNamespace = [{ mlir::transform::tune }];

  let arguments = (ins Builtin_StringAttr:$name,
                       OptionalAttr<APIntAttr>:$selected_region_attr,
                       Optional<TransformParamTypeInterface>:$selected_region_param);
  let results = (outs Variadic<Transform_AnyHandleOrParamType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$alternatives);

  let assemblyFormat = [{
    `<` $name `>`
    (`selected_region` `=` custom<AlternativesOpSelectedRegion>(
        $selected_region_attr, $selected_region_param)^)?
    attr-dict-with-keyword
    (`:` type($selected_region_param)^)?
    (`->` type($results)^)?
    regions
  }];

  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS
