//===-- Passes.td - Transform dialect pass definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_TRANSFORMS_PASSES
#define MLIR_DIALECT_TRANSFORM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def CheckUsesPass : Pass<"transform-dialect-check-uses"> {
  let summary = "warn about potential use-after-free in the transform dialect";
  let description = [{
    This pass analyzes operations from the transform dialect and its extensions
    and warns if a transform IR value may be used by an operation after it was
    "freed" by some other operation, as described by side effects on the
    `TransformMappingResource`. This statically detects situations that lead to
    errors when interpreting the Transform IR.

    The pass is capable of handling branching control flow and reports all
    _potential_ use-after-free situations, e.g., a may-use-after-free is
    reported if at least one of the control flow paths between the definition of
    a value and its use contains an operation with a "free" effect on the
    `TransformMappingResource`. It does not currently perform an SCCP-style data
    flow analysis to prove that some branches are not taken, however, SCCP and
    other control flow simplifications can be performed on the transform IR
    prior to this pass provided that transform ops implement the relevant
    control flow interfaces.
  }];
}

def InferEffectsPass : Pass<"transform-infer-effects"> {
  let summary = "infer transform side effects for symbols";
  let description = [{
    This pass analyzes the definitions of transform dialect callable symbol
    operations, such as `transform.named_sequence`, and annotates the symbol
    arguments with attributes indicating the side effects that the nested
    operations have on them.
  }];
}

def PreloadLibraryPass : Pass<"transform-preload-library"> {
  let summary = "preload transform dialect library";
  let description = [{
    This pass preloads a transform library and makes it available to subsequent
    transform interpreter passes. The preloading occurs into the Transform
    dialect and thus provides very limited functionality that does not scale.

    Warning: Only a single such pass should exist for a given MLIR context.
    This is a temporary solution until a resource-based solution is available.
  }];
  // TODO: investigate using a resource blob if some ownership mode allows it.
  let dependentDialects = ["::mlir::transform::TransformDialect"];
  let options = [
    ListOption<"transformLibraryPaths", "transform-library-paths", "std::string",
    "Optional paths to files with modules that should be merged into the "
    "transform module to provide the definitions of external named sequences.">
  ];
}

def InterpreterPass : Pass<"transform-interpreter"> {
  let summary = "transform dialect interpreter";
  let description = [{
    This pass runs the transform dialect interpreter and applies the named
    sequence transformation specified by the provided name (defaults to
    `TransformDialect::kTransformEntryPointSymbolName` (i.e. `__transform_main`)).
  }];
  let dependentDialects = ["::mlir::transform::TransformDialect"];
  let options = [
    Option<"debugPayloadRootTag", "debug-payload-root-tag", "std::string",
           /*default=*/[{""}],
           "Select the operation with 'transform.target_tag' attribute having "
           "the given value as payload IR root. If empty select the pass "
           "anchor operation as the payload IR root.">,
    Option<"disableExpensiveChecks", "disable-expensive-checks", "bool",
           "false",
           "Disable expensive checks in the interpreter for a faster run.">,
    Option<"entryPoint", "entry-point", "std::string",
           /*default=*/[{TransformDialect::kTransformEntryPointSymbolName.str()}],
           "Entry point of the pass pipeline.">,
  ];
}
#endif // MLIR_DIALECT_TRANSFORM_TRANSFORMS_PASSES
