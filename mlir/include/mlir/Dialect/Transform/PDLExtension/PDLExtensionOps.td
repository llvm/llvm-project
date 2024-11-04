//===- PDLExtensionOps.td - Transform dialect operations ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_PDLEXTENSION_PDLEXTENSIONOPS
#define MLIR_DIALECT_TRANSFORM_PDLEXTENSION_PDLEXTENSIONOPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def PDLMatchOp : TransformDialectOp<"pdl_match",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Finds ops that match the named PDL pattern";
  let description = [{
    Find Payload IR ops nested within the Payload IR op associated with the
    operand that match the PDL pattern identified by its name. The pattern is
    expected to be defined in the closest surrounding `WithPDLPatternsOp`.

    Produces a Transform IR value associated with the list of Payload IR ops
    that matched the pattern. The order of results in the list is that of the
    Operation::walk, clients are advised not to rely on a specific order though.
    If the operand is associated with multiple Payload IR ops, finds matching
    ops nested within each of those and produces a single list containing all
    of the matched ops.

    The transformation is considered successful regardless of whether some
    Payload IR ops actually matched the pattern and only fails if the pattern
    could not be looked up or compiled.
  }];

  let arguments = (ins
    Arg<TransformHandleTypeInterface, "Payload IR scope to match within">:$root,
    SymbolRefAttr:$pattern_name);
  let results = (outs
    Res<TransformHandleTypeInterface, "Handle to the matched Payload IR ops">:$matched);

  let assemblyFormat = "$pattern_name `in` $root attr-dict `:` "
                       "functional-type(operands, results)";
}

def WithPDLPatternsOp : TransformDialectOp<"with_pdl_patterns",
    [DeclareOpInterfaceMethods<TransformOpInterface>, NoTerminator,
     OpAsmOpInterface, PossibleTopLevelTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     SymbolTable]> {
  let summary = "Contains PDL patterns available for use in transforms";
  let description = [{
    This op contains a set of named PDL patterns that are available for the
    Transform dialect operations to be used for pattern matching. For example,
    PDLMatchOp can be used to produce a Transform IR value associated with all
    Payload IR operations that match the pattern as follows:

    ```mlir
    transform.with_pdl_patterns {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @my_pattern : benefit(1) {
        %0 = pdl.operation //...
        // Regular PDL goes here.
        pdl.rewrite %0 with "transform.dialect"
      }

      sequence %arg0 failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %1 = pdl_match @my_pattern in %arg1
        // Use %1 as handle
      }
    }
    ```

    Note that the pattern is expected to finish with a `pdl.rewrite` terminator
    that points to the custom rewriter named "transform.dialect". The rewriter
    actually does nothing, but the transform application will keep track of the
    operations that matched the pattern.

    This op is expected to contain `pdl.pattern` operations and exactly one
    another Transform dialect operation that gets executed with all patterns
    available. This op is a possible top-level Transform IR op, the argument of
    its entry block corresponds to either the root op of the payload IR or the
    ops associated with its operand when provided.
  }];

  let arguments = (ins
    Arg<Optional<TransformHandleTypeInterface>, "Root operation of the Payload IR"
        >:$root);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "($root^ `:` type($root))? attr-dict-with-keyword regions";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "transform"; }
  }];
}

#endif // MLIR_DIALECT_TRANSFORM_PDLEXTENSION_PDLEXTENSIONOPS
