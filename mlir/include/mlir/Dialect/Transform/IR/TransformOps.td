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
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"

def PDLMatchOp : TransformDialectOp<"pdl_match",
    [DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Finds ops that match the named PDL pattern";
  let description = [{
    Find Payload IR ops nested within the Payload IR op associated with the
    operand that match the PDL pattern identified by its name. The pattern is
    expected to be defined in the closest surrounding `WithPDLPatternsOp`.

    Produces a Transform IR value associated with the list of Payload IR ops
    that matched the pattern. The order of results in the list is that of the
    Operation::walk, clients are advised not to rely on a specific order though.
    If the operand is assocaited with multiple Payload IR ops, finds matching
    ops nested within each of those and produces a single list containing all
    of the matched ops.

    The tranfsormation is considered successful regardless of whether some
    Payload IR ops actually matched the pattern and only fails if the pattern
    could not be looked up or compiled.
  }];

  let arguments = (ins
    Arg<PDL_Operation, "Payload IR scope to match within",
        [TransformMappingRead, PayloadIRRead]>:$root, 
    SymbolRefAttr:$pattern_name);
  let results = (outs
    Res<PDL_Operation, "Handle to the matched Payload IR ops",
        [TransformMappingAlloc, TransformMappingWrite]>:$matched);

  let assemblyFormat = "$pattern_name `in` $root attr-dict";
}

def SequenceOp : TransformDialectOp<"sequence",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     OpAsmOpInterface, PossibleTopLevelTransformOpTrait,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">]> {
  let summary = "Contains a sequence of other transform ops to apply";
  let description = [{
    The transformations indicated by the sequence are applied in order of their
    appearance. Each value produced by a transformation within the sequence
    corresponds to an operation or a group of operations in the payload IR.
    During application, if any transformation in the sequence fails, the entire
    sequence fails immediately leaving the payload IR in potentially invalid
    state, i.e., this operation offers no transformation rollback capabilities.

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
  }];

  let hasVerifier = 1;
}

def WithPDLPatternsOp : TransformDialectOp<"with_pdl_patterns",
    [DeclareOpInterfaceMethods<TransformOpInterface>, NoTerminator,
     OpAsmOpInterface, PossibleTopLevelTransformOpTrait, RecursiveSideEffects,
     SymbolTable]> {
  let summary = "Contains PDL patterns available for use in transforms";
  let description = [{
    This op contains a set of named PDL patterns that are available for the
    Transform dialect operations to be used for pattern matching. For example,
    PDLMatchOp can be used to produce a Transform IR value associated with all
    Payload IR operations that match the pattern as follows:

    ```mlir
    transform.with_pdl_patterns {
    ^bb0(%arg0: !pdl.operation):
      pdl.pattern @my_pattern : benefit(1) {
        %0 = pdl.operation //...
        // Regular PDL goes here.
        pdl.rewrite %0 with "transform.dialect"
      }

      sequence %arg0 {
      ^bb0(%arg1: !pdl.operation):
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
    Arg<Optional<PDL_Operation>, "Root operation of the Payload IR",
        [TransformMappingRead]>:$root);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "($root^)? attr-dict-with-keyword regions";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "transform"; }
  }];
}

def YieldOp : TransformDialectOp<"yield", [Terminator]> {
  let summary = "Yields operation handles from a transform IR region";
  let description = [{
    This terminator operation yields operation handles from regions of the
    transform IR ops back to the containing op. It is not itself associated with
    any transformation on the payload IR and is used for flow purposes only.
  }];

  let arguments = (ins 
    Arg<Variadic<AnyType>, "Opration handles yielded back to the parent",
        [TransformMappingRead]>:$operands);
  let assemblyFormat = "operands attr-dict (`:` type($operands)^)?";

  let builders = [
    OpBuilder<(ins), [{
      return build($_builder, $_state, ::mlir::ValueRange());
    }]>
  ];
}

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS
