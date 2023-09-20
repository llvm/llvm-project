//===- SCFTransformOps.td - SCF (loop) transformation ops --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCF_TRANSFORM_OPS
#define SCF_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ApplyForLoopCanonicalizationPatternsOp : Op<Transform_Dialect,
    "apply_patterns.scf.for_loop_canonicalization",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collects patterns for canonicalizing operations inside SCF loop bodies.
    At the moment, only affine.min/max computations with iteration variables,
    loop bounds and loop steps are canonicalized.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplySCFStructuralConversionPatternsOp : Op<Transform_Dialect,
    "apply_conversion_patterns.scf.structural_conversions",
    [DeclareOpInterfaceMethods<ConversionPatternDescriptorOpInterface,
      ["populateConversionTargetRules"]>]> {
  let description = [{
    Collects patterns for performing structural conversions of SCF operations.
  }];

  let assemblyFormat = "attr-dict";
}

def Transform_ScfForOp : Transform_ConcreteOpType<"scf.for">;

def ForallToForOp : Op<Transform_Dialect, "loop.forall_to_for",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Converts scf.forall into a nest of scf.for operations";
  let description = [{
    Converts the `scf.forall` operation pointed to by the given handle into a
    set of nested `scf.for` operations. Each new operation corresponds to one
    induction variable of the original "multifor" loop.

    The operand handle must be associated with exactly one payload operation.

    Loops with shared outputs are currently not supported.

    #### Return Modes

    Consumes the operand handle. Produces a silenceable failure if the operand
    is not associated with a single `scf.forall` payload operation.
    Returns as many handles as the given `forall` op has induction variables
    that are associated with the generated `scf.for` loops.
    Produces a silenceable failure if another number of resulting handles is
    requested.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs Variadic<TransformHandleTypeInterface>:$transformed);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
}

def GetParentForOp : Op<Transform_Dialect, "loop.get_parent_for",
    [NavigationTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Gets a handle to the parent 'for' loop of the given operation";
  let description = [{
    Produces a handle to the n-th (default 1) parent `scf.for` or `affine.for`
    (when the affine flag is true) loop for each Payload IR operation
    associated with the operand. Fails if such a loop cannot be found. The list
    of operations associated with the handle contains parent operations in the
    same order as the list associated with the operand, except for operations
    that are parents to more than one input which are only present once.
  }];

  let arguments =
    (ins TransformHandleTypeInterface:$target,
         DefaultValuedAttr<ConfinedAttr<I64Attr, [IntPositive]>,
                           "1">:$num_loops,
         DefaultValuedAttr<BoolAttr, "false">:$affine);
  let results = (outs TransformHandleTypeInterface : $parent);

  let assemblyFormat =
    "$target attr-dict `:` functional-type(operands, results)";
}

def LoopOutlineOp : Op<Transform_Dialect, "loop.outline",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Outlines a loop into a named function";
  let description = [{
    Moves the loop into a separate function with the specified name and replaces
    the loop in the Payload IR with a call to that function. Takes care of
    forwarding values that are used in the loop as function arguments. If the
    operand is associated with more than one loop, each loop will be outlined
    into a separate function. The provided name is used as a _base_ for forming
    actual function names following `SymbolTable` auto-renaming scheme to avoid
    duplicate symbols. Expects that all ops in the Payload IR have a
    `SymbolTable` ancestor (typically true because of the top-level module).

    #### Return Modes

    Returns a handle to the list of outlined functions and a handle to the
    corresponding function call operations in the same order as the operand
    handle.

    Produces a definite failure if outlining failed for any of the targets.
  }];

  // Note that despite the name of the transform operation and related utility
  // functions, the actual implementation does not require the operation to be
  // a loop.
  let arguments = (ins TransformHandleTypeInterface:$target,
                   StrAttr:$func_name);
  let results = (outs TransformHandleTypeInterface:$function,
                      TransformHandleTypeInterface:$call);

  let assemblyFormat =
    "$target attr-dict `:` functional-type(operands, results)";
}

def LoopPeelOp : Op<Transform_Dialect, "loop.peel",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Peels the last iteration of the loop";
  let description = [{
     Updates the given loop so that its step evenly divides its range and puts
     the remaining iteration into a separate loop or a conditional.

     In the absence of sufficient static information, this op may peel a loop,
     even if the step always divides the range evenly at runtime.

     #### Return modes

     This operation ignores non-scf::ForOp ops and drops them in the return.

     This operation always succeeds and returns the scf::ForOp with the
     postcondition: "the loop trip count is divisible by the step".
     This operation may return the same unmodified loop handle when peeling did
     not modify the IR (i.e. the loop trip count was already divisible).

     Note that even though the Payload IR modification may be performed
     in-place, this operation consumes the operand handle and produces a new
     one.

     TODO: Return both the peeled loop and the remainder loop.
  }];

  let arguments =
      (ins Transform_ScfForOp:$target,
           DefaultValuedAttr<BoolAttr, "false">:$fail_if_already_divisible);
  // TODO: Return both the peeled loop and the remainder loop.
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat =
    "$target attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::scf::ForOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LoopPipelineOp : Op<Transform_Dialect, "loop.pipeline",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Applies software pipelining to the loop";
  let description = [{
    Transforms the given loops one by one to achieve software pipelining for
    each of them. That is, performs some amount of reads from memory before the
    loop rather than inside the loop, the same amount of writes into memory
    after the loop, and updates each iteration to read the data for a following
    iteration rather than the current one.

    The amount is specified by the attributes.

    The values read and about to be stored are transferred as loop iteration
    arguments. Currently supports memref and vector transfer operations as
    memory reads/writes.

    #### Return modes

    This operation ignores non-scf::For ops and drops them in the return.
    If all the operations referred to by the `target` PDLOperation pipeline
    properly, the transform succeeds. Otherwise the transform silently fails.
    The return handle points to only the subset of successfully produced
    pipelined loops, which can be empty.
  }];

  let arguments = (ins Transform_ScfForOp:$target,
                   DefaultValuedAttr<I64Attr, "1">:$iteration_interval,
                   DefaultValuedAttr<I64Attr, "10">:$read_latency);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat =
    "$target attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::scf::ForOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LoopPromoteIfOneIterationOp : Op<Transform_Dialect,
    "loop.promote_if_one_iteration", [
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Promote loop if it has one iteration";
  let description = [{
    Promotes the given target loop op if it has a single iteration. I.e., the
    loop op is removed and only the body remains.

    #### Return modes

    This transform fails if the target is mapped to ops that are loops. Ops are
    considered loops if they implement the `LoopLikeOpInterface`. Otherwise,
    this transform always succeeds. The transform consumes the target handle and
    modifies the payload.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::LoopLikeOpInterface target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LoopUnrollOp : Op<Transform_Dialect, "loop.unroll",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Unrolls the given loop with the given unroll factor";
  let description = [{
    Unrolls each loop associated with the given handle to have up to the given
    number of loop body copies per iteration. If the unroll factor is larger
    than the loop trip count, the latter is used as the unroll factor instead.

    #### Return modes

    This operation ignores non-scf::For, non-affine::For ops and drops them in
    the return.  If all the operations referred to by the `target` PDLOperation
    unroll properly, the transform succeeds. Otherwise the transform silently
    fails.

    Does not return handles as the operation may result in the loop being
    removed after a full unrolling.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       ConfinedAttr<I64Attr, [IntPositive]>:$factor);

  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LoopCoalesceOp : Op<Transform_Dialect, "loop.coalesce", [
  FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
  TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Coalesces the perfect loop nest enclosed by a given loop";
  let description = [{
    Given a perfect loop nest identified by the outermost loop,
    perform loop coalescing in a bottom-up one-by-one manner.

    #### Return modes

    The return handle points to the coalesced loop if coalescing happens, or
    the given input loop if coalescing does not happen.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat =
      "$target attr-dict `:` functional-type($target, $transformed)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TakeAssumedBranchOp : Op<Transform_Dialect, "scf.take_assumed_branch", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Given an scf.if conditional, inject user-defined information that it is
    always safe to execute only the if or else branch.

    This is achieved by just replacing the scf.if by the content of one of its
    branches.

    This is particularly useful for user-controlled rewriting of conditionals
    that exist solely to guard against out-of-bounds behavior.

    At the moment, no assume or assert operation is emitted as it is not always
    desirable. In the future, this may be controlled by a dedicated attribute.

    #### Return modes

    The transform only consumes its operand and does not produce any result.
    The transform definitely fails if `take_else_branch` is specified and the
    `else` region is empty.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target,
                       OptionalAttr<UnitAttr>:$take_else_branch);
  let results = (outs);

  let assemblyFormat = [{
      $target
      (`take_else_branch` $take_else_branch^)?
      attr-dict
       `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::scf::IfOp ifOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LoopFuseSibling : Op<Transform_Dialect, "loop.fuse_sibling",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Fuse a loop into another loop, assuming the fusion is legal.";

  let description = [{
    Fuses the `target` loop into the `source` loop assuming they are
    independent of each other. It is the responsibility of the user to ensure
    that the given two loops are independent of each other, this operation will
    not performa any legality checks and will simply fuse the two given loops.

    Currently, the only fusion supported is when both `target` and `source`
    are `scf.forall` operations. For `scf.forall` fusion, the bounds and the
    mapping must match, otherwise a silencable failure is produced.

    The input handles `target` and `source` must map to exactly one operation,
    a definite failure is produced otherwise.

    #### Return modes

    This operation consumes the `target` and `source` handles and produces the
    `fused_loop` handle, which points to the fused loop.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       TransformHandleTypeInterface:$source);
  let results = (outs TransformHandleTypeInterface:$fused_loop);
  let assemblyFormat = "$target `into` $source attr-dict "
                       " `:` functional-type(operands, results)";

  let builders = [
    OpBuilder<(ins "Value":$loop, "Value":$fused_loop)>
  ];
}

#endif // SCF_TRANSFORM_OPS
