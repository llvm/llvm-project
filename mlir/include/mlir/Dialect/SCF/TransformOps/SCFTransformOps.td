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
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def GetParentForOp : Op<Transform_Dialect, "loop.get_parent_for",
    [NavigationTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Gets a handle to the parent 'for' loop of the given operation";
  let description = [{
    Produces a handle to the n-th (default 1) parent `scf.for` loop for each
    Payload IR operation associated with the operand. Fails if such a loop
    cannot be found. The list of operations associated with the handle contains
    parent operations in the same order as the list associated with the operand,
    except for operations that are parents to more than one input which are only
    present once.
  }];

  let arguments =
    (ins PDL_Operation:$target,
         DefaultValuedAttr<Confined<I64Attr, [IntPositive]>,
                           "1">:$num_loops);
  let results = (outs PDL_Operation:$parent);

  let assemblyFormat = "$target attr-dict";
}

def LoopOutlineOp : Op<Transform_Dialect, "loop.outline",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Outlines a loop into a named function";
  let description = [{
     Moves the loop into a separate function with the specified name and
     replaces the loop in the Payload IR with a call to that function. Takes
     care of forwarding values that are used in the loop as function arguments.
     If the operand is associated with more than one loop, each loop will be
     outlined into a separate function. The provided name is used as a _base_
     for forming actual function names following SymbolTable auto-renaming
     scheme to avoid duplicate symbols. Expects that all ops in the Payload IR
     have a SymbolTable ancestor (typically true because of the top-level
     module). Returns the handle to the list of outlined functions in the same
     order as the operand handle.
  }];

  let arguments = (ins PDL_Operation:$target,
                   StrAttr:$func_name);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
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
      (ins PDL_Operation:$target,
           DefaultValuedAttr<BoolAttr, "false">:$fail_if_already_divisible);
  // TODO: Return both the peeled loop and the remainder loop. 
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::scf::ForOp target, 
        ::llvm::SmallVector<::mlir::Operation *> &results, 
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

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64Attr, "1">:$iteration_interval,
                   DefaultValuedAttr<I64Attr, "10">:$read_latency);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::scf::ForOp target, 
        ::llvm::SmallVector<::mlir::Operation *> &results, 
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
    
    This operation ignores non-scf::For ops and drops them in the return.
    If all the operations referred to by the `target` PDLOperation unroll
    properly, the transform succeeds. Otherwise the transform silently fails.
    
    Does not return handles as the operation may result in the loop being
    removed after a full unrolling.
  }];

  let arguments = (ins PDL_Operation:$target,
                       Confined<I64Attr, [IntPositive]>:$factor);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::scf::ForOp target, 
        ::llvm::SmallVector<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}

#endif // SCF_TRANSFORM_OPS
