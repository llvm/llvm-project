//===- RegionExitInterfaces.td - Region exit interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_REGIONEXITINTERFACES
#define MLIR_IR_REGIONEXITINTERFACES

include "mlir/IR/OpBase.td"

// Indicates that this op may propagate a breaking control-flow event from a
// nested region upward to the addressed HasBreakingControlFlowOpInterface
// operation. The op does NOT consume the break itself; it is merely transparent
// to it. All ops that sit between a breaking terminator and the
// HasBreakingControlFlowOpInterface ancestor that will ultimately receive the
// break must carry this trait.
def PropagateControlFlowBreak : NativeOpTrait<"PropagateControlFlowBreak">;

def HasBreakingControlFlowOpInterface : OpInterface<"HasBreakingControlFlowOpInterface"> {
  let description = [{
    Interface for operations that act as the target of a breaking control-flow
    event (e.g. `scf.break` or `scf.continue`). Every intermediate op must
    carry the `PropagateControlFlowBreak` trait. The terminator may identify the
    receiver with any dialect-defined mechanism, such as a builtin token value,
    an attribute, or a symbolic relationship to an enclosing operation.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/[{
        Return true if this operation accepts the given terminator operation
        as a breaking-control-flow predecessor. This can be implemented through
        the `HasNestedTerminators<[...]>` trait, otherwise an explicit implementation
        must be provided (e.g. `return true;` to accept all, or a type check to
        restrict).
      }],
      /*retTy=*/"bool",
      /*methodName=*/"acceptsTerminator",
      /*args=*/(ins "Operation *":$op)
    >
  ];
  let verifyWithRegions = 1;
  let verify = [{
    for (Operation *terminator :
         ::mlir::getNestedBreakingControlFlowInfo($_op).predecessors)
      if (!ConcreteOp::acceptsTerminator(terminator))
        return $_op->emitOpError("does not accept terminator: ")
               << OpWithFlags(terminator, OpPrintingFlags().skipRegions());
    return success();
  }];
}

// OpInterface for terminators that leave their current region and identify
// potential receiving operations using a dialect-defined mechanism. A normal
// region return may simply target the immediately containing operation; a
// breaking-control-flow region exit may target an ancestor operation that
// implements HasBreakingControlFlowOpInterface.
def RegionExitTerminatorOpInterface : OpInterface<"RegionExitTerminatorOpInterface"> {
  let description = [{
    Interface for terminators that leave their current region. Implemented by
    ops such as `scf.yield`, `scf.break`, and `scf.continue`.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the operations that may receive this region-terminating control
        flow event. Receivers for breaking control flow must implement
        `HasBreakingControlFlowOpInterface`; normal region exits may return the
        immediate parent operation. This set must be a sound over-approximation:
        it must contain every possible target in every defined execution.
        Omitting a defined target makes control-flow analyses unsound. Dialects
        may exclude choices whose execution has undefined behavior.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::Operation *>",
      /*methodName=*/"getPotentialTargets",
      /*args=*/(ins),
      /*methodBody=*/[{}]
    >
  ];

  let verify = [{
    static_assert(ConcreteOp::template hasTrait<OpTrait::IsTerminator>(),
                  "expected operation to be a terminator");
    return success();
  }];
}


#endif // MLIR_IR_REGIONEXITINTERFACES
