//===- RegionKindInterface.td - Region kind interfaces -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces to query the properties of regions
// in an operation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_REGIONKINDINTERFACE
#define MLIR_IR_REGIONKINDINTERFACE

include "mlir/IR/OpBase.td"

// OpInterface to query the properties of regions in an operation
def RegionKindInterface : OpInterface<"RegionKindInterface"> {
  let description = [{
    Interface for operations to describe the abstract semantics of
    their regions. Currently, two kinds of regions are
    supported. RegionKind::Graph represents a graph region without
    control flow semantics. RegionKind::SSACFG represents an
    [SSA-style control flow](../LangRef.md/#modeling-control-flow) region
    with basic blocks, sequential semantics, and reachability.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/[{
        Return the kind of the region with the given index inside this operation.
      }],
      /*retTy=*/"::mlir::RegionKind",
      /*methodName=*/"getRegionKind",
      /*args=*/(ins "unsigned":$index)
    >,
    StaticInterfaceMethod<
      /*desc=*/"Return true if the kind of the given region requires the "
               "SSA-Dominance property",
      /*retTy=*/"bool",
      /*methodName=*/"hasSSADominance",
      /*args=*/(ins "unsigned":$index),
      /*methodBody=*/[{
        return getRegionKind(index) == ::mlir::RegionKind::SSACFG;
      }]
    >,
  ];
}

def HasOnlyGraphRegion : NativeOpTrait<"HasOnlyGraphRegion">;

// Op's regions that don't need a terminator: requires some other traits
// so it defines a list that must be concatenated.
def GraphRegionNoTerminator : TraitList<[
    NoTerminator,
    SingleBlock,
    RegionKindInterface,
    HasOnlyGraphRegion
  ]>;

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
        the `HasNestedTerminator<[...]>` trait, otherwise an explicit implementation
        must be provided (e.g. `return true;` to accept all, or a type check to
        restrict).
      }],
      /*retTy=*/"bool",
      /*methodName=*/"acceptsTerminator",
      /*args=*/(ins "Operation *":$op)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if this operation has at least one breaking terminator
        nested inside it that targets this operation directly. Used to decide
        whether post-dominance analysis must account for early-exit paths.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasNestedPredecessors",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::mlir::hasNestedPredecessors(this->getOperation());
      }]
    >
  ];
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
        immediate parent operation.
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


#endif // MLIR_IR_REGIONKINDINTERFACE
