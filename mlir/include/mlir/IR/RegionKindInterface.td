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

// Indicates that this op may propagate a breaking control-flow event (a
// RegionTerminator with num-breaking-regions > 1) from a nested region upward
// to an enclosing HasBreakingControlFlowOpInterface operation. The op does NOT
// consume the break itself; it is merely transparent to it. All ops that sit
// between a RegionTerminator and the HasBreakingControlFlowOpInterface ancestor
// that will ultimately receive the break must carry this trait.
def PropagateControlFlowBreak : NativeOpTrait<"PropagateControlFlowBreak">;

// OpInterface for operations that can receive a breaking control-flow event
// originating from a RegionTerminator (scf.break / scf.continue) anywhere
// inside their (possibly deeply nested) regions. Every op between the
// RegionTerminator and this receiver must carry the PropagateControlFlowBreak
// trait.
def HasBreakingControlFlowOpInterface : OpInterface<"HasBreakingControlFlowOpInterface"> {
  let description = [{
    Interface for operations that act as the target of a breaking control-flow
    event (e.g. `scf.break` or `scf.continue`). When a `RegionTerminator` has
    `num-breaking-regions = N > 1`, it exits N region levels; the op at the
    N-th level must implement this interface. Every intermediate op (levels
    1 .. N-1) must carry the `PropagateControlFlowBreak` trait.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/[{
        Return true if this operation accepts the given terminator operation
        as a breaking-control-flow predecessor. By default all terminators are
        accepted; override to restrict to specific terminator op types.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"acceptsTerminator",
      /*args=*/(ins "Operation *":$op),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return true;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if this operation has at least one RegionTerminator nested
        inside it that targets this operation directly (i.e.
        num-breaking-regions equals the nesting depth from this op's region to
        the terminator). Used to decide whether post-dominance analysis must
        account for early-exit paths.
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


#endif // MLIR_IR_REGIONKINDINTERFACE
