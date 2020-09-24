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
def GraphRegionNoTerminator {
  list<OpTrait> traits = [
    NoTerminator,
    SingleBlock,
    RegionKindInterface,
    HasOnlyGraphRegion
  ];
}

#endif // MLIR_IR_REGIONKINDINTERFACE
