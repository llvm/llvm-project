//===- LoopInvariantConditionInterface.td - Cond. hoisting --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for if-like operations that can, given enough
// dialect-specific context, prove whether one of their regions is
// unconditionally entered relative to an enclosing loop.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_LOOPINVARIANTCONDITIONINTERFACE
#define MLIR_INTERFACES_LOOPINVARIANTCONDITIONINTERFACE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

def LoopInvariantConditionOpInterface :
    OpInterface<"LoopInvariantConditionOpInterface"> {
  let description = [{
    Interface for ops with one or more conditionally-executed regions
    (if-like ops) that can prove whether a particular 
    region is unconditionally entered relative to an
    enclosing loop.

    Implementations must be conservative: whenever a full, sound proof is
    not available, return false.
  }];

  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns true if region is provably
        entered on every iteration of enclosing loop. Implementations
        should return false whenever region does not belong to this op,
        or whenever the proof cannot be constructed.
      }],
      "bool", "isRegionAlwaysEnteredInLoop",
      (ins "::mlir::Region &":$region,
           "::mlir::LoopLikeOpInterface":$enclosingLoop),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
  ];
}

#endif // MLIR_INTERFACES_LOOPINVARIANTCONDITIONINTERFACE
