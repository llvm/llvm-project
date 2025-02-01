//===- UnrollScopeInterface.td - unroll scope interface ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for unroll region.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_UNROLLSCOPEINTERFACE
#define MLIR_INTERFACES_UNROLLSCOPEINTERFACE

include "mlir/IR/OpBase.td"

def UnrollScopeInterface : OpInterface<"UnrollScopeInterface"> {
  let description = [{
    This interface controls the scope of the loop unroll.It ensures
    that SSA values generated outside the loop when unrolling are 
    in the nearest `UnrollScopeInterface` region.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<[{
      return the `UnrollScopeInterface` region.
    }],
    "::mlir::Region&", "getUnrollBody", (ins),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{
      return $_op->getRegion(0);
    }]>,
  ];
}

#endif // MLIR_INTERFACES_UNROLLSCOPEINTERFACE
