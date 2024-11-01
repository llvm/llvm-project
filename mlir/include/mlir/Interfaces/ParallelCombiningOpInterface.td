//===- ParallelCombiningOpInterface.td - Parallel iface ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for ops that perform parallel combining operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_PARALLELCOMBININGOPINTERFACE
#define MLIR_INTERFACES_PARALLELCOMBININGOPINTERFACE

include "mlir/IR/OpBase.td"

def ParallelCombiningOpInterface : OpInterface<"ParallelCombiningOpInterface"> {
  let description = [{
    A parallel combining op is an op with a region.

    This is useful as a terminator to parallel operations that iterate over
    some set and return tensors while avoiding tight coupling between the
    iterating op, the combining op and the individual subtensor producing ops.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return `idx`^th result of the parent operation.
      }],
      /*retTy=*/"::mlir::OpResult",
      /*methodName=*/"getParentResult",
      /*args=*/(ins "int64_t":$idx),
      /*methodBody=*/[{
        return $_op.getParentResult(idx);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the contained ops that yield subvalues that this op combines to
        yield to its parent.
      }],
      /*retTy=*/"::llvm::iterator_range<Block::iterator>",
      /*methodName=*/"getYieldingOps",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_op.getYieldingOps();
      }]
    >,
  ];
  // TODO: Single region single block interface on interfaces ?
  let verify = [{
    return verifyParallelCombiningOpInterface($_op);
  }];
}

#endif // MLIR_INTERFACES_PARALLELCOMBININGOPINTERFACE
