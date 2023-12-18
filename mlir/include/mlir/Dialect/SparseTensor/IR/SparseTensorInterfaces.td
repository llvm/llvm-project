//===- SparseTensorInterfaces.td --------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARSETENSOR_IR_SPARSETENSORINTERFACES
#define SPARSETENSOR_IR_SPARSETENSORINTERFACES

include "mlir/IR/OpBase.td"

def StageWithSortSparseOpInterface : OpInterface<"StageWithSortSparseOp"> {
  let description = [{
    A stage-with-sort sparse tensor operation is an operation that produces
    unordered intermediate output. An extra sort is required to obtain the final
    ordered result.

    E.g., convert csr -> csc need to be implemented as
          convert csr -> unordered coo -> sort by column -> csc; and
          concatenate csr, csc -> csr can be staged into
          concatenate csr, csr -> unordered coo -> sort by row -> csr.
  }];
  let cppNamespace = "::mlir::sparse_tensor";
  let methods = [
    InterfaceMethod<
    /*desc=*/"Return true if the operation needs an extra sort to produce the final result.",
    /*retTy=*/"bool",
    /*methodName=*/"needsExtraSort",
    /*args=*/(ins),
    /*methodBody=*/"">,
    InterfaceMethod<
    /*desc=*/"Stage the operation, return the final result value after staging.",
    /*retTy=*/"::mlir::LogicalResult",
    /*methodName=*/"stageWithSort",
    /*args=*/(ins "::mlir::PatternRewriter &":$rewriter),
    /*methodBody=*/[{
        return detail::stageWithSortImpl($_op, rewriter);
    }]>,
  ];
}


#endif // SPARSETENSOR_IR_SPARSETENSORINTERFACES
