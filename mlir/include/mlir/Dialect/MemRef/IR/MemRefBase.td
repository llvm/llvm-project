//===- MemRefBase.td - Base definitions for memref dialect -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MEMREF_BASE
#define MEMREF_BASE

include "mlir/IR/OpBase.td"

def MemRef_Dialect : Dialect {
  let name = "memref";
  let cppNamespace = "::mlir::memref";
  let description = [{
    The `memref` dialect is intended to hold core memref creation and
    manipulation ops, which are not strongly associated with any particular
    other dialect or domain abstraction.
  }];
  let dependentDialects = ["arith::ArithmeticDialect", "tensor::TensorDialect"];
  let hasConstantMaterializer = 1;
}

#endif // MEMREF_BASE
