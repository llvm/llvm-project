//===- IndexOps.td - Index operation definitions -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INDEX_OPS
#define INDEX_OPS

include "mlir/Dialect/Index/IR/IndexDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IndexOp
//===----------------------------------------------------------------------===//

/// Base class for Index dialect operations.
class IndexOp<string mnemonic, list<Trait> traits = []>
    : Op<IndexDialect, mnemonic, [NoSideEffect] # traits>;

#endif // INDEX_OPS
