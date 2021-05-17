//===- AsyncDialect.td -------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Async dialect definition.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNC_DIALECT_TD
#define ASYNC_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Async dialect definitions
//===----------------------------------------------------------------------===//

def AsyncDialect : Dialect {
  let name = "async";

  let summary = "Types and operations for async dialect";
  let description = [{
    This dialect contains operations for modeling asynchronous execution.
  }];

  let cppNamespace = "::mlir::async";
}

#endif // ASYNC_DIALECT_TD
