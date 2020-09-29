//===- AsyncBase.td ----------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base definitions for the `async` dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNC_BASE_TD
#define ASYNC_BASE_TD

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

def Async_TokenType : DialectType<AsyncDialect,
    CPred<"$_self.isa<::mlir::async::TokenType>()">, "token type">,
    BuildableType<"$_builder.getType<::mlir::async::TokenType>()"> {
  let typeDescription = [{
    `async.token` is a type returned by asynchronous operations, and it becomes
    `ready` when the asynchronous operations that created it is completed.
  }];
}

#endif // ASYNC_BASE_TD
