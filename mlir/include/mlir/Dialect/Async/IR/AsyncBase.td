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
  let description = [{
    `async.token` is a type returned by asynchronous operations, and it becomes
    `ready` when the asynchronous operations that created it is completed.
  }];
}

class Async_ValueType<Type type>
    : DialectType<AsyncDialect,
        And<[
          CPred<"$_self.isa<::mlir::async::ValueType>()">,
          SubstLeaves<"$_self",
                      "$_self.cast<::mlir::async::ValueType>().getValueType()",
                      type.predicate>
       ]>, "async value type with " # type.summary # " underlying type"> {
  let description = [{
    `async.value` represents a value returned by asynchronous operations,
    which may or may not be available currently, but will be available at some
    point in the future.
  }];

  Type valueType = type;
}

def Async_GroupType : DialectType<AsyncDialect,
    CPred<"$_self.isa<::mlir::async::GroupType>()">, "group type">,
    BuildableType<"$_builder.getType<::mlir::async::GroupType>()"> {
  let description = [{
    `async.group` represent a set of async tokens or values and allows to
    execute async operations on all of them together (e.g. wait for the
    completion of all/any of them).
  }];
}

def Async_AnyValueType : DialectType<AsyncDialect,
                           CPred<"$_self.isa<::mlir::async::ValueType>()">,
                                 "async value type">;

def Async_AnyValueOrTokenType : AnyTypeOf<[Async_AnyValueType,
                                           Async_TokenType]>;

def Async_AnyAsyncType : AnyTypeOf<[Async_AnyValueType,
                                    Async_TokenType,
                                    Async_GroupType]>;

//===----------------------------------------------------------------------===//
// Types for lowering to LLVM + Async Runtime via the LLVM Coroutines.
//===----------------------------------------------------------------------===//

// LLVM coroutines intrinsics use `token` and `i8*` types to represent coroutine
// identifiers and handles. To define type-safe Async Runtime operations and
// build a properly typed intermediate IR during the Async to LLVM lowering we
// define a separate types for values that can be produced by LLVM intrinsics.

def Async_CoroIdType : DialectType<AsyncDialect,
    CPred<"$_self.isa<::mlir::async::CoroIdType>()">, "coro.id type">,
    BuildableType<"$_builder.getType<::mlir::async::CoroIdType>()"> {
  let description = [{
    `async.coro.id` is a type identifying a switched-resume coroutine.
  }];
}

def Async_CoroHandleType : DialectType<AsyncDialect,
    CPred<"$_self.isa<::mlir::async::CoroHandleType>()">, "coro.handle type">,
    BuildableType<"$_builder.getType<::mlir::async::CoroHandleType>()"> {
  let description = [{
    `async.coro.handle` is a handle to the coroutine (pointer to the coroutine
    frame) that can be passed around to resume or destroy the coroutine.
  }];
}

def Async_CoroStateType : DialectType<AsyncDialect,
    CPred<"$_self.isa<::mlir::async::CoroStateType>()">, "coro.state type">,
    BuildableType<"$_builder.getType<::mlir::async::CoroStateType>()"> {
  let description = [{
    `async.coro.state` is a saved coroutine state that should be passed to the
    coroutine suspension operation.
  }];
}

#endif // ASYNC_BASE_TD
