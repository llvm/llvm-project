//===- AsyncTypes.td - Async dialect types -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Async dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ASYNC_IR_ASYNCTYPES
#define MLIR_DIALECT_ASYNC_IR_ASYNCTYPES

include "mlir/Dialect/Async/IR/AsyncDialect.td"

//===----------------------------------------------------------------------===//
// Async Types
//===----------------------------------------------------------------------===//

class Async_Type<string name, string typeMnemonic> : TypeDef<AsyncDialect,
                                                             name> {
  let mnemonic = typeMnemonic;
}

def Async_TokenType : Async_Type<"Token", "token"> {
  let summary = "async token type";
  let description = [{
    `async.token` is a type returned by asynchronous operations, and it becomes
    `available` when the asynchronous operations that created it is completed.
  }];
}

def Async_ValueType : Async_Type<"Value", "value"> {
  let summary = "async value type";
  let description = [{
    `async.value` represents a value returned by asynchronous operations,
    which may or may not be available currently, but will be available at some
    point in the future.
  }];

  let parameters = (ins "Type":$valueType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$valueType), [{
      return $_get(valueType.getContext(), valueType);
    }]>
  ];
  let skipDefaultBuilders = 1;
}

def Async_GroupType : Async_Type<"Group", "group"> {
  let summary = "async group type";
  let description = [{
    `async.group` represent a set of async tokens or values and allows to
    execute async operations on all of them together (e.g. wait for the
    completion of all/any of them).
  }];
}

def Async_AnyValueOrTokenType : AnyTypeOf<[Async_ValueType,
                                           Async_TokenType]>;

def Async_AnyAsyncType : AnyTypeOf<[Async_ValueType,
                                    Async_TokenType,
                                    Async_GroupType]>;

//===----------------------------------------------------------------------===//
// Types for lowering to LLVM + Async Runtime via the LLVM Coroutines.
//===----------------------------------------------------------------------===//

// LLVM coroutines intrinsics use `token` and `i8*` types to represent coroutine
// identifiers and handles. To define type-safe Async Runtime operations and
// build a properly typed intermediate IR during the Async to LLVM lowering we
// define a separate types for values that can be produced by LLVM intrinsics.

def Async_CoroIdType : Async_Type<"CoroId", "coro.id"> {
  let summary = "switched-resume coroutine identifier";
  let description = [{
    `async.coro.id` is a type identifying a switched-resume coroutine.
  }];
}

def Async_CoroHandleType : Async_Type<"CoroHandle", "coro.handle"> {
  let summary = "coroutine handle";
  let description = [{
    `async.coro.handle` is a handle to the coroutine (pointer to the coroutine
    frame) that can be passed around to resume or destroy the coroutine.
  }];
}

def Async_CoroStateType : Async_Type<"CoroState", "coro.state"> {
  let summary = "saved coroutine state";
  let description = [{
    `async.coro.state` is a saved coroutine state that should be passed to the
    coroutine suspension operation.
  }];
}

#endif // MLIR_DIALECT_ASYNC_IR_ASYNCTYPES
