//===- AsyncOps.td - Async operations definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for Async dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNC_OPS
#define ASYNC_OPS

include "mlir/Dialect/Async/IR/AsyncBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Async op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class Async_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<AsyncDialect, mnemonic, traits>;

def Async_ExecuteOp : Async_Op<"execute", [AttrSizedOperandSegments]> {
  let summary = "Asynchronous execute operation";
  let description = [{
    The `body` region attached to the `async.execute` operation semantically
    can be executed concurrently with the successor operation. In the followup
    example "compute0" can be executed concurrently with "compute1".

    The actual concurrency semantics depends on the dialect lowering to the
    executable format. Fully sequential execution ("compute0" completes before
    "compute1" starts) is a completely legal execution.

    Because concurrent execution is not guaranteed, it is illegal to create an
    implicit dependency from "compute1" to "compute0" (e.g. via shared global
    state). All dependencies must be made explicit with async execute arguments
    (`async.token` or `async.value`).

   `async.execute` operation takes `async.token` dependencies and `async.value`
    operands separatly, and starts execution of the attached body region only
    when all tokens and values become ready.

    Example:

    ```mlir
    %dependency = ... : !async.token
    %value = ... : !async.value<f32>

    %token, %results =
      async.execute [%dependency](%value as %unwrapped: !async.value<f32>)
                 -> !async.value<!some.type>
      {
        %0 = "compute0"(%unwrapped): (f32) -> !some.type
        async.yield %0 : !some.type
      }

    %1 = "compute1"(...) : !some.type
    ```

    In the example above asynchronous execution starts only after dependency
    token and value argument become ready. Unwrapped value passed to the
    attached body region as an %unwrapped value of f32 type.
  }];

  let arguments = (ins Variadic<Async_TokenType>:$dependencies,
                       Variadic<Async_AnyValueOrTokenType>:$operands);

  let results = (outs Async_TokenType:$token,
                      Variadic<Async_AnyValueType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let verifier = [{ return ::verify(*this); }];
}

def Async_YieldOp :
    Async_Op<"yield", [HasParent<"ExecuteOp">, NoSideEffect, Terminator]> {
  let summary = "terminator for Async execute operation";
  let description = [{
    The `async.yield` is a special terminator operation for the block inside
    `async.execute` operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let verifier = [{ return ::verify(*this); }];
}

#endif // ASYNC_OPS
