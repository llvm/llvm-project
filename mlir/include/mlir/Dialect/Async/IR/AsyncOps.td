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
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Async op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class Async_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<AsyncDialect, mnemonic, traits>;

def Async_ExecuteOp :
  Async_Op<"execute", [SingleBlockImplicitTerminator<"YieldOp">,
                       DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                                 ["getNumRegionInvocations"]>,
                       AttrSizedOperandSegments]> {
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
    operands separately, and starts execution of the attached body region only
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

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(ins "TypeRange":$resultTypes, "ValueRange":$dependencies,
      "ValueRange":$operands,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
           "nullptr">:$bodyBuilder)>,
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, ValueRange)>;

  }];
}

def Async_YieldOp :
    Async_Op<"yield", [HasParent<"ExecuteOp">, NoSideEffect, Terminator]> {
  let summary = "terminator for Async execute operation";
  let description = [{
    The `async.yield` is a special terminator operation for the block inside
    `async.execute` operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";

  let verifier = [{ return ::verify(*this); }];
}

def Async_AwaitOp : Async_Op<"await"> {
  let summary = "waits for the argument to become ready";
  let description = [{
    The `async.await` operation waits until the argument becomes ready, and for
    the `async.value` arguments it unwraps the underlying value

    Example:

    ```mlir
    %0 = ... : !async.token
    async.await %0 : !async.token

    %1 = ... : !async.value<f32>
    %2 = async.await %1 : !async.value<f32>
    ```
  }];

  let arguments = (ins Async_AnyValueOrTokenType:$operand);
  let results = (outs Optional<AnyType>:$result);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilderDAG<(ins "Value":$operand,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = [{
    Optional<Type> getResultType() {
      if (getResultTypes().empty()) return None;
      return getResultTypes()[0];
    }
  }];

  let assemblyFormat = [{
    $operand `:` custom<AwaitResultType>(
      type($operand), type($result)
    ) attr-dict
  }];

  let verifier = [{ return ::verify(*this); }];
}

def Async_CreateGroupOp : Async_Op<"create_group", [NoSideEffect]> {
  let summary = "creates an empty async group";
  let description = [{
    The `async.create_group` allocates an empty async group. Async tokens or
    values can be added to this group later.

    Example:

    ```mlir
    %0 = async.create_group
    ...
    async.await_all %0
    ```
  }];

  let arguments = (ins );
  let results = (outs Async_GroupType:$result);

  let assemblyFormat = "attr-dict";
}

def Async_AddToGroupOp : Async_Op<"add_to_group", []> {
  let summary = "adds and async token or value to the group";
  let description = [{
    The `async.add_to_group` adds an async token or value to the async group.
    Returns the rank of the added element in the group. This rank is fixed
    for the group lifetime.

    Example:

    ```mlir
    %0 = async.create_group
    %1 = ... : !async.token
    %2 = async.add_to_group %1, %0 : !async.token
    ```
  }];

  let arguments = (ins Async_AnyValueOrTokenType:$operand,
                       Async_GroupType:$group);
  let results = (outs Index:$rank);

  let assemblyFormat = "$operand `,` $group `:` type($operand) attr-dict";
}

def Async_AwaitAllOp : Async_Op<"await_all", []> {
  let summary = "waits for the all async tokens or values in the group to "
                "become ready";
  let description = [{
    The `async.await_all` operation waits until all the tokens or values in the
    group become ready.

    Example:

    ```mlir
    %0 = async.create_group

    %1 = ... : !async.token
    %2 = async.add_to_group %1, %0 : !async.token

    %3 = ... : !async.token
    %4 = async.add_to_group %2, %0 : !async.token

    async.await_all %0
    ```
  }];

  let arguments = (ins Async_GroupType:$operand);
  let results = (outs);

  let assemblyFormat = "$operand attr-dict";
}

#endif // ASYNC_OPS
