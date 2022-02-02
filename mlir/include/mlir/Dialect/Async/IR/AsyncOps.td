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

include "mlir/Dialect/Async/IR/AsyncDialect.td"
include "mlir/Dialect/Async/IR/AsyncTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Async op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class Async_Op<string mnemonic, list<Trait> traits = []> :
    Op<AsyncDialect, mnemonic, traits>;

def Async_ExecuteOp :
  Async_Op<"execute", [SingleBlockImplicitTerminator<"YieldOp">,
                       DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                                 ["getSuccessorEntryOperands"]>,
                       AttrSizedOperandSegments,
                       AutomaticAllocationScope]> {
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
                      Variadic<Async_ValueType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let skipDefaultBuilders = 1;
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$dependencies,
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
    Async_Op<"yield", [
      HasParent<"ExecuteOp">, NoSideEffect, Terminator,
      DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
  let summary = "terminator for Async execute operation";
  let description = [{
    The `async.yield` is a special terminator operation for the block inside
    `async.execute` operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let hasVerifier = 1;
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
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$operand,
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
}

def Async_CreateGroupOp : Async_Op<"create_group", [NoSideEffect]> {
  let summary = "creates an empty async group";
  let description = [{
    The `async.create_group` allocates an empty async group. Async tokens or
    values can be added to this group later. The size of the group must be
    specified at construction time, and `await_all` operation will first
    wait until the number of added tokens or values reaches the group size.

    Example:

    ```mlir
    %size = ... : index
    %group = async.create_group %size : !async.group
    ...
    async.await_all %group
    ```
  }];

  let arguments = (ins Index:$size);
  let results = (outs Async_GroupType:$result);

  let hasCanonicalizeMethod = 1;

  let assemblyFormat = "$size `:` type($result) attr-dict";
}

def Async_AddToGroupOp : Async_Op<"add_to_group", []> {
  let summary = "adds and async token or value to the group";
  let description = [{
    The `async.add_to_group` adds an async token or value to the async group.
    Returns the rank of the added element in the group. This rank is fixed
    for the group lifetime.

    Example:

    ```mlir
    %0 = async.create_group %size : !async.group
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
    %0 = async.create_group %size : !async.group

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

//===----------------------------------------------------------------------===//
// Async Dialect LLVM Coroutines Operations.
//===----------------------------------------------------------------------===//

// Async to LLVM dialect lowering converts async tasks (regions inside async
// execute operations) to LLVM coroutines [1], and relies on switched-resume
// lowering [2] to produce an asynchronous executable.
//
// We define LLVM coro intrinsics in the async dialect to facilitate progressive
// lowering with verifiable and type-safe IR during the multi-step lowering
// pipeline. First we convert from high level async operations (e.g. execute) to
// the explicit calls to coro intrinsics and runtime API, and then finalize
// lowering to LLVM with a simple dialect conversion pass.
//
// [1] https://llvm.org/docs/Coroutines.html
// [2] https://llvm.org/docs/Coroutines.html#switched-resume-lowering

def Async_CoroIdOp : Async_Op<"coro.id"> {
  let summary = "returns a switched-resume coroutine identifier";
  let description = [{
    The `async.coro.id` returns a switched-resume coroutine identifier.
  }];

  let results = (outs Async_CoroIdType:$id);
  let assemblyFormat = "attr-dict";
}

def Async_CoroBeginOp : Async_Op<"coro.begin"> {
  let summary = "returns a handle to the coroutine";
  let description = [{
    The `async.coro.begin` allocates a coroutine frame and returns a handle to
    the coroutine.
  }];

  let arguments = (ins Async_CoroIdType:$id);
  let results = (outs Async_CoroHandleType:$handle);
  let assemblyFormat = "$id attr-dict";
}

def Async_CoroFreeOp : Async_Op<"coro.free"> {
  let summary = "deallocates the coroutine frame";
  let description = [{
    The `async.coro.free` deallocates the coroutine frame created by the
    async.coro.begin operation.
  }];

  let arguments = (ins Async_CoroIdType:$id,
                       Async_CoroHandleType:$handle);
  let assemblyFormat = "$id `,` $handle attr-dict";
}

def Async_CoroEndOp : Async_Op<"coro.end"> {
  let summary = "marks the end of the coroutine in the suspend block";
  let description = [{
    The `async.coro.end` marks the point where a coroutine needs to return
    control back to the caller if it is not an initial invocation of the
    coroutine. It the start part of the coroutine is is no-op.
  }];

  let arguments = (ins Async_CoroHandleType:$handle);
  let assemblyFormat = "$handle attr-dict";
}

def Async_CoroSaveOp : Async_Op<"coro.save"> {
  let summary = "saves the coroutine state";
  let description = [{
    The `async.coro.saves` saves the coroutine state.
  }];

  let arguments = (ins Async_CoroHandleType:$handle);
  let results = (outs Async_CoroStateType:$state);
  let assemblyFormat = "$handle attr-dict";
}

def Async_CoroSuspendOp : Async_Op<"coro.suspend", [Terminator]> {
  let summary = "suspends the coroutine";
  let description = [{
    The `async.coro.suspend` suspends the coroutine and transfers control to the
    `suspend` successor. If suspended coroutine later resumed it will transfer
    control to the `resume` successor. If it is destroyed it will transfer
    control to the the `cleanup` successor.

    In switched-resume lowering coroutine can be already in resumed state when
    suspend operation is called, in this case control will be transferred to the
    `resume` successor skipping the `suspend` successor.
  }];

  let arguments = (ins Async_CoroStateType:$state);
  let successors = (successor AnySuccessor:$suspendDest,
                              AnySuccessor:$resumeDest,
                              AnySuccessor:$cleanupDest);
  let assemblyFormat =
    "$state `,` $suspendDest `,` $resumeDest  `,` $cleanupDest attr-dict";
}

//===----------------------------------------------------------------------===//
// Async Dialect Runtime Operations.
//===----------------------------------------------------------------------===//

// The following operations are intermediate async dialect operations to help
// lowering from high level async operation like `async.execute` to the Async
// Runtime API defined in the `ExecutionEngine/AsyncRuntime.h`.

def Async_RuntimeCreateOp : Async_Op<"runtime.create"> {
  let summary = "creates an async runtime token or value";
  let description = [{
    The `async.runtime.create` operation creates an async dialect token or
    value. Tokens and values are created in the non-ready state.
  }];

  let results = (outs Async_AnyValueOrTokenType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Async_RuntimeCreateGroupOp : Async_Op<"runtime.create_group"> {
  let summary = "creates an async runtime group";
  let description = [{
    The `async.runtime.create_group` operation creates an async dialect group
    of the given size. Group created in the empty state.
  }];

  let arguments = (ins Index:$size);
  let results = (outs Async_GroupType:$result);
  let assemblyFormat = "$size `:` type($result) attr-dict ";
}

def Async_RuntimeSetAvailableOp : Async_Op<"runtime.set_available"> {
  let summary = "switches token or value to available state";
  let description = [{
    The `async.runtime.set_available` operation switches async token or value
    state to available.
  }];

  let arguments = (ins Async_AnyValueOrTokenType:$operand);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def Async_RuntimeSetErrorOp : Async_Op<"runtime.set_error"> {
  let summary = "switches token or value to error state";
  let description = [{
    The `async.runtime.set_error` operation switches async token or value
    state to error.
  }];

  let arguments = (ins Async_AnyValueOrTokenType:$operand);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def Async_RuntimeIsErrorOp : Async_Op<"runtime.is_error"> {
  let summary = "returns true if token, value or group is in error state";
  let description = [{
    The `async.runtime.is_error` operation returns true if the token, value or
    group (any of the async runtime values) is in the error state. It is the
    caller responsibility to check error state after the call to `await` or
    resuming after `await_and_resume`.
  }];

  let arguments = (ins Async_AnyAsyncType:$operand);
  let results = (outs I1:$is_error);

  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def Async_RuntimeAwaitOp : Async_Op<"runtime.await"> {
  let summary = "blocks the caller thread until the operand becomes available";
  let description = [{
    The `async.runtime.await` operation blocks the caller thread until the
    operand becomes available or error.
  }];

  let arguments = (ins Async_AnyAsyncType:$operand);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def Async_RuntimeResumeOp : Async_Op<"runtime.resume"> {
  let summary = "resumes the coroutine on a thread managed by the runtime";
  let description = [{
    The `async.runtime.resume` operation resumes the coroutine on a thread
    managed by the runtime.
  }];

  let arguments = (ins Async_CoroHandleType:$handle);
  let assemblyFormat = "$handle attr-dict";
}

def Async_RuntimeAwaitAndResumeOp : Async_Op<"runtime.await_and_resume"> {
  let summary = "awaits the async operand and resumes the coroutine";
  let description = [{
    The `async.runtime.await_and_resume` operation awaits for the operand to
    become available or error and resumes the coroutine on a thread managed by
    the runtime.
  }];

  let arguments = (ins Async_AnyAsyncType:$operand,
                       Async_CoroHandleType:$handle);
  let assemblyFormat = "$operand `,` $handle attr-dict `:` type($operand)";
}

def Async_RuntimeStoreOp : Async_Op<"runtime.store",
      [TypesMatchWith<"type of 'value' matches element type of 'storage'",
                     "storage", "value",
                     "$_self.cast<ValueType>().getValueType()">]> {
  let summary = "stores the value into the runtime async.value";
  let description = [{
    The `async.runtime.store` operation stores the value into the runtime
    async.value storage.
  }];

  let arguments = (ins AnyType:$value,
                       Async_ValueType:$storage);
  let assemblyFormat = "$value `,` $storage attr-dict `:` type($storage)";
}

def Async_RuntimeLoadOp : Async_Op<"runtime.load",
      [TypesMatchWith<"type of 'value' matches element type of 'storage'",
                     "storage", "result",
                     "$_self.cast<ValueType>().getValueType()">]> {
  let summary = "loads the value from the runtime async.value";
  let description = [{
    The `async.runtime.load` operation loads the value from the runtime
    async.value storage.
  }];

  let arguments = (ins Async_ValueType:$storage);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$storage attr-dict `:` type($storage)";
}

def Async_RuntimeAddToGroupOp : Async_Op<"runtime.add_to_group", []> {
  let summary = "adds and async token or value to the group";
  let description = [{
    The `async.runtime.add_to_group` adds an async token or value to the async
    group. Returns the rank of the added element in the group.
  }];

  let arguments = (ins Async_AnyValueOrTokenType:$operand,
                       Async_GroupType:$group);
  let results = (outs Index:$rank);

  let assemblyFormat = "$operand `,` $group attr-dict `:` type($operand)";
}

// All async values (values, tokens, groups) are reference counted at runtime
// and automatically destructed when reference count drops to 0.
//
// All values are semantically created with a reference count of +1 and it is
// the responsibility of the last async value user to drop reference count.
//
// Async values created when:
//   1. Operation returns async result (e.g. the result of an `async.execute`).
//   2. Async value passed in as a block argument.
//
// It is the responsibility of the async value user to extend the lifetime by
// adding a +1 reference, if the reference counted value captured by the
// asynchronously executed region (`async.execute` operation), and drop it after
// the last nested use.
//
// Reference counting operations can be added to the IR using automatic
// reference count pass, that relies on liveness analysis to find the last uses
// of all reference counted values and automatically inserts
// `drop_ref` operations.
//
// See `AsyncRefCountingPass` documentation for the implementation details.

def Async_RuntimeAddRefOp : Async_Op<"runtime.add_ref"> {
  let summary = "adds a reference to async value";
  let description = [{
    The `async.runtime.add_ref` operation adds a reference(s) to async value
    (token, value or group).
  }];

  let arguments = (ins Async_AnyAsyncType:$operand,
                       Confined<I64Attr, [IntPositive]>:$count);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand)
  }];
}

def Async_RuntimeDropRefOp : Async_Op<"runtime.drop_ref"> {
  let summary = "drops a reference to async value";
  let description = [{
    The `async.runtime.drop_ref` operation drops a reference(s) to async value
    (token, value or group).
  }];

  let arguments = (ins Async_AnyAsyncType:$operand,
                       Confined<I64Attr, [IntPositive]>:$count);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand)
  }];
}

def Async_RuntimeNumWorkerThreadsOp :
  Async_Op<"runtime.num_worker_threads",
           [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "gets the number of threads in the threadpool from the runtime";
  let description = [{
    The `async.runtime.num_worker_threads` operation gets the number of threads
    in the threadpool from the runtime.
  }];

  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

#endif // ASYNC_OPS
