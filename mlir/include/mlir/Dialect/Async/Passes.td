//===-- Passes.td - Async pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ASYNC_PASSES
#define MLIR_DIALECT_ASYNC_PASSES

include "mlir/Pass/PassBase.td"

def AsyncParallelFor : Pass<"async-parallel-for", "ModuleOp"> {
  let summary = "Convert scf.parallel operations to multiple async compute ops "
                "executed concurrently for non-overlapping iteration ranges";
  let constructor = "mlir::createAsyncParallelForPass()";

  let options = [
    Option<"asyncDispatch", "async-dispatch",
      "bool", /*default=*/"true",
      "Dispatch async compute tasks using recursive work splitting. If `false` "
      "async compute tasks will be launched using simple for loop in the "
      "caller thread.">,

    Option<"numWorkerThreads", "num-workers",
      "int32_t", /*default=*/"8",
      "The number of available workers to execute async operations. If `-1` "
      "the value will be retrieved from the runtime.">,

    Option<"minTaskSize", "min-task-size",
      "int32_t", /*default=*/"1000",
      "The minimum task size for sharding parallel operation.">
  ];

  let dependentDialects = [
    "arith::ArithDialect",
    "async::AsyncDialect",
    "scf::SCFDialect"
  ];
}

def AsyncToAsyncRuntime : Pass<"async-to-async-runtime", "ModuleOp"> {
  let summary = "Lower high level async operations (e.g. async.execute) to the"
                "explicit async.runtime and async.coro operations";
  let constructor = "mlir::createAsyncToAsyncRuntimePass()";
  let options = [
    // Temporary for bringup, should become the default.
    Option<"eliminateBlockingAwaitOps", "eliminate-blocking-await-ops", "bool",
            /*default=*/"false",
           "Rewrite functions with blocking async.runtime.await as coroutines "
           "with async.runtime.await_and_resume.">,
  ];
  let dependentDialects = ["async::AsyncDialect"];
}

def AsyncRuntimeRefCounting : Pass<"async-runtime-ref-counting"> {
  let summary = "Automatic reference counting for Async runtime operations";
  let description = [{
    This pass works at the async runtime abtraction level, after all
    `async.execute` and `async.await` operations are lowered to the async
    runtime API calls, and async coroutine operations.

    It relies on the LLVM coroutines switched-resume lowering semantics for
    the correct placing of the reference counting operations.

    See: https://llvm.org/docs/Coroutines.html#switched-resume-lowering
  }];

  let constructor = "mlir::createAsyncRuntimeRefCountingPass()";
  let dependentDialects = ["async::AsyncDialect"];
}

def AsyncRuntimeRefCountingOpt : Pass<"async-runtime-ref-counting-opt"> {
  let summary = "Optimize automatic reference counting operations for the"
                "Async runtime by removing redundant operations";
  let constructor = "mlir::createAsyncRuntimeRefCountingOptPass()";
  let dependentDialects = ["async::AsyncDialect"];
}

def AsyncRuntimePolicyBasedRefCounting
    : Pass<"async-runtime-policy-based-ref-counting"> {
  let summary = "Policy based reference counting for Async runtime operations";
  let description = [{
    This pass works at the async runtime abtraction level, after all
    `async.execute` and `async.await` operations are lowered to the async
    runtime API calls, and async coroutine operations.

    This pass doesn't rely on reference counted values liveness analysis, and
    instead uses simple policy to create reference counting operations. If the
    program violates any of the assumptions, then this pass might lead to
    memory leaks or runtime errors.

    The default reference counting policy assumptions:
      1. Async token can be awaited or added to the group only once.
      2. Async value or group can be awaited only once.

    Under these assumptions reference counting only needs to drop reference:
      1. After `async.runtime.await` operation for async tokens and groups
         (until error handling is not implemented for the sync await).
      2. After `async.runtime.is_error` operation for async tokens and groups
         (this is the last operation in the coroutine resume function).
      3. After `async.runtime.load` operation for async values.

    This pass introduces significanly less runtime overhead compared to the
    automatic reference counting.
  }];

  let constructor = "mlir::createAsyncRuntimePolicyBasedRefCountingPass()";
  let dependentDialects = ["async::AsyncDialect"];
}

#endif // MLIR_DIALECT_ASYNC_PASSES
