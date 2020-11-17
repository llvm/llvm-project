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

def AsyncParallelFor : FunctionPass<"async-parallel-for"> {
  let summary = "Convert scf.parallel operations to multiple async regions "
                "executed concurrently for non-overlapping iteration ranges";
  let constructor = "mlir::createAsyncParallelForPass()";
  let options = [
    Option<"numConcurrentAsyncExecute", "num-concurrent-async-execute",
      "int32_t", /*default=*/"4",
      "The number of async.execute operations that will be used for concurrent "
      "loop execution.">
  ];
  let dependentDialects = ["async::AsyncDialect", "scf::SCFDialect"];
}

#endif // MLIR_DIALECT_ASYNC_PASSES
