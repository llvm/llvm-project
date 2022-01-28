//===-- Passes.td - Bufferization passes definition file ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def BufferDeallocation : Pass<"buffer-deallocation", "FuncOp"> {
  let summary = "Adds all required dealloc operations for all allocations in "
                "the input program";
  let description = [{
    This pass implements an algorithm to automatically introduce all required
    deallocation operations for all buffers in the input program. This ensures
    that the resulting program does not have any memory leaks.


    Input

    ```mlir
    #map0 = affine_map<(d0) -> (d0)>
    module {
      func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
        cond_br %arg0, ^bb1, ^bb2
      ^bb1:
        br ^bb3(%arg1 : memref<2xf32>)
      ^bb2:
        %0 = memref.alloc() : memref<2xf32>
        linalg.generic {
          args_in = 1 : i64,
          args_out = 1 : i64,
          indexing_maps = [#map0, #map0],
          iterator_types = ["parallel"]} %arg1, %0 {
        ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
          %tmp1 = exp %gen1_arg0 : f32
          linalg.yield %tmp1 : f32
        }: memref<2xf32>, memref<2xf32>
        br ^bb3(%0 : memref<2xf32>)
      ^bb3(%1: memref<2xf32>):
        "linalg.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
        return
      }
    }

    ```

    Output

    ```mlir
    #map0 = affine_map<(d0) -> (d0)>
    module {
      func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
        cond_br %arg0, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %0 = memref.alloc() : memref<2xf32>
        linalg.copy(%arg1, %0) : memref<2xf32>, memref<2xf32>
        br ^bb3(%0 : memref<2xf32>)
      ^bb2:  // pred: ^bb0
        %1 = memref.alloc() : memref<2xf32>
        linalg.generic {
          args_in = 1 : i64,
          args_out = 1 : i64,
          indexing_maps = [#map0, #map0],
          iterator_types = ["parallel"]} %arg1, %1 {
        ^bb0(%arg3: f32, %arg4: f32):
          %4 = exp %arg3 : f32
          linalg.yield %4 : f32
        }: memref<2xf32>, memref<2xf32>
        %2 = memref.alloc() : memref<2xf32>
        linalg.copy(%1, %2) : memref<2xf32>, memref<2xf32>
        dealloc %1 : memref<2xf32>
        br ^bb3(%2 : memref<2xf32>)
      ^bb3(%3: memref<2xf32>):  // 2 preds: ^bb1, ^bb2
        linalg.copy(%3, %arg2) : memref<2xf32>, memref<2xf32>
        dealloc %3 : memref<2xf32>
        return
      }

    }
    ```

  }];
  let constructor = "mlir::bufferization::createBufferDeallocationPass()";
}

def BufferHoisting : Pass<"buffer-hoisting", "FuncOp"> {
  let summary = "Optimizes placement of allocation operations by moving them "
                "into common dominators and out of nested regions";
  let description = [{
    This pass implements an approach to aggressively move allocations upwards
    into common dominators and out of nested regions.
  }];
  let constructor = "mlir::bufferization::createBufferHoistingPass()";
}

def BufferLoopHoisting : Pass<"buffer-loop-hoisting", "FuncOp"> {
  let summary = "Optimizes placement of allocation operations by moving them "
                "out of loop nests";
  let description = [{
    This pass implements an approach to aggressively move allocations upwards
    out of loop nests. It does not move allocations into common dominators.
  }];
  let constructor = "mlir::bufferization::createBufferLoopHoistingPass()";
}

def BufferResultsToOutParams : Pass<"buffer-results-to-out-params", "ModuleOp">  {
  let summary = "Converts memref-typed function results to out-params";
  let description = [{
    Some calling conventions prefer to pass output memrefs as "out params". The
    conversion to this calling convention must be done as an atomic
    transformation of the entire program (hence this is a module pass).

    For example, if a call is rewritten, the callee needs to be rewritten
    otherwise the IR will end up invalid. Thus, this transformation
    require an atomic change to the entire program (e.g. the whole module).

    This pass is expected to run immediately after bufferization is finished.
    At that point, tensor-typed results will have been converted to memref-typed
    results, and can be consistently converted to out params.

    All memref-typed results are appended to the function argument list.

    The main issue with this pass (and the out-param calling convention) is that
    buffers for results need to be allocated in the caller. This currently only
    works for static shaped memrefs.
  }];
  let constructor = "mlir::bufferization::createBufferResultsToOutParamsPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def FinalizingBufferize : Pass<"finalizing-bufferize", "FuncOp"> {
  let summary = "Finalize a partial bufferization";
  let description = [{
    A bufferize pass that finalizes a partial bufferization by removing
    remaining `bufferization.to_tensor` and `bufferization.to_buffer` operations.

    The removal of those operations is only possible if the operations only
    exist in pairs, i.e., all uses of `bufferization.to_tensor` operations are
    `bufferization.to_buffer` operations.

    This pass will fail if not all operations can be removed or if any operation
    with tensor typed operands remains.
  }];
  let constructor = "mlir::bufferization::createFinalizingBufferizePass()";
}

def PromoteBuffersToStack : Pass<"promote-buffers-to-stack", "FuncOp"> {
  let summary = "Promotes heap-based allocations to automatically managed "
                "stack-based allocations";
  let description = [{
    This pass implements a simple algorithm to convert heap-based memory
    allocations to stack-based ones. It uses a built-in heuristic to decide
    whether it makes sense to convert an allocation. Furthermore, dynamic
    shaped buffers that are limited by the rank of the tensor can be
    converted. They are only transformed if they are considered to be small.
  }];
  let constructor = "mlir::bufferization::createPromoteBuffersToStackPass()";
  let options = [
    Option<"maxAllocSizeInBytes", "max-alloc-size-in-bytes", "unsigned",
           /*default=*/"1024",
           "Maximal size in bytes to promote allocations to stack.">,
    Option<"maxRankOfAllocatedMemRef", "max-rank-of-allocated-memref", "unsigned",
           /*default=*/"1",
           "Maximal memref rank to promote dynamic buffers.">,
  ];
}

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES
