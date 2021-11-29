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

def BufferDeallocation : FunctionPass<"buffer-deallocation"> {
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
        %0 = alloc() : memref<2xf32>
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
        %0 = alloc() : memref<2xf32>
        linalg.copy(%arg1, %0) : memref<2xf32>, memref<2xf32>
        br ^bb3(%0 : memref<2xf32>)
      ^bb2:  // pred: ^bb0
        %1 = alloc() : memref<2xf32>
        linalg.generic {
          args_in = 1 : i64,
          args_out = 1 : i64,
          indexing_maps = [#map0, #map0],
          iterator_types = ["parallel"]} %arg1, %1 {
        ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
          %4 = exp %arg3 : f32
          linalg.yield %4 : f32
        }: memref<2xf32>, memref<2xf32>
        %2 = alloc() : memref<2xf32>
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

def FinalizingBufferize : FunctionPass<"finalizing-bufferize"> {
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

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES
