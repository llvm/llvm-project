// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

  func.func @warp_extract(%arg0: index, %arg1: memref<1024x1024xf32>, %arg2: index, %arg3: vector<1xf32>) {
    %c0 = arith.constant 0 : index
    vector.warp_execute_on_lane_0(%arg0)[32] {
      // CHECK:%[[val:[0-9]+]] = llvm.extractelement
      // CHECK:%[[base:[0-9]+]] = llvm.extractvalue
      // CHECK:%[[ptr:[0-9]+]] = llvm.getelementptr %[[base]]
      // CHECK:llvm.store %[[val]], %[[ptr]]
      vector.transfer_write %arg3, %arg1[%c0, %c0] {in_bounds = [true]} : vector<1xf32>, memref<1024x1024xf32>
    }
    return
  }
