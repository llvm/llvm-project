// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: func @func
  // CHECK-SAME: %[[IN:.*]]: vector<11xf32>
  func.func @func(%arg: vector<11xf32>) -> vector<11xf32> {
    %cst_41 = arith.constant dense<true> : vector<11xi1>
    // CHECK-NOT: vector.mask
    // CHECK: return %[[IN]] : vector<11xf32>
    %127 = vector.mask %cst_41 { vector.yield %arg : vector<11xf32> } : vector<11xi1> -> vector<11xf32>
    return %127 : vector<11xf32>
  }
}
