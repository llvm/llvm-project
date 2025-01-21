// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module {
  func.func @func(%arg: vector<11xf32>) -> vector<11xf32> {
    %cst_41 = arith.constant dense<true> : vector<11xi1>
    // CHECK: vector.mask
    // CHECK-SAME: vector.yield %arg0
    %127 = vector.mask %cst_41 { vector.yield %arg : vector<11xf32> } : vector<11xi1> -> vector<11xf32>
    return %127 : vector<11xf32>
  }
}
