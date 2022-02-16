// RUN: mlir-opt -allow-unregistered-dialect -gpu-launch-sink-index-computations -split-input-file -verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: @extra_constants
// CHECK-SAME: %[[ARG0:.*]]: memref<?xf32>
func @extra_constants(%arg0: memref<?xf32>) {
  %cst = arith.constant 8 : index
  %cst2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %cst3 = memref.dim %arg0, %c0 : memref<?xf32>
  // CHECK: gpu.launch blocks
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst,
                                       %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst,
                                        %block_z = %cst) {
    // CHECK-NOT: arith.constant 8
    // CHECK: %[[CST2:.*]] = arith.constant 2
    // CHECK-NEXT: %[[CST0:.*]] = arith.constant 0
    // CHECK-NEXT: %[[DIM:.*]] = memref.dim %[[ARG0]], %[[CST0]]
    // CHECK-NEXT: "use"(%[[CST2]], %[[ARG0]], %[[DIM]]) : (index, memref<?xf32>, index) -> ()
    // CHECK-NEXT: gpu.terminator
    "use"(%cst2, %arg0, %cst3) : (index, memref<?xf32>, index) -> ()
    gpu.terminator
  }
  return
}

// -----

// CHECK-LABEL: @extra_constants_not_inlined
// CHECK-SAME: %[[ARG0:.*]]: memref<?xf32>
func @extra_constants_not_inlined(%arg0: memref<?xf32>) {
  %cst = arith.constant 8 : index
  %cst2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[CST_X:.*]] = "secret_constant"()
  %cst3 = "secret_constant"() : () -> index
  // CHECK: gpu.launch blocks
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst,
                                       %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst,
                                        %block_z = %cst) {
    // CHECK-NOT: arith.constant 8
    // CHECK-NOT: "secret_constant"()
    // CHECK: %[[CST2:.*]] = arith.constant 2
    // CHECK-NEXT: "use"(%[[CST2]], %[[ARG0]], %[[CST_X]]) : (index, memref<?xf32>, index) -> ()
    // CHECK-NEXT: gpu.terminator
    "use"(%cst2, %arg0, %cst3) : (index, memref<?xf32>, index) -> ()
    gpu.terminator
  }
  return
}

// -----

// CHECK-LABEL: @multiple_uses
func @multiple_uses(%arg0 : memref<?xf32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: gpu.launch blocks
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1,
                                       %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1,
                                        %block_z = %c1) {
    // CHECK: %[[C2:.*]] = arith.constant 2
    // CHECK-NEXT: "use1"(%[[C2]], %[[C2]])
    // CHECK-NEXT: "use2"(%[[C2]])
    // CHECK-NEXT: gpu.terminator
    "use1"(%c2, %c2) : (index, index) -> ()
    "use2"(%c2) : (index) -> ()
    gpu.terminator
  }
  return
}

// -----

// CHECK-LABEL: @multiple_uses2
func @multiple_uses2(%arg0 : memref<*xf32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d = memref.dim %arg0, %c2 : memref<*xf32>
  // CHECK: gpu.launch blocks
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1,
                                       %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1,
                                        %block_z = %c1) {
    // CHECK: %[[C2:.*]] = arith.constant 2 : index
    // CHECK: %[[D:.*]] = memref.dim %[[ARG:.*]], %[[C2]]
    // CHECK: "use1"(%[[D]])
    // CHECK: "use2"(%[[C2]], %[[C2]])
    // CHECK: "use3"(%[[ARG]])
    // CHECK: gpu.terminator
    "use1"(%d) : (index) -> ()
    "use2"(%c2, %c2) : (index, index) -> ()
    "use3"(%arg0) : (memref<*xf32>) -> ()
    gpu.terminator
  }
  return
}
