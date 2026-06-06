// RUN: mlir-opt -mlir-print-unique-ssa-ids %s | FileCheck %s
// RUN: mlir-opt -mlir-print-op-generic %s | FileCheck %s
// RUN: mlir-opt %s | FileCheck %s --check-prefix=LOCAL_SCOPE

// CHECK: %arg3
// CHECK: %7
// LOCAL_SCOPE-NOT: %arg3
// LOCAL_SCOPE-NOT: %7
module {
  func.func @uniqueSSAIDs(%arg0 : memref<i32>, %arg1 : memref<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %arg2 = %c0 to %c8 step %c1 {
      %a = memref.load %arg0[] : memref<i32>
      %b = memref.load %arg1[] : memref<i32>
      %0 = arith.addi %a, %b : i32
      %1 = arith.subi %a, %b : i32
      scf.yield
    }
    scf.for %arg2 = %c0 to %c8 step %c1 {
      %a = memref.load %arg0[] : memref<i32>
      %b = memref.load %arg1[] : memref<i32>
      %0 = arith.addi %a, %b : i32
      %1 = arith.subi %a, %b : i32
      scf.yield
    }
    return
  }
}
