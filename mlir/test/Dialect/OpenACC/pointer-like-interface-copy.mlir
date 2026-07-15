// RUN: mlir-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(test-acc-pointer-like-interface{test-mode=copy}))" 2>&1 | FileCheck %s

func.func @test_copy_static() {
  %src = memref.alloca() {test.src_ptr} : memref<10x20xf32>
  %dest = memref.alloca() {test.dest_ptr} : memref<10x20xf32>
  
  // CHECK: Successfully generated copy from source: %[[SRC:.*]] = memref.alloca() {test.src_ptr} : memref<10x20xf32> to destination: %[[DEST:.*]] = memref.alloca() {test.dest_ptr} : memref<10x20xf32>
  // CHECK: Generated: memref.copy %[[SRC]], %[[DEST]] : memref<10x20xf32> to memref<10x20xf32>
  return
}

// -----

func.func @test_copy_dynamic() {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %src = memref.alloc(%c10, %c20) {test.src_ptr} : memref<?x?xf32>
  %dest = memref.alloc(%c10, %c20) {test.dest_ptr} : memref<?x?xf32>
  
  // CHECK: Successfully generated copy from source: %[[SRC:.*]] = memref.alloc(%[[C10:.*]], %[[C20:.*]]) {test.src_ptr} : memref<?x?xf32> to destination: %[[DEST:.*]] = memref.alloc(%[[C10]], %[[C20]]) {test.dest_ptr} : memref<?x?xf32>
  // CHECK: Generated: memref.copy %[[SRC]], %[[DEST]] : memref<?x?xf32> to memref<?x?xf32>
  return
}
