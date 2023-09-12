// RUN: mlir-opt -verify-diagnostics -expand-realloc=emit-deallocs=false -ownership-based-buffer-deallocation \
// RUN:  --buffer-deallocation-simplification -split-input-file %s | FileCheck %s

// RUN: mlir-opt %s -buffer-deallocation-pipeline --split-input-file > /dev/null

func.func @auto_dealloc() {
  %c10 = arith.constant 10 : index
  %c100 = arith.constant 100 : index
  %alloc = memref.alloc(%c10) : memref<?xi32>
  %realloc = memref.realloc %alloc(%c100) : memref<?xi32> to memref<?xi32>
  "test.memref_user"(%realloc) : (memref<?xi32>) -> ()
  return
}

// CHECK-LABEL: func @auto_dealloc
//       CHECK:  [[ALLOC:%.*]] = memref.alloc(
//   CHECK-NOT:  bufferization.dealloc
//       CHECK:  [[V0:%.+]]:2 = scf.if
//   CHECK-NOT:  bufferization.dealloc
//       CHECK:  test.memref_user
//  CHECK-NEXT:  [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[V0]]#0
//  CHECK-NEXT:  bufferization.dealloc ([[ALLOC]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, [[V0]]#1)
//  CHECK-NEXT:  return

// -----

func.func @auto_dealloc_inside_nested_region(%arg0: memref<?xi32>, %arg1: i1) {
  %c100 = arith.constant 100 : index
  %0 = scf.if %arg1 -> memref<?xi32> {
    %realloc = memref.realloc %arg0(%c100) : memref<?xi32> to memref<?xi32>
    scf.yield %realloc : memref<?xi32>
  } else {
    scf.yield %arg0 : memref<?xi32>
  }
  "test.memref_user"(%0) : (memref<?xi32>) -> ()
  return
}

// CHECK-LABEL: func @auto_dealloc_inside_nested_region
//  CHECK-SAME: (%arg0: memref<?xi32>, %arg1: i1)
//   CHECK-NOT: dealloc
//       CHECK: "test.memref_user"([[V0:%.+]]#0)
//  CHECK-NEXT: [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//  CHECK-NEXT: bufferization.dealloc ([[BASE]] : memref<i32>) if ([[V0]]#1)
//  CHECK-NEXT: return
