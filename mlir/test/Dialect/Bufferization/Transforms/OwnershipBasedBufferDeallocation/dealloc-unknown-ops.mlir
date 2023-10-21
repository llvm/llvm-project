// RUN: mlir-opt --allow-unregistered-dialect -verify-diagnostics -ownership-based-buffer-deallocation \
// RUN:  -buffer-deallocation-simplification -split-input-file %s | FileCheck %s

func.func private @callee(%arg0: memref<f32>) -> memref<f32> {
  return %arg0 : memref<f32>
}

func.func @generic_ownership_materialization() {
  %a1 = memref.alloc() : memref<f32>
  %a2 = memref.alloca() : memref<f32>
  %0 = "my_dialect.select_randomly"(%a1, %a2, %a1) : (memref<f32>, memref<f32>, memref<f32>) -> memref<f32>
  %1 = call @callee(%0) : (memref<f32>) -> memref<f32>
  return
}

// CHECK-LABEL: func @generic_ownership_materialization
//       CHECK: [[ALLOC:%.+]] = memref.alloc(
//       CHECK: [[ALLOCA:%.+]] = memref.alloca(
//       CHECK: [[SELECT:%.+]] = "my_dialect.select_randomly"([[ALLOC]], [[ALLOCA]], [[ALLOC]])
//       CHECK: [[SELECT_PTR:%.+]] = memref.extract_aligned_pointer_as_index [[SELECT]]
//       CHECK: [[ALLOCA_PTR:%.+]] = memref.extract_aligned_pointer_as_index [[ALLOCA]]
//       CHECK: [[EQ1:%.+]] = arith.cmpi eq, [[SELECT_PTR]], [[ALLOCA_PTR]]
//       CHECK: [[OWN1:%.+]] = arith.select [[EQ1]], %false{{[0-9_]*}}, %true
//       CHECK: [[ALLOC_PTR:%.+]] = memref.extract_aligned_pointer_as_index [[ALLOC]]
//       CHECK: [[EQ2:%.+]] = arith.cmpi eq, [[SELECT_PTR]], [[ALLOC_PTR]]
//       CHECK: [[OWN2:%.+]] = arith.select [[EQ2]], %true{{[0-9_]*}}, [[OWN1]]
//       CHECK: [[CALL:%.+]]:2 = call @callee([[SELECT]], [[OWN2]])
//       CHECK: [[BASE:%.+]],{{.*}} = memref.extract_strided_metadata [[CALL]]#0
//       CHECK: bufferization.dealloc ([[ALLOC]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, [[CALL]]#1)
