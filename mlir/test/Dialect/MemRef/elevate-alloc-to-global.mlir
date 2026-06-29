// RUN: mlir-opt %s --memref-elevate-alloc-to-global | FileCheck %s

// CHECK-DAG: memref.global "private" @func_with_alloc_alloc : memref<10x20xf32> = uninitialized
// CHECK-DAG: memref.global "private" @func_with_alignment_alloc : memref<8xf32> = uninitialized {alignment = 64 : i64}
// CHECK-DAG: memref.global "private" @multiple_allocs_alloc : memref<4xf32> = uninitialized
// CHECK-DAG: memref.global "private" @multiple_allocs_alloc_0 : memref<4xf32> = uninitialized

// CHECK-LABEL: func.func @func_with_alloc
func.func @func_with_alloc(%arg0: index, %arg1: index, %val: f32) -> f32 {
  // CHECK-NOT: memref.alloc
  // CHECK: %[[GLOBAL:.+]] = memref.get_global @func_with_alloc_alloc : memref<10x20xf32>
  %alloc = memref.alloc() : memref<10x20xf32>

  // CHECK: memref.store %{{.+}}, %[[GLOBAL]][%{{.+}}, %{{.+}}]
  memref.store %val, %alloc[%arg0, %arg1] : memref<10x20xf32>
  // CHECK: %[[LOAD:.+]] = memref.load %[[GLOBAL]][%{{.+}}, %{{.+}}]
  %res = memref.load %alloc[%arg0, %arg1] : memref<10x20xf32>

  // CHECK-NOT: memref.dealloc
  memref.dealloc %alloc : memref<10x20xf32>

  return %res : f32
}

// CHECK-LABEL: func.func @func_with_alignment
func.func @func_with_alignment() {
  // CHECK-NOT: memref.alloc
  // CHECK: memref.get_global @func_with_alignment_alloc : memref<8xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
  memref.dealloc %alloc : memref<8xf32>
  return
}

// CHECK-LABEL: func.func @func_with_dynamic_alloc
func.func @func_with_dynamic_alloc(%size: index) {
  // CHECK: memref.alloc
  // CHECK-NOT: memref.get_global
  %alloc = memref.alloc(%size) : memref<?xf32>
  memref.dealloc %alloc : memref<?xf32>
  return
}

// CHECK-LABEL: func.func @multiple_allocs
func.func @multiple_allocs() {
  // CHECK-NOT: memref.alloc
  // CHECK: memref.get_global @multiple_allocs_alloc : memref<4xf32>
  %a = memref.alloc() : memref<4xf32>
  // CHECK: memref.get_global @multiple_allocs_alloc_0 : memref<4xf32>
  %b = memref.alloc() : memref<4xf32>
  return
}
