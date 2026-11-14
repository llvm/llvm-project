// RUN: mlir-opt %s --buffer-deallocation-simplification --split-input-file | FileCheck %s

// Three independent alloc roots are split into separate dealloc operations.
// memref.copy makes allocations observable so they cannot be trivially erased.
func.func @split_distinct_alloc_roots(%ret: memref<1xi8>, %src: memref<1xi8>, %c0: i1, %c1: i1, %c2: i1) -> i1 {
  %a0 = memref.alloc() : memref<1xi8>
  %a1 = memref.alloc() : memref<1xi8>
  %a2 = memref.alloc() : memref<1xi8>
  memref.copy %src, %a0 : memref<1xi8> to memref<1xi8>
  memref.copy %src, %a1 : memref<1xi8> to memref<1xi8>
  memref.copy %src, %a2 : memref<1xi8> to memref<1xi8>
  %out = bufferization.dealloc (%a0, %a1, %a2 : memref<1xi8>, memref<1xi8>, memref<1xi8>) if (%c0, %c1, %c2) retain (%ret : memref<1xi8>)
  return %out : i1
}

// CHECK-LABEL: func.func @split_distinct_alloc_roots
// CHECK-COUNT-3: bufferization.dealloc ({{.*}} : memref<1xi8>) if (
// CHECK-NOT: bufferization.dealloc ({{.*}}, {{.*}} :
// CHECK: return %{{.*}} : i1

// -----

// Two views of the same allocation must NOT be split (possible alias).
func.func @do_not_split_same_allocation_root(%ret: memref<1xi8>, %c0: i1, %c1: i1) -> i1 {
  %a = memref.alloc() : memref<2xi8>
  %v0 = memref.subview %a[0][1][1] : memref<2xi8> to memref<1xi8>
  %v1 = memref.subview %a[1][1][1] : memref<2xi8> to memref<1xi8, strided<[1], offset: 1>>
  %out = bufferization.dealloc (%v0, %v1 : memref<1xi8>, memref<1xi8, strided<[1], offset: 1>>) if (%c0, %c1) retain (%ret : memref<1xi8>)
  return %out : i1
}

// CHECK-LABEL: func.func @do_not_split_same_allocation_root
// CHECK: bufferization.dealloc ({{.*}}, {{.*}} : memref<1xi8>, memref<1xi8, strided<[1], offset: 1>>)
// CHECK: return %{{.*}} : i1

// -----

// Unknown origins use the original conservative alias analysis.
func.func @unknown_origins_keep_conservative_fallback(%a: memref<1xi8>, %b: memref<1xi8>, %ret: memref<1xi8>, %c0: i1, %c1: i1) -> i1 {
  %out = bufferization.dealloc (%a, %b : memref<1xi8>, memref<1xi8>) if (%c0, %c1) retain (%ret : memref<1xi8>)
  return %out : i1
}

// CHECK-LABEL: func.func @unknown_origins_keep_conservative_fallback
// CHECK: bufferization.dealloc ({{.*}}, {{.*}} : memref<1xi8>, memref<1xi8>)
// CHECK: return %{{.*}} : i1


// -----

// A shared-root group stays together while an independent allocation splits.
func.func @same_root_group_plus_one_independent(%src: memref<1xi8>, %ret: memref<1xi8>, %c0: i1, %c1: i1, %c2: i1) -> i1 {
  %a = memref.alloc() : memref<2xi8>
  %v0 = memref.subview %a[0][1][1] : memref<2xi8> to memref<1xi8>
  %v1 = memref.subview %a[1][1][1] : memref<2xi8> to memref<1xi8, strided<[1], offset: 1>>
  %b = memref.alloc() : memref<1xi8>
  memref.copy %src, %b : memref<1xi8> to memref<1xi8>
  %res = bufferization.dealloc (%v0, %v1, %b : memref<1xi8>, memref<1xi8, strided<[1], offset: 1>>, memref<1xi8>) if (%c0, %c1, %c2) retain (%ret : memref<1xi8>)
  return %res : i1
}

// CHECK-LABEL: func.func @same_root_group_plus_one_independent
// CHECK: bufferization.dealloc ({{.*}} : memref<1xi8>) if (
// CHECK: bufferization.dealloc ({{.*}}, {{.*}} : memref<1xi8>, memref<1xi8, strided<[1], offset: 1>>) if (
// CHECK: return %{{.*}} : i1

// -----

// A non-local memref selects the existing alias-analysis path.
func.func @mixed_unknown_and_known_roots(%unknown: memref<1xi8>, %ret: memref<1xi8>, %condition: i1) -> i1 {
  %a0 = memref.alloc() : memref<1xi8>
  %a1 = memref.alloc() : memref<1xi8>
  %res = bufferization.dealloc (%unknown, %a0, %a1 : memref<1xi8>, memref<1xi8>, memref<1xi8>) if (%condition, %condition, %condition) retain (%ret : memref<1xi8>)
  return %res : i1
}

// CHECK-LABEL: func.func @mixed_unknown_and_known_roots
// CHECK: bufferization.dealloc ({{.*}} : memref<1xi8>) if (
// CHECK: return %{{.*}} : i1
