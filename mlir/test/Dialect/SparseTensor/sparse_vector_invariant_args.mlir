// RUN: mlir-opt %s --sparse-vectorization='vl=4' | FileCheck %s

// CHECK-LABEL: func.func @invariant_base(
// CHECK: scf.for {{.*}} step %{{.*}} {
// CHECK: vector.load
// CHECK: arith.muli {{.*}} : vector<4xi32>
// CHECK: vector.store

func.func @invariant_base(%a: memref<?xi32>, %base: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %two = arith.constant 2 : i32

  scf.for %i = %c0 to %c64 step %c1 {
    %p = arith.addi %base, %i : index
    %x = memref.load %a[%p] : memref<?xi32>
    %y = arith.muli %x, %two : i32
    memref.store %y, %a[%p] : memref<?xi32>
  } {"Emitted from" = "linalg.generic"}

  return
}

// CHECK-LABEL: func.func @invariant_shift(
// CHECK: scf.for {{.*}} step %{{.*}} {
// CHECK: vector.load
// CHECK: vector.broadcast {{.*}} : i32 to vector<4xi32>
// CHECK: arith.shli {{.*}} : vector<4xi32>
// CHECK: vector.store

func.func @invariant_shift(%a: memref<?xi32>, %shift: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  scf.for %i = %c0 to %c64 step %c1 {
    %x = memref.load %a[%i] : memref<?xi32>
    %y = arith.shli %x, %shift : i32
    memref.store %y, %a[%i] : memref<?xi32>
  } {"Emitted from" = "linalg.generic"}

  return
}

// CHECK-LABEL: func.func @invariant_base_reversed(
// CHECK: scf.for {{.*}} step %{{.*}} {
// CHECK: vector.load
// CHECK: arith.muli {{.*}} : vector<4xi32>
// CHECK: vector.store

func.func @invariant_base_reversed(%a: memref<?xi32>, %base: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %two = arith.constant 2 : i32

  scf.for %i = %c0 to %c64 step %c1 {
    %p = arith.addi %i, %base : index
    %x = memref.load %a[%p] : memref<?xi32>
    %y = arith.muli %x, %two : i32
    memref.store %y, %a[%p] : memref<?xi32>
  } {"Emitted from" = "linalg.generic"}

  return
}

// CHECK-LABEL: func.func @non_contiguous(
// CHECK: scf.for {{.*}} step %{{.*}} {
// CHECK: arith.muli {{.*}} : index
// CHECK: memref.load
// CHECK: arith.muli {{.*}} : i32
// CHECK: memref.store

func.func @non_contiguous(%a: memref<?xi32>, %stride: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %two = arith.constant 2 : i32

  scf.for %i = %c0 to %c64 step %c1 {
    %p = arith.muli %stride, %i : index
    %x = memref.load %a[%p] : memref<?xi32>
    %y = arith.muli %x, %two : i32
    memref.store %y, %a[%p] : memref<?xi32>
  } {"Emitted from" = "linalg.generic"}

  return
}
