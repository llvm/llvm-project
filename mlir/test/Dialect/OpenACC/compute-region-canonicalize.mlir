// RUN: mlir-opt -canonicalize -split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func @merge_duplicate_ins
func.func @merge_duplicate_ins() -> i32 {
  %c0 = arith.constant 0 : i32
  %m = memref.alloca() : memref<i32>
  memref.store %c0, %m[] : memref<i32>
  acc.compute_region ins(%a = %m, %b = %m) : (memref<i32>, memref<i32>) {
    %c1 = arith.constant 1 : i32
    %v = memref.load %a[] : memref<i32>
    %x = arith.addi %v, %c1 : i32
    memref.store %x, %a[] : memref<i32>
    acc.yield
  } {origin = "acc.serial"}
  %r = memref.load %m[] : memref<i32>
  return %r : i32
}
// CHECK: acc.compute_region ins({{.*}}) : (memref<i32>) {

// -----

// CHECK-LABEL: func @merge_duplicate_ins_complex_pattern
func.func @merge_duplicate_ins_complex_pattern() -> i32 {
  %c0 = arith.constant 0 : i32
  %ma = memref.alloca() : memref<i32>
  %mb = memref.alloca() : memref<i32>
  %mc = memref.alloca() : memref<i32>
  memref.store %c0, %ma[] : memref<i32>
  memref.store %c0, %mb[] : memref<i32>
  memref.store %c0, %mc[] : memref<i32>
  acc.compute_region ins(%a0 = %ma, %b0 = %mb, %a1 = %ma, %mc0 = %mc, %mc1 = %mc, %b1 = %mb, %a2 = %ma) : (memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>) {
    %one = arith.constant 1 : i32
    %v0 = memref.load %a0[] : memref<i32>
    %v1 = memref.load %b0[] : memref<i32>
    %v2 = memref.load %a1[] : memref<i32>
    %v3 = memref.load %mc0[] : memref<i32>
    %v4 = memref.load %mc1[] : memref<i32>
    %v5 = memref.load %b1[] : memref<i32>
    %v6 = memref.load %a2[] : memref<i32>
    %sum1 = arith.addi %v0, %v1 : i32
    %sum2 = arith.addi %sum1, %v2 : i32
    %sum3 = arith.addi %sum2, %v3 : i32
    %sum4 = arith.addi %sum3, %v4 : i32
    %sum5 = arith.addi %sum4, %v5 : i32
    %sum6 = arith.addi %sum5, %v6 : i32
    %out = arith.addi %sum6, %one : i32
    memref.store %out, %a0[] : memref<i32>
    acc.yield
  } {origin = "acc.serial"}
  %r = memref.load %ma[] : memref<i32>
  return %r : i32
}
// CHECK: acc.compute_region ins({{.*}}) : (memref<i32>, memref<i32>, memref<i32>) {

// -----

// CHECK-LABEL: func @drop_unused_ins
func.func @drop_unused_ins() -> i32 {
  %c0 = arith.constant 0 : i32
  %ma = memref.alloca() : memref<i32>
  %mb = memref.alloca() : memref<i32>
  %mc = memref.alloca() : memref<i32>
  memref.store %c0, %ma[] : memref<i32>
  memref.store %c0, %mb[] : memref<i32>
  memref.store %c0, %mc[] : memref<i32>
  acc.compute_region ins(%a = %ma, %b = %mb, %c = %mc) : (memref<i32>, memref<i32>, memref<i32>) {
    %c1 = arith.constant 1 : i32
    %v = memref.load %a[] : memref<i32>
    %x = arith.addi %v, %c1 : i32
    memref.store %x, %a[] : memref<i32>
    acc.yield
  } {origin = "acc.serial"}
  %r = memref.load %ma[] : memref<i32>
  return %r : i32
}
// CHECK: acc.compute_region ins({{.*}}) : (memref<i32>) {
