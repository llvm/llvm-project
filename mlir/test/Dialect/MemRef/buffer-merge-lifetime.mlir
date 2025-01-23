// RUN: mlir-opt -allow-unregistered-dialect -p 'builtin.module(func.func(merge-alloc{analysis-only}))'  %s | FileCheck %s

// CHECK-DAG: func.func @basic() -> memref<8x64xf32>  attributes {__mergealloc_scope = [[TOPSCOPE:[0-9]+]]
func.func @basic() -> memref<8x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %ctrue = arith.constant 1 : i1
  // b is used in return, complex lifetime
  // CHECK-DAG: %[[B:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], -2, -2>}
  %b = memref.alloc() : memref<8x64xf32>
  "test.source"(%b)  : (memref<8x64xf32>) -> ()
  // c and d has overlapping lifetime
  // CHECK-DAG: %[[C:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], 11, 14>}
  %c = memref.alloc() : memref<8x64xf32>
  "test.source"(%c)  : (memref<8x64xf32>) -> ()
  // CHECK-DAG: %[[D:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], 13, 13>}
  %d = memref.alloc() : memref<8x64xf32>
  "test.source"(%d)  : (memref<8x64xf32>) -> ()
  "test.source"(%c)  : (memref<8x64xf32>) -> ()
  // e and f have overlapping lifetime due to the loop
  // CHECK-DAG: %[[E:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], 17, 22>}
  // CHECK-DAG: %[[F:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], 17, 22>}
  %e = memref.alloc() : memref<8x64xf32>
  %f = memref.alloc() : memref<8x64xf32>
  // CHECK: scf.for
  scf.for %i = %c0 to %c5 step %c1 {
    "test.source"(%e)  : (memref<8x64xf32>) -> ()
    "test.source"(%f)  : (memref<8x64xf32>) -> ()
    // CHECK-DAG: %[[G:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], 21, 21>}
    %g = memref.alloc() : memref<8x64xf32>
    "test.source"(%g)  : (memref<8x64xf32>) -> ()
  }
  // CHECK-DAG: %[[H:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], 24, 39>}
  %h = memref.alloc() : memref<8x64xf32>
  // CHECK: scf.forall
  scf.forall (%iv) in (%c5) {
    // check that the alloc in the forall should switch to another scope id
    // CHECK-NOT: array<i64: [[TOPSCOPE]]
    // CHECK-DAG: %[[L:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[FORSCOPE:[0-9]+]], 27, 27>}
    %l = memref.alloc() : memref<8x64xf32>
    "test.source"(%h)  : (memref<8x64xf32>) -> ()
    "test.source"(%l)  : (memref<8x64xf32>) -> ()
    scf.for %i = %c0 to %c5 step %c1 {
      // CHECK-DAG: %[[G:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[FORSCOPE]], 30, 30>}
      %g = memref.alloc() : memref<8x64xf32>
      "test.source"(%g)  : (memref<8x64xf32>) -> ()
    }
    // CHECK-DAG: %[[K:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[FORSCOPE]], 33, 38>}
    %k = memref.alloc() : memref<8x64xf32>
    scf.if %ctrue {
      // CHECK-DAG: %[[J:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[FORSCOPE]], 35, 35>}
      %j = memref.alloc() : memref<8x64xf32>
      "test.source"(%j)  : (memref<8x64xf32>) -> ()
    } else {
      "test.source"(%k)  : (memref<8x64xf32>) -> ()
    }
    // CHECK-DAG: {__mergealloc_scope = [[FORSCOPE]] : i64}
  }
  return %b : memref<8x64xf32>
}

// CHECK-DAG: func.func @basic2() attributes {__mergealloc_scope = [[TOPSCOPE2:[0-9]+]]
func.func @basic2() {
  // CHECK-DAG: %[[B:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE2]], 4, 6>}
  %b = memref.alloc() : memref<8x64xi8>
  %cur = memref.subview %b[1,0][1,64][1,1] : memref<8x64xi8> to memref<1x64xi8, strided<[64, 1], offset: 64>>
  "test.source"(%cur)  : (memref<1x64xi8, strided<[64, 1], offset: 64>>) -> ()
  %cur2 = memref.subview %cur[0,0][1,16][1,1] : memref<1x64xi8, strided<[64, 1], offset: 64>> to memref<1x16xi8, strided<[64, 1], offset: 64>>
  "test.source"(%cur2)  : (memref<1x16xi8, strided<[64, 1], offset: 64>>) -> ()
  // CHECK-DAG: %[[C:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE2]], 8, 8>}
  %c = memref.alloc() : memref<8x64xi8>
  "test.source"(%c)  : (memref<8x64xi8>) -> ()
  return
}

// check that the operations without memory effects do not contribute to the lifetime of the buffer
// CHECK-DAG: func.func @no_mem_effect() attributes {__mergealloc_scope = [[TOPSCOPE3:[0-9]+]]
func.func @no_mem_effect() {
  %c0 = arith.constant 0 : index
  // CHECK: %[[B:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE3]], 5, 5>}
  %b = memref.alloc() : memref<8x64xi8>
  %x = memref.dim %b, %c0 : memref<8x64xi8>
  "test.source"(%b)  : (memref<8x64xi8>) -> ()
  return
}

// check that if pointer is extracted, the alloc is untraceable
// CHECK-DAG: func.func @extract_pointer() attributes {__mergealloc_scope = [[TOPSCOPEExt:[0-9]+]]
func.func @extract_pointer() {
  // CHECK: %[[BExt:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPEExt]], -2, -2>}
  %b = memref.alloc() : memref<8x64xi8>
  %0 = memref.extract_aligned_pointer_as_index %b : memref<8x64xi8> -> index
  "test.source"(%b)  : (memref<8x64xi8>) -> ()
  return
}

// check that Alias Buffers' lifetimes work well
// CHECK-DAG: func.func @alias_ref(%[[ARG0:.*]]: i1) attributes {__mergealloc_scope = [[TOPSCOPE4:[0-9]+]]
func.func @alias_ref(%pred : i1) {
  // CHECK: %[[A:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE4]], 5, 5>}
  %a = memref.alloc() : memref<8x64xi8>
  // CHECK: %[[B:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE4]], 5, 6>}
  %b = memref.alloc() : memref<8x64xi8>
  %c = arith.select %pred, %a, %b : i1, memref<8x64xi8>
  "test.source"(%c)  : (memref<8x64xi8>) -> ()
  "test.source"(%b)  : (memref<8x64xi8>) -> ()
  return
}

// CHECK-DAG: func.func @escape_from_if()  attributes {__mergealloc_scope = [[TOPSCOPE5:[0-9]+]]
func.func @escape_from_if() {
  %ctrue = arith.constant 1 : i1
  // check that f lives at the whole range of the following scf.if 
  // CHECK-DAG: %[[F:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE5]], 4, 13>}
  %f = memref.alloc() : memref<8x64xf32>
  // tick of the scf.if starts from 4 and ends at 14
  // CHECK: scf.if
  %c = scf.if %ctrue -> memref<8x64xf32> {
    "test.source"(%f)  : (memref<8x64xf32>) -> ()
    // CHECK-DAG: %[[G:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE5]], 4, 14>}
    %g = memref.alloc() : memref<8x64xf32>
    "test.source"(%g)  : (memref<8x64xf32>) -> ()
    scf.yield %g : memref<8x64xf32>
  } else {
    // h fully overlaps with g
    // CHECK-DAG: %[[H:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE5]], 4, 14>}
    %h = memref.alloc() : memref<8x64xf32>
    "test.source"(%h)  : (memref<8x64xf32>) -> ()
    // J only used in the scf.if, don't need conservative lifetime
    // CHECK-DAG: %[[J:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE5]], 12, 12>}
    %j = memref.alloc() : memref<8x64xf32>
    "test.source"(%j)  : (memref<8x64xf32>) -> ()
    scf.yield %h : memref<8x64xf32>
  }
  "test.source"(%c)  : (memref<8x64xf32>) -> ()
  return
}

// CHECK-DAG: func.func @escape_from_for()  attributes {__mergealloc_scope = [[TOPSCOPE6:[0-9]+]]
func.func @escape_from_for() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  // check that f has untraceable lifetime, due to being yielded by for loop
  // CHECK-DAG: %[[F:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE6]], -2, -2>}
  %f = memref.alloc() : memref<8x64xf32>
  %out = scf.for %i = %c0 to %c5 step %c1 iter_args(%buf = %f) -> (memref<8x64xf32>) {
    "test.source"(%buf)  : (memref<8x64xf32>) -> ()
    // check that f has untraceable lifetime, due to being yielded by for loop
    // CHECK-DAG: %[[G:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE6]], -2, -2>}
    %g = memref.alloc() : memref<8x64xf32>
    "test.source"(%g)  : (memref<8x64xf32>) -> ()
    %ctrue = "test.source"()  : () -> i1
    %c = scf.if %ctrue -> memref<8x64xf32> {
      scf.yield %g : memref<8x64xf32>
    } else {
      scf.yield %buf : memref<8x64xf32>
    }
    scf.yield %c : memref<8x64xf32>
  }
  "test.source"(%out)  : (memref<8x64xf32>) -> ()
  return
}
