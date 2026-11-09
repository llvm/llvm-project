// RUN: mlir-opt %s -test-strided-metadata-range-analysis -split-input-file 2>&1 | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>} {
  // CHECK: Op: %subview = memref.subview
  // CHECK: offset = [{unsigned : [5, 5] signed : [5, 5]}]
  func.func @offset(%a: memref<16xf32>) -> memref<4xf32, strided<[1], offset: ?>> {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %offset = arith.addi %c2, %c3 : index
    %r = memref.subview %a[%offset] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
    return %r : memref<4xf32, strided<[1], offset: ?>>
  }
}

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>} {
  // CHECK: Op: %subview = memref.subview
  // CHECK: offset = [{unsigned : [5, 5] signed : [5, 5]}]
  func.func @offset(%a: memref<16xf32>) -> memref<4xf32, strided<[1], offset: ?>> {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %offset = arith.addi %c2, %c3 : index
    %r = memref.subview %a[%offset] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
    return %r : memref<4xf32, strided<[1], offset: ?>>
  }
}

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<index = 128 : i32>} {
  // CHECK: Op: %subview = memref.subview
  // CHECK: offset = [{unsigned : [5, 5] signed : [5, 5]}]
  func.func @offset(%a: memref<16xf32>) -> memref<4xf32, strided<[1], offset: ?>> {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %offset = arith.addi %c2, %c3 : index
    %r = memref.subview %a[%offset] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
    return %r : memref<4xf32, strided<[1], offset: ?>>
  }
}

// -----

module {
  // CHECK: Op: %subview = memref.subview
  // CHECK: offset = [{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}]
  func.func @offset(%a: memref<16xf32>) -> memref<4xf32, strided<[1], offset: ?>> {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %offset = arith.addi %c2, %c3 : index
    %r = memref.subview %a[%offset] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
    return %r : memref<4xf32, strided<[1], offset: ?>>
  }
}

// -----

module {
  module attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>} {
    // CHECK: Op: %subview = memref.subview
    // CHECK: offset = [{unsigned : [5, 5] signed : [5, 5]}]
    func.func @nested(%a: memref<16xf32>) -> memref<4xf32, strided<[1], offset: ?>> {
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %offset = arith.addi %c2, %c3 : index
      %r = memref.subview %a[%offset] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
      return %r : memref<4xf32, strided<[1], offset: ?>>
    }
  }
}
