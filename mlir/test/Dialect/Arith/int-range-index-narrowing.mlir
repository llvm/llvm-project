// RUN: mlir-opt %s -arith-int-range-narrowing='int-bitwidths-supported=32,64' | FileCheck %s
// RUN: mlir-opt %s -canonicalize | FileCheck %s

module @index32 attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>} {
  // CHECK-LABEL: func.func @memref_roundtrip
  // CHECK-NEXT: return %arg0
  func.func @memref_roundtrip(%arg: memref<?xi32>) -> memref<?xi32> {
    %i = arith.index_cast %arg : memref<?xi32> to memref<?xindex>
    %r = arith.index_cast %i : memref<?xindex> to memref<?xi32>
    return %r : memref<?xi32>
  }

  // CHECK-LABEL: func.func @index_cast_roundtrip_32
  // CHECK: %[[I:.*]] = arith.index_cast %arg0 : i64 to index
  // CHECK: %[[R:.*]] = arith.index_cast %[[I]] : index to i64
  // CHECK: return %[[R]]
  func.func @index_cast_roundtrip_32(%arg: i64) -> i64 {
    %i = arith.index_cast %arg : i64 to index
    %r = arith.index_cast %i : index to i64
    return %r : i64
  }
}

module @index64 attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>} {
  // CHECK-LABEL: func.func @remove_lossless_casts
  // CHECK-NEXT: return %arg0
  func.func @remove_lossless_casts(%arg: i64) -> i64 {
    %i = arith.index_cast %arg : i64 to index
    %r = arith.index_cast %i : index to i64
    return %r : i64
  }
}

module @unknown {
  // CHECK-LABEL: func.func @memref_roundtrip_unknown
  // CHECK-NEXT: return %arg0
  func.func @memref_roundtrip_unknown(%arg: memref<?xi64>) -> memref<?xi64> {
    %i = arith.index_castui %arg : memref<?xi64> to memref<?xindex>
    %r = arith.index_castui %i : memref<?xindex> to memref<?xi64>
    return %r : memref<?xi64>
  }

  // At an index width of 16, this round trip truncates the input.
  // CHECK-LABEL: func.func @unknown_width_i32
  // CHECK: %[[I:.*]] = arith.index_cast %arg0 : i32 to index
  // CHECK: %[[R:.*]] = arith.index_cast %[[I]] : index to i32
  // CHECK: return %[[R]]
  func.func @unknown_width_i32(%arg: i32) -> i32 {
    %i = arith.index_cast %arg : i32 to index
    %r = arith.index_cast %i : index to i32
    return %r : i32
  }


  // CHECK-LABEL: func.func @unknown_width
  // CHECK: %[[I:.*]] = arith.index_castui %arg0 : i64 to index
  // CHECK: %[[R:.*]] = arith.index_castui %[[I]] : index to i64
  // CHECK: return %[[R]]
  func.func @unknown_width(%arg: i64) -> i64 {
    %i = arith.index_castui %arg : i64 to index
    %r = arith.index_castui %i : index to i64
    return %r : i64
  }
}
