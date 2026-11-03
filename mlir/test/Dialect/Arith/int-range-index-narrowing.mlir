// RUN: mlir-opt %s -arith-int-range-narrowing='int-bitwidths-supported=32,64' | FileCheck %s
// RUN: mlir-opt %s -canonicalize | FileCheck %s

module @index32 attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>} {
  // TODO: Preserve this lossy round trip. The folder currently treats index
  // as 64 bits and returns the input. For input 4294967296, the correct result
  // is 0 after truncation to 32 bits and sign extension to 64 bits.
  // CHECK-LABEL: func.func @index_cast_roundtrip_32
  // CHECK-NEXT: return %arg0
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
  // TODO: Preserve this round trip until the runtime index width is known.
  // The folder currently assumes 64 bits and returns the input.
  // CHECK-LABEL: func.func @unknown_width
  // CHECK-NEXT: return %arg0
  func.func @unknown_width(%arg: i64) -> i64 {
    %i = arith.index_castui %arg : i64 to index
    %r = arith.index_castui %i : index to i64
    return %r : i64
  }
}
