// RUN: mlir-opt %s --transform-interpreter --test-transform-dialect-erase-schedule --split-input-file | FileCheck %s

// Suppressing a target precondition failure must not leave materialized tile
// sizes behind. Do not run CSE or canonicalization: they could erase that IR.
// CHECK-LABEL: func.func @unsupported_target
// CHECK-NEXT: %[[SUM:.*]] = arith.addi
// CHECK-NEXT: return %[[SUM]]
// CHECK-NEXT: }
func.func @unsupported_target(%lhs: i32, %rhs: i32) -> i32 {
  %sum = arith.addi %lhs, %rhs : i32
  return %sum : i32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.sequence %root : !transform.any_op failures(suppress) {
    ^bb0(%scope: !transform.any_op):
      %target = transform.structured.match ops{["arith.addi"]} in %scope : (!transform.any_op) -> !transform.any_op
      %tiled, %loop = transform.structured.tile_using_forall %target tile_sizes [[8], 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
    transform.yield
  }
}

// -----

// Memref pack implements TilingInterface, but its semantics are unsupported.
// CHECK-LABEL: func.func @unsupported_memref_pack
// CHECK-NEXT: linalg.pack
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @unsupported_memref_pack(%source: memref<128x256xf32>, %dest: memref<4x8x32x32xf32>) {
  linalg.pack %source inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %dest : memref<128x256xf32> -> memref<4x8x32x32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.sequence %root : !transform.any_op failures(suppress) {
    ^bb0(%scope: !transform.any_op):
      %target = transform.structured.match ops{["linalg.pack"]} in %scope : (!transform.any_op) -> !transform.any_op
      %tiled, %loop = transform.structured.tile_using_forall %target tile_sizes [[8], 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @unsupported_memref_unpack
// CHECK-NEXT: linalg.unpack
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @unsupported_memref_unpack(%source: memref<4x8x32x32xf32>, %dest: memref<128x256xf32>) {
  linalg.unpack %source inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %dest : memref<4x8x32x32xf32> -> memref<128x256xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.sequence %root : !transform.any_op failures(suppress) {
    ^bb0(%scope: !transform.any_op):
      %target = transform.structured.match ops{["linalg.unpack"]} in %scope : (!transform.any_op) -> !transform.any_op
      %tiled, %loop = transform.structured.tile_using_forall %target tile_sizes [[8], 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
    transform.yield
  }
}
