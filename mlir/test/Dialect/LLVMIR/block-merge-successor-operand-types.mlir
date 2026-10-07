// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=aggressive}))' -split-input-file | FileCheck %s

// Merging identical blocks threads their differing values through the
// predecessors' terminators as new successor operands. LLVM dialect
// terminators only forward LLVM-compatible values, so a merge whose new block
// arguments would be, say, of index type must be refused -- it used to be
// performed and produced an llvm.cond_br that no longer verified.

// CHECK-LABEL: func @no_merge_of_non_llvm_types(
func.func @no_merge_of_non_llvm_types(%m: memref<4xf32>, %c: i1) {
  %f = arith.constant 1.0 : f32
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  // CHECK: llvm.cond_br %{{.*}}, ^[[BB1:.*]], ^[[BB2:.*]]
  llvm.cond_br %c, ^a, ^b
  // CHECK: ^[[BB1]]:
  // CHECK: memref.store
^a:
  memref.store %f, %m[%i0] : memref<4xf32>
  llvm.return
  // CHECK: ^[[BB2]]:
  // CHECK: memref.store
^b:
  memref.store %f, %m[%i1] : memref<4xf32>
  llvm.return
}

// -----

// Differing values of an LLVM-compatible type still merge as before.

llvm.func @use(i32)

// CHECK-LABEL: func @merge_of_llvm_types(
func.func @merge_of_llvm_types(%c: i1) {
  %i0 = llvm.mlir.constant(0 : i32) : i32
  %i1 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.cond_br %{{.*}}, ^[[BB:.*]](%{{.*}} : i32), ^[[BB]](%{{.*}} : i32)
  llvm.cond_br %c, ^a, ^b
  // CHECK: ^[[BB]](%[[ARG:.*]]: i32):
  // CHECK: llvm.call @use(%[[ARG]])
^a:
  llvm.call @use(%i0) : (i32) -> ()
  llvm.return
^b:
  llvm.call @use(%i1) : (i32) -> ()
  llvm.return
}
