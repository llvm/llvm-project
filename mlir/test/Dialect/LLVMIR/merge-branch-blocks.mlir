// RUN: mlir-opt --llvm-merge-branch-blocks %s -split-input-file | FileCheck %s

// This file only proves the standalone pass runs the branch patterns
// independently of full canonicalization, that it leaves
// optimize-none functions untouched, and that merging does not fold
// other operations. The feature itself is tested in canonicalize.mlir.

// A chain of branches to single-predecessor blocks merges into one block.
// CHECK-LABEL: llvm.func @merge_chain
// CHECK-NOT: llvm.br
// CHECK: %[[V:.*]] = llvm.add %{{.*}}, %{{.*}} : i32
// CHECK: llvm.return %[[V]] : i32
llvm.func @merge_chain(%arg0 : i32) -> i32 {
  llvm.br ^bb1
^bb1:
  %0 = llvm.add %arg0, %arg0 : i32
  llvm.br ^bb2
^bb2:
  llvm.return %0 : i32
}

// -----

// Functions with optimize-none are left untouched.
// CHECK-LABEL: llvm.func @no_merge_optnone
// CHECK: llvm.br ^bb1
llvm.func @no_merge_optnone(%arg0 : i32) -> i32 attributes {optimize_none, no_inline} {
  llvm.br ^bb1
^bb1:
  %0 = llvm.add %arg0, %arg0 : i32
  llvm.return %0 : i32
}

// -----

// Merging must not fold other operations: the compare stays even though
// both operands are the same value after the merge.
// CHECK-LABEL: llvm.func @no_fold_after_merge
// CHECK: %[[C:.*]] = llvm.icmp "eq" %{{.*}}, %{{.*}} : i32
// CHECK: llvm.return %[[C]] : i1
llvm.func @no_fold_after_merge(%arg0 : i32) -> i1 {
  llvm.br ^bb1
^bb1:
  %0 = llvm.icmp "eq" %arg0, %arg0 : i32
  llvm.return %0 : i1
}
