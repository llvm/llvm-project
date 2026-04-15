// RUN: mlir-opt %s -split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @subf_canonicalize
llvm.func @subf_canonicalize(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: %[[NEG_ARG1:.*]] = llvm.fneg %arg1 : f32
  // CHECK: %[[ADD_RESULT:.*]] = nvvm.addf %arg0, %[[NEG_ARG1]] : f32
  %0 = nvvm.subf %arg0, %arg1 : f32
  llvm.return %0 : f32
}

// -----

// CHECK-LABEL: @nvvm_barrier_fold_id_zero
llvm.func @nvvm_barrier_fold_id_zero() {
  // CHECK-NOT: llvm.mlir.constant
  // CHECK: nvvm.barrier
  // CHECK-NOT: id =
  %c0 = llvm.mlir.constant(0 : i32) : i32
  nvvm.barrier id = %c0
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_barrier_keep_id_nonzero
llvm.func @nvvm_barrier_keep_id_nonzero() {
  // CHECK: %[[C5:.*]] = llvm.mlir.constant(5 : i32) : i32
  // CHECK: nvvm.barrier id = %[[C5]]
  %c5 = llvm.mlir.constant(5 : i32) : i32
  nvvm.barrier id = %c5
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_barrier_fold_id_zero_with_count
llvm.func @nvvm_barrier_fold_id_zero_with_count(%n : i32) {
  // CHECK: nvvm.barrier number_of_threads = %{{.*}}
  // CHECK-NOT: id =
  %c0 = llvm.mlir.constant(0 : i32) : i32
  nvvm.barrier id = %c0 number_of_threads = %n
  llvm.return
}
