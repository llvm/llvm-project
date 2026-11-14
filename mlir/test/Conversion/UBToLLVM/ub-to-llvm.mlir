// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-ub-to-llvm))" %s -split-input-file | FileCheck %s

// Same below, but using the `ConvertToLLVMPatternInterface` entry point
// and the generic `convert-to-llvm` pass.
// RUN: mlir-opt --convert-to-llvm="filter-dialects=ub" --split-input-file %s | FileCheck %s
// RUN: mlir-opt --convert-to-llvm="filter-dialects=ub allow-pattern-rollback=0" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @check_poison
func.func @check_poison() {
// CHECK: {{.*}} = llvm.mlir.poison : i64
  %0 = ub.poison : index
// CHECK: {{.*}} = llvm.mlir.poison : i16
  %1 = ub.poison : i16
// CHECK: {{.*}} = llvm.mlir.poison : f64
  %2 = ub.poison : f64
// CHECK: {{.*}} = llvm.mlir.poison : !llvm.ptr
  %3 = ub.poison : !llvm.ptr
  return
}

// CHECK-LABEL: @check_unrechable
func.func @check_unrechable() {
// CHECK: llvm.unreachable
  ub.unreachable
}

// CHECK-LABEL: func.func @drop_assumption(
//  CHECK-SAME: %[[VALUE:[^:]+]]: i32) -> i32 {
//  CHECK-NEXT:   return %[[VALUE]] : i32
func.func @drop_assumption(%value: i32) -> i32 {
  %assumed = ub.assume %value, #arith.assume_cmpi<sgt, 0 : i32>() : i32
  return %assumed : i32
}
