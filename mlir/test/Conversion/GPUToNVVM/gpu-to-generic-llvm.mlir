// RUN: mlir-opt %s -convert-gpu-to-nvvm -split-input-file | FileCheck %s

/// Math/arith ops that are not supported by libdevice
/// should be converted by generic LLVM lowering patterns.

gpu.module @generic_llvm_test_module_0 {
  // CHECK-LABEL: @arith_add
  func.func @arith_add(%left: i64, %right: i64) -> i64 {
    // CHECK: llvm.add {{.*}}, {{.*}} : i64
    %result = arith.addi %left, %right : i64
    return %result : i64
  }
}

gpu.module @generic_llvm_test_module_1 {
  // CHECK-LABEL: @math_abs_non_i32
  func.func @math_abs_non_i32(%arg_i64: i64, %arg_i16: i16, %arg_i8: i8, %arg_i1: i1) 
      -> (i64, i16, i8, i1) {
    // CHECK: "llvm.intr.abs"{{.*}} : (i64) -> i64
    %abs_i64 = math.absi %arg_i64 : i64
    // CHECK: "llvm.intr.abs"{{.*}} : (i16) -> i16
    %abs_i16 = math.absi %arg_i16 : i16
    // CHECK: "llvm.intr.abs"{{.*}} : (i8) -> i8
    %abs_i8 = math.absi %arg_i8 : i8
    // CHECK: "llvm.intr.abs"{{.*}} : (i1) -> i1
    %abs_i1 = math.absi %arg_i1 : i1
    return %abs_i64, %abs_i16, %abs_i8, %abs_i1 : i64, i16, i8, i1
  }
}
