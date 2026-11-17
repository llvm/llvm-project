// RUN: mlir-opt %s -canonicalize | FileCheck %s

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = 42 : i32
  spirv.SpecConstant @sc2 = 7 : i32

  // CHECK-LABEL: @spec_constant_operation_one_spec_constant
  spirv.func @spec_constant_operation_one_spec_constant() -> i32 "None" {
    // CHECK-DAG: %[[SC:.*]] = spirv.mlir.referenceof @sc1 : i32
    // CHECK-DAG: %[[CST:.*]] = spirv.Constant 1 : i32
    %0 = spirv.mlir.referenceof @sc1 : i32
    %1 = spirv.Constant 1 : i32

    // CHECK: %[[RES:.*]] = spirv.SpecConstantOperation wraps "spirv.IAdd"(%[[SC]], %[[CST]]) : (i32, i32) -> i32
    %2 = spirv.SpecConstantOperation wraps "spirv.IAdd"(%0, %1) : (i32, i32) -> i32

    // CHECK: spirv.ReturnValue %[[RES]] : i32
    spirv.ReturnValue %2 : i32
  }

  // CHECK-LABEL: @spec_constant_operation_two_spec_constants
  spirv.func @spec_constant_operation_two_spec_constants() -> i32 "None" {
    // CHECK: %[[SC1:.*]] = spirv.mlir.referenceof @sc1 : i32
    // CHECK: %[[SC2:.*]] = spirv.mlir.referenceof @sc2 : i32
    %0 = spirv.mlir.referenceof @sc1 : i32
    %1 = spirv.mlir.referenceof @sc2 : i32

    // CHECK: %[[RES:.*]] = spirv.SpecConstantOperation wraps "spirv.IAdd"(%[[SC1]], %[[SC2]]) : (i32, i32) -> i32
    %2 = spirv.SpecConstantOperation wraps "spirv.IAdd"(%0, %1) : (i32, i32) -> i32

    // CHECK: spirv.ReturnValue %[[RES]] : i32
    spirv.ReturnValue %2 : i32
  }
}
