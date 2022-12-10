// RUN: mlir-opt %s -split-input-file | FileCheck %s

func.func @const() -> () {
  // CHECK: %true
  %0 = spirv.Constant true
  // CHECK: %false
  %1 = spirv.Constant false

  // CHECK: %cst42_i32
  %2 = spirv.Constant 42 : i32
  // CHECK: %cst-42_i32
  %-2 = spirv.Constant -42 : i32
  // CHECK: %cst43_i64
  %3 = spirv.Constant 43 : i64

  // CHECK: %cst_f32
  %4 = spirv.Constant 0.5 : f32
  // CHECK: %cst_f64
  %5 = spirv.Constant 0.5 : f64

  // CHECK: %cst_vec_3xi32
  %6 = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>

  // CHECK: %cst
  %8 = spirv.Constant [dense<3.0> : vector<2xf32>] : !spirv.array<1xvector<2xf32>>

  return
}

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @global_var : !spirv.ptr<f32, Input>

  spirv.func @addressof() -> () "None" {
    // CHECK: %global_var_addr = spirv.mlir.addressof
    %0 = spirv.mlir.addressof @global_var : !spirv.ptr<f32, Input>
    spirv.Return
  }
}

