// RUN: mlir-opt %s -split-input-file | FileCheck %s

func.func @const() -> () {
  // CHECK: %true
  %0 = spv.Constant true
  // CHECK: %false
  %1 = spv.Constant false

  // CHECK: %cst42_i32
  %2 = spv.Constant 42 : i32
  // CHECK: %cst-42_i32
  %-2 = spv.Constant -42 : i32
  // CHECK: %cst43_i64
  %3 = spv.Constant 43 : i64

  // CHECK: %cst_f32
  %4 = spv.Constant 0.5 : f32
  // CHECK: %cst_f64
  %5 = spv.Constant 0.5 : f64

  // CHECK: %cst_vec_3xi32 
  %6 = spv.Constant dense<[1, 2, 3]> : vector<3xi32>

  // CHECK: %cst
  %8 = spv.Constant [dense<3.0> : vector<2xf32>] : !spv.array<1xvector<2xf32>>

  return
}

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @global_var : !spv.ptr<f32, Input>

  spv.func @addressof() -> () "None" {
    // CHECK: %global_var_addr = spv.mlir.addressof 
    %0 = spv.mlir.addressof @global_var : !spv.ptr<f32, Input>
    spv.Return
  }
}

