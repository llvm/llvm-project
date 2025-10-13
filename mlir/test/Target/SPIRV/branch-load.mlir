// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// CHECK: spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []>
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: spirv.func @main() "None"
  spirv.func @main() "None" {
    // CHECK: %[[VAR:.*]] = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    %0 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Branch ^bb1
  ^bb1:  // pred: ^bb0
    // CHECK: spirv.mlir.selection
    spirv.mlir.selection {
      // CHECK: %[[COND:.*]] = spirv.Constant true
      // CHECK: spirv.BranchConditional %[[COND]]
      %true = spirv.Constant true
      spirv.BranchConditional %true, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      // CHECK: %[[CONST:.*]] = spirv.Constant dense<0.000000e+00> : vector<3xf32>
      // CHECK: spirv.Store "Function" %[[VAR]], %[[CONST]] : vector<3xf32>
      %cst_vec_3xf32 = spirv.Constant dense<0.000000e+00> : vector<3xf32>
      spirv.Store "Function" %0, %cst_vec_3xf32 : vector<3xf32>
      spirv.Branch ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      spirv.mlir.merge
    }
    // CHECK: %[[RESULT:.*]] = spirv.Load "Function" %[[VAR]] : vector<3xf32>
    // CHECK: spirv.Return
    %1 = spirv.Load "Function" %0 : vector<3xf32>
    spirv.Return
  }
  // CHECK: spirv.EntryPoint "Fragment" @main
  // CHECK: spirv.ExecutionMode @main "OriginUpperLeft"
  spirv.EntryPoint "Fragment" @main
  spirv.ExecutionMode @main "OriginUpperLeft"
}
