// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, VariablePointers, Linkage], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]> {
  spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.array<4xf32>, StorageBuffer>
  spirv.func @fmain() -> i32 "None" {
    %0 = spirv.Constant 16 : i32
    %1 = spirv.mlir.addressof @var1 : !spirv.ptr<!spirv.array<4xf32>, StorageBuffer>
    // CHECK: {{%.*}} = spirv.FunctionCall @f_0({{%.*}}) : (i32) -> i32
    %3 = spirv.FunctionCall @f_0(%0) : (i32) -> i32
    // CHECK: spirv.FunctionCall @f_1({{%.*}}, {{%.*}}) : (i32, !spirv.ptr<!spirv.array<4 x f32>, StorageBuffer>) -> ()
    spirv.FunctionCall @f_1(%3, %1) : (i32, !spirv.ptr<!spirv.array<4xf32>, StorageBuffer>) ->  ()
    // CHECK: {{%.*}} =  spirv.FunctionCall @f_2({{%.*}}) : (!spirv.ptr<!spirv.array<4 x f32>, StorageBuffer>) -> !spirv.ptr<!spirv.array<4 x f32>, StorageBuffer>
    %4 = spirv.FunctionCall @f_2(%1) : (!spirv.ptr<!spirv.array<4xf32>, StorageBuffer>) -> !spirv.ptr<!spirv.array<4xf32>, StorageBuffer>
    spirv.ReturnValue %3 : i32
  }
  spirv.func @f_0(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
  spirv.func @f_1(%arg0 : i32, %arg1 : !spirv.ptr<!spirv.array<4xf32>, StorageBuffer>) -> () "None" {
    spirv.Return
  }
  spirv.func @f_2(%arg0 : !spirv.ptr<!spirv.array<4xf32>, StorageBuffer>) -> !spirv.ptr<!spirv.array<4xf32>, StorageBuffer> "None" {
    spirv.ReturnValue %arg0 : !spirv.ptr<!spirv.array<4xf32>, StorageBuffer>
  }

  spirv.func @f_loop_with_function_call(%count : i32) -> () "None" {
    %zero = spirv.Constant 0: i32
    %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>
    spirv.mlir.loop {
      spirv.Branch ^header
    ^header:
      %val0 = spirv.Load "Function" %var : i32
      %cmp = spirv.SLessThan %val0, %count : i32
      spirv.BranchConditional %cmp, ^body, ^merge
    ^body:
      spirv.Branch ^continue
    ^continue:
      // CHECK: spirv.FunctionCall @f_inc({{%.*}}) : (!spirv.ptr<i32, Function>) -> ()
      spirv.FunctionCall @f_inc(%var) : (!spirv.ptr<i32, Function>) -> ()
      spirv.Branch ^header
    ^merge:
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.func @f_inc(%arg0 : !spirv.ptr<i32, Function>) -> () "None" {
      %one = spirv.Constant 1 : i32
      %0 = spirv.Load "Function" %arg0 : i32
      %1 = spirv.IAdd %0, %one : i32
      spirv.Store "Function" %arg0, %1 : i32
      spirv.Return
  }
}
