// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.GlobalVariable @var1 : !spv.ptr<!spv.array<4xf32>, Input>
  spv.func @fmain() -> i32 "None" {
    %0 = spv.Constant 16 : i32
    %1 = spv.mlir.addressof @var1 : !spv.ptr<!spv.array<4xf32>, Input>
    // CHECK: {{%.*}} = spv.FunctionCall @f_0({{%.*}}) : (i32) -> i32
    %3 = spv.FunctionCall @f_0(%0) : (i32) -> i32
    // CHECK: spv.FunctionCall @f_1({{%.*}}, {{%.*}}) : (i32, !spv.ptr<!spv.array<4 x f32>, Input>) -> ()
    spv.FunctionCall @f_1(%3, %1) : (i32, !spv.ptr<!spv.array<4xf32>, Input>) ->  ()
    // CHECK: {{%.*}} =  spv.FunctionCall @f_2({{%.*}}) : (!spv.ptr<!spv.array<4 x f32>, Input>) -> !spv.ptr<!spv.array<4 x f32>, Input>
    %4 = spv.FunctionCall @f_2(%1) : (!spv.ptr<!spv.array<4xf32>, Input>) -> !spv.ptr<!spv.array<4xf32>, Input>
    spv.ReturnValue %3 : i32
  }
  spv.func @f_0(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
  spv.func @f_1(%arg0 : i32, %arg1 : !spv.ptr<!spv.array<4xf32>, Input>) -> () "None" {
    spv.Return
  }
  spv.func @f_2(%arg0 : !spv.ptr<!spv.array<4xf32>, Input>) -> !spv.ptr<!spv.array<4xf32>, Input> "None" {
    spv.ReturnValue %arg0 : !spv.ptr<!spv.array<4xf32>, Input>
  }

  spv.func @f_loop_with_function_call(%count : i32) -> () "None" {
    %zero = spv.Constant 0: i32
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>
    spv.mlir.loop {
      spv.Branch ^header
    ^header:
      %val0 = spv.Load "Function" %var : i32
      %cmp = spv.SLessThan %val0, %count : i32
      spv.BranchConditional %cmp, ^body, ^merge
    ^body:
      spv.Branch ^continue
    ^continue:
      // CHECK: spv.FunctionCall @f_inc({{%.*}}) : (!spv.ptr<i32, Function>) -> ()
      spv.FunctionCall @f_inc(%var) : (!spv.ptr<i32, Function>) -> ()
      spv.Branch ^header
    ^merge:
      spv.mlir.merge
    }
    spv.Return
  }
  spv.func @f_inc(%arg0 : !spv.ptr<i32, Function>) -> () "None" {
      %one = spv.Constant 1 : i32
      %0 = spv.Load "Function" %arg0 : i32
      %1 = spv.IAdd %0, %one : i32
      spv.Store "Function" %arg0, %1 : i32
      spv.Return
  }
}
