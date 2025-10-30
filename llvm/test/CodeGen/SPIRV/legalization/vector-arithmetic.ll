; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#main:]] "main"
; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#v4f32:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#c16:]] = OpConstant %[[#int]] 16
; CHECK-DAG: %[[#v16f32:]] = OpTypeArray %[[#float]] %[[#c16]]
; CHECK-DAG: %[[#v16i32:]] = OpTypeArray %[[#int]] %[[#c16]]
; CHECK-DAG: %[[#ptr_ssbo_v16i32:]] = OpTypePointer Private %[[#v16i32]]
; CHECK-DAG: %[[#v4i32:]] = OpTypeVector %[[#int]] 4

@f1 = internal addrspace(10) global [4 x [16 x float] ] zeroinitializer
@f2 = internal addrspace(10) global [4 x [16 x float] ] zeroinitializer
@i1 = internal addrspace(10) global [4 x [16 x i32] ] zeroinitializer
@i2 = internal addrspace(10) global [4 x [16 x i32] ] zeroinitializer

define void @main() local_unnamed_addr #0 {
; CHECK: %[[#main]] = OpFunction
entry:
  %2 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f1, i32 0, i32 0
  %3 = load <16 x float>, ptr addrspace(10) %2, align 4
  %4 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f1, i32 0, i32 1
  %5 = load <16 x float>, ptr addrspace(10) %4, align 4
  %6 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f1, i32 0, i32 2
  %7 = load <16 x float>, ptr addrspace(10) %6, align 4
  %8 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f1, i32 0, i32 3
  %9 = load <16 x float>, ptr addrspace(10) %8, align 4
  
  ; We expect the large vectors to be split into size 4, and the operations performed on them.
  ; CHECK: OpFMul %[[#v4f32]]
  ; CHECK: OpFMul %[[#v4f32]]
  ; CHECK: OpFMul %[[#v4f32]]
  ; CHECK: OpFMul %[[#v4f32]]
  %10 = fmul reassoc nnan ninf nsz arcp afn <16 x float> %3, splat (float 3.000000e+00)

  ; CHECK: OpFAdd %[[#v4f32]]
  ; CHECK: OpFAdd %[[#v4f32]]
  ; CHECK: OpFAdd %[[#v4f32]]
  ; CHECK: OpFAdd %[[#v4f32]]
  %11 = fadd reassoc nnan ninf nsz arcp afn <16 x float> %10, %5

  ; CHECK: OpFAdd %[[#v4f32]]
  ; CHECK: OpFAdd %[[#v4f32]]
  ; CHECK: OpFAdd %[[#v4f32]]
  ; CHECK: OpFAdd %[[#v4f32]]
  %12 = fadd reassoc nnan ninf nsz arcp afn <16 x float> %11, %7

  ; CHECK: OpFSub %[[#v4f32]]
  ; CHECK: OpFSub %[[#v4f32]]
  ; CHECK: OpFSub %[[#v4f32]]
  ; CHECK: OpFSub %[[#v4f32]]
  %13 = fsub reassoc nnan ninf nsz arcp afn <16 x float> %12, %9

  %14 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f2, i32 0, i32 0
  store <16 x float> %13, ptr addrspace(10) %14, align 4
  ret void
}

; Test integer vector arithmetic operations
define void @test_int_vector_arithmetic() local_unnamed_addr #0 {
; CHECK: OpFunction
entry:
  %2 = getelementptr [4 x [16 x i32] ], ptr addrspace(10) @i1, i32 0, i32 0
  %3 = load <16 x i32>, ptr addrspace(10) %2, align 4
  %4 = getelementptr [4 x [16 x i32] ], ptr addrspace(10) @i1, i32 0, i32 1
  %5 = load <16 x i32>, ptr addrspace(10) %4, align 4

  ; CHECK: OpIAdd %[[#v4i32]]
  ; CHECK: OpIAdd %[[#v4i32]]
  ; CHECK: OpIAdd %[[#v4i32]]
  ; CHECK: OpIAdd %[[#v4i32]]
  %6 = add <16 x i32> %3, %5

  ; CHECK: OpISub %[[#v4i32]]
  ; CHECK: OpISub %[[#v4i32]]
  ; CHECK: OpISub %[[#v4i32]]
  ; CHECK: OpISub %[[#v4i32]]
  %7 = sub <16 x i32> %6, %5

  ; CHECK: OpIMul %[[#v4i32]]
  ; CHECK: OpIMul %[[#v4i32]]
  ; CHECK: OpIMul %[[#v4i32]]
  ; CHECK: OpIMul %[[#v4i32]]
  %8 = mul <16 x i32> %7, splat (i32 2)

  ; CHECK: OpSDiv %[[#v4i32]]
  ; CHECK: OpSDiv %[[#v4i32]]
  ; CHECK: OpSDiv %[[#v4i32]]
  ; CHECK: OpSDiv %[[#v4i32]]
  %9 = sdiv <16 x i32> %8, splat (i32 2)

  ; CHECK: OpUDiv %[[#v4i32]]
  ; CHECK: OpUDiv %[[#v4i32]]
  ; CHECK: OpUDiv %[[#v4i32]]
  ; CHECK: OpUDiv %[[#v4i32]]
  %10 = udiv <16 x i32> %9, splat (i32 1)

  ; CHECK: OpSRem %[[#v4i32]]
  ; CHECK: OpSRem %[[#v4i32]]
  ; CHECK: OpSRem %[[#v4i32]]
  ; CHECK: OpSRem %[[#v4i32]]
  %11 = srem <16 x i32> %10, splat (i32 3)

  ; CHECK: OpUMod %[[#v4i32]]
  ; CHECK: OpUMod %[[#v4i32]]
  ; CHECK: OpUMod %[[#v4i32]]
  ; CHECK: OpUMod %[[#v4i32]]
  %12 = urem <16 x i32> %11, splat (i32 3)

  %13 = getelementptr [4 x [16 x i32] ], ptr addrspace(10) @i2, i32 0, i32 0
  store <16 x i32> %12, ptr addrspace(10) %13, align 4
  ret void
}

; Test remaining float vector arithmetic operations
define void @test_float_vector_arithmetic_continued() local_unnamed_addr #0 {
; CHECK: OpFunction
entry:
  %2 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f1, i32 0, i32 0
  %3 = load <16 x float>, ptr addrspace(10) %2, align 4
  %4 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f1, i32 0, i32 1
  %5 = load <16 x float>, ptr addrspace(10) %4, align 4

  ; CHECK: OpFDiv %[[#v4f32]]
  ; CHECK: OpFDiv %[[#v4f32]]
  ; CHECK: OpFDiv %[[#v4f32]]
  ; CHECK: OpFDiv %[[#v4f32]]
  %6 = fdiv reassoc nnan ninf nsz arcp afn <16 x float> %3, splat (float 2.000000e+00)

  ; CHECK: OpFRem %[[#v4f32]]
  ; CHECK: OpFRem %[[#v4f32]]
  ; CHECK: OpFRem %[[#v4f32]]
  ; CHECK: OpFRem %[[#v4f32]]
  %7 = frem reassoc nnan ninf nsz arcp afn <16 x float> %6, splat (float 3.000000e+00)

  ; CHECK: OpExtInst %[[#v4f32]] {{.*}} Fma
  ; CHECK: OpExtInst %[[#v4f32]] {{.*}} Fma
  ; CHECK: OpExtInst %[[#v4f32]] {{.*}} Fma
  ; CHECK: OpExtInst %[[#v4f32]] {{.*}} Fma
  %8 = call reassoc nnan ninf nsz arcp afn <16 x float> @llvm.fma.v16f32(<16 x float> %5, <16 x float> %6, <16 x float> %7)

  %9 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f2, i32 0, i32 0
  store <16 x float> %8, ptr addrspace(10) %9, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }