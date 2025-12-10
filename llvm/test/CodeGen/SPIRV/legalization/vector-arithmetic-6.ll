; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#main:]] "main"
; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#v4f32:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#c6:]] = OpConstant %[[#int]] 6
; CHECK-DAG: %[[#v6f32:]] = OpTypeArray %[[#float]] %[[#c6]]
; CHECK-DAG: %[[#v6i32:]] = OpTypeArray %[[#int]] %[[#c6]]
; CHECK-DAG: %[[#ptr_ssbo_v6i32:]] = OpTypePointer Private %[[#v6i32]]
; CHECK-DAG: %[[#v4i32:]] = OpTypeVector %[[#int]] 4

@f1 = internal addrspace(10) global [4 x [6 x float] ] zeroinitializer
@f2 = internal addrspace(10) global [4 x [6 x float] ] zeroinitializer
@i1 = internal addrspace(10) global [4 x [6 x i32] ] zeroinitializer
@i2 = internal addrspace(10) global [4 x [6 x i32] ] zeroinitializer

define void @main() local_unnamed_addr #0 {
; CHECK: %[[#main]] = OpFunction
entry:
  %2 = getelementptr [4 x [6 x float] ], ptr addrspace(10) @f1, i32 0, i32 0
  %3 = load <6 x float>, ptr addrspace(10) %2, align 4
  %4 = getelementptr [4 x [6 x float] ], ptr addrspace(10) @f1, i32 0, i32 1
  %5 = load <6 x float>, ptr addrspace(10) %4, align 4
  %6 = getelementptr [4 x [6 x float] ], ptr addrspace(10) @f1, i32 0, i32 2
  %7 = load <6 x float>, ptr addrspace(10) %6, align 4
  %8 = getelementptr [4 x [6 x float] ], ptr addrspace(10) @f1, i32 0, i32 3
  %9 = load <6 x float>, ptr addrspace(10) %8, align 4
  
  ; We expect the 6-element vectors to be widened to 8, then split into two vectors of size 4.
  ; CHECK: %[[#Mul1:]] = OpFMul %[[#v4f32]]
  ; CHECK: %[[#Mul2:]] = OpFMul %[[#v4f32]]
  %10 = fmul reassoc nnan ninf nsz arcp afn <6 x float> %3, splat (float 3.000000e+00)

  ; CHECK: %[[#Add1:]] = OpFAdd %[[#v4f32]] %[[#Mul1]]
  ; CHECK: %[[#Add2:]] = OpFAdd %[[#v4f32]] %[[#Mul2]]
  %11 = fadd reassoc nnan ninf nsz arcp afn <6 x float> %10, %5

  ; CHECK: %[[#Sub1:]] = OpFSub %[[#v4f32]] %[[#Add1]]
  ; CHECK: %[[#Sub2:]] = OpFSub %[[#v4f32]] %[[#Add2]]
  %13 = fsub reassoc nnan ninf nsz arcp afn <6 x float> %11, %9

  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub1]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub1]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub1]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub1]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub2]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub2]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  
  %14 = getelementptr [4 x [6 x float] ], ptr addrspace(10) @f2, i32 0, i32 0
  store <6 x float> %13, ptr addrspace(10) %14, align 4
  ret void
}

; Test integer vector arithmetic operations
define void @test_int_vector_arithmetic() local_unnamed_addr #0 {
; CHECK: OpFunction
entry:
  %2 = getelementptr [4 x [6 x i32] ], ptr addrspace(10) @i1, i32 0, i32 0
  %3 = load <6 x i32>, ptr addrspace(10) %2, align 4
  %4 = getelementptr [4 x [6 x i32] ], ptr addrspace(10) @i1, i32 0, i32 1
  %5 = load <6 x i32>, ptr addrspace(10) %4, align 4

  ; CHECK: %[[#Add1:]] = OpIAdd %[[#v4i32]]
  ; CHECK: %[[#Add2:]] = OpIAdd %[[#v4i32]]
  %6 = add <6 x i32> %3, %5

  ; CHECK: %[[#Sub1:]] = OpISub %[[#v4i32]] %[[#Add1]]
  ; CHECK: %[[#Sub2:]] = OpISub %[[#v4i32]] %[[#Add2]]
  %7 = sub <6 x i32> %6, %5

  ; CHECK: %[[#Mul1:]] = OpIMul %[[#v4i32]] %[[#Sub1]]
  ; CHECK: %[[#Mul2:]] = OpIMul %[[#v4i32]] %[[#Sub2]]
  %8 = mul <6 x i32> %7, splat (i32 2)

  ; CHECK: %[[#SDiv1:]] = OpSDiv %[[#v4i32]] %[[#Mul1]]
  ; CHECK: %[[#SDiv2:]] = OpSDiv %[[#v4i32]] %[[#Mul2]]
  %9 = sdiv <6 x i32> %8, splat (i32 2)

  ; CHECK: %[[#UDiv1:]] = OpUDiv %[[#v4i32]] %[[#SDiv1]]
  ; CHECK: %[[#UDiv2:]] = OpUDiv %[[#v4i32]] %[[#SDiv2]]
  %10 = udiv <6 x i32> %9, splat (i32 1)

  ; CHECK: %[[#SRem1:]] = OpSRem %[[#v4i32]] %[[#UDiv1]]
  ; CHECK: %[[#SRem2:]] = OpSRem %[[#v4i32]] %[[#UDiv2]]
  %11 = srem <6 x i32> %10, splat (i32 3)

  ; CHECK: %[[#UMod1:]] = OpUMod %[[#v4i32]] %[[#SRem1]]
  ; CHECK: %[[#UMod2:]] = OpUMod %[[#v4i32]] %[[#SRem2]]
  %12 = urem <6 x i32> %11, splat (i32 3)

  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod1]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod1]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod1]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod1]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod2]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod2]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]

  %13 = getelementptr [4 x [6 x i32] ], ptr addrspace(10) @i2, i32 0, i32 0
  store <6 x i32> %12, ptr addrspace(10) %13, align 4
  ret void
}

; Test remaining float vector arithmetic operations
define void @test_float_vector_arithmetic_continued() local_unnamed_addr #0 {
; CHECK: OpFunction
entry:
  %2 = getelementptr [4 x [6 x float] ], ptr addrspace(10) @f1, i32 0, i32 0
  %3 = load <6 x float>, ptr addrspace(10) %2, align 4
  %4 = getelementptr [4 x [6 x float] ], ptr addrspace(10) @f1, i32 0, i32 1
  %5 = load <6 x float>, ptr addrspace(10) %4, align 4

  ; CHECK: %[[#FDiv1:]] = OpFDiv %[[#v4f32]]
  ; CHECK: %[[#FDiv2:]] = OpFDiv %[[#v4f32]]
  %6 = fdiv reassoc nnan ninf nsz arcp afn <6 x float> %3, splat (float 2.000000e+00)

  ; CHECK: %[[#FRem1:]] = OpFRem %[[#v4f32]] %[[#FDiv1]]
  ; CHECK: %[[#FRem2:]] = OpFRem %[[#v4f32]] %[[#FDiv2]]
  %7 = frem reassoc nnan ninf nsz arcp afn <6 x float> %6, splat (float 3.000000e+00)

  ; CHECK: %[[#Fma1:]] = OpExtInst %[[#v4f32]] {{.*}} Fma {{.*}} %[[#FDiv1]] %[[#FRem1]]
  ; CHECK: %[[#Fma2:]] = OpExtInst %[[#v4f32]] {{.*}} Fma {{.*}} %[[#FDiv2]] %[[#FRem2]]
  %8 = call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.fma.v6f32(<6 x float> %5, <6 x float> %6, <6 x float> %7)

  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma1]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma1]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma1]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma1]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma2]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma2]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]

  %9 = getelementptr [4 x [6 x float] ], ptr addrspace(10) @f2, i32 0, i32 0
  store <6 x float> %8, ptr addrspace(10) %9, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }