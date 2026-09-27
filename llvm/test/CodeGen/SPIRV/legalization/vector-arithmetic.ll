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
  ; CHECK: %[[#Mul1:]] = OpFMul %[[#v4f32]]
  ; CHECK: %[[#Mul2:]] = OpFMul %[[#v4f32]]
  ; CHECK: %[[#Mul3:]] = OpFMul %[[#v4f32]]
  ; CHECK: %[[#Mul4:]] = OpFMul %[[#v4f32]]
  %10 = fmul reassoc nnan ninf nsz arcp afn <16 x float> %3, splat (float 3.000000e+00)

  ; CHECK: %[[#Add1:]] = OpFAdd %[[#v4f32]] %[[#Mul1]]
  ; CHECK: %[[#Add2:]] = OpFAdd %[[#v4f32]] %[[#Mul2]]
  ; CHECK: %[[#Add3:]] = OpFAdd %[[#v4f32]] %[[#Mul3]]
  ; CHECK: %[[#Add4:]] = OpFAdd %[[#v4f32]] %[[#Mul4]]
  %11 = fadd reassoc nnan ninf nsz arcp afn <16 x float> %10, %5

  ; CHECK: %[[#Sub1:]] = OpFSub %[[#v4f32]] %[[#Add1]]
  ; CHECK: %[[#Sub2:]] = OpFSub %[[#v4f32]] %[[#Add2]]
  ; CHECK: %[[#Sub3:]] = OpFSub %[[#v4f32]] %[[#Add3]]
  ; CHECK: %[[#Sub4:]] = OpFSub %[[#v4f32]] %[[#Add4]]
  %13 = fsub reassoc nnan ninf nsz arcp afn <16 x float> %11, %9

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
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub2]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub2]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub3]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub3]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub3]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub3]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub4]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub4]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub4]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Sub4]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  
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

  ; CHECK: %[[#Add1:]] = OpIAdd %[[#v4i32]]
  ; CHECK: %[[#Add2:]] = OpIAdd %[[#v4i32]]
  ; CHECK: %[[#Add3:]] = OpIAdd %[[#v4i32]]
  ; CHECK: %[[#Add4:]] = OpIAdd %[[#v4i32]]
  %6 = add <16 x i32> %3, %5

  ; CHECK: %[[#Sub1:]] = OpISub %[[#v4i32]] %[[#Add1]]
  ; CHECK: %[[#Sub2:]] = OpISub %[[#v4i32]] %[[#Add2]]
  ; CHECK: %[[#Sub3:]] = OpISub %[[#v4i32]] %[[#Add3]]
  ; CHECK: %[[#Sub4:]] = OpISub %[[#v4i32]] %[[#Add4]]
  %7 = sub <16 x i32> %6, %5

  ; CHECK: %[[#Mul1:]] = OpIMul %[[#v4i32]] %[[#Sub1]]
  ; CHECK: %[[#Mul2:]] = OpIMul %[[#v4i32]] %[[#Sub2]]
  ; CHECK: %[[#Mul3:]] = OpIMul %[[#v4i32]] %[[#Sub3]]
  ; CHECK: %[[#Mul4:]] = OpIMul %[[#v4i32]] %[[#Sub4]]
  %8 = mul <16 x i32> %7, splat (i32 2)

  ; CHECK: %[[#SDiv1:]] = OpSDiv %[[#v4i32]] %[[#Mul1]]
  ; CHECK: %[[#SDiv2:]] = OpSDiv %[[#v4i32]] %[[#Mul2]]
  ; CHECK: %[[#SDiv3:]] = OpSDiv %[[#v4i32]] %[[#Mul3]]
  ; CHECK: %[[#SDiv4:]] = OpSDiv %[[#v4i32]] %[[#Mul4]]
  %9 = sdiv <16 x i32> %8, splat (i32 2)

  ; CHECK: %[[#UDiv1:]] = OpUDiv %[[#v4i32]] %[[#SDiv1]]
  ; CHECK: %[[#UDiv2:]] = OpUDiv %[[#v4i32]] %[[#SDiv2]]
  ; CHECK: %[[#UDiv3:]] = OpUDiv %[[#v4i32]] %[[#SDiv3]]
  ; CHECK: %[[#UDiv4:]] = OpUDiv %[[#v4i32]] %[[#SDiv4]]
  %10 = udiv <16 x i32> %9, splat (i32 1)

  ; CHECK: %[[#SRem1:]] = OpSRem %[[#v4i32]] %[[#UDiv1]]
  ; CHECK: %[[#SRem2:]] = OpSRem %[[#v4i32]] %[[#UDiv2]]
  ; CHECK: %[[#SRem3:]] = OpSRem %[[#v4i32]] %[[#UDiv3]]
  ; CHECK: %[[#SRem4:]] = OpSRem %[[#v4i32]] %[[#UDiv4]]
  %11 = srem <16 x i32> %10, splat (i32 3)

  ; CHECK: %[[#UMod1:]] = OpUMod %[[#v4i32]] %[[#SRem1]]
  ; CHECK: %[[#UMod2:]] = OpUMod %[[#v4i32]] %[[#SRem2]]
  ; CHECK: %[[#UMod3:]] = OpUMod %[[#v4i32]] %[[#SRem3]]
  ; CHECK: %[[#UMod4:]] = OpUMod %[[#v4i32]] %[[#SRem4]]
  %12 = urem <16 x i32> %11, splat (i32 3)

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
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod2]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod2]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod3]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod3]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod3]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod3]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod4]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod4]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod4]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#int]] %[[#UMod4]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]

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

  ; CHECK: %[[#FDiv1:]] = OpFDiv %[[#v4f32]]
  ; CHECK: %[[#FDiv2:]] = OpFDiv %[[#v4f32]]
  ; CHECK: %[[#FDiv3:]] = OpFDiv %[[#v4f32]]
  ; CHECK: %[[#FDiv4:]] = OpFDiv %[[#v4f32]]
  %6 = fdiv reassoc nnan ninf nsz arcp afn <16 x float> %3, splat (float 2.000000e+00)

  ; CHECK: %[[#FRem1:]] = OpFRem %[[#v4f32]] %[[#FDiv1]]
  ; CHECK: %[[#FRem2:]] = OpFRem %[[#v4f32]] %[[#FDiv2]]
  ; CHECK: %[[#FRem3:]] = OpFRem %[[#v4f32]] %[[#FDiv3]]
  ; CHECK: %[[#FRem4:]] = OpFRem %[[#v4f32]] %[[#FDiv4]]
  %7 = frem reassoc nnan ninf nsz arcp afn <16 x float> %6, splat (float 3.000000e+00)

  ; CHECK: %[[#Fma1:]] = OpExtInst %[[#v4f32]] {{.*}} Fma {{.*}} %[[#FDiv1]] %[[#FRem1]]
  ; CHECK: %[[#Fma2:]] = OpExtInst %[[#v4f32]] {{.*}} Fma {{.*}} %[[#FDiv2]] %[[#FRem2]]
  ; CHECK: %[[#Fma3:]] = OpExtInst %[[#v4f32]] {{.*}} Fma {{.*}} %[[#FDiv3]] %[[#FRem3]]
  ; CHECK: %[[#Fma4:]] = OpExtInst %[[#v4f32]] {{.*}} Fma {{.*}} %[[#FDiv4]] %[[#FRem4]]
  %8 = call reassoc nnan ninf nsz arcp afn <16 x float> @llvm.fma.v16f32(<16 x float> %5, <16 x float> %6, <16 x float> %7)

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
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma2]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma2]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma3]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma3]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma3]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma3]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma4]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma4]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma4]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma4]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]

  %9 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f2, i32 0, i32 0
  store <16 x float> %8, ptr addrspace(10) %9, align 4
  ret void
}

; Test constrained fma vector arithmetic operations
define void @test_constrained_fma_vector() local_unnamed_addr #0 {
; CHECK: OpFunction
entry:
  %2 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f1, i32 0, i32 0
  %3 = load <16 x float>, ptr addrspace(10) %2, align 4
  %4 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f1, i32 0, i32 1
  %5 = load <16 x float>, ptr addrspace(10) %4, align 4
  %6 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f1, i32 0, i32 2
  %7 = load <16 x float>, ptr addrspace(10) %6, align 4

  ; CHECK: %[[#Fma1:]] = OpExtInst %[[#v4f32]] {{.*}} Fma
  ; CHECK: %[[#Fma2:]] = OpExtInst %[[#v4f32]] {{.*}} Fma
  ; CHECK: %[[#Fma3:]] = OpExtInst %[[#v4f32]] {{.*}} Fma
  ; CHECK: %[[#Fma4:]] = OpExtInst %[[#v4f32]] {{.*}} Fma
  %8 = call <16 x float> @llvm.experimental.constrained.fma.v16f32(<16 x float> %3, <16 x float> %5, <16 x float> %7, metadata !"round.dynamic", metadata !"fpexcept.strict")

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
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma2]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma2]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma3]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma3]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma3]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma3]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma4]] 0
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma4]] 1
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma4]] 2
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]
  ; CHECK: %[[#EXTRACT:]] = OpCompositeExtract %[[#float]] %[[#Fma4]] 3
  ; CHECK: OpStore {{.*}} %[[#EXTRACT]]

  %9 = getelementptr [4 x [16 x float] ], ptr addrspace(10) @f2, i32 0, i32 0
  store <16 x float> %8, ptr addrspace(10) %9, align 4
  ret void
}

declare <16 x float> @llvm.experimental.constrained.fma.v16f32(<16 x float>, <16 x float>, <16 x float>, metadata, metadata)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
