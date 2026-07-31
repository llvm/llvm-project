; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; spirv-val seems to have problems reading OpTypeVectorIdEXT correctly, enable once fixed
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#main:]] "main"
; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#v17f32:]] = OpTypeVectorIdEXT %[[#float]] 17
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#c17:]] = OpConstant %[[#int]] 17
; CHECK-DAG: %[[#v17i32:]] = OpTypeVectorIdEXT %[[#int]] 17
; CHECK-DAG: %[[#ptr_v17i32:]] = OpTypePointer CrossWorkgroup %[[#v17i32]]

@f1 = internal addrspace(1) global [4 x [17 x float] ] zeroinitializer
@f2 = internal addrspace(1) global [4 x [17 x float] ] zeroinitializer
@i1 = internal addrspace(1) global [4 x [17 x i32] ] zeroinitializer
@i2 = internal addrspace(1) global [4 x [17 x i32] ] zeroinitializer

define void @main() local_unnamed_addr {
; CHECK: %[[#main]] = OpFunction
entry:
  %2 = getelementptr [4 x [17 x float] ], ptr addrspace(1) @f1, i32 0, i32 0
  %3 = load <17 x float>, ptr addrspace(1) %2, align 4
  %4 = getelementptr [4 x [17 x float] ], ptr addrspace(1) @f1, i32 0, i32 1
  %5 = load <17 x float>, ptr addrspace(1) %4, align 4
  %6 = getelementptr [4 x [17 x float] ], ptr addrspace(1) @f1, i32 0, i32 2
  %7 = load <17 x float>, ptr addrspace(1) %6, align 4
  %8 = getelementptr [4 x [17 x float] ], ptr addrspace(1) @f1, i32 0, i32 3
  %9 = load <17 x float>, ptr addrspace(1) %8, align 4

  ; CHECK: %[[#Mul:]] = OpFMul %[[#v17f32]]
  %10 = fmul reassoc nnan ninf nsz arcp afn <17 x float> %3, splat (float 3.000000e+00)

  ; CHECK: %[[#Add:]] = OpFAdd %[[#v17f32]] %[[#Mul]]
  %11 = fadd reassoc nnan ninf nsz arcp afn <17 x float> %10, %5

  ; CHECK: %[[#Sub:]] = OpFSub %[[#v17f32]] %[[#Add]]
  %13 = fsub reassoc nnan ninf nsz arcp afn <17 x float> %11, %9

  ; CHECK: OpStore %[[#]] %[[#Sub]]

  %14 = getelementptr [4 x [17 x float] ], ptr addrspace(1) @f2, i32 0, i32 0
  store <17 x float> %13, ptr addrspace(1) %14, align 4
  ret void
}

; Test integer vector arithmetic operations.
define void @test_int_vector_arithmetic() local_unnamed_addr #0 {
; CHECK: OpFunction
entry:
  %2 = getelementptr [4 x [17 x i32] ], ptr addrspace(1) @i1, i32 0, i32 0
  %3 = load <17 x i32>, ptr addrspace(1) %2, align 4
  %4 = getelementptr [4 x [17 x i32] ], ptr addrspace(1) @i1, i32 0, i32 1
  %5 = load <17 x i32>, ptr addrspace(1) %4, align 4

  ; CHECK: %[[#Add1:]] = OpIAdd %[[#v17i32]]
  %6 = add <17 x i32> %3, %5

  ; CHECK: %[[#Sub1:]] = OpISub %[[#v17i32]] %[[#Add1]]
  %7 = sub <17 x i32> %6, %5

  ; CHECK: %[[#Mul1:]] = OpIMul %[[#v17i32]] %[[#Sub1]]
  %8 = mul <17 x i32> %7, splat (i32 2)

  ; CHECK: %[[#SDiv1:]] = OpSDiv %[[#v17i32]] %[[#Mul1]]
  %9 = sdiv <17 x i32> %8, splat (i32 2)

  ; CHECK: %[[#UDiv1:]] = OpUDiv %[[#v17i32]] %[[#SDiv1]]
  %10 = udiv <17 x i32> %9, splat (i32 1)

  ; CHECK: %[[#SRem1:]] = OpSRem %[[#v17i32]] %[[#UDiv1]]
  %11 = srem <17 x i32> %10, splat (i32 3)

  ; CHECK: %[[#UMod1:]] = OpUMod %[[#v17i32]] %[[#SRem1]]
  %12 = urem <17 x i32> %11, splat (i32 3)

  ; CHECK: OpStore {{.*}} %[[#UMod1]]

  %13 = getelementptr [4 x [17 x i32] ], ptr addrspace(1) @i2, i32 0, i32 0
  store <17 x i32> %12, ptr addrspace(1) %13, align 4
  ret void
}

; Test remaining float vector arithmetic operations.
define void @test_float_vector_arithmetic_continued() local_unnamed_addr #0 {
; CHECK: OpFunction
entry:
  %2 = getelementptr [4 x [17 x float] ], ptr addrspace(1) @f1, i32 0, i32 0
  %3 = load <17 x float>, ptr addrspace(1) %2, align 4
  %4 = getelementptr [4 x [17 x float] ], ptr addrspace(1) @f1, i32 0, i32 1
  %5 = load <17 x float>, ptr addrspace(1) %4, align 4

  ; CHECK: %[[#FDiv1:]] = OpFDiv %[[#v17f32]]
  %6 = fdiv reassoc nnan ninf nsz arcp afn <17 x float> %3, splat (float 2.000000e+00)

  ; CHECK: %[[#FRem1:]] = OpFRem %[[#v17f32]] %[[#FDiv1]]
    %7 = frem reassoc nnan ninf nsz arcp afn <17 x float> %6, splat (float 3.000000e+00)

  ; CHECK: %[[#Fma1:]] = OpExtInst %[[#v17f32]] {{.*}} fma {{.*}} %[[#FDiv1]] %[[#FRem1]]
  %8 = call reassoc nnan ninf nsz arcp afn <17 x float> @llvm.fma.v16f32(<17 x float> %5, <17 x float> %6, <17 x float> %7)

  ; CHECK: OpStore {{.*}} %[[#Fma1]]

  %9 = getelementptr [4 x [17 x float] ], ptr addrspace(1) @f2, i32 0, i32 0
  store <17 x float> %8, ptr addrspace(1) %9, align 4
  ret void
}

; Test constrained fma vector arithmetic operations.
define void @test_constrained_fma_vector() local_unnamed_addr #0 {
; CHECK: OpFunction
entry:
  %2 = getelementptr [4 x [17 x float] ], ptr addrspace(1) @f1, i32 0, i32 0
  %3 = load <17 x float>, ptr addrspace(1) %2, align 4
  %4 = getelementptr [4 x [17 x float] ], ptr addrspace(1) @f1, i32 0, i32 1
  %5 = load <17 x float>, ptr addrspace(1) %4, align 4
  %6 = getelementptr [4 x [17 x float] ], ptr addrspace(1) @f1, i32 0, i32 2
  %7 = load <17 x float>, ptr addrspace(1) %6, align 4

  ; CHECK: %[[#Fma2:]] = OpExtInst %[[#v17f32]] {{.*}} fma
  %8 = call <17 x float> @llvm.experimental.constrained.fma.v16f32(<17 x float> %3, <17 x float> %5, <17 x float> %7, metadata !"round.dynamic", metadata !"fpexcept.strict")

  ; CHECK: OpStore {{.*}} %[[#Fma2]]

  %9 = getelementptr [4 x [17 x float] ], ptr addrspace(1) @f2, i32 0, i32 0
  store <17 x float> %8, ptr addrspace(1) %9, align 4
  ret void
}

declare <17 x float> @llvm.experimental.constrained.fma.v16f32(<17 x float>, <17 x float>, <17 x float>, metadata, metadata)
