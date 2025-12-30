; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; CHECK-DAG: OpName %[[#process:]] "_Z7processv"
; CHECK-DAG: OpName %[[#fn:]] "_Z2fnv"
; CHECK-DAG: OpName %[[#fn1:]] "_Z3fn1v"
; CHECK-DAG: OpName %[[#fn2:]] "_Z3fn2v"

; CHECK-DAG: OpName %[[#r2m_a:]] ".reg2mem3"
; CHECK-DAG: OpName %[[#r2m_b:]] ".reg2mem1"
; CHECK-DAG: OpName %[[#r2m_c:]] ".reg2mem"

; CHECK-DAG: %[[#int_ty:]] = OpTypeInt 32 0

; CHECK-DAG: %[[#int_0:]] = OpConstant %[[#]] 0
; CHECK-DAG: %[[#int_1:]] = OpConstant %[[#]] 1
; CHECK-DAG: %[[#true:]] = OpConstantTrue
; CHECK-DAG: %[[#false:]] = OpConstantFalse

declare token @llvm.experimental.convergence.entry() #1

; CHECK: %[[#fn]] = OpFunction %[[#int_ty]]
define spir_func noundef i32 @_Z2fnv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}

; CHECK: %[[#fn1]] = OpFunction %[[#int_ty]]
define spir_func noundef i32 @_Z3fn1v() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 0
}

; CHECK: %[[#fn2]] = OpFunction %[[#int_ty]]
define spir_func noundef i32 @_Z3fn2v() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}

; CHECK: %[[#process]] = OpFunction %[[#int_ty]]
define spir_func noundef i32 @_Z7processv() #0 {

; CHECK:         %[[#entry:]] = OpLabel
; CHECK-DAG:      %[[#r2m_a]] = OpVariable %[[#]] Function
; CHECK:                        OpSelectionMerge %[[#a_merge:]]
; CHECK:                        OpBranchConditional %[[#]] %[[#a_true:]] %[[#a_false:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %var = alloca i32
  br i1 true, label %a_true, label %a_false

; CHECK: %[[#a_false]] = OpLabel
; CHECK:                 OpStore %[[#r2m_a]] %[[#false]]
; CHECK:                 OpBranch %[[#a_merge]]
a_false:
  br label %a_merge

; CHECK: %[[#a_true]] = OpLabel
; CHECK:                OpStore %[[#r2m_a]] %[[#true]]
; CHECK:                OpBranch %[[#a_merge]]
a_true:
  br label %a_merge

; CHECK: %[[#a_merge]] = OpLabel
; CHECK:    %[[#tmp:]] = OpLoad %[[#]] %[[#r2m_a]]
; CHECK:                 OpSelectionMerge %[[#b_merge:]]
; CHECK:                 OpBranchConditional %[[#]] %[[#b_true:]] %[[#b_merge]]
a_merge:
  %1 = phi i1 [ true, %a_true ], [ false, %a_false ]
  br i1 %1, label %b_true, label %b_merge

; CHECK: %[[#b_true]] = OpLabel
; CHECK:                OpBranch %[[#b_merge]]
b_true:
  store i32 0, ptr %var ; Prevents whole branch optimization.
  br label %b_merge

; CHECK: %[[#b_merge]] = OpLabel
; CHECK:                 OpFunctionCall
; CHECK:                 OpSelectionMerge %[[#c_merge:]]
; CHECK:                 OpBranchConditional %[[#]] %[[#c_true:]] %[[#c_false:]]
b_merge:
  %f1 = call spir_func noundef i32 @_Z2fnv() #4 [ "convergencectrl"(token %0) ]
  br i1 true, label %c_true, label %c_false

; CHECK: %[[#c_false]] = OpLabel
; CHECK:        %[[#]] = OpFunctionCall
; CHECK:                 OpStore %[[#r2m_b]] %[[#]]
; CHECK:                 OpBranch %[[#c_merge]]
c_false:
  %f3 = call spir_func noundef i32 @_Z3fn2v() #4 [ "convergencectrl"(token %0) ]
  br label %c_merge

; CHECK: %[[#c_true]] = OpLabel
; CHECK:       %[[#]] = OpFunctionCall
; CHECK:                OpStore %[[#r2m_b]] %[[#]]
; CHECK:                OpBranch %[[#c_merge]]
c_true:
  %f2 = call spir_func noundef i32 @_Z3fn1v() #4 [ "convergencectrl"(token %0) ]
  br label %c_merge


; CHECK: %[[#c_merge]] = OpLabel
; CHECK:    %[[#tmp:]] = OpLoad %[[#]] %[[#r2m_b]]
; CHECK:                 OpStore %[[#r2m_c]] %[[#tmp:]]
; CHECK:                 OpSelectionMerge %[[#d_merge:]]
; CHECK:                 OpBranchConditional %[[#]] %[[#d_true:]] %[[#d_merge]]
c_merge:
  %5 = phi i32 [ %f2, %c_true ], [ %f3, %c_false ]
  br i1 true, label %d_true, label %d_merge

; CHECK: %[[#d_true]] = OpLabel
; CHECK:                OpBranch %[[#d_merge]]
d_true:
  store i32 0, ptr %var ; Prevents whole branch optimization.
  br label %d_merge

; CHECK: %[[#d_merge]] = OpLabel
; CHECK:    %[[#tmp:]] = OpLoad %[[#]] %[[#r2m_c]]
; CHECK:                 OpReturnValue %[[#tmp]]
d_merge:
  ret i32 %5
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal spir_func void @main() #2 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %call1 = call spir_func noundef i32 @_Z7processv() #4 [ "convergencectrl"(token %0) ]
  ret void
}

; Function Attrs: convergent norecurse
define void @main.1() #3 {
entry:
  call void @main()
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { convergent }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
