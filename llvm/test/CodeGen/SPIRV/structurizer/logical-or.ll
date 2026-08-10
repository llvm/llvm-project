; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - --asm-verbose=0 | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

define internal spir_func void @main() #3 {
; CHECK-DAG:   OpName %[[#switch_0:]] "reg1"
; CHECK-DAG:   OpName %[[#switch_1:]] "reg"

; CHECK-DAG:   %[[#int_0:]] = OpConstant %[[#]] 0
; CHECK-DAG:   %[[#int_1:]] = OpConstant %[[#]] 1

; CHECK:       %[[#entry:]] = OpLabel
; CHECK-DAG: %[[#switch_0]] = OpVariable %[[#]] Function
; CHECK-DAG: %[[#switch_1]] = OpVariable %[[#]] Function
; CHECK:                      OpSelectionMerge %[[#merge:]] None
; CHECK:                      OpBranchConditional %[[#]] %[[#new_header:]] %[[#unreachable:]]

; CHECK:       %[[#unreachable]] = OpLabel
; CHECK-NEXT:                      OpUnreachable

; CHECK:     %[[#new_header]] = OpLabel
; CHECK:                        OpSelectionMerge %[[#new_merge:]] None
; CHECK:                        OpBranchConditional %[[#]] %[[#taint_true_merge:]] %[[#br_false:]]

; CHECK:      %[[#br_false]] = OpLabel
; CHECK-DAG:                   OpStore %[[#switch_1]] %[[#int_0]]
; CHECK:                       OpSelectionMerge %[[#taint_merge:]] None
; CHECK:                       OpBranchConditional %[[#]] %[[#taint_merge]] %[[#taint_false:]]

; CHECK:      %[[#taint_false]] = OpLabel
; CHECK:                          OpStore %[[#switch_1]] %[[#int_1]]
; CHECK:                          OpBranch %[[#taint_merge]]

; CHECK:      %[[#taint_merge]] = OpLabel
; CHECK:                          OpStore %[[#switch_0]] %[[#int_0]]
; CHECK:             %[[#tmp:]] = OpLoad %[[#]] %[[#switch_1]]
; CHECK:            %[[#cond:]] = OpIEqual %[[#]] %[[#int_0]] %[[#tmp]]
; CHECK:                          OpBranchConditional %[[#cond]] %[[#taint_false_true:]] %[[#new_merge]]

; CHECK: %[[#taint_false_true]] = OpLabel
; CHECK:                          OpStore %[[#switch_0]] %[[#int_1]]
; CHECK:                          OpBranch %[[#new_merge]]

; CHECK: %[[#taint_true_merge]] = OpLabel
; CHECK:                          OpStore %[[#switch_0]] %[[#int_1]]
; CHECK:                          OpBranch %[[#new_merge]]

; CHECK:      %[[#new_merge]] = OpLabel
; CHECK:             %[[#tmp:]] = OpLoad %[[#]] %[[#switch_0]]
; CHECK:            %[[#cond:]] = OpIEqual %[[#]] %[[#int_0]] %[[#tmp]]
; CHECK:                          OpBranchConditional %[[#cond]] %[[#merge]] %[[#br_true:]]

; CHECK:    %[[#br_true]] = OpLabel
; CHECK:                    OpBranch %[[#merge]]

; CHECK:     %[[#merge]] = OpLabel
; CHECK:                   OpReturn

entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %var = alloca i32, align 4
  br i1 true, label %br_true, label %br_false

br_false:
  store i32 0, ptr %var, align 4
  br i1 true, label %br_true, label %merge

br_true:
  store i32 0, ptr %var, align 4
  br label %merge

merge:
  ret void
}

declare token @llvm.experimental.convergence.entry() #2

attributes #0 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
