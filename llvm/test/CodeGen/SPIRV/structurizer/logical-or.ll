; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - --asm-verbose=0 | FileCheck %s --match-full-lines

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; CHECK-DAG:  OpName %[[#fn:]] "fn"
; CHECK-DAG:  OpName %[[#main:]] "main"
; CHECK-DAG:  OpName %[[#var_a:]] "a"
; CHECK-DAG:  OpName %[[#var_b:]] "b"

; CHECK-DAG:  %[[#bool:]] = OpTypeBool
; CHECK-DAG:  %[[#true:]] = OpConstantTrue %[[#bool]]

; CHECK:  %[[#fn]] = OpFunction %[[#param:]] DontInline %[[#ftype:]]
define spir_func noundef i32 @fn() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}

; CHECK: %[[#main]] = OpFunction %[[#param:]] DontInline %[[#ftype:]]

define internal spir_func void @main() #3 {

; CHECK:     %[[#entry:]] = OpLabel
; CHECK-DAG:  %[[#var_a]] = OpVariable %[[#type:]] Function
; CHECK-DAG:  %[[#var_b]] = OpVariable %[[#type:]] Function
; CHECK:       %[[#tmp:]] = OpLoad %[[#type:]] %[[#var_a]] Aligned 4
; CHECK:      %[[#cond:]] = OpINotEqual %[[#bool]] %[[#tmp]] %[[#const:]]
; CHECK:                    OpSelectionMerge %[[#if_end:]] None
; CHECK:                    OpBranchConditional %[[#true]] %[[#cond1:]] %[[#dead:]]

; CHECK:      %[[#cond1]] = OpLabel
; CHECK:                    OpSelectionMerge %[[#new_exit:]] None
; CHECK:                    OpBranchConditional %[[#cond]] %[[#new_exit]] %[[#lor_lhs_false:]]

; CHECK:       %[[#dead]] = OpLabel
; CHECK-NEXT:               OpUnreachable

; CHECK:  %[[#lor_lhs_false]] = OpLabel
; CHECK:           %[[#tmp:]] = OpLoad %[[#type:]] %[[#var_b]] Aligned 4
; CHECK:          %[[#cond:]] = OpINotEqual %[[#bool]] %[[#tmp]] %[[#value:]]
; CHECK:                        OpBranchConditional %[[#cond]] %[[#new_exit]] %[[#alias_exit:]]

; CHECK: %[[#alias_exit]] = OpLabel
; CHECK:                    OpBranch %[[#new_exit]]

; CHECK:   %[[#new_exit]] = OpLabel
; CHECK:       %[[#tmp:]] = OpPhi %[[#type:]] %[[#A:]] %[[#cond1]] %[[#A:]] %[[#lor_lhs_false]] %[[#B:]] %[[#alias_exit]]
; CHECK:      %[[#cond:]] = OpIEqual %[[#bool]] %[[#A]] %[[#tmp]]
; CHECK:                    OpBranchConditional %[[#cond]] %[[#if_then:]] %[[#if_end]]

; CHECK:    %[[#if_then]] = OpLabel
; CHECK:                    OpBranch %[[#if_end]]

; CHECK:     %[[#if_end]] = OpLabel
; CHECK:                    OpReturn

entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %val = alloca i32, align 4
  store i32 0, ptr %val, align 4
  %1 = load i32, ptr %a, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %if.then, label %lor.lhs.false

lor.lhs.false:
  %2 = load i32, ptr %b, align 4
  %tobool1 = icmp ne i32 %2, 0
  br i1 %tobool1, label %if.then, label %if.end

if.then:
  %8 = load i32, ptr %val, align 4
  %inc = add nsw i32 %8, 1
  store i32 %inc, ptr %val, align 4
  br label %if.end

if.end:
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
