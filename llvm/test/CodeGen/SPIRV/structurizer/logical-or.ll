; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - --asm-verbose=0 | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; CHECK-DAG:  OpName %[[#fn:]] "fn"
; CHECK-DAG:  OpName %[[#main:]] "main"
; CHECK-DAG:  OpName %[[#a:]] "a"
; CHECK-DAG:  OpName %[[#b:]] "b"

; CHECK:  %[[#fn]] = OpFunction %[[#param:]] DontInline %[[#ftype:]]
define spir_func noundef i32 @fn() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}

; CHECK: %[[#main]] = OpFunction %[[#param:]] DontInline %[[#ftype:]]

define internal spir_func void @main() #3 {

; CHECK:     %[[#entry:]] = OpLabel
; CHECK-DAG:      %[[#a]] = OpVariable %[[#type:]] Function
; CHECK-DAG:      %[[#b]] = OpVariable %[[#type:]] Function
; CHECK:       %[[#tmp:]] = OpLoad %[[#type:]] %[[#a]] Aligned 4
; CHECK:                    OpSelectionMerge %[[#if_end:]] None
; CHECK:                    OpBranchConditional %[[#cond:]] %[[#if_then:]] %[[#lor_lhs_false:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %val = alloca i32, align 4
  store i32 0, ptr %val, align 4
  %1 = load i32, ptr %a, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %if.then, label %lor.lhs.false

; CHECK:  %[[#lor_lhs_false]] = OpLabel
; CHECK-NOT:       %[[#tmp:]] = OpLoad %[[#type:]] %[[#a]] Aligned 4
; CHECK:           %[[#tmp:]] = OpLoad %[[#type:]] %[[#b]] Aligned 4
; CHECK:                        OpBranchConditional %[[#cond:]] %[[#if_then]] %[[#new_end:]]

; CHECK: %[[#new_end]] = OpLabel
; CHECK:                 OpBranch %[[#if_end]]
lor.lhs.false:
  %2 = load i32, ptr %b, align 4
  %tobool1 = icmp ne i32 %2, 0
  br i1 %tobool1, label %if.then, label %if.end

; CHECK: %[[#if_end]] = OpLabel
; CHECK:   %[[#tmp:]] = OpFunctionCall %[[#type:]] %[[#fn]]
; CHECK:                OpSelectionMerge %[[#if_end9:]] None
; CHECK:                OpBranchConditional %[[#cond:]] %[[#if_then7:]] %[[#lor_lhs_false4:]]
if.end:
  %call2 = call spir_func noundef i32 @fn() convergent [ "convergencectrl"(token %0) ]
  %tobool3 = icmp ne i32 %call2, 0
  br i1 %tobool3, label %if.then7, label %lor.lhs.false4

; CHECK: %[[#lor_lhs_false4]] = OpLabel
; CHECK:           %[[#tmp:]] = OpFunctionCall %[[#type:]] %[[#fn]]
; CHECK:                        OpBranchConditional %[[#cond:]] %[[#if_then7]] %[[#new_end9:]]

; CHECK: %[[#new_end9]] = OpLabel
; CHECK:                  OpBranch %[[#if_end9]]
lor.lhs.false4:
  %call5 = call spir_func noundef i32 @fn() convergent [ "convergencectrl"(token %0) ]
  %tobool6 = icmp ne i32 %call5, 0
  br i1 %tobool6, label %if.then7, label %if.end9

; CHECK: %[[#if_end9]] = OpLabel
; CHECK:                 OpSelectionMerge %[[#if_end16:]] None
; CHECK:                 OpBranchConditional %[[#cond:]] %[[#if_then14:]] %[[#lor_lhs_false11:]]
if.end9:
  %3 = load i32, ptr %a, align 4
  %tobool10 = icmp ne i32 %3, 0
  br i1 %tobool10, label %if.then14, label %lor.lhs.false11

; CHECK: %[[#lor_lhs_false11]] = OpLabel
; CHECK:            %[[#tmp:]] = OpFunctionCall %[[#type:]] %[[#fn]]
; CHECK:                         OpBranchConditional %[[#cond:]] %[[#if_then14]] %[[#new_end16:]]

; CHECK: %[[#new_end16]] = OpLabel
; CHECK:                   OpBranch %[[#if_end16]]
lor.lhs.false11:
  %call12 = call spir_func noundef i32 @fn() convergent [ "convergencectrl"(token %0) ]
  %tobool13 = icmp ne i32 %call12, 0
  br i1 %tobool13, label %if.then14, label %if.end16

; CHECK: %[[#if_end16]] = OpLabel
; CHECK:     %[[#tmp:]] = OpFunctionCall %[[#type:]] %[[#fn]]
; CHECK:                  OpSelectionMerge %[[#if_end23:]] None
; CHECK:                  OpBranchConditional %[[#cond:]] %[[#if_then21:]] %[[#lor_lhs_false19:]]
if.end16:
  %call17 = call spir_func noundef i32 @fn() convergent [ "convergencectrl"(token %0) ]
  %tobool18 = icmp ne i32 %call17, 0
  br i1 %tobool18, label %if.then21, label %lor.lhs.false19

; CHECK: %[[#lor_lhs_false19]] = OpLabel
; CHECK:                         OpBranchConditional %[[#cond:]] %[[#if_then21]] %[[#new_end32:]]

; CHECK: %[[#new_end32]] = OpLabel
; CHECK:                   OpBranch %[[#if_end23]]
lor.lhs.false19:
  %4 = load i32, ptr %b, align 4
  %tobool20 = icmp ne i32 %4, 0
  br i1 %tobool20, label %if.then21, label %if.end23

; CHECK: %[[#if_end23]] = OpLabel
; CHECK:                  OpReturn
if.end23:
  ret void

; CHECK: %[[#if_then21]] = OpLabel
; CHECK:                   OpBranch %[[#if_end23]]
if.then21:
  %5 = load i32, ptr %val, align 4
  %inc22 = add nsw i32 %5, 1
  store i32 %inc22, ptr %val, align 4
  br label %if.end23

; CHECK: %[[#if_then14]] = OpLabel
; CHECK:                   OpBranch %[[#if_end16]]
if.then14:
  %6 = load i32, ptr %val, align 4
  %inc15 = add nsw i32 %6, 1
  store i32 %inc15, ptr %val, align 4
  br label %if.end16

; CHECK:      %[[#if_then7]] = OpLabel
; CHECK-NOT:      %[[#tmp:]] = OpFunctionCall %[[#type:]] %[[#fn]]
; CHECK:                       OpBranch %[[#if_end9]]
if.then7:
  %7 = load i32, ptr %val, align 4
  %inc8 = add nsw i32 %7, 1
  store i32 %inc8, ptr %val, align 4
  br label %if.end9

; CHECK: %[[#if_then]] = OpLabel
; CHECK:                 OpBranch %[[#if_end]]
if.then:
  %8 = load i32, ptr %val, align 4
  %inc = add nsw i32 %8, 1
  store i32 %inc, ptr %val, align 4
  br label %if.end
}

declare token @llvm.experimental.convergence.entry() #2

attributes #0 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
