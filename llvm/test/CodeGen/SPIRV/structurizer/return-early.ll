; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

define internal spir_func void @main() #0 {
; CHECK: %[[#entry:]] = OpLabel
; CHECK:                OpBranch %[[#while_cond:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %cond = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 1, ptr %cond, align 4
  br label %while.cond

; CHECK: %[[#while_cond]] = OpLabel
; CHECK:                    OpSelectionMerge %[[#while_end:]] None
; CHECK:                    OpBranchConditional %[[#cond:]] %[[#while_body:]] %[[#while_end]]
while.cond:
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %cond, align 4
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %while.body, label %while.end

; CHECK: %[[#while_body]] = OpLabel
; CHECK:                    OpSelectionMerge %[[#switch_end:]] None
; CHECK:                    OpSwitch %[[#cond:]] %[[#switch_end]] 1 %[[#case_1:]] 2 %[[#case_2:]] 5 %[[#case_5:]]
while.body:
  %3 = load i32, ptr %b, align 4
  switch i32 %3, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 5, label %sw.bb2
  ]

; CHECK: %[[#case_1]] = OpLabel
; CHECK:                OpBranch %[[#switch_end]]
sw.bb:
  store i32 1, ptr %a, align 4
  br label %while.end

; CHECK: %[[#case_2]] = OpLabel
; CHECK:                OpBranch %[[#switch_end]]
sw.bb1:
  store i32 3, ptr %a, align 4
  br label %while.end

; CHECK: %[[#case_5]] = OpLabel
; CHECK:                OpBranch %[[#switch_end]]
sw.bb2:
  store i32 5, ptr %a, align 4
  br label %while.end

; CHECK: %[[#switch_end]] = OpLabel
; CHECK:       %[[#phi:]] = OpPhi %[[#type:]] %[[#A:]] %[[#while_body]] %[[#B:]] %[[#case_5]] %[[#B:]] %[[#case_2]] %[[#B:]] %[[#case_1]]
; CHECK:       %[[#tmp:]] = OpIEqual %[[#type:]] %[[#A]] %[[#phi]]
; CHECK:                    OpBranchConditional %[[#tmp]] %[[#sw_default:]] %[[#while_end]]

; CHECK: %[[#sw_default]] = OpLabel
; CHECK:                    OpStore %[[#A:]] %[[#B:]] Aligned 4
; CHECK:                    OpBranch %[[#for_cond:]]
sw.default:
  store i32 0, ptr %i, align 4
  br label %for.cond

; CHECK: %[[#for_cond]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#for_merge:]] None
; CHECK-NEXT:             OpBranchConditional %[[#cond:]] %[[#for_merge]] %[[#for_end:]]
for.cond:
  %4 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %1) ]
  %5 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %5, 10
  br i1 %cmp, label %for.body, label %for.end

; CHECK: %[[#for_end]] = OpLabel
; CHECK:                 OpBranch %[[#for_merge]]
for.end:
  br label %while.end

; CHECK: %[[#for_merge]] = OpLabel
; CHECK:      %[[#phi:]] = OpPhi %[[#type:]] %[[#A:]] %[[#for_cond]] %[[#B:]] %[[#for_end]]
; CHECK:      %[[#tmp:]] = OpIEqual %[[#type:]] %[[#A]] %[[#phi]]
; CHECK:                   OpBranchConditional %[[#tmp]] %[[#for_body:]] %[[#while_end]]

; CHECK: %[[#for_body]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#if_merge:]] None
; CHECK:                  OpBranchConditional %[[#cond:]] %[[#if_merge]] %[[#if_else:]]
for.body:
  %6 = load i32, ptr %cond, align 4
  %tobool3 = icmp ne i32 %6, 0
  br i1 %tobool3, label %if.then, label %if.else

; CHECK: %[[#if_else]] = OpLabel
; CHECK:                 OpBranch %[[#if_merge]]
if.else:
  br label %while.end

; CHECK: %[[#if_merge]] = OpLabel
; CHECK:                  OpBranch %[[#while_end]]
if.then:
  br label %while.end

; CHECK: %[[#while_end]] = OpLabel
; CHECK:                   OpReturn
while.end:
  ret void

; CHECK-NOT: %[[#for_inc:]] = OpLabel
; This block is not emitted since it's unreachable.
for.inc:
  %7 = load i32, ptr %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

}

declare token @llvm.experimental.convergence.entry() #1
declare token @llvm.experimental.convergence.loop() #1

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
