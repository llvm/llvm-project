; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s --match-full-lines

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

define spir_func noundef i32 @_Z3foov() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}

define internal spir_func void @main() #2 {
; CHECK: %[[#entry:]] = OpLabel
; CHECK:                OpBranch %[[#do_body:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %val = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, ptr %val, align 4
  store i32 0, ptr %i, align 4
  br label %do.body

; CHECK: %[[#do_header:]] = OpLabel
; CHECK:                    OpLoopMerge %[[#do_end:]] %[[#do_cond:]] None
; CHECK:                    OpBranch %[[#do_body:]]

; CHECK: %[[#do_body]] = OpLabel
; CHECK:                 OpSelectionMerge %[[#if_then:]] None
; CHECK:                 OpBranchConditional %[[#cond:]] %[[#if_then]] %[[#if_end:]]
do.body:
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %i, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %i, align 4
  %3 = load i32, ptr %i, align 4
  %cmp = icmp sgt i32 %3, 5
  br i1 %cmp, label %if.then, label %if.end

; CHECK: %[[#if_end]] = OpLabel
; CHECK:                OpBranch %[[#if_then]]
if.end:
  %4 = load i32, ptr %i, align 4
  store i32 %4, ptr %val, align 4
  br label %do.cond

; CHECK: %[[#if_then]] = OpLabel
; CHECK:                  OpBranch %[[#do_cond]]
if.then:
  br label %do.cond

; CHECK: %[[#do_cond]] = OpLabel
; CHECK:                 OpBranchConditional %[[#cond:]] %[[#do_header]] %[[#do_end]]
do.cond:
  %5 = load i32, ptr %i, align 4
  %cmp1 = icmp slt i32 %5, 10
  br i1 %cmp1, label %do.body, label %do.end

; CHECK: %[[#do_end]] = OpLabel
; CHECK:                OpBranch %[[#do_body2:]]
do.end:
  br label %do.body2

; CHECK: %[[#do_body2]] = OpLabel
; CHECK:                  OpLoopMerge %[[#do_end11:]] %[[#do_cond9:]] None
; CHECK:                  OpBranch %[[#do_body4:]]
do.body2:
  %6 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %7 = load i32, ptr %i, align 4
  %inc3 = add nsw i32 %7, 1
  store i32 %inc3, ptr %i, align 4
  br label %do.body4

; CHECK: %[[#do_body4]] = OpLabel
; CHECK:                  OpLoopMerge %[[#do_end8:]] %[[#do_cond6:]] None
; CHECK:                  OpBranch %[[#do_cond6]]
do.body4:
  %8 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %6) ]
  %9 = load i32, ptr %val, align 4
  %inc5 = add nsw i32 %9, 1
  store i32 %inc5, ptr %val, align 4
  br label %do.cond6

; CHECK: %[[#do_cond6]] = OpLabel
; CHECK:                  OpBranchConditional %[[#cond:]] %[[#do_body4]] %[[#do_end8]]
do.cond6:
  %10 = load i32, ptr %i, align 4
  %cmp7 = icmp slt i32 %10, 10
  br i1 %cmp7, label %do.body4, label %do.end8

; CHECK: %[[#do_end8]] = OpLabel
; CHECK:                 OpBranch %[[#do_cond9]]
do.end8:
  %11 = load i32, ptr %i, align 4
  %dec = add nsw i32 %11, -1
  store i32 %dec, ptr %i, align 4
  br label %do.cond9

; CHECK: %[[#do_cond9]] = OpLabel
; CHECK:                  OpBranchConditional %[[#cond:]] %[[#do_body2]] %[[#do_end11]]
do.cond9:
  %12 = load i32, ptr %val, align 4
  %cmp10 = icmp slt i32 %12, 10
  br i1 %cmp10, label %do.body2, label %do.end11

; CHECK: %[[#do_end11]] = OpLabel
; CHECK:                  OpReturn
do.end11:
  ret void
}

declare token @llvm.experimental.convergence.entry() #1
declare token @llvm.experimental.convergence.loop() #1

attributes #0 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
