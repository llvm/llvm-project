; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s --match-full-lines

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

define internal spir_func void @main() #0 {
; CHECK:    %[[#entry:]] = OpLabel
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %val = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store i32 0, ptr %val, align 4
  store i32 0, ptr %i, align 4
  store i32 0, ptr %j, align 4
  store i32 0, ptr %k, align 4
  br label %do.body

; CHECK:    %[[#do_1_header:]] = OpLabel
; CHECK:                         OpLoopMerge %[[#end:]] %[[#do_1_latch:]] None
; CHECK:                         OpBranch %[[#do_2_header:]]
do.body:
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %val, align 4
  %3 = load i32, ptr %i, align 4
  %add = add nsw i32 %2, %3
  store i32 %add, ptr %val, align 4
  br label %do.body1

; CHECK:    %[[#do_2_header]] = OpLabel
; CHECK:                        OpLoopMerge %[[#do_2_end:]] %[[#do_2_latch:]] None
; CHECK:                        OpBranch %[[#do_2_body:]]
do.body1:
  %4 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %1) ]
  br label %do.body2

; CHECK:    %[[#do_2_body]] = OpLabel
; CHECK:                      OpLoopMerge %[[#do_3_end:]] %[[#do_3_header:]] None
; CHECK:                      OpBranch %[[#do_3_header]]
do.body2:
  %5 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %4) ]
  %6 = load i32, ptr %k, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, ptr %k, align 4
  br label %do.cond

; CHECK:    %[[#do_3_header]] = OpLabel
; CHECK:                        OpBranchConditional %[[#cond:]] %[[#do_2_body]] %[[#do_3_end]]
do.cond:
  %7 = load i32, ptr %k, align 4
  %cmp = icmp slt i32 %7, 30
  br i1 %cmp, label %do.body2, label %do.end

; CHECK:    %[[#do_3_end]] = OpLabel
; CHECK:                     OpBranch %[[#do_2_latch]]
do.end:
  %8 = load i32, ptr %j, align 4
  %inc3 = add nsw i32 %8, 1
  store i32 %inc3, ptr %j, align 4
  br label %do.cond4

; CHECK:    %[[#do_2_latch]] = OpLabel
; CHECK:                     OpBranchConditional %[[#cond:]] %[[#do_2_header]] %[[#do_2_end]]
do.cond4:
  %9 = load i32, ptr %j, align 4
  %cmp5 = icmp slt i32 %9, 20
  br i1 %cmp5, label %do.body1, label %do.end6

; CHECK:    %[[#do_2_end]] = OpLabel
; CHECK:                     OpBranch %[[#do_1_latch]]
do.end6:
  %10 = load i32, ptr %i, align 4
  %inc7 = add nsw i32 %10, 1
  store i32 %inc7, ptr %i, align 4
  br label %do.cond8

; CHECK:    %[[#do_1_latch]] = OpLabel
; CHECK:                       OpBranchConditional %[[#cond:]] %[[#do_1_header]] %[[#end]]
do.cond8:
  %11 = load i32, ptr %i, align 4
  %cmp9 = icmp slt i32 %11, 10
  br i1 %cmp9, label %do.body, label %do.end10

; CHECK:    %[[#end]] = OpLabel
; CHECK:                OpReturn
do.end10:
  ret void
}

declare token @llvm.experimental.convergence.entry() #1
declare token @llvm.experimental.convergence.loop() #1

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
