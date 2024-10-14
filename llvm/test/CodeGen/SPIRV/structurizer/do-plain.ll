; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s --match-full-lines

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"


define spir_func noundef i32 @_Z3foov() #0 {
; CHECK: %[[#foo:]] = OpLabel
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

; CHECK: %[[#do_body]] = OpLabel
; CHECK:                 OpLoopMerge %[[#do_end:]] %[[#do_cond:]] None
; CHECK:                 OpBranch %[[#do_cond]]
do.body:
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %i, align 4
  store i32 %2, ptr %val, align 4
  br label %do.cond

; CHECK: %[[#do_cond]] = OpLabel
; CHECK:                 OpBranchConditional %[[#cond:]] %[[#do_body]] %[[#do_end]]
do.cond:
  %3 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %3, 10
  br i1 %cmp, label %do.body, label %do.end

; CHECK: %[[#do_end]] = OpLabel
; CHECK:                OpBranch %[[#do_body1:]]
do.end:
  br label %do.body1

; CHECK: %[[#do_body1]] = OpLabel
; CHECK:                  OpLoopMerge %[[#do_end3:]] %[[#do_cond2:]] None
; CHECK:                  OpBranch %[[#do_cond2]]
do.body1:
  %4 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  store i32 0, ptr %val, align 4
  br label %do.cond2

; CHECK: %[[#do_cond2]] = OpLabel
; CHECK:                  OpBranchConditional %[[#cond:]] %[[#do_body1]] %[[#do_end3]]
do.cond2:
  br i1 true, label %do.body1, label %do.end3

; CHECK: %[[#do_end3]] = OpLabel
; CHECK:                 OpBranch %[[#do_body4:]]
do.end3:
  br label %do.body4

; CHECK: %[[#do_body4]] = OpLabel
; CHECK:                  OpLoopMerge %[[#do_end7:]] %[[#do_cond5:]] None
; CHECK:                  OpBranch %[[#do_cond5]]
do.body4:
  %5 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  br label %do.cond5

; CHECK: %[[#do_cond5]] = OpLabel
; CHECK:                  OpBranchConditional %[[#cond:]] %[[#do_body4]] %[[#do_end7]]
do.cond5:
  %6 = load i32, ptr %val, align 4
  %cmp6 = icmp slt i32 %6, 20
  br i1 %cmp6, label %do.body4, label %do.end7

; CHECK: %[[#do_end7]] = OpLabel
; CHECK:                 OpReturn
do.end7:
  ret void
}


declare token @llvm.experimental.convergence.entry() #1
declare token @llvm.experimental.convergence.loop() #1

attributes #0 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
