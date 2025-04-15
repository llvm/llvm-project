; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

define spir_func noundef i32 @_Z3foov() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}

define internal spir_func void @main() #2 {
; CHECK: %[[#entry:]] = OpLabel
; CHECK:                OpBranch %[[#do_header:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %var = alloca i32, align 4
  br label %do_header

; Here a the loop header had to be split in two:
; - 1 header for the loop
; - 1 header for the condition.

; CHECK: %[[#do_header:]] = OpLabel
; CHECK:                    OpLoopMerge %[[#do_merge:]] %[[#do_latch:]] None
; CHECK:                    OpBranch %[[#new_header:]]

; CHECK: %[[#new_header]] = OpLabel
; CHECK:                    OpSelectionMerge %[[#if_merge:]] None
; CHECK:                    OpBranchConditional %[[#]] %[[#if_then:]] %[[#if_end:]]
do_header:
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  store i32 0, ptr %var
  br i1 true, label %if.then, label %if.end

; CHECK: %[[#if_end]] = OpLabel
; CHECK:                OpBranch %[[#if_merge]]
if.end:
  store i32 0, ptr %var
  br label %do_latch

; CHECK: %[[#if_then]] = OpLabel
; CHECK:                 OpBranch %[[#if_merge]]
if.then:
  store i32 0, ptr %var
  br label %do_latch

; CHECK: %[[#if_merge]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#do_latch]] %[[#do_merge]]

; CHECK: %[[#do_merge]] = OpLabel
; CHECK:                  OpBranch %[[#do2_header:]]
do.end:
  store i32 0, ptr %var
  br label %do2_header

; CHECK: %[[#do2_header]] = OpLabel
; CHECK:                    OpLoopMerge %[[#do2_merge:]] %[[#do2_continue:]] None
; CHECK:                    OpBranch %[[#do3_header:]]
do2_header:
  %6 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  store i32 0, ptr %var
  br label %do3_header

; CHECK: %[[#do3_header]] = OpLabel
; CHECK:                  OpLoopMerge %[[#do3_merge:]] %[[#do3_continue:]] None
; CHECK:                  OpBranch %[[#do3_body:]]
do3_header:
  %8 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %6) ]
  store i32 0, ptr %var
  br label %do3_continue

; CHECK: %[[#do3_body]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#do3_continue]] %[[#do3_merge]]

; CHECK: %[[#do3_merge]] = OpLabel
; CHECK:                   OpBranch %[[#do2_new_latch:]]
do3_merge:
  store i32 0, ptr %var
  br label %do2_continue

; CHECK: %[[#do2_new_latch]] = OpLabel
; CHECK:                       OpBranchConditional %[[#]] %[[#do2_continue]] %[[#do2_merge]]

; CHECK: %[[#do2_merge]] = OpLabel
; CHECK:                   OpReturn
do2_merge:
  ret void

; CHECK: %[[#do2_continue]] = OpLabel
; CHECK:                      OpBranch %[[#do2_header]]
do2_continue:
  store i32 0, ptr %var
  br i1 true, label %do2_header, label %do2_merge

; CHECK: %[[#do3_continue]] = OpLabel
; CHECK:                      OpBranch %[[#do3_header]]
do3_continue:
  store i32 0, ptr %var
  br i1 true, label %do3_header, label %do3_merge

; CHECK: %[[#do_latch]] = OpLabel
; CHECK:                  OpBranch %[[#do_header]]
do_latch:
  store i32 0, ptr %var
  br i1 true, label %do_header, label %do.end
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
