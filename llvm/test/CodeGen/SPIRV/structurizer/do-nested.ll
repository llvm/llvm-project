; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

define internal spir_func void @main() #0 {
; CHECK:    %[[#entry:]] = OpLabel
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %var = alloca i32, align 4
  br label %do1_header

; CHECK:    %[[#do1_header:]] = OpLabel
; CHECK:                        OpLoopMerge %[[#do1_merge:]] %[[#do1_continue:]] None
; CHECK:                        OpBranch %[[#do2_header:]]
do1_header:
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  store i32 0, ptr %var
  br label %do2_header

; CHECK:    %[[#do2_header:]] = OpLabel
; CHECK:                        OpLoopMerge %[[#do2_merge:]] %[[#do2_continue:]] None
; CHECK:                        OpBranch %[[#do3_header:]]
do2_header:
  %4 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %1) ]
  store i32 0, ptr %var
  br label %do3_header

; CHECK:    %[[#do3_header:]] = OpLabel
; CHECK:                        OpLoopMerge %[[#do3_merge:]] %[[#do3_continue:]] None
; CHECK:                        OpBranch %[[#do3_cond:]]
do3_header:
  %5 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %4) ]
  store i32 0, ptr %var
  br label %do3_continue

; CHECK:        %[[#do3_cond]] = OpLabel
; CHECK:                         OpBranchConditional %[[#]] %[[#do3_continue]] %[[#do3_merge]]

; CHECK:    %[[#do3_merge]] = OpLabel
; CHECK:                      OpBranch %[[#do2_cond:]]
do3_merge:
  store i32 0, ptr %var
  br label %do2_continue

; CHECK:        %[[#do2_cond]] = OpLabel
; CHECK:                         OpBranchConditional %[[#]] %[[#do2_continue]] %[[#do2_merge]]

; CHECK:    %[[#do2_merge]] = OpLabel
; CHECK:                      OpBranch %[[#do1_cond:]]

; CHECK:        %[[#do1_cond]] = OpLabel
; CHECK:                         OpBranchConditional %[[#]] %[[#do1_continue]] %[[#do1_merge]]
do2_merge:
  store i32 0, ptr %var
  br label %do1_continue

; CHECK:    %[[#do1_merge]] = OpLabel
; CHECK:                      OpReturn
do1_merge:
  ret void

; CHECK:    %[[#do1_continue]] = OpLabel
; CHECK:                         OpBranch %[[#do1_header]]
do1_continue:
  store i32 0, ptr %var
  br i1 true, label %do1_header, label %do1_merge

; CHECK:    %[[#do2_continue]] = OpLabel
; CHECK:                         OpBranch %[[#do2_header]]
do2_continue:
  store i32 0, ptr %var
  br i1 true, label %do2_header, label %do2_merge

; CHECK:    %[[#do3_continue]] = OpLabel
; CHECK:                         OpBranch %[[#do3_header]]
do3_continue:
  store i32 0, ptr %var
  br i1 true, label %do3_header, label %do3_merge
}

declare token @llvm.experimental.convergence.entry() #1
declare token @llvm.experimental.convergence.loop() #1

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
