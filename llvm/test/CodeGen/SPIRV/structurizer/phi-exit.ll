; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

define internal spir_func void @main() #1 {
; CHECK: %[[#entry:]] = OpLabel
; CHECK:                OpBranch %[[#do_body:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  br label %loop_body

loop_body:
  br i1 true, label %left, label %right

left:
  br i1 true, label %loop_exit, label %loop_continue

right:
  br i1 true, label %loop_exit, label %loop_continue

loop_continue:
  br label %loop_body

loop_exit:
  %r = phi i32 [ 0, %left ], [ 1, %right ]
  store i32 %r, ptr %a, align 4
  ret void

}


declare token @llvm.experimental.convergence.entry() #0
declare token @llvm.experimental.convergence.loop() #0

attributes #0 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}

