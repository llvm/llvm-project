
; RUN: llc < %s -mtriple=arm-pc-windows-msvc | FileCheck %s
; Control Flow Guard is currently only available on Windows

; Test that Control Flow Guard checks are not added in modules with the
; cfguard=1 flag (emit tables but no checks).


declare void @target_func()

define void @func_in_module_without_cfguard() #0 {
entry:
  %func_ptr = alloca void ()*, align 8
  store void ()* @target_func, void ()** %func_ptr, align 8
  %0 = load void ()*, void ()** %func_ptr, align 8

  call void %0()
  ret void

  ; CHECK-NOT: __guard_check_icall_fptr
  ; CHECK-NOT: __guard_dispatch_icall_fptr
}
attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a9" "target-features"="+armv7-a,+dsp,+fp16,+neon,+strict-align,+thumb-mode,+vfp3" "use-soft-float"="false"}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 1}
