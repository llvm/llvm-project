; RUN: sed -e s/.tableonly:// %s | llc -mtriple=arm-pc-windows-msvc | FileCheck %s --check-prefixes=CHECK,TABLEONLY
; RUN: sed -e s/.tableonly:// %s | llc -mtriple=arm-w64-windows-gnu | FileCheck %s --check-prefixes=CHECK,TABLEONLY
; RUN: sed -e s/.normal:// %s | llc -mtriple=arm-pc-windows-msvc | FileCheck %s --check-prefixes=CHECK,USECHECK
; RUN: sed -e s/.normal:// %s | llc -mtriple=arm-w64-windows-gnu | FileCheck %s --check-prefixes=CHECK,USECHECK
; RUN: sed -e s/.check:// %s | llc -mtriple=arm-pc-windows-msvc | FileCheck %s --check-prefixes=CHECK,USECHECK
; RUN: sed -e s/.dispatch:// %s | llc -mtriple=arm-pc-windows-msvc | FileCheck %s --check-prefixes=CHECK,USEDISPATCH
; Control Flow Guard is currently only available on Windows

declare void @target_func()

define void @func_in_module_without_cfguard() #0 {
entry:
  %func_ptr = alloca ptr, align 8
  store ptr @target_func, ptr %func_ptr, align 8
  %0 = load ptr, ptr %func_ptr, align 8

  call void %0()
  ret void

  ; CHECK:            movw

  ; USECHECK-SAME:    __guard_check_icall_fptr
  ; USECHECK-NOT:     __guard_dispatch_icall_fptr

  ; USEDISPATCH-SAME: __guard_dispatch_icall_fptr
  ; USEDISPATCH-NOT:  __guard_check_icall_fptr

  ; TABLEONLY-SAME:   target_func
  ; TABLEONLY-NOT:    __guard_dispatch_icall_fptr
  ; TABLEONLY-NOT:    __guard_check_icall_fptr
}
attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a9" "target-features"="+armv7-a,+dsp,+fp16,+neon,+strict-align,+thumb-mode,+vfp3" "use-soft-float"="false"}

; CHECK: .section        .gfids$y,"dr"

!0 = !{i32 2, !"cfguard", i32 1}
!1 = !{i32 2, !"cfguard", i32 2}
!2 = !{i32 2, !"cfguard-mechanism", i32 1}
!3 = !{i32 2, !"cfguard-mechanism", i32 2}
;tableonly: !llvm.module.flags = !{!0}
;normal:    !llvm.module.flags = !{!1}
;check:     !llvm.module.flags = !{!1, !2}
;dispatch:  !llvm.module.flags = !{!1, !3}
