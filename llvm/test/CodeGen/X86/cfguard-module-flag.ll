
; RUN: llc < %s -mtriple=i686-pc-windows-msvc | FileCheck %s -check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=i686-w64-windows-gnu | FileCheck %s -check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-w64-windows-gnu | FileCheck %s -check-prefix=X64
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

  ; X32-NOT: __guard_check_icall_fptr
  ; X64-NOT: __guard_dispatch_icall_fptr
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 1}
