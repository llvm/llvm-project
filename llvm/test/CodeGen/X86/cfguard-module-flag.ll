; RUN: llc < %s -mtriple=i686-pc-windows-msvc | FileCheck %s -check-prefix=X86
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=i686-w64-windows-gnu | FileCheck %s -check-prefix=X86
; RUN: llc < %s -mtriple=x86_64-w64-windows-gnu | FileCheck %s -check-prefix=X64
; Control Flow Guard is currently only available on Windows

; Test that Control Flow Guard checks are not added in modules with the
; cfguard=1 flag (emit tables but no checks).

; If no checks were inserted then the GuardCF bit shouldn't be set in @feat.00.
; CHECK: "@feat.00" = 0
; i686 has SafeSEH (0x1) but should NOT have GuardCF (0x800).
; X86: @feat.00 = 1
; x86_64 has no SafeSEH and should NOT have GuardCF.
; X64: @feat.00 = 0

declare void @target_func()

define void @func_in_module_without_cfguard() #0 {
entry:
  %func_ptr = alloca ptr, align 8
  store ptr @target_func, ptr %func_ptr, align 8
  %0 = load ptr, ptr %func_ptr, align 8

  call void %0()
  ret void

  ; X86-NOT: __guard_check_icall_fptr
  ; X64-NOT: __guard_dispatch_icall_fptr
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 1}
