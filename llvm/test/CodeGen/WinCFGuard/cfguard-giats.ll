; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s
; Control Flow Guard is currently only available on Windows

declare dllimport i32 @target_func()

; Test address-taken functions from imported DLLs are added to the 
; Guard Address-Taken IAT Entry table (.giats).
define i32 @func_cf_giats() {
entry:
  %func_ptr = alloca i32 ()*, align 8
  store i32 ()* @target_func, i32 ()** %func_ptr, align 8
  %0 = load i32 ()*, i32 ()** %func_ptr, align 8
  %1 = call i32 %0()
  ret i32 %1
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}

; CHECK-LABEL: .section .giats$y,"dr"
; CHECK-NEXT:  .symidx __imp_target_func
; CHECK-NOT:   .symidx