; RUN: sed -e s/.tableonly:// %s | llc -mtriple=i686-pc-windows-msvc | FileCheck %s --check-prefixes=CHECK,TABLEONLY
; RUN: sed -e s/.tableonly:// %s | llc -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefixes=CHECK,TABLEONLY
; RUN: sed -e s/.tableonly:// %s | llc -mtriple=i686-w64-windows-gnu | FileCheck %s --check-prefixes=CHECK,TABLEONLY
; RUN: sed -e s/.tableonly:// %s | llc -mtriple=x86_64-w64-windows-gnu | FileCheck %s --check-prefixes=CHECK,TABLEONLY
; RUN: sed -e s/.normal:// %s | llc -mtriple=i686-pc-windows-msvc | FileCheck %s --check-prefixes=CHECK,USECHECK
; RUN: sed -e s/.normal:// %s | llc -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefixes=CHECK,USEDISPATCH
; RUN: sed -e s/.normal:// %s | llc -mtriple=i686-w64-windows-gnu | FileCheck %s --check-prefixes=CHECK,USECHECK
; RUN: sed -e s/.normal:// %s | llc -mtriple=x86_64-w64-windows-gnu | FileCheck %s --check-prefixes=CHECK,USEDISPATCH
; RUN: sed -e s/.check:// %s | llc -mtriple=i686-pc-windows-msvc | FileCheck %s --check-prefixes=CHECK,USECHECK
; RUN: sed -e s/.check:// %s | llc -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefixes=CHECK,USECHECK
; RUN: sed -e s/.dispatch:// %s | llc -mtriple=i686-pc-windows-msvc | FileCheck %s --check-prefixes=CHECK,USEDISPATCH
; RUN: sed -e s/.dispatch:// %s | llc -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefixes=CHECK,USEDISPATCH
; Control Flow Guard is currently only available on Windows

declare void @target_func()

define void @func_in_module_without_cfguard() #0 {
entry:
  %func_ptr = alloca ptr, align 8
  store ptr @target_func, ptr %func_ptr, align 8
  %0 = load ptr, ptr %func_ptr, align 8

  call void %0()
  ret void

  ; CHECK:            call

  ; USECHECK-SAME:    __guard_check_icall_fptr
  ; USECHECK-NOT:     __guard_dispatch_icall_fptr

  ; USEDISPATCH-SAME: __guard_dispatch_icall_fptr
  ; USEDISPATCH-NOT:  __guard_check_icall_fptr

  ; TABLEONLY-SAME:   *%
  ; TABLEONLY-NOT:    __guard_dispatch_icall_fptr
  ; TABLEONLY-NOT:    __guard_check_icall_fptr
}

; CHECK: .section        .gfids$y,"dr"

!0 = !{i32 2, !"cfguard", i32 1}
!1 = !{i32 2, !"cfguard", i32 2}
!2 = !{i32 2, !"cfguard-mechanism", i32 1}
!3 = !{i32 2, !"cfguard-mechanism", i32 2}
;tableonly: !llvm.module.flags = !{!0}
;normal:    !llvm.module.flags = !{!1}
;check:     !llvm.module.flags = !{!1, !2}
;dispatch:  !llvm.module.flags = !{!1, !3}
