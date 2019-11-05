; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s

; Check how constant function pointer casts are handled.

declare void @unprototyped(...)

define i32 @call_unprototyped() {
  call void bitcast (void (...)* @unprototyped to void ()*)()
  ret i32 0
}

; CHECK-LABEL: call_unprototyped:
; CHECK: callq unprototyped
; CHECK: xorl %eax, %eax
; CHECK: retq

declare void @escaped_cast()

define i32 @escape_it_with_cast(i8** %p) {
  store i8* bitcast (void ()* @escaped_cast to i8*), i8** %p
  ret i32 0
}

declare void @dead_constant()

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 1}

!dead_constant_root = !{!1}
!1 = !DITemplateValueParameter(name: "dead_constant", value: i8* bitcast (void ()* @dead_constant to i8*))

; CHECK-LABEL: .section .gfids$y,"dr"
; CHECK-NEXT:  .symidx escaped_cast
; CHECK-NOT:   .symidx

