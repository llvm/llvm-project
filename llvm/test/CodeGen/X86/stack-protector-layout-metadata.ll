; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; "stack-protector-layout" metadata only says how an alloca is laid out if it is
; protected. It composes with the heuristic the function's attributes select:
; here the ordinary sspstrong heuristic decides which allocas are protected, and
; the metadata decides how they are laid out.

declare void @use(ptr, ptr)

; Without metadata a large array is closer to the guard, at the higher offset,
; than a small one.
; CHECK-LABEL: plain:
; CHECK:         leaq 48(%rsp), %rcx
; CHECK-NEXT:    leaq 44(%rsp), %rdx
define void @plain() sspstrong {
  %big = alloca [64 x i8]
  %small = alloca [4 x i8]
  call void @use(ptr %big, ptr %small)
  ret void
}

; The frontend's classification replaces the one the heuristic would make, so
; the order flips. Both allocas are still protected by the heuristic itself.
; CHECK-LABEL: swapped:
; CHECK:         leaq 44(%rsp), %rcx
; CHECK-NEXT:    leaq 108(%rsp), %rdx
; CHECK:         __security_cookie
define void @swapped() sspstrong {
  %big = alloca [64 x i8], !stack-protector-layout !0
  %small = alloca [4 x i8], !stack-protector-layout !1
  call void @use(ptr %big, ptr %small)
  ret void
}

; Metadata on an alloca the heuristic does not protect changes nothing: there is
; no protector for it to be laid out against.
; CHECK-LABEL: unprotected:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @unprotected() sspstrong {
  %x = alloca i32, !stack-protector-layout !1
  store i32 0, ptr %x
  ret void
}

!0 = !{!"small"}
!1 = !{!"large"}
