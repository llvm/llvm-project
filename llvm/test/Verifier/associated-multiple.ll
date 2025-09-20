; RUN: not llvm-as -disable-output < %s -o /dev/null 2>&1 | FileCheck %s

target triple = "unknown-unknown-linux-gnu"

@a = global i32 1
@b = global i32 2
@c = global i32 3, !associated !0, !associated !1

!0 = !{ptr @a}
!1 = !{ptr @b}

; CHECK: only a single associated metadata is supported
; CHECK: ptr @c
