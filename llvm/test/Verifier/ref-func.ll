; RUN: llvm-as < %s -disable-output 2>&1

@a = global i32 1
@b = global i32 2
@c = global i32 3, !implicit.ref !0

define i32 @foo() !implicit.ref !1 {
  ret i32 0
}

!0 = !{ptr @a}
!1 = !{ptr @b}
