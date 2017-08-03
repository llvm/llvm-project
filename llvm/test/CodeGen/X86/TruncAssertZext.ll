; RUN: llc < %s -O2 -march=x86-64 | FileCheck %s
; Checks that a zeroing mov is inserted for the trunc/zext pair even when
; the source of the zext is an AssertSext node
; PR20494

define i64 @main(i64 %a) {
; CHECK-LABEL: main
; CHECK: movl %e{{..}}, %eax
; CHECK: ret
  %or = or i64 %a, -2
  %trunc = trunc i64 %or to i32
  br label %l
l:
  %ext = zext i32 %trunc to i64
  ret i64 %ext
}
