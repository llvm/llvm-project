; RUN: llc < %s | FileCheck %s

define i32 @foo(i64 %a, i64 %b) {
; CHECK: or.b32
; CHECK-NOT: or.b64
entry:
  %or = or i64 %a, %b
  %trunc = trunc i64 %or to i32
  ret i32 %trunc
}
