; RUN: llc < %s | FileCheck %s

define i32 @trunc(i64 %a, i64 %b) {
; CHECK-LABEL: trunc
; CHECK: or.b32
; CHECK-NOT: or.b64
entry:
  %or = or i64 %a, %b
  %trunc = trunc i64 %or to i32
  ret i32 %trunc
}

define i32 @trunc_not(i64 %a, i64 %b, ptr %p) {
; CHECK-LABEL: trunc_not
; CHECK: or.b64
; CHECK-NOT: or.b32
entry:
  %or = or i64 %a, %b
  %trunc = trunc i64 %or to i32
  store i64 %or, ptr %p
  ret i32 %trunc
}
