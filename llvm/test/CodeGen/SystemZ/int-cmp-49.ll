; That that we don't try to use z196 instructions on z10 for TMHH and TMHL.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 -O0 | FileCheck %s

@g = global i32 0

; Check the lowest useful TMHL value.
define void @f1(i64 %a) {
; CHECK-LABEL: f1:
; CHECK-NOT: risblg
; CHECK-NOT: risbhg
; CHECK: tmhl {{%r[0-5]}}, 1
; CHECK-NOT: risblg
; CHECK-NOT: risbhg
; CHECK: br %r14
entry:
  %and = and i64 %a, 4294967296
  %cmp = icmp eq i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, ptr@g
  br label %exit

exit:
  ret void
}

; Check the lowest useful TMHH value.
define void @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK-NOT: risblg
; CHECK-NOT: risbhg
; CHECK: tmhh {{%r[0-5]}}, 1
; CHECK-NOT: risblg
; CHECK-NOT: risbhg
; CHECK: br %r14
entry:
  %and = and i64 %a, 281474976710656
  %cmp = icmp ne i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, ptr@g
  br label %exit

exit:
  ret void
}
