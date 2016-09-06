; Test LOCHI/LOCGHI
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; CHECK-LABEL: bar1:
; CHECK: lhi [[REG:%r[0-5]]], 42
; CHECK: chi %r2, 0
; CHECK: lochie [[REG]], 0
define signext i32 @bar1(i32 signext %x) {
  %cmp = icmp ne i32 %x, 0
  %.x = select i1 %cmp, i32 42, i32 0
  ret i32 %.x
}

; CHECK-LABEL: bar2:
; CHECK: ltgr [[REG:%r[0-5]]], %r2
; CHECK: lghi %r2, 42
; CHECK: locghie %r2, 0
define signext i64 @bar2(i64 signext %x) {
  %cmp = icmp ne i64 %x, 0
  %.x = select i1 %cmp, i64 42, i64 0
  ret i64 %.x
}
