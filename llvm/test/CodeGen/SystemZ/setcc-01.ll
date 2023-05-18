; Test SETCC for every integer condition.  The tests here assume that
; RISBLG isn't available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Test CC in { 0 }, with 3 don't care.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: ipm %r2
; CHECK-NEXT: afi %r2, -268435456
; CHECK-NEXT: srl %r2, 31
; CHECK: br %r14
  %cond = icmp eq i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

; Test CC in { 1 }, with 3 don't care.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 36
; CHECK: br %r14
  %cond = icmp slt i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

; Test CC in { 0, 1 }, with 3 don't care.
define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: ipm %r2
; CHECK-NEXT: afi %r2, -536870912
; CHECK-NEXT: srl %r2, 31
; CHECK: br %r14
  %cond = icmp sle i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

; Test CC in { 2 }, with 3 don't care.
define i32 @f4(i32 %a, i32 %b) {
; CHECK-LABEL: f4:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %cond = icmp sgt i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

; Test CC in { 0, 2 }, with 3 don't care.
define i32 @f5(i32 %a, i32 %b) {
; CHECK-LABEL: f5:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: xilf [[REG]], 4294967295
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 36
; CHECK: br %r14
  %cond = icmp sge i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

; Test CC in { 1, 2 }, with 3 don't care.
define i32 @f6(i32 %a, i32 %b) {
; CHECK-LABEL: f6:
; CHECK: ipm %r2
; CHECK-NEXT: afi %r2, 1879048192
; CHECK-NEXT: srl %r2, 31
; CHECK: br %r14
  %cond = icmp ne i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}
