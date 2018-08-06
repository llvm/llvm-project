; Test SETCC with an i32 result for every integer condition.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Test CC in { 0 }, with 3 don't care.
define i64 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: afi [[REG]], -268435456
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %cond = icmp eq i32 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 1 }, with 3 don't care.
define i64 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 36
; CHECK: br %r14
  %cond = icmp slt i32 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 0, 1 }, with 3 don't care.
define i64 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: afi [[REG]], -536870912
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %cond = icmp sle i32 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 2 }, with 3 don't care.
define i64 @f4(i32 %a, i32 %b) {
; CHECK-LABEL: f4:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %cond = icmp sgt i32 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 0, 2 }, with 3 don't care.
define i64 @f5(i32 %a, i32 %b) {
; CHECK-LABEL: f5:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: xilf [[REG]], 4294967295
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 36
; CHECK: br %r14
  %cond = icmp sge i32 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 1, 2 }, with 3 don't care.
define i64 @f6(i32 %a, i32 %b) {
; CHECK-LABEL: f6:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: afi [[REG]], 1879048192
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %cond = icmp ne i32 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}
