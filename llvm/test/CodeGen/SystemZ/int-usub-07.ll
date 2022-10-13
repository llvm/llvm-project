; Test the three-operand form of 64-bit addition.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i64 @foo(i64, i64, i64)

; Check SLGRK.
define i64 @f1(i64 %dummy, i64 %a, i64 %b, ptr %flag) {
; CHECK-LABEL: f1:
; CHECK: slgrk %r2, %r3, %r4
; CHECK: ipm [[REG1:%r[0-5]]]
; CHECK: afi [[REG1]], -536870912
; CHECK: risbg [[REG2:%r[0-5]]], [[REG1]], 63, 191, 33
; CHECK: stg [[REG2]], 0(%r5)
; CHECK: br %r14
  %t = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  %ext = zext i1 %obit to i64
  store i64 %ext, ptr %flag
  ret i64 %val
}

; Check using the overflow result for a branch.
define i64 @f2(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f2:
; CHECK: slgrk %r2, %r3, %r4
; CHECK-NEXT: bnler %r14
; CHECK: lghi %r2, 0
; CHECK: jg foo@PLT
  %t = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %call, label %exit

call:
  %res = tail call i64 @foo(i64 0, i64 %a, i64 %b)
  ret i64 %res

exit:
  ret i64 %val
}

; ... and the same with the inverted direction.
define i64 @f3(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f3:
; CHECK: slgrk %r2, %r3, %r4
; CHECK-NEXT: bler %r14
; CHECK: lghi %r2, 0
; CHECK: jg foo@PLT
  %t = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %exit, label %call

call:
  %res = tail call i64 @foo(i64 0, i64 %a, i64 %b)
  ret i64 %res

exit:
  ret i64 %val
}

; Check that we can still use SLGR in obvious cases.
define i64 @f4(i64 %a, i64 %b, ptr %flag) {
; CHECK-LABEL: f4:
; CHECK: slgr %r2, %r3
; CHECK: ipm [[REG1:%r[0-5]]]
; CHECK: afi [[REG1]], -536870912
; CHECK: risbg [[REG2:%r[0-5]]], [[REG1]], 63, 191, 33
; CHECK: stg [[REG2]], 0(%r4)
; CHECK: br %r14
  %t = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  %ext = zext i1 %obit to i64
  store i64 %ext, ptr %flag
  ret i64 %val
}

declare {i64, i1} @llvm.usub.with.overflow.i64(i64, i64) nounwind readnone

