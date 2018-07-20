; Test the three-operand form of 32-bit addition.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i32 @foo(i32, i32, i32)

; Check ALRK.
define i32 @f1(i32 %dummy, i32 %a, i32 %b, i32 *%flag) {
; CHECK-LABEL: f1:
; CHECK: alrk %r2, %r3, %r4
; CHECK: ipm [[REG1:%r[0-5]]]
; CHECK: risblg [[REG2:%r[0-5]]], [[REG1]], 31, 159, 35
; CHECK: st [[REG2]], 0(%r5)
; CHECK: br %r14
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  %ext = zext i1 %obit to i32
  store i32 %ext, i32 *%flag
  ret i32 %val
}

; Check using the overflow result for a branch.
define i32 @f2(i32 %dummy, i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: alrk %r2, %r3, %r4
; CHECK-NEXT: bler %r14
; CHECK: lhi %r2, 0
; CHECK: jg foo@PLT
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %call, label %exit

call:
  %res = tail call i32 @foo(i32 0, i32 %a, i32 %b)
  ret i32 %res

exit:
  ret i32 %val
}

; ... and the same with the inverted direction.
define i32 @f3(i32 %dummy, i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: alrk %r2, %r3, %r4
; CHECK-NEXT: bnler %r14
; CHECK: lhi %r2, 0
; CHECK: jg foo@PLT
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %exit, label %call

call:
  %res = tail call i32 @foo(i32 0, i32 %a, i32 %b)
  ret i32 %res

exit:
  ret i32 %val
}

; Check that we can still use ALR in obvious cases.
define i32 @f4(i32 %a, i32 %b, i32 *%flag) {
; CHECK-LABEL: f4:
; CHECK: alr %r2, %r3
; CHECK: ipm [[REG1:%r[0-5]]]
; CHECK: risblg [[REG2:%r[0-5]]], [[REG1]], 31, 159, 35
; CHECK: st [[REG2]], 0(%r4)
; CHECK: br %r14
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  %ext = zext i1 %obit to i32
  store i32 %ext, i32 *%flag
  ret i32 %val
}

declare {i32, i1} @llvm.uadd.with.overflow.i32(i32, i32) nounwind readnone

