; Test 64-bit addition in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @foo()

; Check addition of 1.
define zeroext i1 @f1(i64 %dummy, i64 %a, ptr %res) {
; CHECK-LABEL: f1:
; CHECK: algfi %r3, 1
; CHECK-DAG: stg %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check the high end of the ALGFI range.
define zeroext i1 @f2(i64 %dummy, i64 %a, ptr %res) {
; CHECK-LABEL: f2:
; CHECK: algfi %r3, 4294967295
; CHECK-DAG: stg %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 4294967295)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check the next value up, which must be loaded into a register first.
define zeroext i1 @f3(i64 %dummy, i64 %a, ptr %res) {
; CHECK-LABEL: f3:
; CHECK: llihl [[REG1:%r[0-9]+]], 1
; CHECK: algr [[REG1]], %r3
; CHECK-DAG: stg [[REG1]], 0(%r4)
; CHECK-DAG: ipm [[REG2:%r[0-5]]]
; CHECK-DAG: risbg %r2, [[REG2]], 63, 191, 35
; CHECK: br %r14
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 4294967296)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Likewise for negative values.
define zeroext i1 @f4(i64 %dummy, i64 %a, ptr %res) {
; CHECK-LABEL: f4:
; CHECK: lghi [[REG1:%r[0-9]+]], -1
; CHECK: algr [[REG1]], %r3
; CHECK-DAG: stg [[REG1]], 0(%r4)
; CHECK-DAG: ipm [[REG2:%r[0-5]]]
; CHECK-DAG: risbg %r2, [[REG2]], 63, 191, 35
; CHECK: br %r14
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 -1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check using the overflow result for a branch.
define void @f5(i64 %dummy, i64 %a, ptr %res) {
; CHECK-LABEL: f5:
; CHECK: algfi %r3, 1
; CHECK: stg %r3, 0(%r4)
; CHECK: jgnle foo@PLT
; CHECK: br %r14
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  br i1 %obit, label %call, label %exit

call:
  tail call i64 @foo()
  br label %exit

exit:
  ret void
}

; ... and the same with the inverted direction.
define void @f6(i64 %dummy, i64 %a, ptr %res) {
; CHECK-LABEL: f6:
; CHECK: algfi %r3, 1
; CHECK: stg %r3, 0(%r4)
; CHECK: jgle foo@PLT
; CHECK: br %r14
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  br i1 %obit, label %exit, label %call

call:
  tail call i64 @foo()
  br label %exit

exit:
  ret void
}

declare {i64, i1} @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone

