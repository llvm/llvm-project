; Test 32-bit subtraction in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @foo()

; Check subtraction of 1.
define zeroext i1 @f1(i32 %dummy, i32 %a, ptr %res) {
; CHECK-LABEL: f1:
; CHECK: slfi %r3, 1
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], -536870912
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the high end of the SLFI range.
define zeroext i1 @f2(i32 %dummy, i32 %a, ptr %res) {
; CHECK-LABEL: f2:
; CHECK: slfi %r3, 4294967295
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], -536870912
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 4294967295)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check that negative values are treated as unsigned
define zeroext i1 @f3(i32 %dummy, i32 %a, ptr %res) {
; CHECK-LABEL: f3:
; CHECK: slfi %r3, 4294967295
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], -536870912
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 -1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check using the overflow result for a branch.
define void @f4(i32 %dummy, i32 %a, ptr %res) {
; CHECK-LABEL: f4:
; CHECK: slfi %r3, 1
; CHECK: st %r3, 0(%r4)
; CHECK: jgle foo@PLT
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  br i1 %obit, label %call, label %exit

call:
  tail call i32 @foo()
  br label %exit

exit:
  ret void
}

; ... and the same with the inverted direction.
define void @f5(i32 %dummy, i32 %a, ptr %res) {
; CHECK-LABEL: f5:
; CHECK: slfi %r3, 1
; CHECK: st %r3, 0(%r4)
; CHECK: jgnle foo@PLT
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  br i1 %obit, label %exit, label %call

call:
  tail call i32 @foo()
  br label %exit

exit:
  ret void
}

declare {i32, i1} @llvm.usub.with.overflow.i32(i32, i32) nounwind readnone

