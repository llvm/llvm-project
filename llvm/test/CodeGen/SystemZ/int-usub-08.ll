; Test 32-bit subtraction in which the second operand is constant and in which
; three-operand forms are available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i32 @foo()

; Check subtraction of 1.
define zeroext i1 @f1(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f1:
; CHECK: alhsik [[REG1:%r[0-5]]], %r3, -1
; CHECK-DAG: st [[REG1]], 0(%r4)
; CHECK-DAG: ipm [[REG2:%r[0-5]]]
; CHECK-DAG: afi [[REG2]], -536870912
; CHECK-DAG: risbg %r2, [[REG2]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the high end of the ALHSIK range.
define zeroext i1 @f2(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f2:
; CHECK: alhsik [[REG1:%r[0-5]]], %r3, -32768
; CHECK-DAG: st [[REG1]], 0(%r4)
; CHECK-DAG: ipm [[REG2:%r[0-5]]]
; CHECK-DAG: afi [[REG2]], -536870912
; CHECK-DAG: risbg %r2, [[REG2]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 32768)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the next value down, which must use SLFI instead.
define zeroext i1 @f3(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f3:
; CHECK: slfi %r3, 32769
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG2:%r[0-5]]]
; CHECK-DAG: afi [[REG2]], -536870912
; CHECK-DAG: risbg %r2, [[REG2]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 32769)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the high end of the negative ALHSIK range.
define zeroext i1 @f4(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f4:
; CHECK: alhsik [[REG1:%r[0-5]]], %r3, 1
; CHECK-DAG: st [[REG1]], 0(%r4)
; CHECK-DAG: ipm [[REG2:%r[0-5]]]
; CHECK-DAG: afi [[REG2]], -536870912
; CHECK-DAG: risbg %r2, [[REG2]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 -1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the low end of the ALHSIK range.
define zeroext i1 @f5(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f5:
; CHECK: alhsik [[REG1:%r[0-5]]], %r3, 32767
; CHECK-DAG: st [[REG1]], 0(%r4)
; CHECK-DAG: ipm [[REG2:%r[0-5]]]
; CHECK-DAG: afi [[REG2]], -536870912
; CHECK-DAG: risbg %r2, [[REG2]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 -32767)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the next value down, which must use SLFI instead.
define zeroext i1 @f6(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f6:
; CHECK: slfi %r3, 4294934528
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG2:%r[0-5]]]
; CHECK-DAG: afi [[REG2]], -536870912
; CHECK-DAG: risbg %r2, [[REG2]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 -32768)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check using the overflow result for a branch.
define void @f7(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f7:
; CHECK: alhsik [[REG1:%r[0-5]]], %r3, -1
; CHECK-DAG: st [[REG1]], 0(%r4)
; CHECK: jgle foo@PLT 
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  br i1 %obit, label %call, label %exit

call:
  tail call i32 @foo()
  br label %exit

exit:
  ret void
}

; ... and the same with the inverted direction.
define void @f8(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f8:
; CHECK: alhsik [[REG1:%r[0-5]]], %r3, -1
; CHECK-DAG: st [[REG1]], 0(%r4)
; CHECK: jgnle foo@PLT
; CHECK: br %r14
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  br i1 %obit, label %exit, label %call

call:
  tail call i32 @foo()
  br label %exit

exit:
  ret void
}


declare {i32, i1} @llvm.usub.with.overflow.i32(i32, i32) nounwind readnone

