; Test 32-bit subtraction in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i32 @foo()

; Check subtractions of 1.
define zeroext i1 @f1(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f1:
; CHECK: ahi %r3, -1
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the high end of the AHI range.
define zeroext i1 @f2(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f2:
; CHECK: ahi %r3, -32768
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 32768)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the next value up, which must use AFI instead.
define zeroext i1 @f3(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f3:
; CHECK: afi %r3, -32769
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 32769)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the high end of the signed 32-bit range.
define zeroext i1 @f4(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f4:
; CHECK: afi %r3, -2147483647
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 2147483647)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the next value up, which is treated as a negative value
; and must use a register.
define zeroext i1 @f5(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f5:
; CHECK: llilh [[REG1:%r[0-5]]], 32768
; CHECK: sr %r3, [[REG1]]
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 2147483648)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the next value up, which is treated as a negative value,
; and can use AFI again.
define zeroext i1 @f6(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f6:
; CHECK: afi %r3, 2147483647
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 2147483649)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the high end of the negative AHI range.
define zeroext i1 @f7(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f7:
; CHECK: ahi %r3, 1
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 -1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the low end of the AHI range.
define zeroext i1 @f8(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f8:
; CHECK: ahi %r3, 32767
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 -32767)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the next value down, which must use AFI instead.
define zeroext i1 @f9(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f9:
; CHECK: afi %r3, 32768
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 -32768)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the low end of the signed 32-bit range.
define zeroext i1 @f10(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f10:
; CHECK: afi %r3, 2147483647
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 -2147483647)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the next value down, which must use a register.
define zeroext i1 @f11(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f11:
; CHECK: llilh [[REG1:%r[0-5]]], 32768
; CHECK: sr %r3, [[REG1]]
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 -2147483648)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the next value down, which is treated as a positive value.
define zeroext i1 @f12(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f12:
; CHECK: afi %r3, -2147483647
; CHECK-DAG: st %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 -2147483649)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check using the overflow result for a branch.
define void @f13(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f13:
; CHECK: ahi %r3, -1
; CHECK: st %r3, 0(%r4)
; CHECK: {{jgo foo@PLT|bnor %r14}}
; CHECK: {{br %r14|jg foo@PLT}}
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 1)
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
define void @f14(i32 %dummy, i32 %a, i32 *%res) {
; CHECK-LABEL: f14:
; CHECK: ahi %r3, -1
; CHECK: st %r3, 0(%r4)
; CHECK: {{jgno foo@PLT|bor %r14}}
; CHECK: {{br %r14|jg foo@PLT}}
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 1)
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


declare {i32, i1} @llvm.ssub.with.overflow.i32(i32, i32) nounwind readnone

