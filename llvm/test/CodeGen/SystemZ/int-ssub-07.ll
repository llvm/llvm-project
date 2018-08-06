; Test 64-bit subtraction in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i32 @foo()

; Check subtractions of 1.
define zeroext i1 @f1(i64 %dummy, i64 %a, i64 *%res) {
; CHECK-LABEL: f1:
; CHECK: aghi %r3, -1
; CHECK-DAG: stg %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit

}

; Check the high end of the SGHI range.
define zeroext i1 @f2(i64 %dummy, i64 %a, i64 *%res) {
; CHECK-LABEL: f2:
; CHECK: aghi %r3, -32768
; CHECK-DAG: stg %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 32768)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the next value up, which must use SGFI instead.
define zeroext i1 @f3(i64 %dummy, i64 %a, i64 *%res) {
; CHECK-LABEL: f3:
; CHECK: agfi %r3, -32769
; CHECK-DAG: stg %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 32769)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the high end of the SGFI range.
define zeroext i1 @f4(i64 %dummy, i64 %a, i64 *%res) {
; CHECK-LABEL: f4:
; CHECK: agfi %r3, -2147483648
; CHECK-DAG: stg %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 2147483648)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the next value up, which must be loaded into a register first.
define zeroext i1 @f5(i64 %dummy, i64 %a, i64 *%res) {
; CHECK-LABEL: f5:
; CHECK: llilf [[REG1:%r[0-9]+]], 2147483649
; CHECK: sgr %r3, [[REG1]]
; CHECK-DAG: stg %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 2147483649)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the high end of the negative SGHI range.
define zeroext i1 @f6(i64 %dummy, i64 %a, i64 *%res) {
; CHECK-LABEL: f6:
; CHECK: aghi %r3, 1
; CHECK-DAG: stg %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 -1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the low end of the SGHI range.
define zeroext i1 @f7(i64 %dummy, i64 %a, i64 *%res) {
; CHECK-LABEL: f7:
; CHECK: aghi %r3, 32767
; CHECK-DAG: stg %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 -32767)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the next value down, which must use SGFI instead.
define zeroext i1 @f8(i64 %dummy, i64 %a, i64 *%res) {
; CHECK-LABEL: f8:
; CHECK: agfi %r3, 32768
; CHECK-DAG: stg %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 -32768)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the low end of the SGFI range.
define zeroext i1 @f9(i64 %dummy, i64 %a, i64 *%res) {
; CHECK-LABEL: f9:
; CHECK: agfi %r3, 2147483647
; CHECK-DAG: stg %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 -2147483647)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the next value down, which must use register subtraction instead.
define zeroext i1 @f10(i64 %dummy, i64 %a, i64 *%res) {
; CHECK-LABEL: f10:
; CHECK: lgfi [[REG1:%r[0-9]+]], -2147483648
; CHECK: sgr %r3, [[REG1]]
; CHECK-DAG: stg %r3, 0(%r4)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 -2147483648)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check using the overflow result for a branch.
define void @f11(i64 %dummy, i64 %a, i64 *%res) {
; CHECK-LABEL: f11:
; CHECK: aghi %r3, -1
; CHECK: stg %r3, 0(%r4)
; CHECK: {{jgo foo@PLT|bnor %r14}}
; CHECK: {{br %r14|jg foo@PLT}}
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  br i1 %obit, label %call, label %exit

call:
  tail call i32 @foo()
  br label %exit

exit:
  ret void
}

; ... and the same with the inverted direction.
define void @f12(i64 %dummy, i64 %a, i64 *%res) {
; CHECK-LABEL: f12:
; CHECK: aghi %r3, -1
; CHECK: stg %r3, 0(%r4)
; CHECK: {{jgno foo@PLT|bor %r14}}
; CHECK: {{br %r14|jg foo@PLT}}
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  br i1 %obit, label %exit, label %call

call:
  tail call i32 @foo()
  br label %exit

exit:
  ret void
}

declare {i64, i1} @llvm.ssub.with.overflow.i64(i64, i64) nounwind readnone

