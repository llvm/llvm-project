; Test 32-bit subtraction in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @foo()

; Check SR.
define zeroext i1 @f1(i32 %dummy, i32 %a, i32 %b, ptr %res) {
; CHECK-LABEL: f1:
; CHECK: sr %r3, %r4
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check using the overflow result for a branch.
define void @f2(i32 %dummy, i32 %a, i32 %b, ptr %res) {
; CHECK-LABEL: f2:
; CHECK: sr %r3, %r4
; CHECK: st %r3, 0(%r5)
; CHECK: jgo foo@PLT
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
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
define void @f3(i32 %dummy, i32 %a, i32 %b, ptr %res) {
; CHECK-LABEL: f3:
; CHECK: sr %r3, %r4
; CHECK: st %r3, 0(%r5)
; CHECK: jgno foo@PLT
; CHECK: br %r14
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
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

; Check the low end of the S range.
define zeroext i1 @f4(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f4:
; CHECK: s %r3, 0(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %b = load i32, ptr %src
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the high end of the aligned S range.
define zeroext i1 @f5(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f5:
; CHECK: s %r3, 4092(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 1023
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the next word up, which should use SY instead of S.
define zeroext i1 @f6(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f6:
; CHECK: sy %r3, 4096(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 1024
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the high end of the aligned SY range.
define zeroext i1 @f7(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f7:
; CHECK: sy %r3, 524284(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131071
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f8(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f8:
; CHECK: agfi %r4, 524288
; CHECK: s %r3, 0(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131072
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the high end of the negative aligned SY range.
define zeroext i1 @f9(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f9:
; CHECK: sy %r3, -4(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -1
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the low end of the SY range.
define zeroext i1 @f10(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f10:
; CHECK: sy %r3, -524288(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131072
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f11(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f11:
; CHECK: agfi %r4, -524292
; CHECK: s %r3, 0(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131073
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check that S allows an index.
define zeroext i1 @f12(i64 %src, i64 %index, i32 %a, ptr %res) {
; CHECK-LABEL: f12:
; CHECK: s %r4, 4092({{%r3,%r2|%r2,%r3}})
; CHECK-DAG: st %r4, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to ptr
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check that SY allows an index.
define zeroext i1 @f13(i64 %src, i64 %index, i32 %a, ptr %res) {
; CHECK-LABEL: f13:
; CHECK: sy %r4, 4096({{%r3,%r2|%r2,%r3}})
; CHECK-DAG: st %r4, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to ptr
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check that subtractions of spilled values can use S rather than SR.
define zeroext i1 @f14(ptr %ptr0) {
; CHECK-LABEL: f14:
; CHECK: brasl %r14, foo@PLT
; CHECK: s %r2, 16{{[04]}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i32, ptr %ptr0, i64 2
  %ptr2 = getelementptr i32, ptr %ptr0, i64 4
  %ptr3 = getelementptr i32, ptr %ptr0, i64 6
  %ptr4 = getelementptr i32, ptr %ptr0, i64 8
  %ptr5 = getelementptr i32, ptr %ptr0, i64 10
  %ptr6 = getelementptr i32, ptr %ptr0, i64 12
  %ptr7 = getelementptr i32, ptr %ptr0, i64 14
  %ptr8 = getelementptr i32, ptr %ptr0, i64 16
  %ptr9 = getelementptr i32, ptr %ptr0, i64 18

  %val0 = load i32, ptr %ptr0
  %val1 = load i32, ptr %ptr1
  %val2 = load i32, ptr %ptr2
  %val3 = load i32, ptr %ptr3
  %val4 = load i32, ptr %ptr4
  %val5 = load i32, ptr %ptr5
  %val6 = load i32, ptr %ptr6
  %val7 = load i32, ptr %ptr7
  %val8 = load i32, ptr %ptr8
  %val9 = load i32, ptr %ptr9

  %ret = call i32 @foo()

  %t0 = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %ret, i32 %val0)
  %add0 = extractvalue {i32, i1} %t0, 0
  %obit0 = extractvalue {i32, i1} %t0, 1
  %t1 = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %add0, i32 %val1)
  %add1 = extractvalue {i32, i1} %t1, 0
  %obit1 = extractvalue {i32, i1} %t1, 1
  %res1 = or i1 %obit0, %obit1
  %t2 = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %add1, i32 %val2)
  %add2 = extractvalue {i32, i1} %t2, 0
  %obit2 = extractvalue {i32, i1} %t2, 1
  %res2 = or i1 %res1, %obit2
  %t3 = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %add2, i32 %val3)
  %add3 = extractvalue {i32, i1} %t3, 0
  %obit3 = extractvalue {i32, i1} %t3, 1
  %res3 = or i1 %res2, %obit3
  %t4 = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %add3, i32 %val4)
  %add4 = extractvalue {i32, i1} %t4, 0
  %obit4 = extractvalue {i32, i1} %t4, 1
  %res4 = or i1 %res3, %obit4
  %t5 = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %add4, i32 %val5)
  %add5 = extractvalue {i32, i1} %t5, 0
  %obit5 = extractvalue {i32, i1} %t5, 1
  %res5 = or i1 %res4, %obit5
  %t6 = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %add5, i32 %val6)
  %add6 = extractvalue {i32, i1} %t6, 0
  %obit6 = extractvalue {i32, i1} %t6, 1
  %res6 = or i1 %res5, %obit6
  %t7 = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %add6, i32 %val7)
  %add7 = extractvalue {i32, i1} %t7, 0
  %obit7 = extractvalue {i32, i1} %t7, 1
  %res7 = or i1 %res6, %obit7
  %t8 = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %add7, i32 %val8)
  %add8 = extractvalue {i32, i1} %t8, 0
  %obit8 = extractvalue {i32, i1} %t8, 1
  %res8 = or i1 %res7, %obit8
  %t9 = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %add8, i32 %val9)
  %add9 = extractvalue {i32, i1} %t9, 0
  %obit9 = extractvalue {i32, i1} %t9, 1
  %res9 = or i1 %res8, %obit9

  ret i1 %res9
}

declare {i32, i1} @llvm.ssub.with.overflow.i32(i32, i32) nounwind readnone

