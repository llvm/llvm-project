; Test 64-bit addition in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @foo()

; Check ALGR.
define zeroext i1 @f1(i64 %dummy, i64 %a, i64 %b, i64 *%res) {
; CHECK-LABEL: f1:
; CHECK: algr %r3, %r4
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check using the overflow result for a branch.
define void @f2(i64 %dummy, i64 %a, i64 %b, i64 *%res) {
; CHECK-LABEL: f2:
; CHECK: algr %r3, %r4
; CHECK: stg %r3, 0(%r5)
; CHECK: jgnle foo@PLT
; CHECK: br %r14
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  br i1 %obit, label %call, label %exit

call:
  tail call i64 @foo()
  br label %exit

exit:
  ret void
}

; ... and the same with the inverted direction.
define void @f3(i64 %dummy, i64 %a, i64 %b, i64 *%res) {
; CHECK-LABEL: f3:
; CHECK: algr %r3, %r4
; CHECK: stg %r3, 0(%r5)
; CHECK: jgle foo@PLT
; CHECK: br %r14
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  br i1 %obit, label %exit, label %call

call:
  tail call i64 @foo()
  br label %exit

exit:
  ret void
}

; Check ALG with no displacement.
define zeroext i1 @f4(i64 %dummy, i64 %a, i64 *%src, i64 *%res) {
; CHECK-LABEL: f4:
; CHECK: alg %r3, 0(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %b = load i64, i64 *%src
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the high end of the aligned ALG range.
define zeroext i1 @f5(i64 %dummy, i64 %a, i64 *%src, i64 *%res) {
; CHECK-LABEL: f5:
; CHECK: alg %r3, 524280(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65535
  %b = load i64, i64 *%ptr
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f6(i64 %dummy, i64 %a, i64 *%src, i64 *%res) {
; CHECK-LABEL: f6:
; CHECK: agfi %r4, 524288
; CHECK: alg %r3, 0(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65536
  %b = load i64, i64 *%ptr
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the high end of the negative aligned ALG range.
define zeroext i1 @f7(i64 %dummy, i64 %a, i64 *%src, i64 *%res) {
; CHECK-LABEL: f7:
; CHECK: alg %r3, -8(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -1
  %b = load i64, i64 *%ptr
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the low end of the ALG range.
define zeroext i1 @f8(i64 %dummy, i64 %a, i64 *%src, i64 *%res) {
; CHECK-LABEL: f8:
; CHECK: alg %r3, -524288(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65536
  %b = load i64, i64 *%ptr
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f9(i64 %dummy, i64 %a, i64 *%src, i64 *%res) {
; CHECK-LABEL: f9:
; CHECK: agfi %r4, -524296
; CHECK: alg %r3, 0(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65537
  %b = load i64, i64 *%ptr
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check that ALG allows an index.
define zeroext i1 @f10(i64 %src, i64 %index, i64 %a, i64 *%res) {
; CHECK-LABEL: f10:
; CHECK: alg %r4, 524280({{%r3,%r2|%r2,%r3}})
; CHECK-DAG: stg %r4, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524280
  %ptr = inttoptr i64 %add2 to i64 *
  %b = load i64, i64 *%ptr
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check that additions of spilled values can use ALG rather than ALGR.
define zeroext i1 @f11(i64 *%ptr0) {
; CHECK-LABEL: f11:
; CHECK: brasl %r14, foo@PLT
; CHECK: alg %r2, 160(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i64, i64 *%ptr0, i64 2
  %ptr2 = getelementptr i64, i64 *%ptr0, i64 4
  %ptr3 = getelementptr i64, i64 *%ptr0, i64 6
  %ptr4 = getelementptr i64, i64 *%ptr0, i64 8
  %ptr5 = getelementptr i64, i64 *%ptr0, i64 10
  %ptr6 = getelementptr i64, i64 *%ptr0, i64 12
  %ptr7 = getelementptr i64, i64 *%ptr0, i64 14
  %ptr8 = getelementptr i64, i64 *%ptr0, i64 16
  %ptr9 = getelementptr i64, i64 *%ptr0, i64 18

  %val0 = load i64, i64 *%ptr0
  %val1 = load i64, i64 *%ptr1
  %val2 = load i64, i64 *%ptr2
  %val3 = load i64, i64 *%ptr3
  %val4 = load i64, i64 *%ptr4
  %val5 = load i64, i64 *%ptr5
  %val6 = load i64, i64 *%ptr6
  %val7 = load i64, i64 *%ptr7
  %val8 = load i64, i64 *%ptr8
  %val9 = load i64, i64 *%ptr9

  %ret = call i64 @foo()

  %t0 = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %ret, i64 %val0)
  %add0 = extractvalue {i64, i1} %t0, 0
  %obit0 = extractvalue {i64, i1} %t0, 1
  %t1 = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %add0, i64 %val1)
  %add1 = extractvalue {i64, i1} %t1, 0
  %obit1 = extractvalue {i64, i1} %t1, 1
  %res1 = or i1 %obit0, %obit1
  %t2 = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %add1, i64 %val2)
  %add2 = extractvalue {i64, i1} %t2, 0
  %obit2 = extractvalue {i64, i1} %t2, 1
  %res2 = or i1 %res1, %obit2
  %t3 = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %add2, i64 %val3)
  %add3 = extractvalue {i64, i1} %t3, 0
  %obit3 = extractvalue {i64, i1} %t3, 1
  %res3 = or i1 %res2, %obit3
  %t4 = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %add3, i64 %val4)
  %add4 = extractvalue {i64, i1} %t4, 0
  %obit4 = extractvalue {i64, i1} %t4, 1
  %res4 = or i1 %res3, %obit4
  %t5 = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %add4, i64 %val5)
  %add5 = extractvalue {i64, i1} %t5, 0
  %obit5 = extractvalue {i64, i1} %t5, 1
  %res5 = or i1 %res4, %obit5
  %t6 = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %add5, i64 %val6)
  %add6 = extractvalue {i64, i1} %t6, 0
  %obit6 = extractvalue {i64, i1} %t6, 1
  %res6 = or i1 %res5, %obit6
  %t7 = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %add6, i64 %val7)
  %add7 = extractvalue {i64, i1} %t7, 0
  %obit7 = extractvalue {i64, i1} %t7, 1
  %res7 = or i1 %res6, %obit7
  %t8 = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %add7, i64 %val8)
  %add8 = extractvalue {i64, i1} %t8, 0
  %obit8 = extractvalue {i64, i1} %t8, 1
  %res8 = or i1 %res7, %obit8
  %t9 = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %add8, i64 %val9)
  %add9 = extractvalue {i64, i1} %t9, 0
  %obit9 = extractvalue {i64, i1} %t9, 1
  %res9 = or i1 %res8, %obit9

  ret i1 %res9
}

declare {i64, i1} @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone

