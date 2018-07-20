; Test subtractions between an i64 and a sign-extended i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @foo()

; Check SGFR.
define zeroext i1 @f1(i64 %dummy, i64 %a, i32 %b, i64 *%res) {
; CHECK-LABEL: f1:
; CHECK: sgfr %r3, %r4
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %bext = sext i32 %b to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %bext)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check using the overflow result for a branch.
define void @f2(i64 %dummy, i64 %a, i32 %b, i64 *%res) {
; CHECK-LABEL: f2:
; CHECK: sgfr %r3, %r4
; CHECK: stg %r3, 0(%r5)
; CHECK: jgo foo@PLT
; CHECK: br %r14
  %bext = sext i32 %b to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %bext)
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
define void @f3(i64 %dummy, i64 %a, i32 %b, i64 *%res) {
; CHECK-LABEL: f3:
; CHECK: sgfr %r3, %r4
; CHECK: stg %r3, 0(%r5)
; CHECK: jgno foo@PLT
; CHECK: br %r14
  %bext = sext i32 %b to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %bext)
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

; Check SGF with no displacement.
define zeroext i1 @f4(i64 %dummy, i64 %a, i32 *%src, i64 *%res) {
; CHECK-LABEL: f4:
; CHECK: sgf %r3, 0(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %b = load i32, i32 *%src
  %bext = sext i32 %b to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %bext)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the high end of the aligned SGF range.
define zeroext i1 @f5(i64 %dummy, i64 %a, i32 *%src, i64 *%res) {
; CHECK-LABEL: f5:
; CHECK: sgf %r3, 524284(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131071
  %b = load i32, i32 *%ptr
  %bext = sext i32 %b to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %bext)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f6(i64 %dummy, i64 %a, i32 *%src, i64 *%res) {
; CHECK-LABEL: f6:
; CHECK: agfi %r4, 524288
; CHECK: sgf %r3, 0(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131072
  %b = load i32, i32 *%ptr
  %bext = sext i32 %b to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %bext)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the high end of the negative aligned SGF range.
define zeroext i1 @f7(i64 %dummy, i64 %a, i32 *%src, i64 *%res) {
; CHECK-LABEL: f7:
; CHECK: sgf %r3, -4(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -1
  %b = load i32, i32 *%ptr
  %bext = sext i32 %b to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %bext)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the low end of the SGF range.
define zeroext i1 @f8(i64 %dummy, i64 %a, i32 *%src, i64 *%res) {
; CHECK-LABEL: f8:
; CHECK: sgf %r3, -524288(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131072
  %b = load i32, i32 *%ptr
  %bext = sext i32 %b to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %bext)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f9(i64 %dummy, i64 %a, i32 *%src, i64 *%res) {
; CHECK-LABEL: f9:
; CHECK: agfi %r4, -524292
; CHECK: sgf %r3, 0(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131073
  %b = load i32, i32 *%ptr
  %bext = sext i32 %b to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %bext)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check that SGF allows an index.
define zeroext i1 @f10(i64 %src, i64 %index, i64 %a, i64 *%res) {
; CHECK-LABEL: f10:
; CHECK: sgf %r4, 524284({{%r3,%r2|%r2,%r3}})
; CHECK-DAG: stg %r4, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524284
  %ptr = inttoptr i64 %add2 to i32 *
  %b = load i32, i32 *%ptr
  %bext = sext i32 %b to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %bext)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64 *%res
  ret i1 %obit
}

; Check that subtractions of spilled values can use SGF rather than SGFR.
define zeroext i1 @f11(i32 *%ptr0) {
; CHECK-LABEL: f11:
; CHECK: brasl %r14, foo@PLT
; CHECK: sgf %r2, 16{{[04]}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i32, i32 *%ptr0, i64 2
  %ptr2 = getelementptr i32, i32 *%ptr0, i64 4
  %ptr3 = getelementptr i32, i32 *%ptr0, i64 6
  %ptr4 = getelementptr i32, i32 *%ptr0, i64 8
  %ptr5 = getelementptr i32, i32 *%ptr0, i64 10
  %ptr6 = getelementptr i32, i32 *%ptr0, i64 12
  %ptr7 = getelementptr i32, i32 *%ptr0, i64 14
  %ptr8 = getelementptr i32, i32 *%ptr0, i64 16
  %ptr9 = getelementptr i32, i32 *%ptr0, i64 18

  %val0 = load i32, i32 *%ptr0
  %val1 = load i32, i32 *%ptr1
  %val2 = load i32, i32 *%ptr2
  %val3 = load i32, i32 *%ptr3
  %val4 = load i32, i32 *%ptr4
  %val5 = load i32, i32 *%ptr5
  %val6 = load i32, i32 *%ptr6
  %val7 = load i32, i32 *%ptr7
  %val8 = load i32, i32 *%ptr8
  %val9 = load i32, i32 *%ptr9

  %frob0 = add i32 %val0, 100
  %frob1 = add i32 %val1, 100
  %frob2 = add i32 %val2, 100
  %frob3 = add i32 %val3, 100
  %frob4 = add i32 %val4, 100
  %frob5 = add i32 %val5, 100
  %frob6 = add i32 %val6, 100
  %frob7 = add i32 %val7, 100
  %frob8 = add i32 %val8, 100
  %frob9 = add i32 %val9, 100

  store i32 %frob0, i32 *%ptr0
  store i32 %frob1, i32 *%ptr1
  store i32 %frob2, i32 *%ptr2
  store i32 %frob3, i32 *%ptr3
  store i32 %frob4, i32 *%ptr4
  store i32 %frob5, i32 *%ptr5
  store i32 %frob6, i32 *%ptr6
  store i32 %frob7, i32 *%ptr7
  store i32 %frob8, i32 *%ptr8
  store i32 %frob9, i32 *%ptr9

  %ret = call i64 @foo()

  %ext0 = sext i32 %frob0 to i64
  %ext1 = sext i32 %frob1 to i64
  %ext2 = sext i32 %frob2 to i64
  %ext3 = sext i32 %frob3 to i64
  %ext4 = sext i32 %frob4 to i64
  %ext5 = sext i32 %frob5 to i64
  %ext6 = sext i32 %frob6 to i64
  %ext7 = sext i32 %frob7 to i64
  %ext8 = sext i32 %frob8 to i64
  %ext9 = sext i32 %frob9 to i64

  %t0 = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %ret, i64 %ext0)
  %add0 = extractvalue {i64, i1} %t0, 0
  %obit0 = extractvalue {i64, i1} %t0, 1
  %t1 = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %add0, i64 %ext1)
  %add1 = extractvalue {i64, i1} %t1, 0
  %obit1 = extractvalue {i64, i1} %t1, 1
  %res1 = or i1 %obit0, %obit1
  %t2 = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %add1, i64 %ext2)
  %add2 = extractvalue {i64, i1} %t2, 0
  %obit2 = extractvalue {i64, i1} %t2, 1
  %res2 = or i1 %res1, %obit2
  %t3 = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %add2, i64 %ext3)
  %add3 = extractvalue {i64, i1} %t3, 0
  %obit3 = extractvalue {i64, i1} %t3, 1
  %res3 = or i1 %res2, %obit3
  %t4 = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %add3, i64 %ext4)
  %add4 = extractvalue {i64, i1} %t4, 0
  %obit4 = extractvalue {i64, i1} %t4, 1
  %res4 = or i1 %res3, %obit4
  %t5 = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %add4, i64 %ext5)
  %add5 = extractvalue {i64, i1} %t5, 0
  %obit5 = extractvalue {i64, i1} %t5, 1
  %res5 = or i1 %res4, %obit5
  %t6 = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %add5, i64 %ext6)
  %add6 = extractvalue {i64, i1} %t6, 0
  %obit6 = extractvalue {i64, i1} %t6, 1
  %res6 = or i1 %res5, %obit6
  %t7 = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %add6, i64 %ext7)
  %add7 = extractvalue {i64, i1} %t7, 0
  %obit7 = extractvalue {i64, i1} %t7, 1
  %res7 = or i1 %res6, %obit7
  %t8 = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %add7, i64 %ext8)
  %add8 = extractvalue {i64, i1} %t8, 0
  %obit8 = extractvalue {i64, i1} %t8, 1
  %res8 = or i1 %res7, %obit8
  %t9 = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %add8, i64 %ext9)
  %add9 = extractvalue {i64, i1} %t9, 0
  %obit9 = extractvalue {i64, i1} %t9, 1
  %res9 = or i1 %res8, %obit9

  ret i1 %res9
}

declare {i64, i1} @llvm.ssub.with.overflow.i64(i64, i64) nounwind readnone

