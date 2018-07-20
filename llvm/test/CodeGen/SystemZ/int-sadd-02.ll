; Test 32-bit addition in which the second operand is a sign-extended
; i16 memory value.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @foo()

; Check the low end of the AH range.
define zeroext i1 @f1(i32 %dummy, i32 %a, i16 *%src, i32 *%res) {
; CHECK-LABEL: f1:
; CHECK: ah %r3, 0(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %half = load i16, i16 *%src
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the high end of the aligned AH range.
define zeroext i1 @f2(i32 %dummy, i32 %a, i16 *%src, i32 *%res) {
; CHECK-LABEL: f2:
; CHECK: ah %r3, 4094(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 2047
  %half = load i16, i16 *%ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the next halfword up, which should use AHY instead of AH.
define zeroext i1 @f3(i32 %dummy, i32 %a, i16 *%src, i32 *%res) {
; CHECK-LABEL: f3:
; CHECK: ahy %r3, 4096(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 2048
  %half = load i16, i16 *%ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the high end of the aligned AHY range.
define zeroext i1 @f4(i32 %dummy, i32 %a, i16 *%src, i32 *%res) {
; CHECK-LABEL: f4:
; CHECK: ahy %r3, 524286(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 262143
  %half = load i16, i16 *%ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f5(i32 %dummy, i32 %a, i16 *%src, i32 *%res) {
; CHECK-LABEL: f5:
; CHECK: agfi %r4, 524288
; CHECK: ah %r3, 0(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 262144
  %half = load i16, i16 *%ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the high end of the negative aligned AHY range.
define zeroext i1 @f6(i32 %dummy, i32 %a, i16 *%src, i32 *%res) {
; CHECK-LABEL: f6:
; CHECK: ahy %r3, -2(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -1
  %half = load i16, i16 *%ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the low end of the AHY range.
define zeroext i1 @f7(i32 %dummy, i32 %a, i16 *%src, i32 *%res) {
; CHECK-LABEL: f7:
; CHECK: ahy %r3, -524288(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -262144
  %half = load i16, i16 *%ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f8(i32 %dummy, i32 %a, i16 *%src, i32 *%res) {
; CHECK-LABEL: f8:
; CHECK: agfi %r4, -524290
; CHECK: ah %r3, 0(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -262145
  %half = load i16, i16 *%ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check that AH allows an index.
define zeroext i1 @f9(i64 %src, i64 %index, i32 %a, i32 *%res) {
; CHECK-LABEL: f9:
; CHECK: ah %r4, 4094({{%r3,%r2|%r2,%r3}})
; CHECK-DAG: st %r4, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4094
  %ptr = inttoptr i64 %add2 to i16 *
  %half = load i16, i16 *%ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check that AHY allows an index.
define zeroext i1 @f10(i64 %src, i64 %index, i32 %a, i32 *%res) {
; CHECK-LABEL: f10:
; CHECK: ahy %r4, 4096({{%r3,%r2|%r2,%r3}})
; CHECK-DAG: st %r4, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i16 *
  %half = load i16, i16 *%ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32 *%res
  ret i1 %obit
}

; Check using the overflow result for a branch.
define void @f11(i32 %dummy, i32 %a, i16 *%src, i32 *%res) {
; CHECK-LABEL: f11:
; CHECK: ah %r3, 0(%r4)
; CHECK: st %r3, 0(%r5)
; CHECK: jgo foo@PLT
; CHECK: br %r14
  %half = load i16, i16 *%src
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
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
define void @f12(i32 %dummy, i32 %a, i16 *%src, i32 *%res) {
; CHECK-LABEL: f12:
; CHECK: ah %r3, 0(%r4)
; CHECK: st %r3, 0(%r5)
; CHECK: jgno foo@PLT
; CHECK: br %r14
  %half = load i16, i16 *%src
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
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


declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone

