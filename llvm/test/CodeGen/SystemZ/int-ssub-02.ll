; Test 32-bit subtraction in which the second operand is a sign-extended
; i16 memory value.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @foo()

; Check the low end of the SH range.
define zeroext i1 @f1(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f1:
; CHECK: sh %r3, 0(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %half = load i16, ptr %src
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the high end of the aligned SH range.
define zeroext i1 @f2(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f2:
; CHECK: sh %r3, 4094(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 2047
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the next halfword up, which should use SHY instead of SH.
define zeroext i1 @f3(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f3:
; CHECK: shy %r3, 4096(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 2048
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the high end of the aligned SHY range.
define zeroext i1 @f4(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f4:
; CHECK: shy %r3, 524286(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 262143
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f5(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f5:
; CHECK: agfi %r4, 524288
; CHECK: sh %r3, 0(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 262144
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the high end of the negative aligned SHY range.
define zeroext i1 @f6(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f6:
; CHECK: shy %r3, -2(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -1
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the low end of the SHY range.
define zeroext i1 @f7(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f7:
; CHECK: shy %r3, -524288(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -262144
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f8(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f8:
; CHECK: agfi %r4, -524290
; CHECK: sh %r3, 0(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -262145
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check that SH allows an index.
define zeroext i1 @f9(i64 %src, i64 %index, i32 %a, ptr %res) {
; CHECK-LABEL: f9:
; CHECK: sh %r4, 4094({{%r3,%r2|%r2,%r3}})
; CHECK-DAG: st %r4, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4094
  %ptr = inttoptr i64 %add2 to ptr
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check that SHY allows an index.
define zeroext i1 @f10(i64 %src, i64 %index, i32 %a, ptr %res) {
; CHECK-LABEL: f10:
; CHECK: shy %r4, 4096({{%r3,%r2|%r2,%r3}})
; CHECK-DAG: st %r4, 0(%r5)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to ptr
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i32
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check using the overflow result for a branch.
define void @f11(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f11:
; CHECK: sh %r3, 0(%r4)
; CHECK: st %r3, 0(%r5)
; CHECK: jgo foo@PLT
; CHECK: br %r14
  %half = load i16, ptr %src
  %b = sext i16 %half to i32
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
define void @f12(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f12:
; CHECK: sh %r3, 0(%r4)
; CHECK: st %r3, 0(%r5)
; CHECK: jgno foo@PLT
; CHECK: br %r14
  %half = load i16, ptr %src
  %b = sext i16 %half to i32
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


declare {i32, i1} @llvm.ssub.with.overflow.i32(i32, i32) nounwind readnone

