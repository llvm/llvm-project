; Test subtractions between an i64 and a sign-extended i16 on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare i64 @foo()

; Check SGH with no displacement.
define zeroext i1 @f1(i64 %dummy, i64 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f1:
; CHECK: sgh %r3, 0(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: lghi %r2, 0
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %half = load i16, ptr %src
  %b = sext i16 %half to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check the high end of the aligned SGH range.
define zeroext i1 @f4(i64 %dummy, i64 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f4:
; CHECK: sgh %r3, 524286(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: lghi %r2, 0
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 262143
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f5(i64 %dummy, i64 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f5:
; CHECK: agfi %r4, 524288
; CHECK: sgh %r3, 0(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: lghi %r2, 0
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 262144
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check the high end of the negative aligned SGH range.
define zeroext i1 @f6(i64 %dummy, i64 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f6:
; CHECK: sgh %r3, -2(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: lghi %r2, 0
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -1
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check the low end of the SGH range.
define zeroext i1 @f7(i64 %dummy, i64 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f7:
; CHECK: sgh %r3, -524288(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: lghi %r2, 0
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -262144
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f8(i64 %dummy, i64 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f8:
; CHECK: agfi %r4, -524290
; CHECK: sgh %r3, 0(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: lghi %r2, 0
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -262145
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check that SGH allows an index.
define zeroext i1 @f9(i64 %src, i64 %index, i64 %a, ptr %res) {
; CHECK-LABEL: f9:
; CHECK: sgh %r4, 524284({{%r3,%r2|%r2,%r3}})
; CHECK-DAG: stg %r4, 0(%r5)
; CHECK-DAG: lghi %r2, 0
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524284
  %ptr = inttoptr i64 %add2 to ptr
  %half = load i16, ptr %ptr
  %b = sext i16 %half to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check using the overflow result for a branch.
define void @f11(i64 %dummy, i64 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f11:
; CHECK: sgh %r3, 0(%r4)
; CHECK: stg %r3, 0(%r5)
; CHECK: jgo foo@PLT
; CHECK: br %r14
  %half = load i16, ptr %src
  %b = sext i16 %half to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %b)
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
define void @f12(i64 %dummy, i64 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f12:
; CHECK: sgh %r3, 0(%r4)
; CHECK: stg %r3, 0(%r5)
; CHECK: jgno foo@PLT
; CHECK: br %r14
  %half = load i16, ptr %src
  %b = sext i16 %half to i64
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %a, i64 %b)
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


declare {i64, i1} @llvm.ssub.with.overflow.i64(i64, i64) nounwind readnone

