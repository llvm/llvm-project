; Test signed multiplication with overflow on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s --check-prefix=Z13

declare i32 @foo()

; Check MSRKC, using the overflow result as a value.
define zeroext i1 @f1(i32 %dummy, i32 %a, i32 %b, ptr %res) {
; CHECK-LABEL: f1:
; CHECK: msrkc [[REG:%r[0-5]]], %r3, %r4
; CHECK-DAG: st [[REG]], 0(%r5)
; CHECK-DAG: lghi %r2, 0
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
; Z13-LABEL: f1:
; Z13-NOT: msrkc
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check MSGRKC, using the overflow result as a value.
define zeroext i1 @f2(i64 %dummy, i64 %a, i64 %b, ptr %res) {
; CHECK-LABEL: f2:
; CHECK: msgrkc [[REG:%r[0-5]]], %r3, %r4
; CHECK-DAG: stg [[REG]], 0(%r5)
; CHECK-DAG: lghi %r2, 0
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
; Z13-LABEL: f2:
; Z13-NOT: msgrkc
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check using the overflow result for a branch.
define void @f3(i64 %dummy, i64 %a, i64 %b, ptr %res) {
; CHECK-LABEL: f3:
; CHECK: msgrkc [[REG:%r[0-5]]], %r3, %r4
; CHECK: stg [[REG]], 0(%r5)
; CHECK: jgo foo@PLT
; CHECK: br %r14
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  br i1 %obit, label %call, label %exit

call:
  tail call i32 @foo()
  br label %exit

exit:
  ret void
}

; ... and the same with the inverted direction.
define void @f4(i32 %dummy, i32 %a, i32 %b, ptr %res) {
; CHECK-LABEL: f4:
; CHECK: msrkc [[REG:%r[0-5]]], %r3, %r4
; CHECK: st [[REG]], 0(%r5)
; CHECK: jgno foo@PLT
; CHECK: br %r14
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
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

; Check MSC with no displacement.
define zeroext i1 @f5(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f5:
; CHECK: msc %r3, 0(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %b = load i32, ptr %src
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check MSC with the load as the first operand.
define zeroext i1 @f6(i32 %dummy, ptr %src, i32 %b, ptr %res) {
; CHECK-LABEL: f6:
; CHECK: msc %r4, 0(%r3)
; CHECK-DAG: st %r4, 0(%r5)
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %a = load i32, ptr %src
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the high end of the MSC range.
define zeroext i1 @f7(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f7:
; CHECK: msc %r3, 524284(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131071
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
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
; CHECK: msc %r3, 0(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131072
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check the low end of the MSC range.
define zeroext i1 @f9(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f9:
; CHECK: msc %r3, -524288(%r4)
; CHECK-DAG: st %r3, 0(%r5)
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131072
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check MSC with an index.
define zeroext i1 @f10(i32 %dummy, i32 %a, i64 %base, i64 %index, ptr %res) {
; CHECK-LABEL: f10:
; CHECK: msc %r3, 4(%r5,%r4)
; CHECK-DAG: st %r3, 0(%r6)
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4
  %ptr = inttoptr i64 %add2 to ptr
  %b = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check MSGC with no displacement.
define zeroext i1 @f11(i64 %dummy, i64 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f11:
; CHECK: msgc %r3, 0(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %b = load i64, ptr %src
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check the high end of the aligned MSGC range.
define zeroext i1 @f12(i64 %dummy, i64 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f12:
; CHECK: msgc %r3, 524280(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 65535
  %b = load i64, ptr %ptr
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check the low end of the MSGC range.
define zeroext i1 @f13(i64 %dummy, i64 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f13:
; CHECK: msgc %r3, -524288(%r4)
; CHECK-DAG: stg %r3, 0(%r5)
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -65536
  %b = load i64, ptr %ptr
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  ret i1 %obit
}

; Check a plain multiplication of a loaded value still uses MS/MSG rather
; than the CC-setting forms.
define i32 @f14(i32 %a, ptr %src) {
; CHECK-LABEL: f14:
; CHECK: ms %r2, 0(%r3)
; CHECK: br %r14
  %b = load i32, ptr %src
  %mul = mul i32 %a, %b
  ret i32 %mul
}

define i64 @f15(i64 %a, ptr %src) {
; CHECK-LABEL: f15:
; CHECK: msg %r2, 0(%r3)
; CHECK: br %r14
  %b = load i64, ptr %src
  %mul = mul i64 %a, %b
  ret i64 %mul
}

; Check MSC with the overflow result used for a branch.
define void @f16(i32 %dummy, i32 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f16:
; CHECK: msc %r3, 0(%r4)
; CHECK: st %r3, 0(%r5)
; CHECK: jgo foo@PLT
; CHECK: br %r14
  %b = load i32, ptr %src
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
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

; ... and MSGC with the inverted direction.
define void @f17(i64 %dummy, i64 %a, ptr %src, ptr %res) {
; CHECK-LABEL: f17:
; CHECK: msgc %r3, 0(%r4)
; CHECK: stg %r3, 0(%r5)
; CHECK: jgno foo@PLT
; CHECK: br %r14
  %b = load i64, ptr %src
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, ptr %res
  br i1 %obit, label %exit, label %call

call:
  tail call i32 @foo()
  br label %exit

exit:
  ret void
}

; Check a constant operand.  There is no immediate form that sets the
; condition code, so the constant is loaded into a register first.
define zeroext i1 @f18(i32 %dummy, i32 %a, ptr %res) {
; CHECK-LABEL: f18:
; CHECK: lhi [[REG:%r[0-5]]], 100
; CHECK: msrkc [[REG]], %r3, [[REG]]
; CHECK-DAG: st [[REG]], 0(%r4)
; CHECK-DAG: locghio %r2, 1
; CHECK: br %r14
; Z13-LABEL: f18:
; Z13-NOT: msrkc
; Z13: mghi
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 100)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %res
  ret i1 %obit
}

; Check using the overflow result in a select.
define i64 @f19(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f19:
; CHECK: msgrkc %r2, %r3, %r4
; CHECK: locghio %r2, 0
; CHECK: br %r14
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  %sel = select i1 %obit, i64 0, i64 %val
  ret i64 %sel
}

; Check that a type narrower than i32 is promoted and the overflow is detected
; by checking that the product still fits, not via the multiply's CC.
define zeroext i1 @f20(i16 %a, i16 %b, ptr %res) {
; CHECK-LABEL: f20:
; CHECK: msrkc [[REG:%r[0-5]]], {{%r[0-5]}}, {{%r[0-5]}}
; CHECK: lhr [[EXT:%r[0-5]]], [[REG]]
; CHECK: cr [[EXT]], [[REG]]
; CHECK: sth [[REG]], 0(%r4)
; CHECK: br %r14
  %t = call {i16, i1} @llvm.smul.with.overflow.i16(i16 %a, i16 %b)
  %val = extractvalue {i16, i1} %t, 0
  %obit = extractvalue {i16, i1} %t, 1
  store i16 %val, ptr %res
  ret i1 %obit
}

declare {i16, i1} @llvm.smul.with.overflow.i16(i16, i16) nounwind readnone
declare {i32, i1} @llvm.smul.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.smul.with.overflow.i64(i64, i64) nounwind readnone
