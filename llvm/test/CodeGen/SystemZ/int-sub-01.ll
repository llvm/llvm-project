; Test 32-bit subtraction.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i32 @foo()

; Check SR.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: sr %r2, %r3
; CHECK: br %r14
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the low end of the S range.
define i32 @f2(i32 %a, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: s %r2, 0(%r3)
; CHECK: br %r14
  %b = load i32, ptr %src
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the high end of the aligned S range.
define i32 @f3(i32 %a, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: s %r2, 4092(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 1023
  %b = load i32, ptr %ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the next word up, which should use SY instead of S.
define i32 @f4(i32 %a, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: sy %r2, 4096(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 1024
  %b = load i32, ptr %ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the high end of the aligned SY range.
define i32 @f5(i32 %a, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: sy %r2, 524284(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131071
  %b = load i32, ptr %ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f6(i32 %a, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r3, 524288
; CHECK: s %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131072
  %b = load i32, ptr %ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the high end of the negative aligned SY range.
define i32 @f7(i32 %a, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: sy %r2, -4(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -1
  %b = load i32, ptr %ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the low end of the SY range.
define i32 @f8(i32 %a, ptr %src) {
; CHECK-LABEL: f8:
; CHECK: sy %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131072
  %b = load i32, ptr %ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f9(i32 %a, ptr %src) {
; CHECK-LABEL: f9:
; CHECK: agfi %r3, -524292
; CHECK: s %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131073
  %b = load i32, ptr %ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check that S allows an index.
define i32 @f10(i32 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: s %r2, 4092({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to ptr
  %b = load i32, ptr %ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check that SY allows an index.
define i32 @f11(i32 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: sy %r2, 4096({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to ptr
  %b = load i32, ptr %ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check that subtractions of spilled values can use S rather than SR.
define i32 @f12(ptr %ptr0) {
; CHECK-LABEL: f12:
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

  %sub0 = sub i32 %ret, %val0
  %sub1 = sub i32 %sub0, %val1
  %sub2 = sub i32 %sub1, %val2
  %sub3 = sub i32 %sub2, %val3
  %sub4 = sub i32 %sub3, %val4
  %sub5 = sub i32 %sub4, %val5
  %sub6 = sub i32 %sub5, %val6
  %sub7 = sub i32 %sub6, %val7
  %sub8 = sub i32 %sub7, %val8
  %sub9 = sub i32 %sub8, %val9

  ret i32 %sub9
}
