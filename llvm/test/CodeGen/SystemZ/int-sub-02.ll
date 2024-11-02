; Test subtractions of a sign-extended i32 from an i64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @foo()

; Check SGFR.
define i64 @f1(i64 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: sgfr %r2, %r3
; CHECK: br %r14
  %bext = sext i32 %b to i64
  %sub = sub i64 %a, %bext
  ret i64 %sub
}

; Check SGF with no displacement.
define i64 @f2(i64 %a, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: sgf %r2, 0(%r3)
; CHECK: br %r14
  %b = load i32, ptr %src
  %bext = sext i32 %b to i64
  %sub = sub i64 %a, %bext
  ret i64 %sub
}

; Check the high end of the aligned SGF range.
define i64 @f3(i64 %a, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: sgf %r2, 524284(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131071
  %b = load i32, ptr %ptr
  %bext = sext i32 %b to i64
  %sub = sub i64 %a, %bext
  ret i64 %sub
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f4(i64 %a, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: agfi %r3, 524288
; CHECK: sgf %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131072
  %b = load i32, ptr %ptr
  %bext = sext i32 %b to i64
  %sub = sub i64 %a, %bext
  ret i64 %sub
}

; Check the high end of the negative aligned SGF range.
define i64 @f5(i64 %a, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: sgf %r2, -4(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -1
  %b = load i32, ptr %ptr
  %bext = sext i32 %b to i64
  %sub = sub i64 %a, %bext
  ret i64 %sub
}

; Check the low end of the SGF range.
define i64 @f6(i64 %a, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: sgf %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131072
  %b = load i32, ptr %ptr
  %bext = sext i32 %b to i64
  %sub = sub i64 %a, %bext
  ret i64 %sub
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f7(i64 %a, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: agfi %r3, -524292
; CHECK: sgf %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131073
  %b = load i32, ptr %ptr
  %bext = sext i32 %b to i64
  %sub = sub i64 %a, %bext
  ret i64 %sub
}

; Check that SGF allows an index.
define i64 @f8(i64 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f8:
; CHECK: sgf %r2, 524284({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524284
  %ptr = inttoptr i64 %add2 to ptr
  %b = load i32, ptr %ptr
  %bext = sext i32 %b to i64
  %sub = sub i64 %a, %bext
  ret i64 %sub
}

; Check that subtractions of spilled values can use SGF rather than SGFR.
define i64 @f9(ptr %ptr0) {
; CHECK-LABEL: f9:
; CHECK: brasl %r14, foo@PLT
; CHECK: sgf %r2, 160(%r15)
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

  store i32 %frob0, ptr %ptr0
  store i32 %frob1, ptr %ptr1
  store i32 %frob2, ptr %ptr2
  store i32 %frob3, ptr %ptr3
  store i32 %frob4, ptr %ptr4
  store i32 %frob5, ptr %ptr5
  store i32 %frob6, ptr %ptr6
  store i32 %frob7, ptr %ptr7
  store i32 %frob8, ptr %ptr8
  store i32 %frob9, ptr %ptr9

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

  %sub0 = sub i64 %ret, %ext0
  %sub1 = sub i64 %sub0, %ext1
  %sub2 = sub i64 %sub1, %ext2
  %sub3 = sub i64 %sub2, %ext3
  %sub4 = sub i64 %sub3, %ext4
  %sub5 = sub i64 %sub4, %ext5
  %sub6 = sub i64 %sub5, %ext6
  %sub7 = sub i64 %sub6, %ext7
  %sub8 = sub i64 %sub7, %ext8
  %sub9 = sub i64 %sub8, %ext9

  ret i64 %sub9
}
