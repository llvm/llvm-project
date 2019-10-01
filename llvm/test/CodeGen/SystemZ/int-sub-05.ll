; Test 128-bit subtraction in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i128 *@foo()

; Test register addition.
define void @f1(i128 *%ptr, i64 %high, i64 %low) {
; CHECK-LABEL: f1:
; CHECK: slgr {{%r[0-5]}}, %r4
; CHECK: slbgr {{%r[0-5]}}, %r3
; CHECK: br %r14
  %a = load i128, i128 *%ptr
  %highx = zext i64 %high to i128
  %lowx = zext i64 %low to i128
  %bhigh = shl i128 %highx, 64
  %b = or i128 %bhigh, %lowx
  %sub = sub i128 %a, %b
  store i128 %sub, i128 *%ptr
  ret void
}

; Test memory addition with no offset.
define void @f2(i64 %addr) {
; CHECK-LABEL: f2:
; CHECK: slg {{%r[0-5]}}, 8(%r2)
; CHECK: slbg {{%r[0-5]}}, 0(%r2)
; CHECK: br %r14
  %bptr = inttoptr i64 %addr to i128 *
  %aptr = getelementptr i128, i128 *%bptr, i64 -8
  %a = load i128, i128 *%aptr
  %b = load i128, i128 *%bptr
  %sub = sub i128 %a, %b
  store i128 %sub, i128 *%aptr
  ret void
}

; Test the highest aligned offset that is in range of both SLG and SLBG.
define void @f3(i64 %base) {
; CHECK-LABEL: f3:
; CHECK: slg {{%r[0-5]}}, 524280(%r2)
; CHECK: slbg {{%r[0-5]}}, 524272(%r2)
; CHECK: br %r14
  %addr = add i64 %base, 524272
  %bptr = inttoptr i64 %addr to i128 *
  %aptr = getelementptr i128, i128 *%bptr, i64 -8
  %a = load i128, i128 *%aptr
  %b = load i128, i128 *%bptr
  %sub = sub i128 %a, %b
  store i128 %sub, i128 *%aptr
  ret void
}

; Test the next doubleword up, which requires separate address logic for SLG.
define void @f4(i64 %base) {
; CHECK-LABEL: f4:
; CHECK: lay [[BASE:%r[1-5]]], 524280(%r2)
; CHECK: slg {{%r[0-5]}}, 8([[BASE]])
; CHECK: slbg {{%r[0-5]}}, 524280(%r2)
; CHECK: br %r14
  %addr = add i64 %base, 524280
  %bptr = inttoptr i64 %addr to i128 *
  %aptr = getelementptr i128, i128 *%bptr, i64 -8
  %a = load i128, i128 *%aptr
  %b = load i128, i128 *%bptr
  %sub = sub i128 %a, %b
  store i128 %sub, i128 *%aptr
  ret void
}

; Test the next doubleword after that, which requires separate logic for
; both instructions.
define void @f5(i64 %base) {
; CHECK-LABEL: f5:
; CHECK: slg {{%r[0-5]}}, 8({{%r[1-5]}})
; CHECK: slbg {{%r[0-5]}}, 0({{%r[1-5]}})
; CHECK: br %r14
  %addr = add i64 %base, 524288
  %bptr = inttoptr i64 %addr to i128 *
  %aptr = getelementptr i128, i128 *%bptr, i64 -8
  %a = load i128, i128 *%aptr
  %b = load i128, i128 *%bptr
  %sub = sub i128 %a, %b
  store i128 %sub, i128 *%aptr
  ret void
}

; Test the lowest displacement that is in range of both SLG and SLBG.
define void @f6(i64 %base) {
; CHECK-LABEL: f6:
; CHECK: slg {{%r[0-5]}}, -524280(%r2)
; CHECK: slbg {{%r[0-5]}}, -524288(%r2)
; CHECK: br %r14
  %addr = add i64 %base, -524288
  %bptr = inttoptr i64 %addr to i128 *
  %aptr = getelementptr i128, i128 *%bptr, i64 -8
  %a = load i128, i128 *%aptr
  %b = load i128, i128 *%bptr
  %sub = sub i128 %a, %b
  store i128 %sub, i128 *%aptr
  ret void
}

; Test the next doubleword down, which is out of range of the SLBG.
define void @f7(i64 %base) {
; CHECK-LABEL: f7:
; CHECK: slg {{%r[0-5]}}, -524288(%r2)
; CHECK: slbg {{%r[0-5]}}, 0({{%r[1-5]}})
; CHECK: br %r14
  %addr = add i64 %base, -524296
  %bptr = inttoptr i64 %addr to i128 *
  %aptr = getelementptr i128, i128 *%bptr, i64 -8
  %a = load i128, i128 *%aptr
  %b = load i128, i128 *%bptr
  %sub = sub i128 %a, %b
  store i128 %sub, i128 *%aptr
  ret void
}

; Check that subtractions of spilled values can use SLG and SLBG rather than
; SLGR and SLBGR.
define void @f8(i128 *%ptr0) {
; CHECK-LABEL: f8:
; CHECK: brasl %r14, foo@PLT
; CHECK: slg {{%r[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: slbg {{%r[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i128, i128 *%ptr0, i128 2
  %ptr2 = getelementptr i128, i128 *%ptr0, i128 4
  %ptr3 = getelementptr i128, i128 *%ptr0, i128 6
  %ptr4 = getelementptr i128, i128 *%ptr0, i128 8
  %ptr5 = getelementptr i128, i128 *%ptr0, i128 10

  %val0 = load i128, i128 *%ptr0
  %val1 = load i128, i128 *%ptr1
  %val2 = load i128, i128 *%ptr2
  %val3 = load i128, i128 *%ptr3
  %val4 = load i128, i128 *%ptr4
  %val5 = load i128, i128 *%ptr5

  %retptr = call i128 *@foo()

  %ret = load i128, i128 *%retptr
  %sub0 = sub i128 %ret, %val0
  %sub1 = sub i128 %sub0, %val1
  %sub2 = sub i128 %sub1, %val2
  %sub3 = sub i128 %sub2, %val3
  %sub4 = sub i128 %sub3, %val4
  %sub5 = sub i128 %sub4, %val5
  store i128 %sub5, i128 *%retptr

  ret void
}
