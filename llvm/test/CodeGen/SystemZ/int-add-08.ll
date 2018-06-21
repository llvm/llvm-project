; Test 128-bit addition in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i128 *@foo()

; Test register addition.
define void @f1(i128 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: algr
; CHECK: alcgr
; CHECK: br %r14
  %value = load i128 , i128 *%ptr
  %add = add i128 %value, %value
  store i128 %add, i128 *%ptr
  ret void
}

; Test memory addition with no offset.  Making the load of %a volatile
; should force the memory operand to be %b.
define void @f2(i128 *%aptr, i64 %addr) {
; CHECK-LABEL: f2:
; CHECK: alg {{%r[0-5]}}, 8(%r3)
; CHECK: alcg {{%r[0-5]}}, 0(%r3)
; CHECK: br %r14
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 , i128 *%aptr
  %b = load i128 , i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

; Test the highest aligned offset that is in range of both ALG and ALCG.
define void @f3(i128 *%aptr, i64 %base) {
; CHECK-LABEL: f3:
; CHECK: alg {{%r[0-5]}}, 524280(%r3)
; CHECK: alcg {{%r[0-5]}}, 524272(%r3)
; CHECK: br %r14
  %addr = add i64 %base, 524272
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 , i128 *%aptr
  %b = load i128 , i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

; Test the next doubleword up, which requires separate address logic for ALG.
define void @f4(i128 *%aptr, i64 %base) {
; CHECK-LABEL: f4:
; CHECK: lgr [[BASE:%r[1-5]]], %r3
; CHECK: agfi [[BASE]], 524288
; CHECK: alg {{%r[0-5]}}, 0([[BASE]])
; CHECK: alcg {{%r[0-5]}}, 524280(%r3)
; CHECK: br %r14
  %addr = add i64 %base, 524280
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 , i128 *%aptr
  %b = load i128 , i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

; Test the next doubleword after that, which requires separate logic for
; both instructions.  It would be better to create an anchor at 524288
; that both instructions can use, but that isn't implemented yet.
define void @f5(i128 *%aptr, i64 %base) {
; CHECK-LABEL: f5:
; CHECK: alg {{%r[0-5]}}, 0({{%r[1-5]}})
; CHECK: alcg {{%r[0-5]}}, 0({{%r[1-5]}})
; CHECK: br %r14
  %addr = add i64 %base, 524288
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 , i128 *%aptr
  %b = load i128 , i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

; Test the lowest displacement that is in range of both ALG and ALCG.
define void @f6(i128 *%aptr, i64 %base) {
; CHECK-LABEL: f6:
; CHECK: alg {{%r[0-5]}}, -524280(%r3)
; CHECK: alcg {{%r[0-5]}}, -524288(%r3)
; CHECK: br %r14
  %addr = add i64 %base, -524288
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 , i128 *%aptr
  %b = load i128 , i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

; Test the next doubleword down, which is out of range of the ALCG.
define void @f7(i128 *%aptr, i64 %base) {
; CHECK-LABEL: f7:
; CHECK: alg {{%r[0-5]}}, -524288(%r3)
; CHECK: alcg {{%r[0-5]}}, 0({{%r[1-5]}})
; CHECK: br %r14
  %addr = add i64 %base, -524296
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 , i128 *%aptr
  %b = load i128 , i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

; Check that additions of spilled values can use ALG and ALCG rather than
; ALGR and ALCGR.
define void @f8(i128 *%ptr0) {
; CHECK-LABEL: f8:
; CHECK: brasl %r14, foo@PLT
; CHECK: alg {{%r[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: alcg {{%r[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i128, i128 *%ptr0, i128 2
  %ptr2 = getelementptr i128, i128 *%ptr0, i128 4
  %ptr3 = getelementptr i128, i128 *%ptr0, i128 6
  %ptr4 = getelementptr i128, i128 *%ptr0, i128 8

  %val0 = load i128 , i128 *%ptr0
  %val1 = load i128 , i128 *%ptr1
  %val2 = load i128 , i128 *%ptr2
  %val3 = load i128 , i128 *%ptr3
  %val4 = load i128 , i128 *%ptr4

  %retptr = call i128 *@foo()

  %ret = load i128 , i128 *%retptr
  %add0 = add i128 %ret, %val0
  %add1 = add i128 %add0, %val1
  %add2 = add i128 %add1, %val2
  %add3 = add i128 %add2, %val3
  %add4 = add i128 %add3, %val4
  store i128 %add4, i128 *%retptr

  ret void
}
