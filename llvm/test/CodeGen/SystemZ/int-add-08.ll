; Test 128-bit addition in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare ptr@foo()

; Test register addition.
define void @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: algr
; CHECK: alcgr
; CHECK: br %r14
  %value = load i128, ptr %ptr
  %add = add i128 %value, %value
  store i128 %add, ptr %ptr
  ret void
}

; Test memory addition with no offset.  Making the load of %a volatile
; should force the memory operand to be %b.
define void @f2(ptr %aptr, i64 %addr) {
; CHECK-LABEL: f2:
; CHECK: alg {{%r[0-5]}}, 8(%r3)
; CHECK: alcg {{%r[0-5]}}, 0(%r3)
; CHECK: br %r14
  %bptr = inttoptr i64 %addr to ptr
  %a = load volatile i128, ptr %aptr
  %b = load i128, ptr %bptr
  %add = add i128 %a, %b
  store i128 %add, ptr %aptr
  ret void
}

; Test the highest aligned offset that is in range of both ALG and ALCG.
define void @f3(ptr %aptr, i64 %base) {
; CHECK-LABEL: f3:
; CHECK: alg {{%r[0-5]}}, 524280(%r3)
; CHECK: alcg {{%r[0-5]}}, 524272(%r3)
; CHECK: br %r14
  %addr = add i64 %base, 524272
  %bptr = inttoptr i64 %addr to ptr
  %a = load volatile i128, ptr %aptr
  %b = load i128, ptr %bptr
  %add = add i128 %a, %b
  store i128 %add, ptr %aptr
  ret void
}

; Test the next doubleword up, which requires separate address logic for ALG.
define void @f4(ptr %aptr, i64 %base) {
; CHECK-LABEL: f4:
; CHECK: lay [[BASE:%r[1-5]]], 524280(%r3)
; CHECK: alg {{%r[0-5]}}, 8([[BASE]])
; CHECK: alcg {{%r[0-5]}}, 524280(%r3)
; CHECK: br %r14
  %addr = add i64 %base, 524280
  %bptr = inttoptr i64 %addr to ptr
  %a = load volatile i128, ptr %aptr
  %b = load i128, ptr %bptr
  %add = add i128 %a, %b
  store i128 %add, ptr %aptr
  ret void
}

; Test the next doubleword after that, which requires separate logic for
; both instructions.
define void @f5(ptr %aptr, i64 %base) {
; CHECK-LABEL: f5:
; CHECK: alg {{%r[0-5]}}, 8({{%r[1-5]}})
; CHECK: alcg {{%r[0-5]}}, 0({{%r[1-5]}})
; CHECK: br %r14
  %addr = add i64 %base, 524288
  %bptr = inttoptr i64 %addr to ptr
  %a = load volatile i128, ptr %aptr
  %b = load i128, ptr %bptr
  %add = add i128 %a, %b
  store i128 %add, ptr %aptr
  ret void
}

; Test the lowest displacement that is in range of both ALG and ALCG.
define void @f6(ptr %aptr, i64 %base) {
; CHECK-LABEL: f6:
; CHECK: alg {{%r[0-5]}}, -524280(%r3)
; CHECK: alcg {{%r[0-5]}}, -524288(%r3)
; CHECK: br %r14
  %addr = add i64 %base, -524288
  %bptr = inttoptr i64 %addr to ptr
  %a = load volatile i128, ptr %aptr
  %b = load i128, ptr %bptr
  %add = add i128 %a, %b
  store i128 %add, ptr %aptr
  ret void
}

; Test the next doubleword down, which is out of range of the ALCG.
define void @f7(ptr %aptr, i64 %base) {
; CHECK-LABEL: f7:
; CHECK: alg {{%r[0-5]}}, -524288(%r3)
; CHECK: alcg {{%r[0-5]}}, 0({{%r[1-5]}})
; CHECK: br %r14
  %addr = add i64 %base, -524296
  %bptr = inttoptr i64 %addr to ptr
  %a = load volatile i128, ptr %aptr
  %b = load i128, ptr %bptr
  %add = add i128 %a, %b
  store i128 %add, ptr %aptr
  ret void
}

; Check that additions of spilled values can use ALG and ALCG rather than
; ALGR and ALCGR.
define void @f8(ptr %ptr0) {
; CHECK-LABEL: f8:
; CHECK: brasl %r14, foo@PLT
; CHECK: alcg {{%r[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: alcg {{%r[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i128, ptr %ptr0, i128 2
  %ptr2 = getelementptr i128, ptr %ptr0, i128 4
  %ptr3 = getelementptr i128, ptr %ptr0, i128 6
  %ptr4 = getelementptr i128, ptr %ptr0, i128 8
  %ptr5 = getelementptr i128, ptr %ptr0, i128 10

  %val0 = load i128, ptr %ptr0
  %val1 = load i128, ptr %ptr1
  %val2 = load i128, ptr %ptr2
  %val3 = load i128, ptr %ptr3
  %val4 = load i128, ptr %ptr4
  %val5 = load i128, ptr %ptr5

  %retptr = call ptr@foo()

  %ret = load i128, ptr %retptr
  %add0 = add i128 %ret, %val0
  %add1 = add i128 %add0, %val1
  %add2 = add i128 %add1, %val2
  %add3 = add i128 %add2, %val3
  %add4 = add i128 %add3, %val4
  %add5 = add i128 %add4, %val5
  store i128 %add5, ptr %retptr

  ret void
}
