; Test 64-bit additions of constants to memory.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check additions of 1.
define void @f1(i64 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: agsi 0(%r2), 1
; CHECK: br %r14
  %val = load i64, i64 *%ptr
  %add = add i64 %val, 127
  store i64 %add, i64 *%ptr
  ret void
}

; Check the high end of the constant range.
define void @f2(i64 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: agsi 0(%r2), 127
; CHECK: br %r14
  %val = load i64, i64 *%ptr
  %add = add i64 %val, 127
  store i64 %add, i64 *%ptr
  ret void
}

; Check the next constant up, which must use an addition and a store.
; Both LG/AGHI and LGHI/AG would be OK.
define void @f3(i64 *%ptr) {
; CHECK-LABEL: f3:
; CHECK-NOT: agsi
; CHECK: stg %r0, 0(%r2)
; CHECK: br %r14
  %val = load i64, i64 *%ptr
  %add = add i64 %val, 128
  store i64 %add, i64 *%ptr
  ret void
}

; Check the low end of the constant range.
define void @f4(i64 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: agsi 0(%r2), -128
; CHECK: br %r14
  %val = load i64, i64 *%ptr
  %add = add i64 %val, -128
  store i64 %add, i64 *%ptr
  ret void
}

; Check the next value down, with the same comment as f3.
define void @f5(i64 *%ptr) {
; CHECK-LABEL: f5:
; CHECK-NOT: agsi
; CHECK: stg %r0, 0(%r2)
; CHECK: br %r14
  %val = load i64, i64 *%ptr
  %add = add i64 %val, -129
  store i64 %add, i64 *%ptr
  ret void
}

; Check the high end of the aligned AGSI range.
define void @f6(i64 *%base) {
; CHECK-LABEL: f6:
; CHECK: agsi 524280(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 65535
  %val = load i64, i64 *%ptr
  %add = add i64 %val, 1
  store i64 %add, i64 *%ptr
  ret void
}

; Check the next doubleword up, which must use separate address logic.
; Other sequences besides this one would be OK.
define void @f7(i64 *%base) {
; CHECK-LABEL: f7:
; CHECK: agfi %r2, 524288
; CHECK: agsi 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 65536
  %val = load i64, i64 *%ptr
  %add = add i64 %val, 1
  store i64 %add, i64 *%ptr
  ret void
}

; Check the low end of the AGSI range.
define void @f8(i64 *%base) {
; CHECK-LABEL: f8:
; CHECK: agsi -524288(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 -65536
  %val = load i64, i64 *%ptr
  %add = add i64 %val, 1
  store i64 %add, i64 *%ptr
  ret void
}

; Check the next doubleword down, which must use separate address logic.
; Other sequences besides this one would be OK.
define void @f9(i64 *%base) {
; CHECK-LABEL: f9:
; CHECK: agfi %r2, -524296
; CHECK: agsi 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 -65537
  %val = load i64, i64 *%ptr
  %add = add i64 %val, 1
  store i64 %add, i64 *%ptr
  ret void
}

; Check that AGSI does not allow indices.
define void @f10(i64 %base, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: agr %r2, %r3
; CHECK: agsi 8(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 8
  %ptr = inttoptr i64 %add2 to i64 *
  %val = load i64, i64 *%ptr
  %add = add i64 %val, 1
  store i64 %add, i64 *%ptr
  ret void
}

; Check that adding 127 to a spilled value can use AGSI.
define void @f11(i64 *%ptr, i32 %sel) {
; CHECK-LABEL: f11:
; CHECK: agsi {{[0-9]+}}(%r15), 127
; CHECK: br %r14
entry:
  %val0 = load volatile i64, i64 *%ptr
  %val1 = load volatile i64, i64 *%ptr
  %val2 = load volatile i64, i64 *%ptr
  %val3 = load volatile i64, i64 *%ptr
  %val4 = load volatile i64, i64 *%ptr
  %val5 = load volatile i64, i64 *%ptr
  %val6 = load volatile i64, i64 *%ptr
  %val7 = load volatile i64, i64 *%ptr
  %val8 = load volatile i64, i64 *%ptr
  %val9 = load volatile i64, i64 *%ptr
  %val10 = load volatile i64, i64 *%ptr
  %val11 = load volatile i64, i64 *%ptr
  %val12 = load volatile i64, i64 *%ptr
  %val13 = load volatile i64, i64 *%ptr
  %val14 = load volatile i64, i64 *%ptr
  %val15 = load volatile i64, i64 *%ptr

  %test = icmp ne i32 %sel, 0
  br i1 %test, label %add, label %store

add:
  %add0 = add i64 %val0, 127
  %add1 = add i64 %val1, 127
  %add2 = add i64 %val2, 127
  %add3 = add i64 %val3, 127
  %add4 = add i64 %val4, 127
  %add5 = add i64 %val5, 127
  %add6 = add i64 %val6, 127
  %add7 = add i64 %val7, 127
  %add8 = add i64 %val8, 127
  %add9 = add i64 %val9, 127
  %add10 = add i64 %val10, 127
  %add11 = add i64 %val11, 127
  %add12 = add i64 %val12, 127
  %add13 = add i64 %val13, 127
  %add14 = add i64 %val14, 127
  %add15 = add i64 %val15, 127
  br label %store

store:
  %new0 = phi i64 [ %val0, %entry ], [ %add0, %add ]
  %new1 = phi i64 [ %val1, %entry ], [ %add1, %add ]
  %new2 = phi i64 [ %val2, %entry ], [ %add2, %add ]
  %new3 = phi i64 [ %val3, %entry ], [ %add3, %add ]
  %new4 = phi i64 [ %val4, %entry ], [ %add4, %add ]
  %new5 = phi i64 [ %val5, %entry ], [ %add5, %add ]
  %new6 = phi i64 [ %val6, %entry ], [ %add6, %add ]
  %new7 = phi i64 [ %val7, %entry ], [ %add7, %add ]
  %new8 = phi i64 [ %val8, %entry ], [ %add8, %add ]
  %new9 = phi i64 [ %val9, %entry ], [ %add9, %add ]
  %new10 = phi i64 [ %val10, %entry ], [ %add10, %add ]
  %new11 = phi i64 [ %val11, %entry ], [ %add11, %add ]
  %new12 = phi i64 [ %val12, %entry ], [ %add12, %add ]
  %new13 = phi i64 [ %val13, %entry ], [ %add13, %add ]
  %new14 = phi i64 [ %val14, %entry ], [ %add14, %add ]
  %new15 = phi i64 [ %val15, %entry ], [ %add15, %add ]

  store volatile i64 %new0, i64 *%ptr
  store volatile i64 %new1, i64 *%ptr
  store volatile i64 %new2, i64 *%ptr
  store volatile i64 %new3, i64 *%ptr
  store volatile i64 %new4, i64 *%ptr
  store volatile i64 %new5, i64 *%ptr
  store volatile i64 %new6, i64 *%ptr
  store volatile i64 %new7, i64 *%ptr
  store volatile i64 %new8, i64 *%ptr
  store volatile i64 %new9, i64 *%ptr
  store volatile i64 %new10, i64 *%ptr
  store volatile i64 %new11, i64 *%ptr
  store volatile i64 %new12, i64 *%ptr
  store volatile i64 %new13, i64 *%ptr
  store volatile i64 %new14, i64 *%ptr
  store volatile i64 %new15, i64 *%ptr

  ret void
}

; Check that adding -128 to a spilled value can use AGSI.
define void @f12(i64 *%ptr, i32 %sel) {
; CHECK-LABEL: f12:
; CHECK: agsi {{[0-9]+}}(%r15), -128
; CHECK: br %r14
entry:
  %val0 = load volatile i64, i64 *%ptr
  %val1 = load volatile i64, i64 *%ptr
  %val2 = load volatile i64, i64 *%ptr
  %val3 = load volatile i64, i64 *%ptr
  %val4 = load volatile i64, i64 *%ptr
  %val5 = load volatile i64, i64 *%ptr
  %val6 = load volatile i64, i64 *%ptr
  %val7 = load volatile i64, i64 *%ptr
  %val8 = load volatile i64, i64 *%ptr
  %val9 = load volatile i64, i64 *%ptr
  %val10 = load volatile i64, i64 *%ptr
  %val11 = load volatile i64, i64 *%ptr
  %val12 = load volatile i64, i64 *%ptr
  %val13 = load volatile i64, i64 *%ptr
  %val14 = load volatile i64, i64 *%ptr
  %val15 = load volatile i64, i64 *%ptr

  %test = icmp ne i32 %sel, 0
  br i1 %test, label %add, label %store

add:
  %add0 = add i64 %val0, -128
  %add1 = add i64 %val1, -128
  %add2 = add i64 %val2, -128
  %add3 = add i64 %val3, -128
  %add4 = add i64 %val4, -128
  %add5 = add i64 %val5, -128
  %add6 = add i64 %val6, -128
  %add7 = add i64 %val7, -128
  %add8 = add i64 %val8, -128
  %add9 = add i64 %val9, -128
  %add10 = add i64 %val10, -128
  %add11 = add i64 %val11, -128
  %add12 = add i64 %val12, -128
  %add13 = add i64 %val13, -128
  %add14 = add i64 %val14, -128
  %add15 = add i64 %val15, -128
  br label %store

store:
  %new0 = phi i64 [ %val0, %entry ], [ %add0, %add ]
  %new1 = phi i64 [ %val1, %entry ], [ %add1, %add ]
  %new2 = phi i64 [ %val2, %entry ], [ %add2, %add ]
  %new3 = phi i64 [ %val3, %entry ], [ %add3, %add ]
  %new4 = phi i64 [ %val4, %entry ], [ %add4, %add ]
  %new5 = phi i64 [ %val5, %entry ], [ %add5, %add ]
  %new6 = phi i64 [ %val6, %entry ], [ %add6, %add ]
  %new7 = phi i64 [ %val7, %entry ], [ %add7, %add ]
  %new8 = phi i64 [ %val8, %entry ], [ %add8, %add ]
  %new9 = phi i64 [ %val9, %entry ], [ %add9, %add ]
  %new10 = phi i64 [ %val10, %entry ], [ %add10, %add ]
  %new11 = phi i64 [ %val11, %entry ], [ %add11, %add ]
  %new12 = phi i64 [ %val12, %entry ], [ %add12, %add ]
  %new13 = phi i64 [ %val13, %entry ], [ %add13, %add ]
  %new14 = phi i64 [ %val14, %entry ], [ %add14, %add ]
  %new15 = phi i64 [ %val15, %entry ], [ %add15, %add ]

  store volatile i64 %new0, i64 *%ptr
  store volatile i64 %new1, i64 *%ptr
  store volatile i64 %new2, i64 *%ptr
  store volatile i64 %new3, i64 *%ptr
  store volatile i64 %new4, i64 *%ptr
  store volatile i64 %new5, i64 *%ptr
  store volatile i64 %new6, i64 *%ptr
  store volatile i64 %new7, i64 *%ptr
  store volatile i64 %new8, i64 *%ptr
  store volatile i64 %new9, i64 *%ptr
  store volatile i64 %new10, i64 *%ptr
  store volatile i64 %new11, i64 *%ptr
  store volatile i64 %new12, i64 *%ptr
  store volatile i64 %new13, i64 *%ptr
  store volatile i64 %new14, i64 *%ptr
  store volatile i64 %new15, i64 *%ptr

  ret void
}
