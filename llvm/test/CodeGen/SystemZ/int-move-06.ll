; Test 32-bit GPR stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test an i32 store.
define void @f1(ptr %dst, i32 %val) {
; CHECK-LABEL: f1:
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  store i32 %val, ptr %dst
  ret void
}

; Test a truncating i64 store.
define void @f2(ptr %dst, i64 %val) {
  %word = trunc i64 %val to i32
  store i32 %word, ptr %dst
  ret void
}

; Check the high end of the aligned ST range.
define void @f3(ptr %dst, i32 %val) {
; CHECK-LABEL: f3:
; CHECK: st %r3, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %dst, i64 1023
  store i32 %val, ptr %ptr
  ret void
}

; Check the next word up, which should use STY instead of ST.
define void @f4(ptr %dst, i32 %val) {
; CHECK-LABEL: f4:
; CHECK: sty %r3, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %dst, i64 1024
  store i32 %val, ptr %ptr
  ret void
}

; Check the high end of the aligned STY range.
define void @f5(ptr %dst, i32 %val) {
; CHECK-LABEL: f5:
; CHECK: sty %r3, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %dst, i64 131071
  store i32 %val, ptr %ptr
  ret void
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f6(ptr %dst, i32 %val) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, 524288
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %dst, i64 131072
  store i32 %val, ptr %ptr
  ret void
}

; Check the high end of the negative aligned STY range.
define void @f7(ptr %dst, i32 %val) {
; CHECK-LABEL: f7:
; CHECK: sty %r3, -4(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %dst, i64 -1
  store i32 %val, ptr %ptr
  ret void
}

; Check the low end of the STY range.
define void @f8(ptr %dst, i32 %val) {
; CHECK-LABEL: f8:
; CHECK: sty %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %dst, i64 -131072
  store i32 %val, ptr %ptr
  ret void
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f9(ptr %dst, i32 %val) {
; CHECK-LABEL: f9:
; CHECK: agfi %r2, -524292
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %dst, i64 -131073
  store i32 %val, ptr %ptr
  ret void
}

; Check that ST allows an index.
define void @f10(i64 %dst, i64 %index, i32 %val) {
; CHECK-LABEL: f10:
; CHECK: st %r4, 4095(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %dst, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to ptr
  store i32 %val, ptr %ptr
  ret void
}

; Check that STY allows an index.
define void @f11(i64 %dst, i64 %index, i32 %val) {
; CHECK-LABEL: f11:
; CHECK: sty %r4, 4096(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %dst, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to ptr
  store i32 %val, ptr %ptr
  ret void
}
