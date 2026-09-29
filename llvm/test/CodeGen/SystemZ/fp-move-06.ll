; Test 32-bit floating-point stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test the low end of the STE range.
define void @f1(ptr %ptr, float %val) {
; CHECK-LABEL: f1:
; CHECK: ste %f0, 0(%r2)
; CHECK: br %r14
  store float %val, ptr %ptr
  ret void
}

; Test the high end of the STE range.
define void @f2(ptr %src, float %val) {
; CHECK-LABEL: f2:
; CHECK: ste %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %src, i64 1023
  store float %val, ptr %ptr
  ret void
}

; Check the next word up, which should use STEY instead of STE.
define void @f3(ptr %src, float %val) {
; CHECK-LABEL: f3:
; CHECK: stey %f0, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %src, i64 1024
  store float %val, ptr %ptr
  ret void
}

; Check the high end of the aligned STEY range.
define void @f4(ptr %src, float %val) {
; CHECK-LABEL: f4:
; CHECK: stey %f0, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %src, i64 131071
  store float %val, ptr %ptr
  ret void
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f5(ptr %src, float %val) {
; CHECK-LABEL: f5:
; CHECK: agfi %r2, 524288
; CHECK: ste %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %src, i64 131072
  store float %val, ptr %ptr
  ret void
}

; Check the high end of the negative aligned STEY range.
define void @f6(ptr %src, float %val) {
; CHECK-LABEL: f6:
; CHECK: stey %f0, -4(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %src, i64 -1
  store float %val, ptr %ptr
  ret void
}

; Check the low end of the STEY range.
define void @f7(ptr %src, float %val) {
; CHECK-LABEL: f7:
; CHECK: stey %f0, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %src, i64 -131072
  store float %val, ptr %ptr
  ret void
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8(ptr %src, float %val) {
; CHECK-LABEL: f8:
; CHECK: agfi %r2, -524292
; CHECK: ste %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %src, i64 -131073
  store float %val, ptr %ptr
  ret void
}

; Check that STE allows an index.
define void @f9(i64 %src, i64 %index, float %val) {
; CHECK-LABEL: f9:
; CHECK: ste %f0, 4092({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to ptr
  store float %val, ptr %ptr
  ret void
}

; Check that STEY allows an index.
define void @f10(i64 %src, i64 %index, float %val) {
; CHECK-LABEL: f10:
; CHECK: stey %f0, 4096({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to ptr
  store float %val, ptr %ptr
  ret void
}
