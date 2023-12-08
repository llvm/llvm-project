; Test extensions of f32 to f128.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check register extension.
define void @f1(ptr %dst, float %val) {
; CHECK-LABEL: f1:
; CHECK: lxebr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %res = fpext float %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Check the low end of the LXEB range.
define void @f2(ptr %dst, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: lxeb %f0, 0(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %val = load float, ptr %ptr
  %res = fpext float %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Check the high end of the aligned LXEB range.
define void @f3(ptr %dst, ptr %base) {
; CHECK-LABEL: f3:
; CHECK: lxeb %f0, 4092(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 1023
  %val = load float, ptr %ptr
  %res = fpext float %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f4(ptr %dst, ptr %base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r3, 4096
; CHECK: lxeb %f0, 0(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 1024
  %val = load float, ptr %ptr
  %res = fpext float %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Check negative displacements, which also need separate address logic.
define void @f5(ptr %dst, ptr %base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r3, -4
; CHECK: lxeb %f0, 0(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 -1
  %val = load float, ptr %ptr
  %res = fpext float %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Check that LXEB allows indices.
define void @f6(ptr %dst, ptr %base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r4, 2
; CHECK: lxeb %f0, 400(%r1,%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float, ptr %base, i64 %index
  %ptr2 = getelementptr float, ptr %ptr1, i64 100
  %val = load float, ptr %ptr2
  %res = fpext float %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Test a case where we spill the source of at least one LXEBR.  We want
; to use LXEB if possible.
define void @f7(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f7:
; CHECK: lxeb {{%f[0-9]+}}, 16{{[04]}}(%r15)
; CHECK: br %r14
  %val0 = load volatile float, ptr %ptr2
  %val1 = load volatile float, ptr %ptr2
  %val2 = load volatile float, ptr %ptr2
  %val3 = load volatile float, ptr %ptr2
  %val4 = load volatile float, ptr %ptr2
  %val5 = load volatile float, ptr %ptr2
  %val6 = load volatile float, ptr %ptr2
  %val7 = load volatile float, ptr %ptr2
  %val8 = load volatile float, ptr %ptr2
  %val9 = load volatile float, ptr %ptr2
  %val10 = load volatile float, ptr %ptr2
  %val11 = load volatile float, ptr %ptr2
  %val12 = load volatile float, ptr %ptr2
  %val13 = load volatile float, ptr %ptr2
  %val14 = load volatile float, ptr %ptr2
  %val15 = load volatile float, ptr %ptr2
  %val16 = load volatile float, ptr %ptr2

  %ext0 = fpext float %val0 to fp128
  %ext1 = fpext float %val1 to fp128
  %ext2 = fpext float %val2 to fp128
  %ext3 = fpext float %val3 to fp128
  %ext4 = fpext float %val4 to fp128
  %ext5 = fpext float %val5 to fp128
  %ext6 = fpext float %val6 to fp128
  %ext7 = fpext float %val7 to fp128
  %ext8 = fpext float %val8 to fp128
  %ext9 = fpext float %val9 to fp128
  %ext10 = fpext float %val10 to fp128
  %ext11 = fpext float %val11 to fp128
  %ext12 = fpext float %val12 to fp128
  %ext13 = fpext float %val13 to fp128
  %ext14 = fpext float %val14 to fp128
  %ext15 = fpext float %val15 to fp128
  %ext16 = fpext float %val16 to fp128

  store volatile float %val0, ptr %ptr2
  store volatile float %val1, ptr %ptr2
  store volatile float %val2, ptr %ptr2
  store volatile float %val3, ptr %ptr2
  store volatile float %val4, ptr %ptr2
  store volatile float %val5, ptr %ptr2
  store volatile float %val6, ptr %ptr2
  store volatile float %val7, ptr %ptr2
  store volatile float %val8, ptr %ptr2
  store volatile float %val9, ptr %ptr2
  store volatile float %val10, ptr %ptr2
  store volatile float %val11, ptr %ptr2
  store volatile float %val12, ptr %ptr2
  store volatile float %val13, ptr %ptr2
  store volatile float %val14, ptr %ptr2
  store volatile float %val15, ptr %ptr2
  store volatile float %val16, ptr %ptr2

  store volatile fp128 %ext0, ptr %ptr1
  store volatile fp128 %ext1, ptr %ptr1
  store volatile fp128 %ext2, ptr %ptr1
  store volatile fp128 %ext3, ptr %ptr1
  store volatile fp128 %ext4, ptr %ptr1
  store volatile fp128 %ext5, ptr %ptr1
  store volatile fp128 %ext6, ptr %ptr1
  store volatile fp128 %ext7, ptr %ptr1
  store volatile fp128 %ext8, ptr %ptr1
  store volatile fp128 %ext9, ptr %ptr1
  store volatile fp128 %ext10, ptr %ptr1
  store volatile fp128 %ext11, ptr %ptr1
  store volatile fp128 %ext12, ptr %ptr1
  store volatile fp128 %ext13, ptr %ptr1
  store volatile fp128 %ext14, ptr %ptr1
  store volatile fp128 %ext15, ptr %ptr1
  store volatile fp128 %ext16, ptr %ptr1

  ret void
}
