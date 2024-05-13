; Test extensions of f64 to f128.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check register extension.
define void @f1(ptr %dst, double %val) {
; CHECK-LABEL: f1:
; CHECK: lxdbr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %res = fpext double %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Check the low end of the LXDB range.
define void @f2(ptr %dst, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: lxdb %f0, 0(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %val = load double, ptr %ptr
  %res = fpext double %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Check the high end of the aligned LXDB range.
define void @f3(ptr %dst, ptr %base) {
; CHECK-LABEL: f3:
; CHECK: lxdb %f0, 4088(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 511
  %val = load double, ptr %ptr
  %res = fpext double %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f4(ptr %dst, ptr %base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r3, 4096
; CHECK: lxdb %f0, 0(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 512
  %val = load double, ptr %ptr
  %res = fpext double %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Check negative displacements, which also need separate address logic.
define void @f5(ptr %dst, ptr %base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r3, -8
; CHECK: lxdb %f0, 0(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 -1
  %val = load double, ptr %ptr
  %res = fpext double %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Check that LXDB allows indices.
define void @f6(ptr %dst, ptr %base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r4, 3
; CHECK: lxdb %f0, 800(%r1,%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr1 = getelementptr double, ptr %base, i64 %index
  %ptr2 = getelementptr double, ptr %ptr1, i64 100
  %val = load double, ptr %ptr2
  %res = fpext double %val to fp128
  store fp128 %res, ptr %dst
  ret void
}

; Test a case where we spill the source of at least one LXDBR.  We want
; to use LXDB if possible.
define void @f7(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f7:
; CHECK: lxdb {{%f[0-9]+}}, 160(%r15)
; CHECK: br %r14
  %val0 = load volatile double, ptr %ptr2
  %val1 = load volatile double, ptr %ptr2
  %val2 = load volatile double, ptr %ptr2
  %val3 = load volatile double, ptr %ptr2
  %val4 = load volatile double, ptr %ptr2
  %val5 = load volatile double, ptr %ptr2
  %val6 = load volatile double, ptr %ptr2
  %val7 = load volatile double, ptr %ptr2
  %val8 = load volatile double, ptr %ptr2
  %val9 = load volatile double, ptr %ptr2
  %val10 = load volatile double, ptr %ptr2
  %val11 = load volatile double, ptr %ptr2
  %val12 = load volatile double, ptr %ptr2
  %val13 = load volatile double, ptr %ptr2
  %val14 = load volatile double, ptr %ptr2
  %val15 = load volatile double, ptr %ptr2
  %val16 = load volatile double, ptr %ptr2

  %ext0 = fpext double %val0 to fp128
  %ext1 = fpext double %val1 to fp128
  %ext2 = fpext double %val2 to fp128
  %ext3 = fpext double %val3 to fp128
  %ext4 = fpext double %val4 to fp128
  %ext5 = fpext double %val5 to fp128
  %ext6 = fpext double %val6 to fp128
  %ext7 = fpext double %val7 to fp128
  %ext8 = fpext double %val8 to fp128
  %ext9 = fpext double %val9 to fp128
  %ext10 = fpext double %val10 to fp128
  %ext11 = fpext double %val11 to fp128
  %ext12 = fpext double %val12 to fp128
  %ext13 = fpext double %val13 to fp128
  %ext14 = fpext double %val14 to fp128
  %ext15 = fpext double %val15 to fp128
  %ext16 = fpext double %val16 to fp128

  store volatile double %val0, ptr %ptr2
  store volatile double %val1, ptr %ptr2
  store volatile double %val2, ptr %ptr2
  store volatile double %val3, ptr %ptr2
  store volatile double %val4, ptr %ptr2
  store volatile double %val5, ptr %ptr2
  store volatile double %val6, ptr %ptr2
  store volatile double %val7, ptr %ptr2
  store volatile double %val8, ptr %ptr2
  store volatile double %val9, ptr %ptr2
  store volatile double %val10, ptr %ptr2
  store volatile double %val11, ptr %ptr2
  store volatile double %val12, ptr %ptr2
  store volatile double %val13, ptr %ptr2
  store volatile double %val14, ptr %ptr2
  store volatile double %val15, ptr %ptr2
  store volatile double %val16, ptr %ptr2

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
