; Test f64 conditional stores that are presented as selects.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @foo(double *)

; Test with the loaded value first.
define void @f1(double *%ptr, double %alt, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: std %f0, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load double , double *%ptr
  %res = select i1 %cond, double %orig, double %alt
  store double %res, double *%ptr
  ret void
}

; ...and with the loaded value second
define void @f2(double *%ptr, double %alt, i32 %limit) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: std %f0, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load double , double *%ptr
  %res = select i1 %cond, double %alt, double %orig
  store double %res, double *%ptr
  ret void
}

; Check the high end of the aligned STD range.
define void @f3(double *%base, double %alt, i32 %limit) {
; CHECK-LABEL: f3:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: std %f0, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 511
  %cond = icmp ult i32 %limit, 420
  %orig = load double , double *%ptr
  %res = select i1 %cond, double %orig, double %alt
  store double %res, double *%ptr
  ret void
}

; Check the next doubleword up, which should use STDY instead of STD.
define void @f4(double *%base, double %alt, i32 %limit) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stdy %f0, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 512
  %cond = icmp ult i32 %limit, 420
  %orig = load double , double *%ptr
  %res = select i1 %cond, double %orig, double %alt
  store double %res, double *%ptr
  ret void
}

; Check the high end of the aligned STDY range.
define void @f5(double *%base, double %alt, i32 %limit) {
; CHECK-LABEL: f5:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stdy %f0, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 65535
  %cond = icmp ult i32 %limit, 420
  %orig = load double , double *%ptr
  %res = select i1 %cond, double %orig, double %alt
  store double %res, double *%ptr
  ret void
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f6(double *%base, double %alt, i32 %limit) {
; CHECK-LABEL: f6:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: agfi %r2, 524288
; CHECK: std %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 65536
  %cond = icmp ult i32 %limit, 420
  %orig = load double , double *%ptr
  %res = select i1 %cond, double %orig, double %alt
  store double %res, double *%ptr
  ret void
}

; Check the low end of the STDY range.
define void @f7(double *%base, double %alt, i32 %limit) {
; CHECK-LABEL: f7:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stdy %f0, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 -65536
  %cond = icmp ult i32 %limit, 420
  %orig = load double , double *%ptr
  %res = select i1 %cond, double %orig, double %alt
  store double %res, double *%ptr
  ret void
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8(double *%base, double %alt, i32 %limit) {
; CHECK-LABEL: f8:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: agfi %r2, -524296
; CHECK: std %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 -65537
  %cond = icmp ult i32 %limit, 420
  %orig = load double , double *%ptr
  %res = select i1 %cond, double %orig, double %alt
  store double %res, double *%ptr
  ret void
}

; Check that STDY allows an index.
define void @f9(i64 %base, i64 %index, double %alt, i32 %limit) {
; CHECK-LABEL: f9:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stdy %f0, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to double *
  %cond = icmp ult i32 %limit, 420
  %orig = load double , double *%ptr
  %res = select i1 %cond, double %orig, double %alt
  store double %res, double *%ptr
  ret void
}

; Check that volatile loads are not matched.
define void @f10(double *%ptr, double %alt, i32 %limit) {
; CHECK-LABEL: f10:
; CHECK: ld {{%f[0-5]}}, 0(%r2)
; CHECK: {{jl|jnl}} [[LABEL:[^ ]*]]
; CHECK: [[LABEL]]:
; CHECK: std {{%f[0-5]}}, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load volatile double , double *%ptr
  %res = select i1 %cond, double %orig, double %alt
  store double %res, double *%ptr
  ret void
}

; ...likewise stores.  In this case we should have a conditional load into %f0.
define void @f11(double *%ptr, double %alt, i32 %limit) {
; CHECK-LABEL: f11:
; CHECK: jhe [[LABEL:[^ ]*]]
; CHECK: ld %f0, 0(%r2)
; CHECK: [[LABEL]]:
; CHECK: std %f0, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load double , double *%ptr
  %res = select i1 %cond, double %orig, double %alt
  store volatile double %res, double *%ptr
  ret void
}

; Try a frame index base.
define void @f12(double %alt, i32 %limit) {
; CHECK-LABEL: f12:
; CHECK: brasl %r14, foo@PLT
; CHECK-NOT: %r15
; CHECK: jl [[LABEL:[^ ]*]]
; CHECK-NOT: %r15
; CHECK: std {{%f[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: [[LABEL]]:
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %ptr = alloca double
  call void @foo(double *%ptr)
  %cond = icmp ult i32 %limit, 420
  %orig = load double , double *%ptr
  %res = select i1 %cond, double %orig, double %alt
  store double %res, double *%ptr
  call void @foo(double *%ptr)
  ret void
}
