; Test f32 conditional stores that are presented as selects.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @foo(float *)

; Test with the loaded value first.
define void @f1(float *%ptr, float %alt, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: ste %f0, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load float , float *%ptr
  %res = select i1 %cond, float %orig, float %alt
  store float %res, float *%ptr
  ret void
}

; ...and with the loaded value second
define void @f2(float *%ptr, float %alt, i32 %limit) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: ste %f0, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load float , float *%ptr
  %res = select i1 %cond, float %alt, float %orig
  store float %res, float *%ptr
  ret void
}

; Check the high end of the aligned STE range.
define void @f3(float *%base, float %alt, i32 %limit) {
; CHECK-LABEL: f3:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: ste %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1023
  %cond = icmp ult i32 %limit, 420
  %orig = load float , float *%ptr
  %res = select i1 %cond, float %orig, float %alt
  store float %res, float *%ptr
  ret void
}

; Check the next word up, which should use STEY instead of STE.
define void @f4(float *%base, float %alt, i32 %limit) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stey %f0, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1024
  %cond = icmp ult i32 %limit, 420
  %orig = load float , float *%ptr
  %res = select i1 %cond, float %orig, float %alt
  store float %res, float *%ptr
  ret void
}

; Check the high end of the aligned STEY range.
define void @f5(float *%base, float %alt, i32 %limit) {
; CHECK-LABEL: f5:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stey %f0, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 131071
  %cond = icmp ult i32 %limit, 420
  %orig = load float , float *%ptr
  %res = select i1 %cond, float %orig, float %alt
  store float %res, float *%ptr
  ret void
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f6(float *%base, float %alt, i32 %limit) {
; CHECK-LABEL: f6:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: agfi %r2, 524288
; CHECK: ste %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 131072
  %cond = icmp ult i32 %limit, 420
  %orig = load float , float *%ptr
  %res = select i1 %cond, float %orig, float %alt
  store float %res, float *%ptr
  ret void
}

; Check the low end of the STEY range.
define void @f7(float *%base, float %alt, i32 %limit) {
; CHECK-LABEL: f7:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stey %f0, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 -131072
  %cond = icmp ult i32 %limit, 420
  %orig = load float , float *%ptr
  %res = select i1 %cond, float %orig, float %alt
  store float %res, float *%ptr
  ret void
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8(float *%base, float %alt, i32 %limit) {
; CHECK-LABEL: f8:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: agfi %r2, -524292
; CHECK: ste %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 -131073
  %cond = icmp ult i32 %limit, 420
  %orig = load float , float *%ptr
  %res = select i1 %cond, float %orig, float %alt
  store float %res, float *%ptr
  ret void
}

; Check that STEY allows an index.
define void @f9(i64 %base, i64 %index, float %alt, i32 %limit) {
; CHECK-LABEL: f9:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stey %f0, 4096(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to float *
  %cond = icmp ult i32 %limit, 420
  %orig = load float , float *%ptr
  %res = select i1 %cond, float %orig, float %alt
  store float %res, float *%ptr
  ret void
}

; Check that volatile loads are not matched.
define void @f10(float *%ptr, float %alt, i32 %limit) {
; CHECK-LABEL: f10:
; CHECK: le {{%f[0-5]}}, 0(%r2)
; CHECK: {{jl|jnl}} [[LABEL:[^ ]*]]
; CHECK: [[LABEL]]:
; CHECK: ste {{%f[0-5]}}, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load volatile float , float *%ptr
  %res = select i1 %cond, float %orig, float %alt
  store float %res, float *%ptr
  ret void
}

; ...likewise stores.  In this case we should have a conditional load into %f0.
define void @f11(float *%ptr, float %alt, i32 %limit) {
; CHECK-LABEL: f11:
; CHECK: jhe [[LABEL:[^ ]*]]
; CHECK: le %f0, 0(%r2)
; CHECK: [[LABEL]]:
; CHECK: ste %f0, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load float , float *%ptr
  %res = select i1 %cond, float %orig, float %alt
  store volatile float %res, float *%ptr
  ret void
}

; Try a frame index base.
define void @f12(float %alt, i32 %limit) {
; CHECK-LABEL: f12:
; CHECK: brasl %r14, foo@PLT
; CHECK-NOT: %r15
; CHECK: jl [[LABEL:[^ ]*]]
; CHECK-NOT: %r15
; CHECK: ste {{%f[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: [[LABEL]]:
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %ptr = alloca float
  call void @foo(float *%ptr)
  %cond = icmp ult i32 %limit, 420
  %orig = load float , float *%ptr
  %res = select i1 %cond, float %orig, float %alt
  store float %res, float *%ptr
  call void @foo(float *%ptr)
  ret void
}
