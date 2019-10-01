; Verify that strict FP operations are not rescheduled
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
declare float @llvm.sqrt.f32(float)
declare void @llvm.s390.sfpc(i32)

; For non-strict operations, we expect the post-RA scheduler to
; separate the two square root instructions on z13.
define void @f1(float %f1, float %f2, float %f3, float %f4, float *%ptr0) {
; CHECK-LABEL: f1:
; CHECK: sqebr
; CHECK: {{aebr|sebr}}
; CHECK: sqebr
; CHECK: br %r14

  %add = fadd float %f1, %f2
  %sub = fsub float %f3, %f4
  %sqrt1 = call float @llvm.sqrt.f32(float %f2)
  %sqrt2 = call float @llvm.sqrt.f32(float %f4)

  %ptr1 = getelementptr float, float *%ptr0, i64 1
  %ptr2 = getelementptr float, float *%ptr0, i64 2
  %ptr3 = getelementptr float, float *%ptr0, i64 3

  store float %add, float *%ptr0
  store float %sub, float *%ptr1
  store float %sqrt1, float *%ptr2
  store float %sqrt2, float *%ptr3

  ret void
}

; But for strict operations, this must not happen.
define void @f2(float %f1, float %f2, float %f3, float %f4, float *%ptr0) {
; CHECK-LABEL: f2:
; CHECK: {{aebr|sebr}}
; CHECK: {{aebr|sebr}}
; CHECK: sqebr
; CHECK: sqebr
; CHECK: br %r14

  %add = call float @llvm.experimental.constrained.fadd.f32(
                        float %f1, float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %sub = call float @llvm.experimental.constrained.fsub.f32(
                        float %f3, float %f4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %sqrt1 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %sqrt2 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")

  %ptr1 = getelementptr float, float *%ptr0, i64 1
  %ptr2 = getelementptr float, float *%ptr0, i64 2
  %ptr3 = getelementptr float, float *%ptr0, i64 3

  store float %add, float *%ptr0
  store float %sub, float *%ptr1
  store float %sqrt1, float *%ptr2
  store float %sqrt2, float *%ptr3

  ret void
}

; On the other hand, strict operations that use the fpexcept.ignore
; exception behaviour should be scheduled freely.
define void @f3(float %f1, float %f2, float %f3, float %f4, float *%ptr0) {
; CHECK-LABEL: f3:
; CHECK: sqebr
; CHECK: {{aebr|sebr}}
; CHECK: sqebr
; CHECK: br %r14

  %add = call float @llvm.experimental.constrained.fadd.f32(
                        float %f1, float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore")
  %sub = call float @llvm.experimental.constrained.fsub.f32(
                        float %f3, float %f4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore")
  %sqrt1 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore")
  %sqrt2 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore")

  %ptr1 = getelementptr float, float *%ptr0, i64 1
  %ptr2 = getelementptr float, float *%ptr0, i64 2
  %ptr3 = getelementptr float, float *%ptr0, i64 3

  store float %add, float *%ptr0
  store float %sub, float *%ptr1
  store float %sqrt1, float *%ptr2
  store float %sqrt2, float *%ptr3

  ret void
}

; However, even non-strict operations must not be scheduled across an SFPC.
define void @f4(float %f1, float %f2, float %f3, float %f4, float *%ptr0) {
; CHECK-LABEL: f4:
; CHECK: {{aebr|sebr}}
; CHECK: {{aebr|sebr}}
; CHECK: sfpc
; CHECK: sqebr
; CHECK: sqebr
; CHECK: br %r14

  %add = fadd float %f1, %f2
  %sub = fsub float %f3, %f4
  call void @llvm.s390.sfpc(i32 0)
  %sqrt1 = call float @llvm.sqrt.f32(float %f2)
  %sqrt2 = call float @llvm.sqrt.f32(float %f4)

  %ptr1 = getelementptr float, float *%ptr0, i64 1
  %ptr2 = getelementptr float, float *%ptr0, i64 2
  %ptr3 = getelementptr float, float *%ptr0, i64 3

  store float %add, float *%ptr0
  store float %sub, float *%ptr1
  store float %sqrt1, float *%ptr2
  store float %sqrt2, float *%ptr3

  ret void
}

