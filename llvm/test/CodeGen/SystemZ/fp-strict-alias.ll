; Verify that strict FP operations are not rescheduled
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
declare float @llvm.sqrt.f32(float)
declare void @llvm.s390.sfpc(i32)

; The basic assumption of all following tests is that on z13, we never
; want to see two square root instructions directly in a row, so the
; post-RA scheduler will always schedule something else in between
; whenever possible.

; We can move any FP operation across a (normal) store.

define void @f1(float %f1, float %f2, float *%ptr1, float *%ptr2) {
; CHECK-LABEL: f1:
; CHECK: sqebr
; CHECK: ste
; CHECK: sqebr
; CHECK: ste
; CHECK: br %r14

  %sqrt1 = call float @llvm.sqrt.f32(float %f1)
  %sqrt2 = call float @llvm.sqrt.f32(float %f2)

  store float %sqrt1, float *%ptr1
  store float %sqrt2, float *%ptr2

  ret void
}

define void @f2(float %f1, float %f2, float *%ptr1, float *%ptr2) #0 {
; CHECK-LABEL: f2:
; CHECK: sqebr
; CHECK: ste
; CHECK: sqebr
; CHECK: ste
; CHECK: br %r14

  %sqrt1 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore") #0
  %sqrt2 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore") #0

  store float %sqrt1, float *%ptr1
  store float %sqrt2, float *%ptr2

  ret void
}

define void @f3(float %f1, float %f2, float *%ptr1, float *%ptr2) #0 {
; CHECK-LABEL: f3:
; CHECK: sqebr
; CHECK: ste
; CHECK: sqebr
; CHECK: ste
; CHECK: br %r14

  %sqrt1 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %sqrt2 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  store float %sqrt1, float *%ptr1
  store float %sqrt2, float *%ptr2

  ret void
}


; We can move a non-strict FP operation or a fpexcept.ignore
; operation even across a volatile store, but not a fpexcept.strict
; operation.

define void @f4(float %f1, float %f2, float *%ptr1, float *%ptr2) {
; CHECK-LABEL: f4:
; CHECK: sqebr
; CHECK: ste
; CHECK: sqebr
; CHECK: ste
; CHECK: br %r14

  %sqrt1 = call float @llvm.sqrt.f32(float %f1)
  %sqrt2 = call float @llvm.sqrt.f32(float %f2)

  store volatile float %sqrt1, float *%ptr1
  store volatile float %sqrt2, float *%ptr2

  ret void
}

define void @f5(float %f1, float %f2, float *%ptr1, float *%ptr2) #0 {
; CHECK-LABEL: f5:
; CHECK: sqebr
; CHECK: ste
; CHECK: sqebr
; CHECK: ste
; CHECK: br %r14

  %sqrt1 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore") #0
  %sqrt2 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore") #0

  store volatile float %sqrt1, float *%ptr1
  store volatile float %sqrt2, float *%ptr2

  ret void
}

define void @f6(float %f1, float %f2, float *%ptr1, float *%ptr2) #0 {
; CHECK-LABEL: f6:
; CHECK: sqebr
; CHECK: sqebr
; CHECK: ste
; CHECK: ste
; CHECK: br %r14

  %sqrt1 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %sqrt2 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  store volatile float %sqrt1, float *%ptr1
  store volatile float %sqrt2, float *%ptr2

  ret void
}


; No variant of FP operations can be scheduled across a SPFC.

define void @f7(float %f1, float %f2, float *%ptr1, float *%ptr2) {
; CHECK-LABEL: f7:
; CHECK: sqebr
; CHECK: sqebr
; CHECK: ste
; CHECK: ste
; CHECK: br %r14

  %sqrt1 = call float @llvm.sqrt.f32(float %f1)
  %sqrt2 = call float @llvm.sqrt.f32(float %f2)

  call void @llvm.s390.sfpc(i32 0)

  store float %sqrt1, float *%ptr1
  store float %sqrt2, float *%ptr2

  ret void
}

define void @f8(float %f1, float %f2, float *%ptr1, float *%ptr2) #0 {
; CHECK-LABEL: f8:
; CHECK: sqebr
; CHECK: sqebr
; CHECK: ste
; CHECK: ste
; CHECK: br %r14

  %sqrt1 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore") #0
  %sqrt2 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore") #0

  call void @llvm.s390.sfpc(i32 0) #0

  store float %sqrt1, float *%ptr1
  store float %sqrt2, float *%ptr2

  ret void
}

define void @f9(float %f1, float %f2, float *%ptr1, float *%ptr2) #0 {
; CHECK-LABEL: f9:
; CHECK: sqebr
; CHECK: sqebr
; CHECK: ste
; CHECK: ste
; CHECK: br %r14

  %sqrt1 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %sqrt2 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  call void @llvm.s390.sfpc(i32 0) #0

  store float %sqrt1, float *%ptr1
  store float %sqrt2, float *%ptr2

  ret void
}

attributes #0 = { strictfp }
