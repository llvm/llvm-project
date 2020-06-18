; Verify that strict FP operations are not rescheduled
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
declare float @llvm.sqrt.f32(float)
declare void @llvm.s390.sfpc(i32)
declare void @bar()

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
                        metadata !"fpexcept.maytrap") #0
  %sqrt2 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.maytrap") #0

  store float %sqrt1, float *%ptr1
  store float %sqrt2, float *%ptr2

  ret void
}

define void @f4(float %f1, float %f2, float *%ptr1, float *%ptr2) #0 {
; CHECK-LABEL: f4:
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
; operation even across a volatile store, but not a fpexcept.maytrap
; or fpexcept.strict operation.

define void @f5(float %f1, float %f2, float *%ptr1, float *%ptr2) {
; CHECK-LABEL: f5:
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

define void @f6(float %f1, float %f2, float *%ptr1, float *%ptr2) #0 {
; CHECK-LABEL: f6:
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

define void @f7(float %f1, float %f2, float *%ptr1, float *%ptr2) #0 {
; CHECK-LABEL: f7:
; CHECK: sqebr
; CHECK: sqebr
; CHECK: ste
; CHECK: ste
; CHECK: br %r14

  %sqrt1 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.maytrap") #0
  %sqrt2 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.maytrap") #0

  store volatile float %sqrt1, float *%ptr1
  store volatile float %sqrt2, float *%ptr2

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

define void @f9(float %f1, float %f2, float *%ptr1, float *%ptr2) {
; CHECK-LABEL: f9:
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

define void @f10(float %f1, float %f2, float *%ptr1, float *%ptr2) #0 {
; CHECK-LABEL: f10:
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

define void @f11(float %f1, float %f2, float *%ptr1, float *%ptr2) #0 {
; CHECK-LABEL: f11:
; CHECK: sqebr
; CHECK: sqebr
; CHECK: ste
; CHECK: ste
; CHECK: br %r14

  %sqrt1 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.maytrap") #0
  %sqrt2 = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.maytrap") #0

  call void @llvm.s390.sfpc(i32 0) #0

  store float %sqrt1, float *%ptr1
  store float %sqrt2, float *%ptr2

  ret void
}

define void @f12(float %f1, float %f2, float *%ptr1, float *%ptr2) #0 {
; CHECK-LABEL: f12:
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

; If the result of any FP operation is unused, it can be removed
; -- except for fpexcept.strict operations.

define void @f13(float %f1) {
; CHECK-LABEL: f13:
; CHECK-NOT: sqeb
; CHECK: br %r14

  %sqrt = call float @llvm.sqrt.f32(float %f1)

  ret void
}

define void @f14(float %f1) #0 {
; CHECK-LABEL: f14:
; CHECK-NOT: sqeb
; CHECK: br %r14

  %sqrt = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore") #0

  ret void
}

define void @f15(float %f1) #0 {
; CHECK-LABEL: f15:
; CHECK-NOT: sqeb
; CHECK: br %r14

  %sqrt = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.maytrap") #0

  ret void
}

define void @f16(float %f1) #0 {
; CHECK-LABEL: f16:
; CHECK: sqebr
; CHECK: br %r14

  %sqrt = call float @llvm.experimental.constrained.sqrt.f32(
                        float %f1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  ret void
}


; Verify that constrained intrinsics and memory operations get their
; chains linked up properly.

define void @f17(float %in, float* %out) #0 {
; CHECK-LABEL: f17:
; CHECK: sqebr
; CHECK: ste
; CHECK: jg bar
  %sqrt = call float @llvm.experimental.constrained.sqrt.f32(
                        float %in,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore") #0
  store float %sqrt, float* %out, align 4
  tail call void @bar() #0
  ret void
}

define void @f18(float %in, float* %out) #0 {
; CHECK-LABEL: f18:
; CHECK: sqebr
; CHECK: ste
; CHECK: jg bar
  %sqrt = call float @llvm.experimental.constrained.sqrt.f32(
                        float %in,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.ignore") #0
  store float %sqrt, float* %out, align 4
  tail call void @bar() #0
  ret void
}

define void @f19(float %in, float* %out) #0 {
; CHECK-LABEL: f19:
; CHECK: sqebr
; CHECK: ste
; CHECK: jg bar
  %sqrt = call float @llvm.experimental.constrained.sqrt.f32(
                        float %in,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.maytrap") #0
  store float %sqrt, float* %out, align 4
  tail call void @bar() #0
  ret void
}

define void @f20(float %in, float* %out) #0 {
; CHECK-LABEL: f20:
; CHECK: sqebr
; CHECK: ste
; CHECK: jg bar
  %sqrt = call float @llvm.experimental.constrained.sqrt.f32(
                        float %in,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  store float %sqrt, float* %out, align 4
  tail call void @bar() #0
  ret void
}

attributes #0 = { strictfp }
