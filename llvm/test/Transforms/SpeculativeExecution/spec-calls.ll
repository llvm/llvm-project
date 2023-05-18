; RUN: opt < %s -S -passes=speculative-execution \
; RUN:   -spec-exec-max-speculation-cost 4 -spec-exec-max-not-hoisted 3 \
; RUN:   | FileCheck %s

declare float @llvm.fabs.f32(float) nounwind readnone
declare i32 @llvm.ctlz.i32(i32, i1) nounwind readnone

declare float @unknown(float)
declare float @unknown_readnone(float) nounwind readnone

; CHECK-LABEL: @ifThen_fabs(
; CHECK: call float @llvm.fabs.f32(
; CHECK: br i1 true
define void @ifThen_fabs() {
  br i1 true, label %a, label %b

a:
  %x = call float @llvm.fabs.f32(float 1.0)
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_ctlz(
; CHECK: call i32 @llvm.ctlz.i32(
; CHECK: br i1 true
define void @ifThen_ctlz() {
  br i1 true, label %a, label %b

a:
  %x = call i32 @llvm.ctlz.i32(i32 0, i1 true)
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_call_sideeffects(
; CHECK: br i1 true
; CHECK: call float @unknown(
define void @ifThen_call_sideeffects() {
  br i1 true, label %a, label %b

a:
  %x = call float @unknown(float 1.0)
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_call_readnone(
; CHECK: br i1 true
; CHECK: call float @unknown_readnone(
define void @ifThen_call_readnone() {
  br i1 true, label %a, label %b
a:
  %x = call float @unknown_readnone(float 1.0)
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_fpclass(
; CHECK: %class = call i1 @llvm.is.fpclass.f32(float %x, i32 11)
; CHECK-NEXT: br i1 true
define void @ifThen_fpclass(float %x) {
  br i1 true, label %a, label %b

a:
  %class = call i1 @llvm.is.fpclass.f32(float %x, i32 11)
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_arithmetic_fence(
; CHECK: %fence = call float @llvm.arithmetic.fence.f32(float %x)
; CHECK-NEXT: br i1 true
define void @ifThen_arithmetic_fence(float %x) {
  br i1 true, label %a, label %b

a:
  %fence = call float @llvm.arithmetic.fence.f32(float %x)
  br label %b

b:
  ret void
}

declare i1 @llvm.is.fpclass.f32(float, i32)
declare float @llvm.arithmetic.fence.f32(float)

; CHECK-LABEL: @ifThen_fptrunc_round(
; CHECK: %round = call half @llvm.fptrunc.round.f16.f32(float %x, metadata !"round.downward")
; CHECK-NEXT: br i1 true
define void @ifThen_fptrunc_round(float %x) {
  br i1 true, label %a, label %b

a:
  %round = call half @llvm.fptrunc.round.f16.f32(float %x, metadata !"round.downward")
  br label %b

b:
  ret void
}

declare half @llvm.fptrunc.round.f16.f32(float, metadata)

; CHECK-LABEL: @ifThen_vector_reduce_fadd(
; CHECK: %reduce = call float @llvm.vector.reduce.fadd.v2f32(float %x, <2 x float> %y)
; CHECK-NEXT: br i1 true
define void @ifThen_vector_reduce_fadd(float %x, <2 x float> %y) {
  br i1 true, label %a, label %b

a:
  %reduce = call float @llvm.vector.reduce.fadd.v2f32(float %x, <2 x float> %y)
  br label %b

b:
  ret void
}

declare float @llvm.vector.reduce.fadd.v2f32(float, <2 x float>)

; CHECK-LABEL: @ifThen_vector_reduce_fmul(
; CHECK: %reduce = call float @llvm.vector.reduce.fmul.v2f32(float %x, <2 x float> %y)
; CHECK-NEXT: br i1 true
define void @ifThen_vector_reduce_fmul(float %x, <2 x float> %y) {
  br i1 true, label %a, label %b

a:
  %reduce = call float @llvm.vector.reduce.fmul.v2f32(float %x, <2 x float> %y)
  br label %b

b:
  ret void
}

declare float @llvm.vector.reduce.fmul.v2f32(float, <2 x float>)

; CHECK-LABEL: @ifThen_vector_reduce_add(
; CHECK: %reduce = call i32 @llvm.vector.reduce.add.v2i32(<2 x i32> %x)
; CHECK-NEXT: br i1 true
define void @ifThen_vector_reduce_add(<2 x i32> %x) {
  br i1 true, label %a, label %b

a:
  %reduce = call i32 @llvm.vector.reduce.add.v2i32(<2 x i32> %x)
  br label %b

b:
  ret void
}

declare i32 @llvm.vector.reduce.add.v2i32(<2 x i32>)

; CHECK-LABEL: @ifThen_vector_reduce_mul(
; CHECK: %reduce = call i32 @llvm.vector.reduce.mul.v2i32(<2 x i32> %x)
; CHECK-NEXT: br i1 true
define void @ifThen_vector_reduce_mul(<2 x i32> %x) {
  br i1 true, label %a, label %b

a:
  %reduce = call i32 @llvm.vector.reduce.mul.v2i32(<2 x i32> %x)
  br label %b

b:
  ret void
}

declare i32 @llvm.vector.reduce.mul.v2i32(<2 x i32>)


; CHECK-LABEL: @ifThen_vector_reduce_and(
; CHECK: %reduce = call i32 @llvm.vector.reduce.and.v2i32(<2 x i32> %x)
; CHECK-NEXT: br i1 true
define void @ifThen_vector_reduce_and(<2 x i32> %x) {
  br i1 true, label %a, label %b

a:
  %reduce = call i32 @llvm.vector.reduce.and.v2i32(<2 x i32> %x)
  br label %b

b:
  ret void
}

declare i32 @llvm.vector.reduce.and.v2i32(<2 x i32>)

; CHECK-LABEL: @ifThen_vector_reduce_or(
; CHECK: %reduce = call i32 @llvm.vector.reduce.or.v2i32(<2 x i32> %x)
; CHECK-NEXT: br i1 true
define void @ifThen_vector_reduce_or(<2 x i32> %x) {
  br i1 true, label %a, label %b

a:
  %reduce = call i32 @llvm.vector.reduce.or.v2i32(<2 x i32> %x)
  br label %b

b:
  ret void
}

declare i32 @llvm.vector.reduce.or.v2i32(<2 x i32>)

; CHECK-LABEL: @ifThen_vector_reduce_xor(
; CHECK: %reduce = call i32 @llvm.vector.reduce.xor.v2i32(<2 x i32> %x)
; CHECK-NEXT: br i1 true
define void @ifThen_vector_reduce_xor(<2 x i32> %x) {
  br i1 true, label %a, label %b

a:
  %reduce = call i32 @llvm.vector.reduce.xor.v2i32(<2 x i32> %x)
  br label %b

b:
  ret void
}

declare i32 @llvm.vector.reduce.xor.v2i32(<2 x i32>)

; CHECK-LABEL: @ifThen_vector_reduce_smax(
; CHECK: %reduce = call i32 @llvm.vector.reduce.smax.v2i32(<2 x i32> %x)
; CHECK-NEXT: br i1 true
define void @ifThen_vector_reduce_smax(<2 x i32> %x) {
  br i1 true, label %a, label %b

a:
  %reduce = call i32 @llvm.vector.reduce.smax.v2i32(<2 x i32> %x)
  br label %b

b:
  ret void
}

declare i32 @llvm.vector.reduce.smax.v2i32(<2 x i32>)

; CHECK-LABEL: @ifThen_vector_reduce_umax(
; CHECK: %reduce = call i32 @llvm.vector.reduce.umax.v2i32(<2 x i32> %x)
; CHECK-NEXT: br i1 true
define void @ifThen_vector_reduce_umax(<2 x i32> %x) {
  br i1 true, label %a, label %b

a:
  %reduce = call i32 @llvm.vector.reduce.umax.v2i32(<2 x i32> %x)
  br label %b

b:
  ret void
}

declare i32 @llvm.vector.reduce.umax.v2i32(<2 x i32>)

; CHECK-LABEL: @ifThen_vector_reduce_umin(
; CHECK: %reduce = call i32 @llvm.vector.reduce.umin.v2i32(<2 x i32> %x)
; CHECK-NEXT: br i1 true
define void @ifThen_vector_reduce_umin(<2 x i32> %x) {
  br i1 true, label %a, label %b

a:
  %reduce = call i32 @llvm.vector.reduce.umin.v2i32(<2 x i32> %x)
  br label %b

b:
  ret void
}

declare i32 @llvm.vector.reduce.umin.v2i32(<2 x i32>)

; CHECK-LABEL: @ifThen_vector_reduce_fmax(
; CHECK: %reduce = call float @llvm.vector.reduce.fmax.v2f32(<2 x float> %x)
; CHECK-NEXT: br i1 true
define void @ifThen_vector_reduce_fmax(<2 x float> %x) {
  br i1 true, label %a, label %b

a:
  %reduce = call float @llvm.vector.reduce.fmax.v2f32(<2 x float> %x)
  br label %b

b:
  ret void
}

declare float @llvm.vector.reduce.fmax.v2f32(<2 x float>)

; CHECK-LABEL: @ifThen_vector_reduce_fmin(
; CHECK: %reduce = call float @llvm.vector.reduce.fmin.v2f32(<2 x float> %x)
; CHECK-NEXT: br i1 true
define void @ifThen_vector_reduce_fmin(<2 x float> %x) {
  br i1 true, label %a, label %b

a:
  %reduce = call float @llvm.vector.reduce.fmin.v2f32(<2 x float> %x)
  br label %b

b:
  ret void
}

declare float @llvm.vector.reduce.fmin.v2f32(<2 x float>)

; CHECK-LABEL: @ifThen_ldexp(
; CHECK: %ldexp = call float @llvm.ldexp.f32.i32(float %x, i32 %y)
; CHECK-NEXT: br i1 true
define void @ifThen_ldexp(float %x, i32 %y) {
  br i1 true, label %a, label %b

a:
  %ldexp = call float @llvm.ldexp.f32.i32(float %x, i32 %y)
  br label %b

b:
  ret void
}

declare float @llvm.ldexp.f32.i32(float, i32)
