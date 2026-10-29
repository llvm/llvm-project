; air-emulate-f64 makes the constrained floating point intrinsics of every
; format AIR: to nearest (round.dynamic too: AIR has no floating point
; environment) the operation; in a directed rounding the operands as doubles,
; the operation in soft-f64 rounded in that direction and the result rounded
; to the format in it. bfloat is computed in float (or soft-f64), its
; conversions on the bits. strictfp goes.
; RUN: opt -mtriple=air -passes=air-emulate-f64 -S < %s \
; RUN:   | FileCheck %s --implicit-check-not='@llvm.experimental.constrained' --implicit-check-not=strictfp
; RUN: opt -mtriple=air -passes=air-emulate-f64 -S < %s | FileCheck %s --check-prefix=HINT

target triple = "air"

; CHECK-LABEL: define void @nearest(
; CHECK: %[[A:[0-9]+]] = fadd float %x, %y
; CHECK: %[[B:[0-9]+]] = fmul float %[[A]], %y
; CHECK: %[[C:[0-9]+]] = call float @llvm.fma.f32(float %[[B]], float %x, float %y)
; CHECK: %[[D:[0-9]+]] = call float @llvm.sqrt.f32(float %[[C]])
; CHECK: %[[E:[0-9]+]] = call float @llvm.exp2.f32(float %[[D]])
; CHECK: fcmp olt float %[[E]], %x
; CHECK: fadd <2 x half> %v, %v
; CHECK: fptrunc float %{{[0-9]+}} to half
; CHECK: ret void
define void @nearest(ptr addrspace(1) %p, float %x, float %y, <2 x half> %v) "air-kernel" strictfp {
  %a = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.ignore") strictfp
  %b = call float @llvm.experimental.constrained.fmul.f32(float %a, float %y, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  %c = call float @llvm.experimental.constrained.fma.f32(float %b, float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") strictfp
  %d = call float @llvm.experimental.constrained.sqrt.f32(float %c, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  %e = call float @llvm.experimental.constrained.exp2.f32(float %d, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  %l = call i1 @llvm.experimental.constrained.fcmp.f32(float %e, float %x, metadata !"olt", metadata !"fpexcept.ignore") strictfp
  store i1 %l, ptr addrspace(1) %p
  %w = call <2 x half> @llvm.experimental.constrained.fadd.v2f16(<2 x half> %v, <2 x half> %v, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  store <2 x half> %w, ptr addrspace(1) %p
  %h = call half @llvm.experimental.constrained.fptrunc.f16.f32(float %e, metadata !"round.tonearest", metadata !"fpexcept.ignore") strictfp
  store half %h, ptr addrspace(1) %p
  ret void
}

; Directed: hints (llvm/IR/AIR.h) that name the constrained operation, for
; a consumer that has it (AGX); their bodies are the exact emulations (float
; multiplication toward zero by the error-free product where it is exact and
; soft-f64 elsewhere, half addition upward by the float error-free sum, integer
; to float downward), which the Metal pipeline inlines.
; CHECK-LABEL: define void @directed(
; CHECK: call float @__air_hint.{{[0-9]+}}(float %x, float %y)
; CHECK: call half @__air_hint.{{[0-9]+}}(half %h, half %h)
; CHECK: call float @__air_hint.{{[0-9]+}}(i64 %i)
; CHECK: ret void
; HINT-DAG: "air-hint"="call:llvm.experimental.constrained.fmul.f32|---|round.towardzero|fpexcept.ignore"
; HINT-DAG: "air-hint"="call:llvm.experimental.constrained.fadd.f16|---|round.upward|fpexcept.ignore"
; HINT-DAG: "air-hint"="call:llvm.experimental.constrained.sitofp.f32.i64|--|round.downward|fpexcept.ignore"
define void @directed(ptr addrspace(1) %p, float %x, float %y, half %h, i64 %i) "air-kernel" strictfp {
  %m = call float @llvm.experimental.constrained.fmul.f32(float %x, float %y, metadata !"round.towardzero", metadata !"fpexcept.ignore") strictfp
  store float %m, ptr addrspace(1) %p
  %a = call half @llvm.experimental.constrained.fadd.f16(half %h, half %h, metadata !"round.upward", metadata !"fpexcept.ignore") strictfp
  store half %a, ptr addrspace(1) %p
  %c = call float @llvm.experimental.constrained.sitofp.f32.i64(i64 %i, metadata !"round.downward", metadata !"fpexcept.ignore") strictfp
  store float %c, ptr addrspace(1) %p
  ret void
}

; bfloat: no bfloat arithmetic is left.
; CHECK-LABEL: define void @bf16(
; CHECK-NOT: {{(fadd|fmul|fma|fpext|fptrunc).*bfloat}}
; CHECK: %[[X:[0-9]+]] = bitcast bfloat %x to i16
; CHECK: fadd float
; fma in float where exact (rounded to odd, then to nearest bfloat), else
; soft-f64 rounded to odd (toward zero, odd where rounding away differs)
; CHECK: bf16.fma.slow
; CHECK: call i64 @__air_f64_fma_r(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i32 0)
; CHECK: call i64 @__air_f64_fma_r(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i32 %{{[0-9]+}})
; CHECK: call i32 @__air_f64_to_bf16_r(i64 %{{[0-9]+}}, i32 1)
; CHECK: fmul <2 x float>
; CHECK: bitcast <2 x i16> %{{[0-9]+}} to <2 x bfloat>
; CHECK: ret void
define void @bf16(ptr addrspace(1) %p, bfloat %x, <2 x bfloat> %v, float %f) "air-kernel" strictfp {
  %a = call bfloat @llvm.experimental.constrained.fadd.bf16(bfloat %x, bfloat %x, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  %b = call bfloat @llvm.experimental.constrained.fma.bf16(bfloat %a, bfloat %x, bfloat %x, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  store bfloat %b, ptr addrspace(1) %p
  %e = call float @llvm.experimental.constrained.fpext.f32.bf16(bfloat %b, metadata !"fpexcept.ignore") strictfp
  %t = call bfloat @llvm.experimental.constrained.fptrunc.bf16.f32(float %e, metadata !"round.tonearest", metadata !"fpexcept.ignore") strictfp
  store bfloat %t, ptr addrspace(1) %p
  %w = call <2 x bfloat> @llvm.experimental.constrained.fmul.v2bf16(<2 x bfloat> %v, <2 x bfloat> %v, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  store <2 x bfloat> %w, ptr addrspace(1) %p
  ret void
}

; Where float denormals flush, bfloat's (float denormals in float) would
; too: soft-f64 instead.
; CHECK-LABEL: define void @bf16_ftz(
; CHECK: call i64 @__air_f64_add_r(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i32 1)
; CHECK: call i32 @__air_f64_to_bf16_r(i64 %{{[0-9]+}}, i32 1)
; CHECK: ret void
define void @bf16_ftz(ptr addrspace(1) %p, bfloat %x) "air-kernel" strictfp denormal_fpenv(float: preservesign) {
  %a = call bfloat @llvm.experimental.constrained.fadd.bf16(bfloat %x, bfloat %x, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  store bfloat %a, ptr addrspace(1) %p
  ret void
}

; Double precision in a directed rounding: hints too, whose bodies call the
; _r functions.
; CHECK-LABEL: define void @f64(
; CHECK: call i64 @__air_hint.{{[0-9]+}}(i64 %x, i64 %y)
; CHECK: call float @__air_hint.{{[0-9]+}}(i64 %d)
; CHECK: call i64 @__air_hint.{{[0-9]+}}(i64 %i)
; CHECK: ret void
; HINT-DAG: call i64 @__air_f64_div_r(i64 %0, i64 %1, i32 2)
; HINT-DAG: call i32 @__air_f64_to_f32_r(i64 %{{[0-9]+}}, i32 0)
; HINT-DAG: call i64 @__air_f64_from_u64_r(i64 %0, i32 3)
; HINT-DAG: "air-hint"="call:llvm.experimental.constrained.fdiv.f64|ddd|round.upward|fpexcept.ignore"
define void @f64(ptr addrspace(1) %p, double %x, double %y, i64 %i) "air-kernel" strictfp {
  %d = call double @llvm.experimental.constrained.fdiv.f64(double %x, double %y, metadata !"round.upward", metadata !"fpexcept.ignore") strictfp
  store double %d, ptr addrspace(1) %p
  %f = call float @llvm.experimental.constrained.fptrunc.f32.f64(double %d, metadata !"round.towardzero", metadata !"fpexcept.ignore") strictfp
  store float %f, ptr addrspace(1) %p
  %u = call double @llvm.experimental.constrained.uitofp.f64.i64(i64 %i, metadata !"round.downward", metadata !"fpexcept.ignore") strictfp
  store double %u, ptr addrspace(1) %p
  ret void
}
