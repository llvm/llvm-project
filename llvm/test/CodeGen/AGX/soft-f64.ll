; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-soft-f64 < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM

; Double precision arithmetic becomes integer arithmetic on the bits, with
; 128-bit intermediate values; values that are only moved stay double.

; CHECK-LABEL: @arith(
; CHECK-NOT: {{fadd|fmul|fdiv|fpext|fptrunc|fcmp}} double
; CHECK: mul i128
; CHECK: call i128 @llvm.ctlz.i128
; CHECK: bitcast i64 {{.*}} to double
; ASM-LABEL: arith:
; ASM-NOT: error
define agx_kernel void @arith(ptr addrspace(1) %out, double %a, float %b) {
  %e = fpext float %b to double
  %s = fadd double %a, %e
  %m = fmul double %s, %a
  %d = fdiv double %m, %e
  %t = fptrunc double %d to float
  store float %t, ptr addrspace(1) %out
  %o = getelementptr double, ptr addrspace(1) %out, i32 1
  store double %d, ptr addrspace(1) %o
  ret void
}

; The square root is a loop producing a bit of the root per iteration.
; CHECK-LABEL: @root(
; CHECK: isqrt:
; CHECK: phi i128
; CHECK: br i1 {{.*}}, label %isqrt.end, label %isqrt
define agx_kernel void @root(ptr addrspace(1) %out, double %a) {
  %r = call double @llvm.sqrt.f64(double %a)
  store double %r, ptr addrspace(1) %out
  ret void
}

; A directed rounding mode is applied to the software operation.
; CHECK-LABEL: @toward_zero(
; CHECK-NOT: call {{.*}}constrained
; CHECK-NOT: fadd double
define agx_kernel void @toward_zero(ptr addrspace(1) %out, double %a, double %b) #0 {
  %r = call double @llvm.experimental.constrained.fadd.f64(double %a, double %b, metadata !"round.towardzero", metadata !"fpexcept.ignore") #0
  store double %r, ptr addrspace(1) %out
  ret void
}

; Comparisons are integer comparisons of an order key.
; CHECK-LABEL: @compare(
; CHECK-NOT: fcmp
; CHECK: icmp slt i64
define agx_kernel void @compare(ptr addrspace(1) %out, double %a, double %b) {
  %c = fcmp olt double %a, %b
  %z = zext i1 %c to i32
  store i32 %z, ptr addrspace(1) %out
  ret void
}

declare double @llvm.sqrt.f64(double)
declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)

attributes #0 = { strictfp }
