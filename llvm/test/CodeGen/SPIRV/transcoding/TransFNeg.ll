; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpFNegate
; CHECK-SPIRV: OpFNegate
; CHECK-SPIRV: OpFNegate
; CHECK-SPIRV: OpFNegate

;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable
;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;;
;; __kernel void foo(double a1, __global half *h, __global float *b0, __global double *b1, __global double8 *d) {
;;    *h = -*h;
;;    *b0 = -*b0;
;;    *b1 = -a1;
;;    *d = -*d;
;; }

define dso_local spir_kernel void @foo(double noundef %a1, ptr addrspace(1) noundef %h, ptr addrspace(1) noundef %b0, ptr addrspace(1) noundef %b1, ptr addrspace(1) noundef %d) {
entry:
  %a1.addr = alloca double, align 8
  %h.addr = alloca ptr addrspace(1), align 4
  %b0.addr = alloca ptr addrspace(1), align 4
  %b1.addr = alloca ptr addrspace(1), align 4
  %d.addr = alloca ptr addrspace(1), align 4
  store double %a1, ptr %a1.addr, align 8
  store ptr addrspace(1) %h, ptr %h.addr, align 4
  store ptr addrspace(1) %b0, ptr %b0.addr, align 4
  store ptr addrspace(1) %b1, ptr %b1.addr, align 4
  store ptr addrspace(1) %d, ptr %d.addr, align 4
  %0 = load ptr addrspace(1), ptr %h.addr, align 4
  %1 = load half, ptr addrspace(1) %0, align 2
  %fneg = fneg half %1
  %2 = load ptr addrspace(1), ptr %h.addr, align 4
  store half %fneg, ptr addrspace(1) %2, align 2
  %3 = load ptr addrspace(1), ptr %b0.addr, align 4
  %4 = load float, ptr addrspace(1) %3, align 4
  %fneg1 = fneg float %4
  %5 = load ptr addrspace(1), ptr %b0.addr, align 4
  store float %fneg1, ptr addrspace(1) %5, align 4
  %6 = load double, ptr %a1.addr, align 8
  %fneg2 = fneg double %6
  %7 = load ptr addrspace(1), ptr %b1.addr, align 4
  store double %fneg2, ptr addrspace(1) %7, align 8
  %8 = load ptr addrspace(1), ptr %d.addr, align 4
  %9 = load <8 x double>, ptr addrspace(1) %8, align 64
  %fneg3 = fneg <8 x double> %9
  %10 = load ptr addrspace(1), ptr %d.addr, align 4
  store <8 x double> %fneg3, ptr addrspace(1) %10, align 64
  ret void
}
