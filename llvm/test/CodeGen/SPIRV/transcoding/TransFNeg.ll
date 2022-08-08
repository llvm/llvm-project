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

define dso_local spir_kernel void @foo(double noundef %a1, half addrspace(1)* noundef %h, float addrspace(1)* noundef %b0, double addrspace(1)* noundef %b1, <8 x double> addrspace(1)* noundef %d) {
entry:
  %a1.addr = alloca double, align 8
  %h.addr = alloca half addrspace(1)*, align 4
  %b0.addr = alloca float addrspace(1)*, align 4
  %b1.addr = alloca double addrspace(1)*, align 4
  %d.addr = alloca <8 x double> addrspace(1)*, align 4
  store double %a1, double* %a1.addr, align 8
  store half addrspace(1)* %h, half addrspace(1)** %h.addr, align 4
  store float addrspace(1)* %b0, float addrspace(1)** %b0.addr, align 4
  store double addrspace(1)* %b1, double addrspace(1)** %b1.addr, align 4
  store <8 x double> addrspace(1)* %d, <8 x double> addrspace(1)** %d.addr, align 4
  %0 = load half addrspace(1)*, half addrspace(1)** %h.addr, align 4
  %1 = load half, half addrspace(1)* %0, align 2
  %fneg = fneg half %1
  %2 = load half addrspace(1)*, half addrspace(1)** %h.addr, align 4
  store half %fneg, half addrspace(1)* %2, align 2
  %3 = load float addrspace(1)*, float addrspace(1)** %b0.addr, align 4
  %4 = load float, float addrspace(1)* %3, align 4
  %fneg1 = fneg float %4
  %5 = load float addrspace(1)*, float addrspace(1)** %b0.addr, align 4
  store float %fneg1, float addrspace(1)* %5, align 4
  %6 = load double, double* %a1.addr, align 8
  %fneg2 = fneg double %6
  %7 = load double addrspace(1)*, double addrspace(1)** %b1.addr, align 4
  store double %fneg2, double addrspace(1)* %7, align 8
  %8 = load <8 x double> addrspace(1)*, <8 x double> addrspace(1)** %d.addr, align 4
  %9 = load <8 x double>, <8 x double> addrspace(1)* %8, align 64
  %fneg3 = fneg <8 x double> %9
  %10 = load <8 x double> addrspace(1)*, <8 x double> addrspace(1)** %d.addr, align 4
  store <8 x double> %fneg3, <8 x double> addrspace(1)* %10, align 64
  ret void
}
