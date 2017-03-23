; RUN: llc < %s -march=amdgcn -mcpu=tahiti -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

; CHECK: {{^}}fconst_f64:
; CHECK-DAG: s_mov_b32 {{s[0-9]+}}, 0x40140000
; CHECK-DAG: s_mov_b32 {{s[0-9]+}}, 0

define amdgpu_kernel void @fconst_f64(double addrspace(1)* %out, double addrspace(1)* %in) {
   %r1 = load double, double addrspace(1)* %in
   %r2 = fadd double %r1, 5.000000e+00
   store double %r2, double addrspace(1)* %out
   ret void
}
