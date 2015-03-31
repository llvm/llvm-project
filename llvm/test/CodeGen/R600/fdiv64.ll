; RUN: llc < %s -march=amdgcn -mcpu=tahiti -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

; CHECK: {{^}}fdiv_f64:
; CHECK: v_rcp_f64_e32 {{v\[[0-9]+:[0-9]+\]}}
; CHECK: v_mul_f64 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}

define void @fdiv_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                      double addrspace(1)* %in2) {
   %r0 = load double addrspace(1)* %in1
   %r1 = load double addrspace(1)* %in2
   %r2 = fdiv double %r0, %r1
   store double %r2, double addrspace(1)* %out
   ret void
}
