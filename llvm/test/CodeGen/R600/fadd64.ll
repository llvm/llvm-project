; RUN: llc < %s -march=amdgcn -mcpu=tahiti -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

; CHECK: {{^}}fadd_f64:
; CHECK: v_add_f64 {{v[[0-9]+:[0-9]+]}}, {{v[[0-9]+:[0-9]+]}}, {{v[[0-9]+:[0-9]+]}}

define void @fadd_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                      double addrspace(1)* %in2) {
   %r0 = load double addrspace(1)* %in1
   %r1 = load double addrspace(1)* %in2
   %r2 = fadd double %r0, %r1
   store double %r2, double addrspace(1)* %out
   ret void
}
