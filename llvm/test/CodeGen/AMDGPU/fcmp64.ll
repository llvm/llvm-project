; RUN: llc < %s -march=amdgcn -mcpu=tahiti -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: {{^}}flt_f64:
; CHECK: v_cmp_nge_f64_e32 vcc, {{s\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @flt_f64(ptr addrspace(1) %out, ptr addrspace(1) %in1,
                     ptr addrspace(1) %in2) {
   %r0 = load double, ptr addrspace(1) %in1
   %r1 = load double, ptr addrspace(1) %in2
   %r2 = fcmp ult double %r0, %r1
   %r3 = zext i1 %r2 to i32
   store i32 %r3, ptr addrspace(1) %out
   ret void
}

; CHECK-LABEL: {{^}}fle_f64:
; CHECK: v_cmp_ngt_f64_e32 vcc, {{s\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fle_f64(ptr addrspace(1) %out, ptr addrspace(1) %in1,
                     ptr addrspace(1) %in2) {
   %r0 = load double, ptr addrspace(1) %in1
   %r1 = load double, ptr addrspace(1) %in2
   %r2 = fcmp ule double %r0, %r1
   %r3 = zext i1 %r2 to i32
   store i32 %r3, ptr addrspace(1) %out
   ret void
}

; CHECK-LABEL: {{^}}fgt_f64:
; CHECK: v_cmp_nle_f64_e32 vcc, {{s\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fgt_f64(ptr addrspace(1) %out, ptr addrspace(1) %in1,
                     ptr addrspace(1) %in2) {
   %r0 = load double, ptr addrspace(1) %in1
   %r1 = load double, ptr addrspace(1) %in2
   %r2 = fcmp ugt double %r0, %r1
   %r3 = zext i1 %r2 to i32
   store i32 %r3, ptr addrspace(1) %out
   ret void
}

; CHECK-LABEL: {{^}}fge_f64:
; CHECK: v_cmp_nlt_f64_e32 vcc, {{s\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fge_f64(ptr addrspace(1) %out, ptr addrspace(1) %in1,
                     ptr addrspace(1) %in2) {
   %r0 = load double, ptr addrspace(1) %in1
   %r1 = load double, ptr addrspace(1) %in2
   %r2 = fcmp uge double %r0, %r1
   %r3 = zext i1 %r2 to i32
   store i32 %r3, ptr addrspace(1) %out
   ret void
}

; CHECK-LABEL: {{^}}fne_f64:
; CHECK: v_cmp_neq_f64_e32 vcc, {{s\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fne_f64(ptr addrspace(1) %out, ptr addrspace(1) %in1,
                     ptr addrspace(1) %in2) {
   %r0 = load double, ptr addrspace(1) %in1
   %r1 = load double, ptr addrspace(1) %in2
   %r2 = fcmp une double %r0, %r1
   %r3 = select i1 %r2, double %r0, double %r1
   store double %r3, ptr addrspace(1) %out
   ret void
}

; CHECK-LABEL: {{^}}feq_f64:
; CHECK: v_cmp_nlg_f64_e32 vcc, {{s\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @feq_f64(ptr addrspace(1) %out, ptr addrspace(1) %in1,
                     ptr addrspace(1) %in2) {
   %r0 = load double, ptr addrspace(1) %in1
   %r1 = load double, ptr addrspace(1) %in2
   %r2 = fcmp ueq double %r0, %r1
   %r3 = select i1 %r2, double %r0, double %r1
   store double %r3, ptr addrspace(1) %out
   ret void
}
