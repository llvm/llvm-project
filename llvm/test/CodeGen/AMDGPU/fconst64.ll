; RUN: llc < %s -march=amdgcn -mcpu=tahiti -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

; CHECK: {{^}}fconst_f64:
; CHECK-DAG: s_mov_b32 {{s[0-9]+}}, 0x40140000
; CHECK-DAG: s_mov_b32 {{s[0-9]+}}, 0

define amdgpu_kernel void @fconst_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
   %tid = call i32 @llvm.amdgcn.workitem.id.x()
   %gep = getelementptr inbounds double, ptr addrspace(1) %in, i32 %tid
   %r1 = load double, ptr addrspace(1) %gep
   %r2 = fadd double %r1, 5.000000e+00
   store double %r2, ptr addrspace(1) %out
   ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
