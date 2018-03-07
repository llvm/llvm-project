; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn-amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-HSA -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}constant_load_f64:
; GCN: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}]
; GCN-NOHSA: buffer_store_dwordx2
; GCN-HSA: flat_store_dwordx2
define amdgpu_kernel void @constant_load_f64(double addrspace(1)* %out, double addrspace(4)* %in) #0 {
  %ld = load double, double addrspace(4)* %in
  store double %ld, double addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }

; Tests whether a load-chain of 8 constants of 64bit each gets vectorized into a wider load.
; FUNC-LABEL: {{^}}constant_load_2v4f64:
; GCN: s_load_dwordx16
define amdgpu_kernel void @constant_load_2v4f64(double addrspace(4)* noalias nocapture readonly %weights, double addrspace(1)* noalias nocapture %out_ptr) {
entry:
  %out_ptr.promoted = load double, double addrspace(1)* %out_ptr, align 4
  %tmp = load double, double addrspace(4)* %weights, align 4
  %add = fadd double %tmp, %out_ptr.promoted
  %arrayidx.1 = getelementptr inbounds double, double addrspace(4)* %weights, i64 1
  %tmp1 = load double, double addrspace(4)* %arrayidx.1, align 4
  %add.1 = fadd double %tmp1, %add
  %arrayidx.2 = getelementptr inbounds double, double addrspace(4)* %weights, i64 2
  %tmp2 = load double, double addrspace(4)* %arrayidx.2, align 4
  %add.2 = fadd double %tmp2, %add.1
  %arrayidx.3 = getelementptr inbounds double, double addrspace(4)* %weights, i64 3
  %tmp3 = load double, double addrspace(4)* %arrayidx.3, align 4
  %add.3 = fadd double %tmp3, %add.2
  %arrayidx.4 = getelementptr inbounds double, double addrspace(4)* %weights, i64 4
  %tmp4 = load double, double addrspace(4)* %arrayidx.4, align 4
  %add.4 = fadd double %tmp4, %add.3
  %arrayidx.5 = getelementptr inbounds double, double addrspace(4)* %weights, i64 5
  %tmp5 = load double, double addrspace(4)* %arrayidx.5, align 4
  %add.5 = fadd double %tmp5, %add.4
  %arrayidx.6 = getelementptr inbounds double, double addrspace(4)* %weights, i64 6
  %tmp6 = load double, double addrspace(4)* %arrayidx.6, align 4
  %add.6 = fadd double %tmp6, %add.5
  %arrayidx.7 = getelementptr inbounds double, double addrspace(4)* %weights, i64 7
  %tmp7 = load double, double addrspace(4)* %arrayidx.7, align 4
  %add.7 = fadd double %tmp7, %add.6
  store double %add.7, double addrspace(1)* %out_ptr, align 4
  ret void
}
