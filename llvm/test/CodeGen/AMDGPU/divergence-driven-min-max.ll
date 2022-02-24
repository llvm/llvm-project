; RUN: llc -march=amdgcn -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: name:            uniform_imin
; GCN: S_MIN_I32
define amdgpu_kernel void @uniform_imin(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %cmp = icmp sle i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            divergent_imin
; GCN: V_MIN_I32_e64
define void @divergent_imin(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %cmp = icmp sle i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            uniform_umin
; GCN: S_MIN_U32
define amdgpu_kernel void @uniform_umin(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %tmp = icmp ule i32 %a, %b
  %val = select i1 %tmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: name:            divergent_umin
; GCN: V_MIN_U32_e64
define void @divergent_umin(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %tmp = icmp ule i32 %a, %b
  %val = select i1 %tmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: name:            uniform_imax
; GCN: S_MAX_I32
define amdgpu_kernel void @uniform_imax(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp sge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            divergent_imax
; GCN: V_MAX_I32_e64
define void @divergent_imax(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp sge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            uniform_umax
; GCN: S_MAX_U32
define amdgpu_kernel void @uniform_umax(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp uge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            divergent_umax
; GCN: V_MAX_U32_e64
define void @divergent_umax(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp uge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}
