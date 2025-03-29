; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}expand_atomicrmw_agent:
; GCN: global_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9:]+}}], v[{{[0-9:]+}}], off glc{{$}}
define void @expand_atomicrmw_agent(ptr addrspace(1) nocapture %arg) {
entry:
  %ret = atomicrmw fadd ptr addrspace(1) %arg, float 1.000000e+00 syncscope("agent") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}expand_atomicrmw_workgroup:
; GCN: global_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9:]+}}], v[{{[0-9:]+}}], off glc{{$}}
define void @expand_atomicrmw_workgroup(ptr addrspace(1) nocapture %arg) {
entry:
  %ret = atomicrmw fadd ptr addrspace(1) %arg, float 1.000000e+00 syncscope("workgroup") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}expand_atomicrmw_wavefront:
; GCN: global_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9:]+}}], v[{{[0-9:]+}}], off glc{{$}}
define void @expand_atomicrmw_wavefront(ptr addrspace(1) nocapture %arg) {
entry:
  %ret = atomicrmw fadd ptr addrspace(1) %arg, float 1.000000e+00 syncscope("wavefront") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}expand_atomicrmw_agent_one_as:
; GCN: global_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9:]+}}], v[{{[0-9:]+}}], off glc{{$}}
define void @expand_atomicrmw_agent_one_as(ptr addrspace(1) nocapture %arg) {
entry:
  %ret = atomicrmw fadd ptr addrspace(1) %arg, float 1.000000e+00 syncscope("agent-one-as") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}expand_atomicrmw_workgroup_one_as:
; GCN: global_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9:]+}}], v[{{[0-9:]+}}], off glc{{$}}
define void @expand_atomicrmw_workgroup_one_as(ptr addrspace(1) nocapture %arg) {
entry:
  %ret = atomicrmw fadd ptr addrspace(1) %arg, float 1.000000e+00 syncscope("workgroup-one-as") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}expand_atomicrmw_wavefront_one_as:
; GCN: global_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9:]+}}], v[{{[0-9:]+}}], off glc{{$}}
define void @expand_atomicrmw_wavefront_one_as(ptr addrspace(1) nocapture %arg) {
entry:
  %ret = atomicrmw fadd ptr addrspace(1) %arg, float 1.000000e+00 syncscope("wavefront-one-as") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}expand_atomicrmw_singlethread_one_as:
; GCN: global_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9:]+}}], v[{{[0-9:]+}}], off glc{{$}}
define void @expand_atomicrmw_singlethread_one_as(ptr addrspace(1) nocapture %arg) {
entry:
  %ret = atomicrmw fadd ptr addrspace(1) %arg, float 1.000000e+00 syncscope("singlethread-one-as") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}expand_atomicrmw_one_as:
; GCN: global_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9:]+}}], v[{{[0-9:]+}}], off glc{{$}}
define void @expand_atomicrmw_one_as(ptr addrspace(1) nocapture %arg) {
entry:
  %ret = atomicrmw fadd ptr addrspace(1) %arg, float 1.000000e+00 syncscope("one-as") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}expand_atomicrmw_system:
; GCN: global_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9:]+}}], v[{{[0-9:]+}}], off glc{{$}}
define void @expand_atomicrmw_system(ptr addrspace(1) nocapture %arg) {
entry:
  %ret = atomicrmw fadd ptr addrspace(1) %arg, float 1.000000e+00 monotonic, align 4
  ret void
}
