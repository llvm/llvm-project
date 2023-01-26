; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC -check-prefix=SI %s

; FIXME: This currently doesn't do a great job of clustering the
; loads, which end up with extra moves between them. Right now, it
; seems the only things areLoadsFromSameBasePtr is accomplishing is
; ordering the loads so that the lower address loads come first.

; FUNC-LABEL: {{^}}cluster_global_arg_loads:
; SI-DAG: buffer_load_dword [[REG0:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; SI-DAG: buffer_load_dword [[REG1:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:8
; SI: buffer_store_dword [[REG0]]
; SI: buffer_store_dword [[REG1]]
define amdgpu_kernel void @cluster_global_arg_loads(ptr addrspace(1) %out0, ptr addrspace(1) %out1, ptr addrspace(1) %ptr) #0 {
  %load0 = load i32, ptr addrspace(1) %ptr, align 4
  %gep = getelementptr i32, ptr addrspace(1) %ptr, i32 2
  %load1 = load i32, ptr addrspace(1) %gep, align 4
  store i32 %load0, ptr addrspace(1) %out0, align 4
  store i32 %load1, ptr addrspace(1) %out1, align 4
  ret void
}

; Test for a crach in SIInstrInfo::areLoadsFromSameBasePtr() when checking
; an MUBUF load which does not have a vaddr operand.
; FUNC-LABEL: {{^}}same_base_ptr_crash:
; SI: buffer_load_dword
; SI: buffer_load_dword
define amdgpu_kernel void @same_base_ptr_crash(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %offset) {
entry:
  %out1 = getelementptr i32, ptr addrspace(1) %out, i32 %offset
  %tmp0 = load i32, ptr addrspace(1) %out
  %tmp1 = load i32, ptr addrspace(1) %out1
  %tmp2 = add i32 %tmp0, %tmp1
  store i32 %tmp2, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
