; RUN: llc -O0 -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -mattr=+promote-alloca < %s | FileCheck -check-prefix=NOOPTS -check-prefix=ALL %s
; RUN: llc -O1 -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -mattr=+promote-alloca < %s | FileCheck -check-prefix=OPTS -check-prefix=ALL %s

; ALL-LABEL: {{^}}promote_alloca_i32_array_array:
; NOOPTS: .amdhsa_group_segment_fixed_size 0
; NOOPTS-NOT: ds_write
; OPTS: ds_write
define amdgpu_kernel void @promote_alloca_i32_array_array(ptr addrspace(1) %out, i32 %index) #0 {
entry:
  %alloca = alloca [2 x [2 x i32]], addrspace(5)
  %gep1 = getelementptr inbounds [2 x [2 x i32]], ptr addrspace(5) %alloca, i32 0, i32 0, i32 1
  store i32 0, ptr addrspace(5) %alloca
  store i32 1, ptr addrspace(5) %gep1
  %gep2 = getelementptr inbounds [2 x [2 x i32]], ptr addrspace(5) %alloca, i32 0, i32 0, i32 %index
  %load = load i32, ptr addrspace(5) %gep2
  store i32 %load, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}optnone_promote_alloca_i32_array_array:
; ALL: .amdhsa_group_segment_fixed_size 0
; ALL-NOT: ds_write
define amdgpu_kernel void @optnone_promote_alloca_i32_array_array(ptr addrspace(1) %out, i32 %index) #1 {
entry:
  %alloca = alloca [2 x [2 x i32]], addrspace(5)
  %gep1 = getelementptr inbounds [2 x [2 x i32]], ptr addrspace(5) %alloca, i32 0, i32 0, i32 1
  store i32 0, ptr addrspace(5) %alloca
  store i32 1, ptr addrspace(5) %gep1
  %gep2 = getelementptr inbounds [2 x [2 x i32]], ptr addrspace(5) %alloca, i32 0, i32 0, i32 %index
  %load = load i32, ptr addrspace(5) %gep2
  store i32 %load, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="64,64" }
attributes #1 = { nounwind optnone noinline "amdgpu-flat-work-group-size"="64,64" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
