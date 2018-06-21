; RUN: llc -O0 -march=amdgcn -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -mattr=+promote-alloca < %s | FileCheck -check-prefix=NOOPTS -check-prefix=ALL %s
; RUN: llc -O1 -march=amdgcn -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -mattr=+promote-alloca < %s | FileCheck -check-prefix=OPTS -check-prefix=ALL %s

; ALL-LABEL: {{^}}promote_alloca_i32_array_array:
; NOOPTS: workgroup_group_segment_byte_size = 0{{$}}
; NOOPTS-NOT ds_write
; OPTS: ds_write
define amdgpu_kernel void @promote_alloca_i32_array_array(i32 addrspace(1)* %out, i32 %index) #0 {
entry:
  %alloca = alloca [2 x [2 x i32]]
  %gep0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %alloca, i32 0, i32 0, i32 0
  %gep1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %alloca, i32 0, i32 0, i32 1
  store i32 0, i32* %gep0
  store i32 1, i32* %gep1
  %gep2 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %alloca, i32 0, i32 0, i32 %index
  %load = load i32, i32* %gep2
  store i32 %load, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}optnone_promote_alloca_i32_array_array:
; ALL: workgroup_group_segment_byte_size = 0{{$}}
; ALL-NOT ds_write
define amdgpu_kernel void @optnone_promote_alloca_i32_array_array(i32 addrspace(1)* %out, i32 %index) #1 {
entry:
  %alloca = alloca [2 x [2 x i32]]
  %gep0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %alloca, i32 0, i32 0, i32 0
  %gep1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %alloca, i32 0, i32 0, i32 1
  store i32 0, i32* %gep0
  store i32 1, i32* %gep1
  %gep2 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %alloca, i32 0, i32 0, i32 %index
  %load = load i32, i32* %gep2
  store i32 %load, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="64,64" }
attributes #1 = { nounwind optnone noinline "amdgpu-flat-work-group-size"="64,64" }
