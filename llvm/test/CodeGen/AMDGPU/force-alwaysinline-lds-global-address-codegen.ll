; RUN: llc -mtriple=amdgcn-amd-amdhsa -amdgpu-function-calls -amdgpu-stress-function-calls < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -amdgpu-stress-function-calls < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa < %s | FileCheck -check-prefix=GCN %s

@lds0 = addrspace(3) global i32 undef, align 4

; GCN-NOT: load_lds_simple

define internal i32 @load_lds_simple() {
  %load = load i32, i32 addrspace(3)* @lds0, align 4
  ret i32 %load
}

; GCN-LABEL: {{^}}kernel:
; GCN: v_mov_b32_e32 [[ADDR:v[0-9]+]], 0
; GCN: ds_read_b32 v{{[0-9]+}}, [[ADDR]]
define amdgpu_kernel void @kernel(i32 addrspace(1)* %out) {
  %call = call i32 @load_lds_simple()
  store i32 %call, i32 addrspace(1)* %out
  ret void
}
