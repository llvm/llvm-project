; RUN: llc -mtriple=amdgpu12.00-amd-amdpal -filetype=null < %s 2>&1 | FileCheck %s

; CHECK: warning: {{.*}}dynamic VGPR entry point vector registers (21) exceeds limit (16) in function 'entry_block16'
define amdgpu_cs void @entry_block16() #0 {
  call void asm sideeffect "", "~{v20}"()
  ret void
}

; CHECK: warning: {{.*}}dynamic VGPR entry point vector registers (21) exceeds limit (16) in function 'kernel_block16'
define amdgpu_kernel void @kernel_block16() #0 {
  call void asm sideeffect "", "~{v20}"()
  ret void
}

; CHECK-NOT: warning{{.*}}'entry_block32'
define amdgpu_cs void @entry_block32() #1 {
  call void asm sideeffect "", "~{v20}"()
  ret void
}

; CHECK-NOT: warning{{.*}}'chain_block16'
define amdgpu_cs_chain void @chain_block16() #0 {
  call void asm sideeffect "", "~{v20}"()
  unreachable
}

define amdgpu_gfx void @gfx_func_high_pressure() #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"()
  ret void
}

; CHECK: warning: {{.*}}dynamic VGPR entry point vector registers (40) exceeds limit (16) in function 'entry_calls_high_pressure'
define amdgpu_cs void @entry_calls_high_pressure() #0 {
  call amdgpu_gfx void @gfx_func_high_pressure()
  ret void
}

attributes #0 = { nounwind "amdgpu-dynamic-vgpr-block-size"="16" }
attributes #1 = { nounwind "amdgpu-dynamic-vgpr-block-size"="32" }
