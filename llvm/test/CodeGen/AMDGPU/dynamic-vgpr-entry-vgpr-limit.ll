; RUN: not llc -mtriple=amdgpu12.00-amd-amdpal -filetype=null < %s 2>&1 | FileCheck %s

; CHECK: error: {{.*}}dynamic VGPR entry point vector registers (21) exceeds limit (16) in function 'entry_block16'
define amdgpu_cs void @entry_block16() #0 {
  call void asm sideeffect "", "~{v20}"()
  ret void
}

; CHECK: error: {{.*}}dynamic VGPR entry point vector registers (21) exceeds limit (16) in function 'kernel_block16'
define amdgpu_kernel void @kernel_block16() #0 {
  call void asm sideeffect "", "~{v20}"()
  ret void
}

; CHECK-NOT: error{{.*}}'entry_block32'
define amdgpu_cs void @entry_block32() #1 {
  call void asm sideeffect "", "~{v20}"()
  ret void
}

; CHECK-NOT: error{{.*}}'chain_block16'
define amdgpu_cs_chain void @chain_block16() #0 {
  call void asm sideeffect "", "~{v20}"()
  unreachable
}

attributes #0 = { nounwind "amdgpu-dynamic-vgpr-block-size"="16" }
attributes #1 = { nounwind "amdgpu-dynamic-vgpr-block-size"="32" }
