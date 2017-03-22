; RUN: llc -march=amdgcn -enable-no-signed-zeros-fp-math=0 < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-SAFE %s
; RUN: llc -march=amdgcn -enable-no-signed-zeros-fp-math=1 < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-UNSAFE %s
; RUN: llc -march=amdgcn -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-UNSAFE %s

; Test that the -enable-no-signed-zeros-fp-math flag works

; GCN-LABEL: {{^}}fneg_fsub_f32:
; GCN: v_subrev_f32_e32 [[SUB:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[SUB]]

; GCN-UNSAFE-NOT: xor
define amdgpu_kernel void @fneg_fsub_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %b_ptr = getelementptr float, float addrspace(1)* %in, i32 1
  %a = load float, float addrspace(1)* %in, align 4
  %b = load float, float addrspace(1)* %b_ptr, align 4
  %result = fsub float %a, %b
  %neg.result = fsub float -0.0, %result
  store float %neg.result, float addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind }
