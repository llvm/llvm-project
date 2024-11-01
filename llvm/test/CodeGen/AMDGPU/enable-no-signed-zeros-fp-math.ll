; RUN: llc -march=amdgcn < %s | FileCheck --check-prefixes=GCN,GCN-FMF,GCN-SAFE %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; Test that the -enable-no-signed-zeros-fp-math flag works

; GCN-LABEL: {{^}}fneg_fsub_f32_fmf:
; GCN: v_sub_f32_e32 [[SUB:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}
; GCN-FMF-NOT: xor
define amdgpu_kernel void @fneg_fsub_f32_fmf(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %add = add i32 %tid, 1
  %gep = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %b_ptr = getelementptr float, ptr addrspace(1) %in, i32 %add
  %a = load float, ptr addrspace(1) %gep, align 4
  %b = load float, ptr addrspace(1) %b_ptr, align 4
  %result = fsub fast float %a, %b
  %neg.result = fsub fast float -0.0, %result
  store float %neg.result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fneg_fsub_f32_safe:
; GCN: v_sub_f32_e32 [[SUB:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[SUB]]
define amdgpu_kernel void @fneg_fsub_f32_safe(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %add = add i32 %tid, 1
  %gep = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %b_ptr = getelementptr float, ptr addrspace(1) %in, i32 %add
  %a = load float, ptr addrspace(1) %gep, align 4
  %b = load float, ptr addrspace(1) %b_ptr, align 4
  %result = fsub float %a, %b
  %neg.result = fsub float -0.0, %result
  store float %neg.result, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind }
