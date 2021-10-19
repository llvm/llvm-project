; RUN: llc < %s -march=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck %s --check-prefix=GCN
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}full_mask:
; GCN: s_mov_b64 exec, -1
; GCN: v_add_f32_e32 v0,
define amdgpu_ps float @full_mask(float %a, float %b) {
main_body:
  %s = fadd float %a, %b
  call void @llvm.amdgcn.init.exec(i64 -1)
  ret float %s
}

; GCN-LABEL: {{^}}partial_mask:
; GCN: s_mov_b64 exec, 0x1e240
; GCN: v_add_f32_e32 v0,
define amdgpu_ps float @partial_mask(float %a, float %b) {
main_body:
  %s = fadd float %a, %b
  call void @llvm.amdgcn.init.exec(i64 123456)
  ret float %s
}

; GCN-LABEL: {{^}}input_s3off8:
; GCN: s_bfe_u32 s0, s3, 0x70008
; GCN: s_bfm_b64 exec, s0, 0
; GCN: s_cmp_eq_u32 s0, 64
; GCN: s_cmov_b64 exec, -1
; GCN: v_add_f32_e32 v0,
define amdgpu_ps float @input_s3off8(i32 inreg, i32 inreg, i32 inreg, i32 inreg %count, float %a, float %b) {
main_body:
  %s = fadd float %a, %b
  call void @llvm.amdgcn.init.exec.from.input(i32 %count, i32 8)
  ret float %s
}

; GCN-LABEL: {{^}}input_s0off19:
; GCN: s_bfe_u32 s0, s0, 0x70013
; GCN: s_bfm_b64 exec, s0, 0
; GCN: s_cmp_eq_u32 s0, 64
; GCN: s_cmov_b64 exec, -1
; GCN: v_add_f32_e32 v0,
define amdgpu_ps float @input_s0off19(i32 inreg %count, float %a, float %b) {
main_body:
  %s = fadd float %a, %b
  call void @llvm.amdgcn.init.exec.from.input(i32 %count, i32 19)
  ret float %s
}

; GCN-LABEL: {{^}}reuse_input:
; GCN: s_bfe_u32 s1, s0, 0x70013
; GCN: s_bfm_b64 exec, s1, 0
; GCN: s_cmp_eq_u32 s1, 64
; GCN: s_cmov_b64 exec, -1
; GCN: v_add{{(_nc)?}}_u32_e32 v0, s0, v0
define amdgpu_ps float @reuse_input(i32 inreg %count, i32 %a) {
main_body:
  call void @llvm.amdgcn.init.exec.from.input(i32 %count, i32 19)
  %s = add i32 %a, %count
  %f = sitofp i32 %s to float
  ret float %f
}

; GCN-LABEL: {{^}}reuse_input2:
; GCN: s_bfe_u32 s1, s0, 0x70013
; GCN: s_bfm_b64 exec, s1, 0
; GCN: s_cmp_eq_u32 s1, 64
; GCN: s_cmov_b64 exec, -1
; GCN: v_add{{(_nc)?}}_u32_e32 v0, s0, v0
define amdgpu_ps float @reuse_input2(i32 inreg %count, i32 %a) {
main_body:
  %s = add i32 %a, %count
  %f = sitofp i32 %s to float
  call void @llvm.amdgcn.init.exec.from.input(i32 %count, i32 19)
  ret float %f
}

; GCN-LABEL: {{^}}init_unreachable:
;
; This used to crash.
define amdgpu_ps void @init_unreachable() {
main_body:
  call void @llvm.amdgcn.init.exec(i64 -1)
  unreachable
}

; GCN-LABEL: {{^}}init_exec_before_frame_materialize:
; GCN-NOT: {{^}}v_
; GCN: s_mov_b64 exec, -1
; GCN: v_mov
; GCN: v_add
define amdgpu_ps float @init_exec_before_frame_materialize(i32 inreg %a, i32 inreg %b) {
main_body:
  %array0 = alloca [1024 x i32], align 16, addrspace(5)
  %array1 = alloca [20 x i32], align 16, addrspace(5)
  call void @llvm.amdgcn.init.exec(i64 -1)

  %ptr0 = getelementptr inbounds [1024 x i32], [1024 x i32] addrspace(5)* %array0, i32 0, i32 1
  store i32 %a, i32 addrspace(5)* %ptr0, align 4

  %ptr1 = getelementptr inbounds [20 x i32], [20 x i32] addrspace(5)* %array1, i32 0, i32 1
  store i32 %a, i32 addrspace(5)* %ptr1, align 4

  %ptr2 = getelementptr inbounds [20 x i32], [20 x i32] addrspace(5)* %array1, i32 0, i32 2
  store i32 %b, i32 addrspace(5)* %ptr2, align 4

  %ptr3 = getelementptr inbounds [20 x i32], [20 x i32] addrspace(5)* %array1, i32 0, i32 %b
  %v3 = load i32, i32 addrspace(5)* %ptr3, align 4

  %ptr4 = getelementptr inbounds [1024 x i32], [1024 x i32] addrspace(5)* %array0, i32 0, i32 %b
  %v4 = load i32, i32 addrspace(5)* %ptr4, align 4

  %v5 = add i32 %v3, %v4
  %v = bitcast i32 %v5 to float
  ret float %v
}

; GCN-LABEL: {{^}}init_exec_input_before_frame_materialize:
; GCN-NOT: {{^}}v_
; GCN: s_bfe_u32 s2, s2, 0x70008
; GCN-NEXT: s_bfm_b64 exec, s2, 0
; GCN-NEXT: s_cmp_eq_u32 s2, 64
; GCN-NEXT: s_cmov_b64 exec, -1
; GCN: v_mov
; GCN: v_add
define amdgpu_ps float @init_exec_input_before_frame_materialize(i32 inreg %a, i32 inreg %b, i32 inreg %count) {
main_body:
  %array0 = alloca [1024 x i32], align 16, addrspace(5)
  %array1 = alloca [20 x i32], align 16, addrspace(5)
  call void @llvm.amdgcn.init.exec.from.input(i32 %count, i32 8)

  %ptr0 = getelementptr inbounds [1024 x i32], [1024 x i32] addrspace(5)* %array0, i32 0, i32 1
  store i32 %a, i32 addrspace(5)* %ptr0, align 4

  %ptr1 = getelementptr inbounds [20 x i32], [20 x i32] addrspace(5)* %array1, i32 0, i32 1
  store i32 %a, i32 addrspace(5)* %ptr1, align 4

  %ptr2 = getelementptr inbounds [20 x i32], [20 x i32] addrspace(5)* %array1, i32 0, i32 2
  store i32 %b, i32 addrspace(5)* %ptr2, align 4

  %ptr3 = getelementptr inbounds [20 x i32], [20 x i32] addrspace(5)* %array1, i32 0, i32 %b
  %v3 = load i32, i32 addrspace(5)* %ptr3, align 4

  %ptr4 = getelementptr inbounds [1024 x i32], [1024 x i32] addrspace(5)* %array0, i32 0, i32 %b
  %v4 = load i32, i32 addrspace(5)* %ptr4, align 4

  %v5 = add i32 %v3, %v4
  %v = bitcast i32 %v5 to float
  ret float %v
}

; GCN-LABEL: {{^}}init_exec_input_before_frame_materialize_nonentry:
; GCN-NOT: {{^}}v_
; GCN: %endif
; GCN: s_bfe_u32 s3, s2, 0x70008
; GCN-NEXT: s_bfm_b64 exec, s3, 0
; GCN-NEXT: s_cmp_eq_u32 s3, 64
; GCN-NEXT: s_cmov_b64 exec, -1
; GCN: v_mov
; GCN: v_add
define amdgpu_ps float @init_exec_input_before_frame_materialize_nonentry(i32 inreg %a, i32 inreg %b, i32 inreg %count) {
main_body:
  ; ideally these alloca would be in %endif, but this causes problems on Windows GlobalISel
  %array0 = alloca [1024 x i32], align 16, addrspace(5)
  %array1 = alloca [20 x i32], align 16, addrspace(5)

  %cc = icmp uge i32 %count, 32
  br i1 %cc, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  call void @llvm.amdgcn.init.exec.from.input(i32 %count, i32 8)

  %ptr0 = getelementptr inbounds [1024 x i32], [1024 x i32] addrspace(5)* %array0, i32 0, i32 1
  store i32 %a, i32 addrspace(5)* %ptr0, align 4

  %ptr1 = getelementptr inbounds [20 x i32], [20 x i32] addrspace(5)* %array1, i32 0, i32 1
  store i32 %a, i32 addrspace(5)* %ptr1, align 4

  %ptr2 = getelementptr inbounds [20 x i32], [20 x i32] addrspace(5)* %array1, i32 0, i32 2
  store i32 %b, i32 addrspace(5)* %ptr2, align 4

  %ptr3 = getelementptr inbounds [20 x i32], [20 x i32] addrspace(5)* %array1, i32 0, i32 %b
  %v3 = load i32, i32 addrspace(5)* %ptr3, align 4

  %ptr4 = getelementptr inbounds [1024 x i32], [1024 x i32] addrspace(5)* %array0, i32 0, i32 %b
  %v4 = load i32, i32 addrspace(5)* %ptr4, align 4

  %v5 = add i32 %v3, %v4
  %v6 = add i32 %v5, %count
  %v = bitcast i32 %v6 to float
  ret float %v
}

declare void @llvm.amdgcn.init.exec(i64) #1
declare void @llvm.amdgcn.init.exec.from.input(i32, i32) #1

attributes #1 = { convergent }
