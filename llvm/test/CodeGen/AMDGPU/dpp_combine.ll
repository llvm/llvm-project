; RUN: llc -march=amdgcn -mcpu=tonga -amdgpu-dpp-combine -verify-machineinstrs < %s | FileCheck %s

; VOP2 with literal cannot be combined
; CHECK-LABEL: {{^}}dpp_combine_i32_literal:
; CHECK: v_mov_b32_dpp [[OLD:v[0-9]+]], {{v[0-9]+}} quad_perm:[1,0,0,0] row_mask:0x2 bank_mask:0x1 bound_ctrl:0
; CHECK: v_add_u32_e32 {{v[0-9]+}}, vcc, 42, [[OLD]]
define amdgpu_kernel void @dpp_combine_i32_literal(i32 addrspace(1)* %out, i32 %in) {
  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 undef, i32 %in, i32 1, i32 2, i32 1, i1 1) #0
  %res = add nsw i32 %dpp, 42
  store i32 %res, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}dpp_combine_i32_bz:
; CHECK: v_add_u32_dpp {{v[0-9]+}}, vcc, {{v[0-9]+}}, v0  quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:0
define amdgpu_kernel void @dpp_combine_i32_bz(i32 addrspace(1)* %out, i32 %in) {
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 undef, i32 %in, i32 1, i32 1, i32 1, i1 1) #0
  %res = add nsw i32 %dpp, %x
  store i32 %res, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}dpp_combine_i32_boff_undef:
; CHECK: v_add_u32_dpp {{v[0-9]+}}, vcc, {{v[0-9]+}}, v0  quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1
define amdgpu_kernel void @dpp_combine_i32_boff_undef(i32 addrspace(1)* %out, i32 %in) {
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 undef, i32 %in, i32 1, i32 1, i32 1, i1 0) #0
  %res = add nsw i32 %dpp, %x
  store i32 %res, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}dpp_combine_i32_boff_0:
; CHECK: v_add_u32_dpp {{v[0-9]+}}, vcc, {{v[0-9]+}}, v0  quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:0
define amdgpu_kernel void @dpp_combine_i32_boff_0(i32 addrspace(1)* %out, i32 %in) {
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %in, i32 1, i32 1, i32 1, i1 0) #0
  %res = add nsw i32 %dpp, %x
  store i32 %res, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}dpp_combine_i32_boff_max:
; CHECK: v_bfrev_b32_e32 [[OLD:v[0-9]+]], -2
; CHECK: v_max_i32_dpp [[OLD]], {{v[0-9]+}}, v0  quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1
define amdgpu_kernel void @dpp_combine_i32_boff_max(i32 addrspace(1)* %out, i32 %in) {
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 2147483647, i32 %in, i32 1, i32 1, i32 1, i1 0) #0
  %cmp = icmp sge i32 %dpp, %x
  %res = select i1 %cmp, i32 %dpp, i32 %x
  store i32 %res, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}dpp_combine_i32_boff_min:
; CHECK: v_bfrev_b32_e32 [[OLD:v[0-9]+]], 1
; CHECK: v_min_i32_dpp [[OLD]], {{v[0-9]+}}, v0  quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1
define amdgpu_kernel void @dpp_combine_i32_boff_min(i32 addrspace(1)* %out, i32 %in) {
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 -2147483648, i32 %in, i32 1, i32 1, i32 1, i1 0) #0
  %cmp = icmp sle i32 %dpp, %x
  %res = select i1 %cmp, i32 %dpp, i32 %x
  store i32 %res, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}dpp_combine_i32_boff_mul:
; CHECK: v_mul_i32_i24_dpp v0, v3, v0  quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1
define amdgpu_kernel void @dpp_combine_i32_boff_mul(i32 addrspace(1)* %out, i32 %in) {
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 1, i32 %in, i32 1, i32 1, i32 1, i1 0) #0

  %dpp.shl = shl i32 %dpp, 8
  %dpp.24 = ashr i32 %dpp.shl, 8
  %x.shl = shl i32 %x, 8
  %x.24 = ashr i32 %x.shl, 8
  %res = mul i32 %dpp.24, %x.24
  store i32 %res, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}dpp_combine_i32_commute:
; CHECK: v_subrev_u32_dpp {{v[0-9]+}}, vcc, {{v[0-9]+}}, v0  quad_perm:[2,0,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:0
define amdgpu_kernel void @dpp_combine_i32_commute(i32 addrspace(1)* %out, i32 %in) {
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 undef, i32 %in, i32 2, i32 1, i32 1, i1 1) #0
  %res = sub nsw i32 %x, %dpp
  store i32 %res, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}dpp_combine_f32:
; CHECK: v_add_f32_dpp {{v[0-9]+}}, {{v[0-9]+}}, v0  quad_perm:[3,0,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:0
define amdgpu_kernel void @dpp_combine_f32(i32 addrspace(1)* %out, i32 %in) {
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()

  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 undef, i32 %in, i32 3, i32 1, i32 1, i1 1) #0
  %dpp.f32 = bitcast i32 %dpp to float
  %x.f32 = bitcast i32 %x to float
  %res.f32 = fadd float %x.f32, %dpp.f32
  %res = bitcast float %res.f32 to i32
  store i32 %res, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}dpp_combine_test_f32_mods:
; CHECK: v_mul_f32_dpp {{v[0-9]+}}, |{{v[0-9]+}}|, -v0  quad_perm:[0,1,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:0
define amdgpu_kernel void @dpp_combine_test_f32_mods(i32 addrspace(1)* %out, i32 %in) {
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()

  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 undef, i32 %in, i32 4, i32 1, i32 1, i1 1) #0

  %x.f32 = bitcast i32 %x to float
  %x.f32.neg = fsub float -0.000000e+00, %x.f32

  %dpp.f32 = bitcast i32 %dpp to float
  %dpp.f32.cmp = fcmp fast olt float %dpp.f32, 0.000000e+00
  %dpp.f32.sign = select i1 %dpp.f32.cmp, float -1.000000e+00, float 1.000000e+00
  %dpp.f32.abs = fmul fast float %dpp.f32, %dpp.f32.sign

  %res.f32 = fmul float %x.f32.neg, %dpp.f32.abs
  %res = bitcast float %res.f32 to i32
  store i32 %res, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}dpp_combine_mac:
; CHECK: v_mac_f32_dpp v0, {{v[0-9]+}}, v1  quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:0
define amdgpu_kernel void @dpp_combine_mac(float addrspace(1)* %out, i32 %in) {
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 undef, i32 %in, i32 1, i32 1, i32 1, i1 1) #0
  %dpp.f32 = bitcast i32 %dpp to float
  %x.f32 = bitcast i32 %x to float
  %y.f32 = bitcast i32 %y to float

  %mult = fmul float %dpp.f32, %y.f32
  %res = fadd float %mult, %x.f32
  store float %res, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}dpp_combine_sequence:
define amdgpu_kernel void @dpp_combine_sequence(i32 addrspace(1)* %out, i32 %in, i1 %cmp) {
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 undef, i32 %in, i32 1, i32 1, i32 1, i1 1) #0
  br i1 %cmp, label %bb1, label %bb2
bb1:
; CHECK: v_add_u32_dpp {{v[0-9]+}}, vcc, {{v[0-9]+}}, v0  quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:0
  %resadd = add nsw i32 %dpp, %x
  br label %bb3
bb2:
; CHECK: v_subrev_u32_dpp {{v[0-9]+}}, vcc, {{v[0-9]+}}, v0  quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:0
  %ressub = sub nsw i32 %x, %dpp
  br label %bb3
bb3:
  %res = phi i32 [%resadd, %bb1], [%ressub, %bb2]
  store i32 %res, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}dpp_combine_sequence_negative:
; CHECK: v_mov_b32_dpp v1, v1  quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:0
define amdgpu_kernel void @dpp_combine_sequence_negative(i32 addrspace(1)* %out, i32 %in, i1 %cmp) {
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 undef, i32 %in, i32 1, i32 1, i32 1, i1 1) #0
  br i1 %cmp, label %bb1, label %bb2
bb1:
  %resadd = add nsw i32 %dpp, %x
  br label %bb3
bb2:
  %ressub = sub nsw i32 2, %dpp ; break seq
  br label %bb3
bb3:
  %res = phi i32 [%resadd, %bb1], [%ressub, %bb2]
  store i32 %res, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workitem.id.y()
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1) #0

attributes #0 = { nounwind readnone convergent }
