; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}bfe_i32_arg_arg_arg:
; GCN: v_bfe_i32
define amdgpu_kernel void @bfe_i32_arg_arg_arg(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 %src0, i32 %src1, i32 %src1)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_arg_arg_imm:
; GCN: v_bfe_i32
define amdgpu_kernel void @bfe_i32_arg_arg_imm(i32 addrspace(1)* %out, i32 %src0, i32 %src1) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 %src0, i32 %src1, i32 123)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_arg_imm_arg:
; GCN: v_bfe_i32
define amdgpu_kernel void @bfe_i32_arg_imm_arg(i32 addrspace(1)* %out, i32 %src0, i32 %src2) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 %src0, i32 123, i32 %src2)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_imm_arg_arg:
; GCN: v_bfe_i32
define amdgpu_kernel void @bfe_i32_imm_arg_arg(i32 addrspace(1)* %out, i32 %src1, i32 %src2) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 123, i32 %src1, i32 %src2)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_bfe_print_arg:
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 2, 8
define amdgpu_kernel void @v_bfe_print_arg(i32 addrspace(1)* %out, i32 addrspace(1)* %src0) #0 {
  %load = load i32, i32 addrspace(1)* %src0, align 4
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 %load, i32 2, i32 8)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_arg_0_width_reg_offset:
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_arg_0_width_reg_offset(i32 addrspace(1)* %out, i32 %src0, i32 %src1) #0 {
  %bfe_u32 = call i32 @llvm.amdgcn.sbfe.i32(i32 %src0, i32 %src1, i32 0)
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_arg_0_width_imm_offset:
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_arg_0_width_imm_offset(i32 addrspace(1)* %out, i32 %src0, i32 %src1) #0 {
  %bfe_u32 = call i32 @llvm.amdgcn.sbfe.i32(i32 %src0, i32 8, i32 0)
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_test_6:
; GCN: v_lshlrev_b32_e32 v{{[0-9]+}}, 31, v{{[0-9]+}}
; GCN: v_ashrrev_i32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_test_6(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %shl, i32 1, i32 31)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_test_7:
; GCN-NOT: shl
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_test_7(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %shl, i32 0, i32 31)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_test_8:
; GCN: buffer_load_dword
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 1
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_test_8(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %shl, i32 31, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_test_9:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_ashrrev_i32_e32 v{{[0-9]+}}, 31, v{{[0-9]+}}
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_test_9(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %x, i32 31, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_test_10:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_ashrrev_i32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_test_10(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %x, i32 1, i32 31)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_test_11:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_ashrrev_i32_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_test_11(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %x, i32 8, i32 24)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_test_12:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_ashrrev_i32_e32 v{{[0-9]+}}, 24, v{{[0-9]+}}
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_test_12(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %x, i32 24, i32 8)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_test_13:
; GCN: v_ashrrev_i32_e32 {{v[0-9]+}}, 31, {{v[0-9]+}}
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_test_13(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = ashr i32 %x, 31
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %shl, i32 31, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4 ret void
}

; GCN-LABEL: {{^}}bfe_i32_test_14:
; GCN-NOT: lshr
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_test_14(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = lshr i32 %x, 31
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %shl, i32 31, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4 ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_0:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_0(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 0, i32 0, i32 0)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_1:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_1(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 12334, i32 0, i32 0)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_2:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_2(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 0, i32 0, i32 1)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_3:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], -1
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_3(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 1, i32 0, i32 1)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_4:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], -1
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_4(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 4294967295, i32 0, i32 1)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_5:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], -1
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_5(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 128, i32 7, i32 1)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_6:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 0xffffff80
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_6(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 128, i32 0, i32 8)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_7:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 0x7f
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_7(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 127, i32 0, i32 8)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_8:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 1
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_8(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 127, i32 6, i32 8)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_9:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 1
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_9(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 65536, i32 16, i32 8)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_10:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_10(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 65535, i32 16, i32 16)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_11:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], -6
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_11(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 160, i32 4, i32 4)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_12:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_12(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 160, i32 31, i32 1)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_13:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 1
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_13(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 131070, i32 16, i32 16)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_14:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 40
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_14(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 160, i32 2, i32 30)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_15:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 10
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_15(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 160, i32 4, i32 28)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_16:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], -1
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_16(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 4294967295, i32 1, i32 7)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_17:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 0x7f
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_17(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 255, i32 1, i32 31)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_i32_constant_fold_test_18:
; GCN-NOT: {{[^@]}}bfe
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; GCN: buffer_store_dword [[VREG]],
; GCN: s_endpgm
define amdgpu_kernel void @bfe_i32_constant_fold_test_18(i32 addrspace(1)* %out) #0 {
  %bfe_i32 = call i32 @llvm.amdgcn.sbfe.i32(i32 255, i32 31, i32 1)
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_sext_in_reg_i24:
; GCN: buffer_load_dword [[LOAD:v[0-9]+]],
; GCN-NOT: v_lshl
; GCN-NOT: v_ashr
; GCN: v_bfe_i32 [[BFE:v[0-9]+]], [[LOAD]], 0, 24
; GCN: buffer_store_dword [[BFE]],
define amdgpu_kernel void @bfe_sext_in_reg_i24(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %x, i32 0, i32 24)
  %shl = shl i32 %bfe, 8
  %ashr = ashr i32 %shl, 8
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: @simplify_demanded_bfe_sdiv
; GCN: buffer_load_dword [[LOAD:v[0-9]+]]
; GCN: v_bfe_i32 [[BFE:v[0-9]+]], [[LOAD]], 1, 16
; GCN: v_lshrrev_b32_e32 [[TMP0:v[0-9]+]], 31, [[BFE]]
; GCN: v_add_{{[iu]}}32_e32 [[TMP1:v[0-9]+]], vcc, [[TMP0]], [[BFE]]
; GCN: v_ashrrev_i32_e32 [[TMP2:v[0-9]+]], 1, [[TMP1]]
; GCN: buffer_store_dword [[TMP2]]
define amdgpu_kernel void @simplify_demanded_bfe_sdiv(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %src = load i32, i32 addrspace(1)* %in, align 4
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %src, i32 1, i32 16)
  %div = sdiv i32 %bfe, 2
  store i32 %div, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_0_width:
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @bfe_0_width(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %load, i32 8, i32 0)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_8_bfe_8:
; GCN: v_bfe_i32
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @bfe_8_bfe_8(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe0 = call i32 @llvm.amdgcn.sbfe.i32(i32 %load, i32 0, i32 8)
  %bfe1 = call i32 @llvm.amdgcn.sbfe.i32(i32 %bfe0, i32 0, i32 8)
  store i32 %bfe1, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_8_bfe_16:
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 8
; GCN: s_endpgm
define amdgpu_kernel void @bfe_8_bfe_16(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe0 = call i32 @llvm.amdgcn.sbfe.i32(i32 %load, i32 0, i32 8)
  %bfe1 = call i32 @llvm.amdgcn.sbfe.i32(i32 %bfe0, i32 0, i32 16)
  store i32 %bfe1, i32 addrspace(1)* %out, align 4
  ret void
}

; This really should be folded into 1
; GCN-LABEL: {{^}}bfe_16_bfe_8:
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 8
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @bfe_16_bfe_8(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe0 = call i32 @llvm.amdgcn.sbfe.i32(i32 %load, i32 0, i32 16)
  %bfe1 = call i32 @llvm.amdgcn.sbfe.i32(i32 %bfe0, i32 0, i32 8)
  store i32 %bfe1, i32 addrspace(1)* %out, align 4
  ret void
}

; Make sure there isn't a redundant BFE
; GCN-LABEL: {{^}}sext_in_reg_i8_to_i32_bfe:
; GCN: s_sext_i32_i8 s{{[0-9]+}}, s{{[0-9]+}}
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @sext_in_reg_i8_to_i32_bfe(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %c, i32 0, i32 8)
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sext_in_reg_i8_to_i32_bfe_wrong:
define amdgpu_kernel void @sext_in_reg_i8_to_i32_bfe_wrong(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %c, i32 8, i32 0)
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sextload_i8_to_i32_bfe:
; GCN: buffer_load_sbyte
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @sextload_i8_to_i32_bfe(i32 addrspace(1)* %out, i8 addrspace(1)* %ptr) #0 {
  %load = load i8, i8 addrspace(1)* %ptr, align 1
  %sext = sext i8 %load to i32
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %sext, i32 0, i32 8)
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN: .text
; GCN-LABEL: {{^}}sextload_i8_to_i32_bfe_0:{{.*$}}
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @sextload_i8_to_i32_bfe_0(i32 addrspace(1)* %out, i8 addrspace(1)* %ptr) #0 {
  %load = load i8, i8 addrspace(1)* %ptr, align 1
  %sext = sext i8 %load to i32
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %sext, i32 8, i32 0)
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sext_in_reg_i1_bfe_offset_0:
; GCN-NOT: shr
; GCN-NOT: shl
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 1
; GCN: s_endpgm
define amdgpu_kernel void @sext_in_reg_i1_bfe_offset_0(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %shr = ashr i32 %shl, 31
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %shr, i32 0, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sext_in_reg_i1_bfe_offset_1:
; GCN: buffer_load_dword
; GCN-NOT: shl
; GCN-NOT: shr
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 1, 1
; GCN: s_endpgm
define amdgpu_kernel void @sext_in_reg_i1_bfe_offset_1(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 30
  %shr = ashr i32 %shl, 30
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %shr, i32 1, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sext_in_reg_i2_bfe_offset_1:
; GCN: buffer_load_dword
; GCN-NOT: v_lshl
; GCN-NOT: v_ashr
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 2
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 1, 2
; GCN: s_endpgm
define amdgpu_kernel void @sext_in_reg_i2_bfe_offset_1(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 30
  %shr = ashr i32 %shl, 30
  %bfe = call i32 @llvm.amdgcn.sbfe.i32(i32 %shr, i32 1, i32 2)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.sbfe.i32(i32, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
