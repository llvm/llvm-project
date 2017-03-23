; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=FUNC -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=FUNC -check-prefix=GCN %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.bfe.u32(i32, i32, i32) nounwind readnone

; FUNC-LABEL: {{^}}bfe_u32_arg_arg_arg:
; SI: v_bfe_u32
; EG: BFE_UINT
define amdgpu_kernel void @bfe_u32_arg_arg_arg(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 %src0, i32 %src1, i32 %src1) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_arg_arg_imm:
; SI: v_bfe_u32
; EG: BFE_UINT
define amdgpu_kernel void @bfe_u32_arg_arg_imm(i32 addrspace(1)* %out, i32 %src0, i32 %src1) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 %src0, i32 %src1, i32 123) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_arg_imm_arg:
; SI: v_bfe_u32
; EG: BFE_UINT
define amdgpu_kernel void @bfe_u32_arg_imm_arg(i32 addrspace(1)* %out, i32 %src0, i32 %src2) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 %src0, i32 123, i32 %src2) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_imm_arg_arg:
; SI: v_bfe_u32
; EG: BFE_UINT
define amdgpu_kernel void @bfe_u32_imm_arg_arg(i32 addrspace(1)* %out, i32 %src1, i32 %src2) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 123, i32 %src1, i32 %src2) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_arg_0_width_reg_offset:
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_arg_0_width_reg_offset(i32 addrspace(1)* %out, i32 %src0, i32 %src1) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 %src0, i32 %src1, i32 0) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_arg_0_width_imm_offset:
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_arg_0_width_imm_offset(i32 addrspace(1)* %out, i32 %src0, i32 %src1) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 %src0, i32 8, i32 0) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_zextload_i8:
; SI: buffer_load_ubyte
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_zextload_i8(i32 addrspace(1)* %out, i8 addrspace(1)* %in) nounwind {
  %load = load i8, i8 addrspace(1)* %in
  %ext = zext i8 %load to i32
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %ext, i32 0, i32 8)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_zext_in_reg_i8:
; GCN: buffer_load_dword
; SI: v_add_i32
; SI-NEXT: v_and_b32_e32
; FIXME: Should be using s_add_i32
; VI: v_add_i32
; VI-NEXT: v_and_b32_e32
; SI-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define amdgpu_kernel void @bfe_u32_zext_in_reg_i8(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %load = load i32, i32 addrspace(1)* %in, align 4
  %add = add i32 %load, 1
  %ext = and i32 %add, 255
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %ext, i32 0, i32 8)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_zext_in_reg_i16:
; SI: buffer_load_dword
; SI: v_add_i32
; SI-NEXT: v_and_b32_e32
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_zext_in_reg_i16(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %load = load i32, i32 addrspace(1)* %in, align 4
  %add = add i32 %load, 1
  %ext = and i32 %add, 65535
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %ext, i32 0, i32 16)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_zext_in_reg_i8_offset_1:
; SI: buffer_load_dword
; SI: v_add_i32
; SI: bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_zext_in_reg_i8_offset_1(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %load = load i32, i32 addrspace(1)* %in, align 4
  %add = add i32 %load, 1
  %ext = and i32 %add, 255
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %ext, i32 1, i32 8)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_zext_in_reg_i8_offset_3:
; SI: buffer_load_dword
; SI: v_add_i32
; SI-NEXT: v_and_b32_e32 {{v[0-9]+}}, 0xf8
; SI-NEXT: bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_zext_in_reg_i8_offset_3(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %load = load i32, i32 addrspace(1)* %in, align 4
  %add = add i32 %load, 1
  %ext = and i32 %add, 255
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %ext, i32 3, i32 8)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_zext_in_reg_i8_offset_7:
; SI: buffer_load_dword
; SI: v_add_i32
; SI-NEXT: v_and_b32_e32 {{v[0-9]+}}, 0x80
; SI-NEXT: bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_zext_in_reg_i8_offset_7(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %load = load i32, i32 addrspace(1)* %in, align 4
  %add = add i32 %load, 1
  %ext = and i32 %add, 255
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %ext, i32 7, i32 8)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_zext_in_reg_i16_offset_8:
; SI: buffer_load_dword
; SI: v_add_i32
; SI-NEXT: bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_zext_in_reg_i16_offset_8(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %load = load i32, i32 addrspace(1)* %in, align 4
  %add = add i32 %load, 1
  %ext = and i32 %add, 65535
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %ext, i32 8, i32 8)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_test_1:
; SI: buffer_load_dword
; SI: v_and_b32_e32 {{v[0-9]+}}, 1, {{v[0-9]+}}
; SI: s_endpgm
; EG: AND_INT T{{[0-9]\.[XYZW]}}, T{{[0-9]\.[XYZW]}}, 1,
define amdgpu_kernel void @bfe_u32_test_1(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %x, i32 0, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

define amdgpu_kernel void @bfe_u32_test_2(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %shl, i32 0, i32 8)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

define amdgpu_kernel void @bfe_u32_test_3(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %shl, i32 0, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_test_4:
; SI-NOT: lshl
; SI-NOT: shr
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_test_4(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %shr = lshr i32 %shl, 31
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %shr, i32 31, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_test_5:
; SI: buffer_load_dword
; SI-NOT: lshl
; SI-NOT: shr
; SI: v_bfe_i32 {{v[0-9]+}}, {{v[0-9]+}}, 0, 1
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_test_5(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %shr = ashr i32 %shl, 31
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %shr, i32 0, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_test_6:
; SI: v_lshlrev_b32_e32 v{{[0-9]+}}, 31, v{{[0-9]+}}
; SI: v_lshrrev_b32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_test_6(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %shl, i32 1, i32 31)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_test_7:
; SI: v_lshlrev_b32_e32 v{{[0-9]+}}, 31, v{{[0-9]+}}
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_test_7(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %shl, i32 0, i32 31)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_test_8:
; SI-NOT: {{[^@]}}bfe
; SI: v_and_b32_e32 {{v[0-9]+}}, 1, {{v[0-9]+}}
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_test_8(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %shl, i32 31, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_test_9:
; SI-NOT: {{[^@]}}bfe
; SI: v_lshrrev_b32_e32 v{{[0-9]+}}, 31, v{{[0-9]+}}
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_test_9(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %x, i32 31, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_test_10:
; SI-NOT: {{[^@]}}bfe
; SI: v_lshrrev_b32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_test_10(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %x, i32 1, i32 31)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_test_11:
; SI-NOT: {{[^@]}}bfe
; SI: v_lshrrev_b32_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_test_11(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %x, i32 8, i32 24)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_test_12:
; SI-NOT: {{[^@]}}bfe
; SI: v_lshrrev_b32_e32 v{{[0-9]+}}, 24, v{{[0-9]+}}
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_test_12(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %x, i32 24, i32 8)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_test_13:
; V_ASHRREV_U32_e32 {{v[0-9]+}}, 31, {{v[0-9]+}}
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_test_13(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = ashr i32 %x, 31
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %shl, i32 31, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4 ret void
}

; FUNC-LABEL: {{^}}bfe_u32_test_14:
; SI-NOT: lshr
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define amdgpu_kernel void @bfe_u32_test_14(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = lshr i32 %x, 31
  %bfe = call i32 @llvm.AMDGPU.bfe.u32(i32 %shl, i32 31, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4 ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_0:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_0(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 0, i32 0, i32 0) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_1:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_1(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 12334, i32 0, i32 0) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_2:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_2(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 0, i32 0, i32 1) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_3:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 1
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_3(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 1, i32 0, i32 1) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_4:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], -1
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_4(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 4294967295, i32 0, i32 1) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_5:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 1
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_5(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 128, i32 7, i32 1) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_6:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 0x80
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_6(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 128, i32 0, i32 8) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_7:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 0x7f
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_7(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 127, i32 0, i32 8) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_8:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 1
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_8(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 127, i32 6, i32 8) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_9:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 1
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_9(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 65536, i32 16, i32 8) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_10:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_10(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 65535, i32 16, i32 16) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_11:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 10
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_11(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 160, i32 4, i32 4) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_12:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_12(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 160, i32 31, i32 1) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_13:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 1
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_13(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 131070, i32 16, i32 16) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_14:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 40
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_14(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 160, i32 2, i32 30) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_15:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 10
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_15(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 160, i32 4, i32 28) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_16:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 0x7f
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_16(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 4294967295, i32 1, i32 7) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_17:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 0x7f
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_17(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 255, i32 1, i32 31) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_u32_constant_fold_test_18:
; SI-NOT: {{[^@]}}bfe
; SI: v_mov_b32_e32 [[VREG:v[0-9]+]], 0
; SI: buffer_store_dword [[VREG]],
; SI: s_endpgm
; EG-NOT: BFE
define amdgpu_kernel void @bfe_u32_constant_fold_test_18(i32 addrspace(1)* %out) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 255, i32 31, i32 1) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; Make sure that SimplifyDemandedBits doesn't cause the and to be
; reduced to the bits demanded by the bfe.

; XXX: The operand to v_bfe_u32 could also just directly be the load register.
; FUNC-LABEL: {{^}}simplify_bfe_u32_multi_use_arg:
; SI: buffer_load_dword [[ARG:v[0-9]+]]
; SI: v_and_b32_e32 [[AND:v[0-9]+]], 63, [[ARG]]
; SI: v_bfe_u32 [[BFE:v[0-9]+]], [[AND]], 2, 2
; SI-DAG: buffer_store_dword [[AND]]
; SI-DAG: buffer_store_dword [[BFE]]
; SI: s_endpgm
define amdgpu_kernel void @simplify_bfe_u32_multi_use_arg(i32 addrspace(1)* %out0,
                                            i32 addrspace(1)* %out1,
                                            i32 addrspace(1)* %in) nounwind {
  %src = load i32, i32 addrspace(1)* %in, align 4
  %and = and i32 %src, 63
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 %and, i32 2, i32 2) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out0, align 4
  store i32 %and, i32 addrspace(1)* %out1, align 4
  ret void
}

; FUNC-LABEL: {{^}}lshr_and:
; SI: s_bfe_u32 {{s[0-9]+}}, {{s[0-9]+}}, 0x30006
; SI: buffer_store_dword
define amdgpu_kernel void @lshr_and(i32 addrspace(1)* %out, i32 %a) nounwind {
  %b = lshr i32 %a, 6
  %c = and i32 %b, 7
  store i32 %c, i32 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_lshr_and:
; SI: v_bfe_u32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}, 3
; SI: buffer_store_dword
define amdgpu_kernel void @v_lshr_and(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %c = lshr i32 %a, %b
  %d = and i32 %c, 7
  store i32 %d, i32 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}and_lshr:
; SI: s_bfe_u32 {{s[0-9]+}}, {{s[0-9]+}}, 0x30006
; SI: buffer_store_dword
define amdgpu_kernel void @and_lshr(i32 addrspace(1)* %out, i32 %a) nounwind {
  %b = and i32 %a, 448
  %c = lshr i32 %b, 6
  store i32 %c, i32 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}and_lshr2:
; SI: s_bfe_u32 {{s[0-9]+}}, {{s[0-9]+}}, 0x30006
; SI: buffer_store_dword
define amdgpu_kernel void @and_lshr2(i32 addrspace(1)* %out, i32 %a) nounwind {
  %b = and i32 %a, 511
  %c = lshr i32 %b, 6
  store i32 %c, i32 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}shl_lshr:
; SI: s_bfe_u32 {{s[0-9]+}}, {{s[0-9]+}}, 0x150002
; SI: buffer_store_dword
define amdgpu_kernel void @shl_lshr(i32 addrspace(1)* %out, i32 %a) nounwind {
  %b = shl i32 %a, 9
  %c = lshr i32 %b, 11
  store i32 %c, i32 addrspace(1)* %out, align 8
  ret void
}
