; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI,MUBUF %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX9-MUBUF,MUBUF %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-promote-alloca,+enable-flat-scratch -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX9-FLATSCR %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s | FileCheck --check-prefixes=GFX11 %s

; Test that non-entry function frame indices are expanded properly to
; give an index relative to the scratch wave offset register

; Materialize into a mov. Make sure there isn't an unnecessary copy.
; GCN-LABEL: {{^}}func_mov_fi_i32:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)

; CI-NEXT: v_lshr_b32_e64 v0, s32, 6
; GFX9-MUBUF-NEXT: v_lshrrev_b32_e64 v0, 6, s32

; GFX9-FLATSCR:     v_mov_b32_e32 v0, s32
; GFX9-FLATSCR-NOT: v_lshrrev_b32_e64

; MUBUF-NOT: v_mov

; GCN: ds_write_b32 v0, v0
define void @func_mov_fi_i32() #0 {
  %alloca = alloca i32, addrspace(5)
  store volatile ptr addrspace(5) %alloca, ptr addrspace(3) undef
  ret void
}

; Offset due to different objects
; GCN-LABEL: {{^}}func_mov_fi_i32_offset:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)

; CI-DAG: v_lshr_b32_e64 v0, s32, 6
; CI-NOT: v_mov
; CI: ds_write_b32 v0, v0
; CI-NEXT: v_lshr_b32_e64 [[SCALED:v[0-9]+]], s32, 6
; CI-NEXT: v_add_i32_e{{32|64}} v0, {{s\[[0-9]+:[0-9]+\]|vcc}}, 4, [[SCALED]]
; CI-NEXT: ds_write_b32 v0, v0

; GFX9-MUBUF-NEXT:   v_lshrrev_b32_e64 v0, 6, s32
; GFX9-FLATSCR:      v_mov_b32_e32 v0, s32
; GFX9-FLATSCR:      s_add_i32 [[ADD:[^,]+]], s32, 4
; GFX9-NEXT:         ds_write_b32 v0, v0
; GFX9-MUBUF-NEXT:   v_lshrrev_b32_e64 [[SCALED:v[0-9]+]], 6, s32
; GFX9-MUBUF-NEXT:   v_add_u32_e32 v0, 4, [[SCALED]]
; GFX9-FLATSCR-NEXT: v_mov_b32_e32 v0, [[ADD]]
; GFX9-NEXT:         ds_write_b32 v0, v0
define void @func_mov_fi_i32_offset() #0 {
  %alloca0 = alloca i32, addrspace(5)
  %alloca1 = alloca i32, addrspace(5)
  store volatile ptr addrspace(5) %alloca0, ptr addrspace(3) undef
  store volatile ptr addrspace(5) %alloca1, ptr addrspace(3) undef
  ret void
}

; Materialize into an add of a constant offset from the FI.
; FIXME: Should be able to merge adds

; GCN-LABEL: {{^}}func_add_constant_to_fi_i32:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)

; CI: v_lshr_b32_e64 [[SCALED:v[0-9]+]], s32, 6
; CI-NEXT: v_add_i32_e32 v0, vcc, 4, [[SCALED]]

; GFX9-MUBUF:       v_lshrrev_b32_e64 [[SCALED:v[0-9]+]], 6, s32
; GFX9-MUBUF-NEXT:  v_add_u32_e32 v0, 4, [[SCALED]]

; FIXME: Should commute and shrink
; GFX9-FLATSCR: v_add_u32_e64 v0, 4, s32

; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @func_add_constant_to_fi_i32() #0 {
  %alloca = alloca [2 x i32], align 4, addrspace(5)
  %gep0 = getelementptr inbounds [2 x i32], ptr addrspace(5) %alloca, i32 0, i32 1
  store volatile ptr addrspace(5) %gep0, ptr addrspace(3) undef
  ret void
}

; A user the materialized frame index can't be meaningfully folded
; into.
; FIXME: Should use s_mul but the frame index always gets materialized into a
; vgpr

; GCN-LABEL: {{^}}func_other_fi_user_i32:
; MUBUF: s_lshr_b32 [[SCALED:s[0-9]+]], s32, 6
; MUBUF: s_mul_i32 [[MUL:s[0-9]+]], [[SCALED]], 9
; MUBUF: v_mov_b32_e32 v0, [[MUL]]

; GFX9-FLATSCR: s_mul_i32 [[MUL:s[0-9]+]], s32, 9
; GFX9-FLATSCR: v_mov_b32_e32 v0, [[MUL]]

; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @func_other_fi_user_i32() #0 {
  %alloca = alloca [2 x i32], align 4, addrspace(5)
  %ptrtoint = ptrtoint ptr addrspace(5) %alloca to i32
  %mul = mul i32 %ptrtoint, 9
  store volatile i32 %mul, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}func_store_private_arg_i32_ptr:
; GCN: v_mov_b32_e32 v1, 15{{$}}
; MUBUF:        buffer_store_dword v1, v0, s[0:3], 0 offen{{$}}
; GFX9-FLATSCR: scratch_store_dword v0, v1, off{{$}}
define void @func_store_private_arg_i32_ptr(ptr addrspace(5) %ptr) #0 {
  store volatile i32 15, ptr addrspace(5) %ptr
  ret void
}

; GCN-LABEL: {{^}}func_load_private_arg_i32_ptr:
; GCN: s_waitcnt
; MUBUF-NEXT:        buffer_load_dword v0, v0, s[0:3], 0 offen glc{{$}}
; GFX9-FLATSCR-NEXT: scratch_load_dword v0, v0, off glc{{$}}
define void @func_load_private_arg_i32_ptr(ptr addrspace(5) %ptr) #0 {
  %val = load volatile i32, ptr addrspace(5) %ptr
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_ptr:
; GCN: s_waitcnt

; CI: v_lshr_b32_e64 [[SHIFT:v[0-9]+]], s32, 6
; CI-NEXT: v_or_b32_e32 v0, 4, [[SHIFT]]

; GFX9-MUBUF:      v_lshrrev_b32_e64 [[SHIFT:v[0-9]+]], 6, s32
; GFX9-MUBUF-NEXT: v_or_b32_e32 v0, 4, [[SHIFT]]

; GFX9-FLATSCR: v_or_b32_e64 v0, s32, 4

; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @void_func_byval_struct_i8_i32_ptr(ptr addrspace(5) byval({ i8, i32 }) %arg0) #0 {
  %gep0 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %arg0, i32 0, i32 1
  %load1 = load i32, ptr addrspace(5) %gep1
  store volatile ptr addrspace(5) %gep1, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_ptr_value:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; MUBUF-NEXT: buffer_load_ubyte v0, off, s[0:3], s32
; MUBUF-NEXT: buffer_load_dword v1, off, s[0:3], s32 offset:4
; GFX9-FLATSCR-NEXT: scratch_load_ubyte v0, off, s32
; GFX9-FLATSCR-NEXT: scratch_load_dword v1, off, s32 offset:4
define void @void_func_byval_struct_i8_i32_ptr_value(ptr addrspace(5) byval({ i8, i32 }) %arg0) #0 {
  %gep0 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %arg0, i32 0, i32 1
  %load0 = load i8, ptr addrspace(5) %gep0
  %load1 = load i32, ptr addrspace(5) %gep1
  store volatile i8 %load0, ptr addrspace(3) undef
  store volatile i32 %load1, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_ptr_nonentry_block:

; GCN: s_and_saveexec_b64

; CI: buffer_load_dword v{{[0-9]+}}, off, s[0:3], s32 offset:4 glc{{$}}
; GFX9-MUBUF:   buffer_load_dword v{{[0-9]+}}, off, s[0:3], s32 offset:4 glc{{$}}
; GFX9-FLATSCR: scratch_load_dword v{{[0-9]+}}, off, s32 offset:4 glc{{$}}

; CI: v_lshr_b32_e64 [[SHIFT:v[0-9]+]], s32, 6
; CI: v_add_i32_e64 [[GEP:v[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 4, [[SHIFT]]

; GFX9-MUBUF: v_lshrrev_b32_e64 [[SP:v[0-9]+]], 6, s32
; GFX9-MUBUF: v_add_u32_e32 [[GEP:v[0-9]+]], 4, [[SP]]

; GFX9-FLATSCR: v_add_u32_e64 [[GEP:v[0-9]+]], 4, s32

; GCN: ds_write_b32 v{{[0-9]+}}, [[GEP]]
define void @void_func_byval_struct_i8_i32_ptr_nonentry_block(ptr addrspace(5) byval({ i8, i32 }) %arg0, i32 %arg2) #0 {
  %cmp = icmp eq i32 %arg2, 0
  br i1 %cmp, label %bb, label %ret

bb:
  %gep0 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %arg0, i32 0, i32 1
  %load1 = load volatile i32, ptr addrspace(5) %gep1
  store volatile ptr addrspace(5) %gep1, ptr addrspace(3) undef
  br label %ret

ret:
  ret void
}

; Added offset can't be used with VOP3 add
; GCN-LABEL: {{^}}func_other_fi_user_non_inline_imm_offset_i32:

; MUBUF: s_lshr_b32 [[SCALED:s[0-9]+]], s32, 6
; MUBUF: s_addk_i32 [[SCALED]], 0x200

; MUBUF: s_mul_i32 [[Z:s[0-9]+]], [[SCALED]], 9
; MUBUF: v_mov_b32_e32 [[VZ:v[0-9]+]], [[Z]]

; GFX9-FLATSCR: s_add_i32 [[SZ:[^,]+]], s32, 0x200
; GFX9-FLATSCR: s_mul_i32 [[MUL:s[0-9]+]], [[SZ]], 9
; GFX9-FLATSCR: v_mov_b32_e32 [[VZ:v[0-9]+]], [[MUL]]

; GCN: ds_write_b32 v0, [[VZ]]
define void @func_other_fi_user_non_inline_imm_offset_i32() #0 {
  %alloca0 = alloca [128 x i32], align 4, addrspace(5)
  %alloca1 = alloca [8 x i32], align 4, addrspace(5)
  %gep0 = getelementptr inbounds [128 x i32], ptr addrspace(5) %alloca0, i32 0, i32 65
  store volatile i32 7, ptr addrspace(5) %gep0
  %ptrtoint = ptrtoint ptr addrspace(5) %alloca1 to i32
  %mul = mul i32 %ptrtoint, 9
  store volatile i32 %mul, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}func_other_fi_user_non_inline_imm_offset_i32_vcc_live:

; MUBUF: s_lshr_b32 [[SCALED:s[0-9]+]], s32, 6
; MUBUF: s_addk_i32 [[SCALED]], 0x200
; MUBUF: s_mul_i32 [[Z:s[0-9]+]], [[SCALED]], 9
; MUBUF: v_mov_b32_e32 [[VZ:v[0-9]+]], [[Z]]

; GFX9-FLATSCR: s_add_i32 [[SZ:[^,]+]], s32, 0x200
; GFX9-FLATSCR: s_mul_i32 [[MUL:s[0-9]+]], [[SZ]], 9
; GFX9-FLATSCR: v_mov_b32_e32 [[VZ:v[0-9]+]], [[MUL]]

; GCN: ds_write_b32 v0, [[VZ]]
define void @func_other_fi_user_non_inline_imm_offset_i32_vcc_live() #0 {
  %alloca0 = alloca [128 x i32], align 4, addrspace(5)
  %alloca1 = alloca [8 x i32], align 4, addrspace(5)
  %vcc = call i64 asm sideeffect "; def $0", "={vcc}"()
  %gep0 = getelementptr inbounds [128 x i32], ptr addrspace(5) %alloca0, i32 0, i32 65
  store volatile i32 7, ptr addrspace(5) %gep0
  call void asm sideeffect "; use $0", "{vcc}"(i64 %vcc)
  %ptrtoint = ptrtoint ptr addrspace(5) %alloca1 to i32
  %mul = mul i32 %ptrtoint, 9
  store volatile i32 %mul, ptr addrspace(3) undef
  ret void
}

declare void @func(ptr addrspace(5) nocapture) #0

; undef flag not preserved in eliminateFrameIndex when handling the
; stores in the middle block.

; GCN-LABEL: {{^}}undefined_stack_store_reg:
; GCN: s_and_saveexec_b64
; MUBUF: buffer_store_dword v0, off, s[0:3], s33 offset:
; MUBUF: buffer_store_dword v0, off, s[0:3], s33 offset:
; MUBUF: buffer_store_dword v0, off, s[0:3], s33 offset:
; MUBUF: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s33 offset:
; FLATSCR: scratch_store_dword v0, off, s33 offset:
; FLATSCR: scratch_store_dword v0, off, s33 offset:
; FLATSCR: scratch_store_dword v0, off, s33 offset:
; FLATSCR: scratch_store_dword v{{[0-9]+}}, off, s33 offset:
define void @undefined_stack_store_reg(float %arg, i32 %arg1) #0 {
bb:
  %tmp = alloca <4 x float>, align 16, addrspace(5)
  %tmp2 = insertelement <4 x float> undef, float %arg, i32 0
  store <4 x float> %tmp2, ptr addrspace(5) undef
  %tmp3 = icmp eq i32 %arg1, 0
  br i1 %tmp3, label %bb4, label %bb5

bb4:
  call void @func(ptr addrspace(5) nonnull undef)
  store <4 x float> %tmp2, ptr addrspace(5) %tmp, align 16
  call void @func(ptr addrspace(5) nonnull %tmp)
  br label %bb5

bb5:
  ret void
}

; GCN-LABEL: {{^}}alloca_ptr_nonentry_block:
; GCN: s_and_saveexec_b64
; MUBUF:   buffer_load_dword v{{[0-9]+}}, off, s[0:3], s32 offset:4
; FLATSCR: scratch_load_dword v{{[0-9]+}}, off, s32 offset:4

; CI: v_lshr_b32_e64 [[SHIFT:v[0-9]+]], s32, 6
; CI-NEXT: v_or_b32_e32 [[PTR:v[0-9]+]], 4, [[SHIFT]]

; GFX9-MUBUF: v_lshrrev_b32_e64 [[SHIFT:v[0-9]+]], 6, s32
; GFX9-MUBUF-NEXT: v_or_b32_e32 [[PTR:v[0-9]+]], 4, [[SHIFT]]

; GFX9-FLATSCR: v_or_b32_e64 [[PTR:v[0-9]+]], s32, 4

; GCN: ds_write_b32 v{{[0-9]+}}, [[PTR]]
define void @alloca_ptr_nonentry_block(i32 %arg0) #0 {
  %alloca0 = alloca { i8, i32 }, align 8, addrspace(5)
  %cmp = icmp eq i32 %arg0, 0
  br i1 %cmp, label %bb, label %ret

bb:
  %gep0 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %alloca0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %alloca0, i32 0, i32 1
  %load1 = load volatile i32, ptr addrspace(5) %gep1
  store volatile ptr addrspace(5) %gep1, ptr addrspace(3) undef
  br label %ret

ret:
  ret void
}

%struct0 = type { [4224 x %type.i16] }
%type.i16 = type { i16 }
@_ZZN0 = external hidden addrspace(3) global %struct0, align 8

; GFX11-LABEL: tied_operand_test:
; GFX11:       ; %bb.0: ; %entry
; GFX11:     scratch_load_u16 [[LDRESULT:v[0-9]+]], off, off
; GFX11:     v_dual_mov_b32 [[C:v[0-9]+]], 0x7b :: v_dual_mov_b32 v{{[0-9]+}}, s{{[0-9]+}}
; GFX11-DAG:     ds_store_b16 v{{[0-9]+}}, [[LDRESULT]]  offset:10
; GFX11-DAG:     ds_store_b16 v{{[0-9]+}}, [[C]]  offset:8
; GFX11-NEXT:    s_endpgm
define protected amdgpu_kernel void @tied_operand_test(i1 %c1, i1 %c2, i32 %val) {
entry:
  %scratch0 = alloca i16, align 4, addrspace(5)
  %scratch1 = alloca i16, align 4, addrspace(5)
  %first = select i1 %c1, ptr addrspace(5) %scratch0, ptr addrspace(5) %scratch1
  %spec.select = select i1 %c2, ptr addrspace(5) %first, ptr addrspace(5) %scratch0
  %dead.load = load i16, ptr addrspace(5) %spec.select, align 2
  %scratch0.load = load i16, ptr addrspace(5) %scratch0, align 4
  %add4 = add nuw nsw i32 %val, 4
  %addr0 = getelementptr inbounds %struct0, ptr addrspace(3) @_ZZN0, i32 0, i32 0, i32 %add4, i32 0
  store i16 123, ptr addrspace(3) %addr0, align 2
  %add5 = add nuw nsw i32 %val, 5
  %addr1 = getelementptr inbounds %struct0, ptr addrspace(3) @_ZZN0, i32 0, i32 0, i32 %add5, i32 0
  store i16 %scratch0.load, ptr addrspace(3) %addr1, align 2
  ret void
}

; GCN-LABEL: {{^}}fi_vop3_literal_error:
; CI: v_lshr_b32_e64 [[SCALED_FP:v[0-9]+]], s33, 6
; CI: s_movk_i32 vcc_lo, 0x3000
; CI-NEXT: v_add_i32_e32 [[SCALED_FP]], vcc, vcc_lo, [[SCALED_FP]]
; CI-NEXT: v_add_i32_e32 v0, vcc, 64, [[SCALED_FP]]

; GFX9-MUBUF: v_lshrrev_b32_e64 [[SCALED_FP:v[0-9]+]], 6, s33
; GFX9-MUBUF-NEXT: v_add_u32_e32 [[SCALED_FP]], 0x3000, [[SCALED_FP]]
; GFX9-MUBUF-NEXT: v_add_u32_e32 v0, 64, [[SCALED_FP]]
define void @fi_vop3_literal_error() {
entry:
  %pin.low = alloca i32, align 8192, addrspace(5)
  %local.area = alloca [1060 x i64], align 4096, addrspace(5)
  store i32 0, ptr addrspace(5) %pin.low, align 4
  %gep.small.offset = getelementptr i8, ptr addrspace(5) %local.area, i64 64
  %load1 = load volatile i64, ptr addrspace(5) %gep.small.offset, align 4
  ret void
}

attributes #0 = { nounwind }
