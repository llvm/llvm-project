; RUN: llc -O0 -march=amdgcn -mcpu=gfx900 -amdgpu-dpp-combine=false -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX9,GFX9-O0 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -amdgpu-dpp-combine=false -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX9,GFX9-O3 %s

; NOTE: llvm.amdgcn.wwm is deprecated, use llvm.amdgcn.strict.wwm instead.

; GFX9-LABEL: {{^}}no_cfg:
define amdgpu_cs void @no_cfg(<4 x i32> inreg %tmp14) {
  %tmp100 = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %tmp14, i32 0, i32 0, i32 0)
  %tmp101 = bitcast <2 x float> %tmp100 to <2 x i32>
  %tmp102 = extractelement <2 x i32> %tmp101, i32 0
  %tmp103 = extractelement <2 x i32> %tmp101, i32 1
  %tmp105 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 %tmp102, i32 0)
  %tmp107 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 %tmp103, i32 0)

; GFX9: s_or_saveexec_b64 s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, -1

; GFX9-DAG: v_mov_b32_dpp v[[FIRST_MOV:[0-9]+]], v{{[0-9]+}} row_bcast:31 row_mask:0xc bank_mask:0xf
; GFX9-O3-DAG: v_add_u32_e32 v[[FIRST_ADD:[0-9]+]], v{{[0-9]+}}, v[[FIRST_MOV]]
; GFX9-O0-DAG: v_add_u32_e64 v[[FIRST_ADD:[0-9]+]], v{{[0-9]+}}, v[[FIRST_MOV]]
; GFX9-DAG: v_mov_b32_e32 v[[FIRST:[0-9]+]], v[[FIRST_ADD]]
  %tmp120 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %tmp105, i32 323, i32 12, i32 15, i1 false)
  %tmp121 = add i32 %tmp105, %tmp120
  %tmp122 = tail call i32 @llvm.amdgcn.wwm.i32(i32 %tmp121)

; GFX9-DAG: v_mov_b32_dpp v[[SECOND_MOV:[0-9]+]], v{{[0-9]+}} row_bcast:31 row_mask:0xc bank_mask:0xf
; GFX9-O3-DAG: v_add_u32_e32 v[[SECOND_ADD:[0-9]+]], v{{[0-9]+}}, v[[SECOND_MOV]]
; GFX9-O0-DAG: v_add_u32_e64 v[[SECOND_ADD:[0-9]+]], v{{[0-9]+}}, v[[SECOND_MOV]]
; GFX9-DAG: v_mov_b32_e32 v[[SECOND:[0-9]+]], v[[SECOND_ADD]]
  %tmp135 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %tmp107, i32 323, i32 12, i32 15, i1 false)
  %tmp136 = add i32 %tmp107, %tmp135
  %tmp137 = tail call i32 @llvm.amdgcn.wwm.i32(i32 %tmp136)

; GFX9-O3: v_cmp_eq_u32_e32 vcc, v[[FIRST]], v[[SECOND]]
; GFX9-O0: v_cmp_eq_u32_e64 s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, v[[FIRST]], v[[SECOND]]
  %tmp138 = icmp eq i32 %tmp122, %tmp137
  %tmp139 = sext i1 %tmp138 to i32
  %tmp140 = shl nsw i32 %tmp139, 1
  %tmp141 = and i32 %tmp140, 2
  %tmp145 = bitcast i32 %tmp141 to float
  call void @llvm.amdgcn.raw.buffer.store.f32(float %tmp145, <4 x i32> %tmp14, i32 4, i32 0, i32 0)
  ret void
}

; GFX9-LABEL: {{^}}cfg:
define amdgpu_cs void @cfg(<4 x i32> inreg %tmp14, i32 %arg) {
entry:
  %tmp100 = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %tmp14, i32 0, i32 0, i32 0)
  %tmp101 = bitcast <2 x float> %tmp100 to <2 x i32>
  %tmp102 = extractelement <2 x i32> %tmp101, i32 0
  %tmp105 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 %tmp102, i32 0)

; GFX9: v_mov_b32_dpp v[[FIRST_MOV:[0-9]+]], v{{[0-9]+}} row_bcast:31 row_mask:0xc bank_mask:0xf
; GFX9-O3: v_add_u32_e32 v[[FIRST_ADD:[0-9]+]], v{{[0-9]+}}, v[[FIRST_MOV]]
; GFX9-O0: v_add_u32_e64 v[[FIRST_ADD:[0-9]+]], v{{[0-9]+}}, v[[FIRST_MOV]]
; GFX9: v_mov_b32_e32 v[[FIRST:[0-9]+]], v[[FIRST_ADD]]
; GFX9-O0: buffer_store_dword v[[FIRST]], off, s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, 0 offset:[[FIRST_IMM_OFFSET:[0-9]+]]
  %tmp120 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %tmp105, i32 323, i32 12, i32 15, i1 false)
  %tmp121 = add i32 %tmp105, %tmp120
  %tmp122 = tail call i32 @llvm.amdgcn.wwm.i32(i32 %tmp121)

  %cond = icmp eq i32 %arg, 0
  br i1 %cond, label %if, label %merge
if:
  %tmp103 = extractelement <2 x i32> %tmp101, i32 1
  %tmp107 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 %tmp103, i32 0)

; GFX9: v_mov_b32_dpp v[[SECOND_MOV:[0-9]+]], v{{[0-9]+}} row_bcast:31 row_mask:0xc bank_mask:0xf
; GFX9-O3: v_add_u32_e32 v[[SECOND_ADD:[0-9]+]], v{{[0-9]+}}, v[[SECOND_MOV]]
; GFX9-O0: v_add_u32_e64 v[[SECOND_ADD:[0-9]+]], v{{[0-9]+}}, v[[SECOND_MOV]]
; GFX9: v_mov_b32_e32 v[[SECOND:[0-9]+]], v[[SECOND_ADD]]
; GFX9-O0: buffer_store_dword v[[SECOND]], off, s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, 0 offset:[[SECOND_IMM_OFFSET:[0-9]+]]
  %tmp135 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %tmp107, i32 323, i32 12, i32 15, i1 false)
  %tmp136 = add i32 %tmp107, %tmp135
  %tmp137 = tail call i32 @llvm.amdgcn.wwm.i32(i32 %tmp136)
  br label %merge

merge:
  %merge_value = phi i32 [ 0, %entry ], [%tmp137, %if ]
; GFX9-O3: v_cmp_eq_u32_e32 vcc, v[[FIRST]], v[[SECOND]]
; GFX9-O0: buffer_load_dword v[[FIRST:[0-9]+]], off, s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, 0 offset:[[FIRST_IMM_OFFSET]]
; GFX9-O0: buffer_load_dword v[[SECOND:[0-9]+]], off, s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, 0 offset:[[SECOND_IMM_OFFSET]]
; GFX9-O0: v_cmp_eq_u32_e64 s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, v[[FIRST]], v[[SECOND]]
  %tmp138 = icmp eq i32 %tmp122, %merge_value
  %tmp139 = sext i1 %tmp138 to i32
  %tmp140 = shl nsw i32 %tmp139, 1
  %tmp141 = and i32 %tmp140, 2
  %tmp145 = bitcast i32 %tmp141 to float
  call void @llvm.amdgcn.raw.buffer.store.f32(float %tmp145, <4 x i32> %tmp14, i32 4, i32 0, i32 0)
  ret void
}

; GFX9-LABEL: {{^}}called:
define hidden i32 @called(i32 %a) noinline {
; GFX9-O3: v_add_u32_e32 v1, v0, v0
; GFX9-O0: v_add_u32_e64 v1, v0, v0
  %add = add i32 %a, %a
; GFX9: v_mul_lo_u32 v0, v1, v0
  %mul = mul i32 %add, %a
; GFX9-O3: v_sub_u32_e32 v0, v0, v1
; GFX9-O0: v_sub_u32_e64 v0, v0, v1
  %sub = sub i32 %mul, %add
  ret i32 %sub
}

; GFX9-LABEL: {{^}}call:
define amdgpu_kernel void @call(<4 x i32> inreg %tmp14, i32 inreg %arg) {
; GFX9-DAG: s_load_dword [[ARG:s[0-9]+]]
; GFX9-O0-DAG: s_mov_b32 s4, 0{{$}}
; GFX9-O0-DAG: v_mov_b32_e32 v0, [[ARG]]
; GFX9-O0-DAG: v_mov_b32_e32 v2, v0

; GFX9-O3: v_mov_b32_e32 v2, [[ARG]]

; GFX9-NEXT: s_not_b64 exec, exec
; GFX9-O0-NEXT: v_mov_b32_e32 v2, s4
; GFX9-O3-NEXT: v_mov_b32_e32 v2, 0
; GFX9-NEXT: s_not_b64 exec, exec
  %tmp107 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 %arg, i32 0)
; GFX9: v_mov_b32_e32 v0, v2
; GFX9: s_swappc_b64
  %tmp134 = call i32 @called(i32 %tmp107)
; GFX9: v_mov_b32_e32 v1, v0
; GFX9-O3: v_add_u32_e32 v1, v1, v2
; GFX9-O0: v_add_u32_e64 v1, v1, v2
  %tmp136 = add i32 %tmp134, %tmp107
  %tmp137 = tail call i32 @llvm.amdgcn.wwm.i32(i32 %tmp136)
; GFX9: buffer_store_dword v0
  call void @llvm.amdgcn.raw.buffer.store.i32(i32 %tmp137, <4 x i32> %tmp14, i32 4, i32 0, i32 0)
  ret void
}

; GFX9-LABEL: {{^}}called_i64:
define i64 @called_i64(i64 %a) noinline {
  %add = add i64 %a, %a
  %mul = mul i64 %add, %a
  %sub = sub i64 %mul, %add
  ret i64 %sub
}

; GFX9-LABEL: {{^}}call_i64:
define amdgpu_kernel void @call_i64(<4 x i32> inreg %tmp14, i64 inreg %arg) {
; GFX9: s_load_dwordx2 s{{\[}}[[ARG_LO:[0-9]+]]:[[ARG_HI:[0-9]+]]{{\]}}

; GFX9-O0: s_mov_b64 s{{\[}}[[ZERO_LO:[0-9]+]]:[[ZERO_HI:[0-9]+]]{{\]}}, 0{{$}}
; GFX9-O0: v_mov_b32_e32 v0, s[[ARG_LO]]
; GFX9-O0: v_mov_b32_e32 v1, s[[ARG_HI]]
; GFX9-O0-DAG: v_mov_b32_e32 v9, v1
; GFX9-O0-DAG: v_mov_b32_e32 v8, v0

; GFX9-O3-DAG: v_mov_b32_e32 v7, s[[ARG_HI]]
; GFX9-O3-DAG: v_mov_b32_e32 v6, s[[ARG_LO]]

; GFX9: s_not_b64 exec, exec
; GFX9-O0-NEXT: v_mov_b32_e32 v8, s[[ZERO_LO]]
; GFX9-O0-NEXT: v_mov_b32_e32 v9, s[[ZERO_HI]]
; GFX9-O3-NEXT: v_mov_b32_e32 v6, 0
; GFX9-O3-NEXT: v_mov_b32_e32 v7, 0
; GFX9-NEXT: s_not_b64 exec, exec
  %tmp107 = tail call i64 @llvm.amdgcn.set.inactive.i64(i64 %arg, i64 0)
; GFX9: s_swappc_b64
  %tmp134 = call i64 @called_i64(i64 %tmp107)
  %tmp136 = add i64 %tmp134, %tmp107
  %tmp137 = tail call i64 @llvm.amdgcn.wwm.i64(i64 %tmp136)
  %tmp138 = bitcast i64 %tmp137 to <2 x i32>
; GFX9: buffer_store_dwordx2
  call void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32> %tmp138, <4 x i32> %tmp14, i32 4, i32 0, i32 0)
  ret void
}

; GFX9-LABEL: {{^}}_amdgpu_cs_main:
define amdgpu_cs void @_amdgpu_cs_main(<4 x i32> inreg %desc, i32 %index) {
  %tmp17 = shl i32 %index, 5
; GFX9: buffer_load_dwordx4
  %tmp18 = tail call <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32> %desc, i32 %tmp17, i32 0)
  %.i0.upto1.bc = bitcast <4 x i32> %tmp18 to <2 x i64>
  %tmp19 = or i32 %tmp17, 16
; GFX9: buffer_load_dwordx2
  %tmp20 = tail call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> %desc, i32 %tmp19, i32 0)
  %.i0.upto1.extract = extractelement <2 x i64> %.i0.upto1.bc, i32 0
  %tmp22 = tail call i64 @llvm.amdgcn.set.inactive.i64(i64 %.i0.upto1.extract, i64 9223372036854775807)
  %tmp97 = tail call i64 @llvm.amdgcn.wwm.i64(i64 %tmp22)
  %.i1.upto1.extract = extractelement <2 x i64> %.i0.upto1.bc, i32 1
  %tmp99 = tail call i64 @llvm.amdgcn.set.inactive.i64(i64 %.i1.upto1.extract, i64 9223372036854775807)
  %tmp174 = tail call i64 @llvm.amdgcn.wwm.i64(i64 %tmp99)
  %.i25 = bitcast <2 x i32> %tmp20 to i64
  %tmp176 = tail call i64 @llvm.amdgcn.set.inactive.i64(i64 %.i25, i64 9223372036854775807)
  %tmp251 = tail call i64 @llvm.amdgcn.wwm.i64(i64 %tmp176)
  %.cast = bitcast i64 %tmp97 to <2 x float>
  %.cast6 = bitcast i64 %tmp174 to <2 x float>
  %.cast7 = bitcast i64 %tmp251 to <2 x float>
  %tmp254 = shufflevector <2 x float> %.cast, <2 x float> %.cast6, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; GFX9: buffer_store_dwordx4
  tail call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> %tmp254, <4 x i32> %desc, i32 %tmp17, i32 0, i32 0)
  ; GFX9: buffer_store_dwordx2
  tail call void @llvm.amdgcn.raw.buffer.store.v2f32(<2 x float> %.cast7, <4 x i32> %desc, i32 %tmp19, i32 0, i32 0)
  ret void
}


; GFX9-LABEL: {{^}}strict_wwm_no_cfg:
define amdgpu_cs void @strict_wwm_no_cfg(<4 x i32> inreg %tmp14) {
  %tmp100 = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %tmp14, i32 0, i32 0, i32 0)
  %tmp101 = bitcast <2 x float> %tmp100 to <2 x i32>
  %tmp102 = extractelement <2 x i32> %tmp101, i32 0
  %tmp103 = extractelement <2 x i32> %tmp101, i32 1
  %tmp105 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 %tmp102, i32 0)
  %tmp107 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 %tmp103, i32 0)

; GFX9: s_or_saveexec_b64 s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, -1

; GFX9-DAG: v_mov_b32_dpp v[[FIRST_MOV:[0-9]+]], v{{[0-9]+}} row_bcast:31 row_mask:0xc bank_mask:0xf
; GFX9-O3-DAG: v_add_u32_e32 v[[FIRST_ADD:[0-9]+]], v{{[0-9]+}}, v[[FIRST_MOV]]
; GFX9-O0-DAG: v_add_u32_e64 v[[FIRST_ADD:[0-9]+]], v{{[0-9]+}}, v[[FIRST_MOV]]
; GFX9-DAG: v_mov_b32_e32 v[[FIRST:[0-9]+]], v[[FIRST_ADD]]
  %tmp120 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %tmp105, i32 323, i32 12, i32 15, i1 false)
  %tmp121 = add i32 %tmp105, %tmp120
  %tmp122 = tail call i32 @llvm.amdgcn.strict.wwm.i32(i32 %tmp121)

; GFX9-DAG: v_mov_b32_dpp v[[SECOND_MOV:[0-9]+]], v{{[0-9]+}} row_bcast:31 row_mask:0xc bank_mask:0xf
; GFX9-O3-DAG: v_add_u32_e32 v[[SECOND_ADD:[0-9]+]], v{{[0-9]+}}, v[[SECOND_MOV]]
; GFX9-O0-DAG: v_add_u32_e64 v[[SECOND_ADD:[0-9]+]], v{{[0-9]+}}, v[[SECOND_MOV]]
; GFX9-DAG: v_mov_b32_e32 v[[SECOND:[0-9]+]], v[[SECOND_ADD]]
  %tmp135 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %tmp107, i32 323, i32 12, i32 15, i1 false)
  %tmp136 = add i32 %tmp107, %tmp135
  %tmp137 = tail call i32 @llvm.amdgcn.strict.wwm.i32(i32 %tmp136)

; GFX9-O3: v_cmp_eq_u32_e32 vcc, v[[FIRST]], v[[SECOND]]
; GFX9-O0: v_cmp_eq_u32_e64 s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, v[[FIRST]], v[[SECOND]]
  %tmp138 = icmp eq i32 %tmp122, %tmp137
  %tmp139 = sext i1 %tmp138 to i32
  %tmp140 = shl nsw i32 %tmp139, 1
  %tmp141 = and i32 %tmp140, 2
  %tmp145 = bitcast i32 %tmp141 to float
  call void @llvm.amdgcn.raw.buffer.store.f32(float %tmp145, <4 x i32> %tmp14, i32 4, i32 0, i32 0)
  ret void
}

; GFX9-LABEL: {{^}}strict_wwm_cfg:
define amdgpu_cs void @strict_wwm_cfg(<4 x i32> inreg %tmp14, i32 %arg) {
entry:
  %tmp100 = call <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32> %tmp14, i32 0, i32 0, i32 0)
  %tmp101 = bitcast <2 x float> %tmp100 to <2 x i32>
  %tmp102 = extractelement <2 x i32> %tmp101, i32 0
  %tmp105 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 %tmp102, i32 0)

; GFX9: v_mov_b32_dpp v[[FIRST_MOV:[0-9]+]], v{{[0-9]+}} row_bcast:31 row_mask:0xc bank_mask:0xf
; GFX9-O3: v_add_u32_e32 v[[FIRST_ADD:[0-9]+]], v{{[0-9]+}}, v[[FIRST_MOV]]
; GFX9-O0: v_add_u32_e64 v[[FIRST_ADD:[0-9]+]], v{{[0-9]+}}, v[[FIRST_MOV]]
; GFX9: v_mov_b32_e32 v[[FIRST:[0-9]+]], v[[FIRST_ADD]]
; GFX9-O0: buffer_store_dword v[[FIRST]], off, s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, 0 offset:[[FIRST_IMM_OFFSET:[0-9]+]]
  %tmp120 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %tmp105, i32 323, i32 12, i32 15, i1 false)
  %tmp121 = add i32 %tmp105, %tmp120
  %tmp122 = tail call i32 @llvm.amdgcn.strict.wwm.i32(i32 %tmp121)

  %cond = icmp eq i32 %arg, 0
  br i1 %cond, label %if, label %merge
if:
  %tmp103 = extractelement <2 x i32> %tmp101, i32 1
  %tmp107 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 %tmp103, i32 0)

; GFX9: v_mov_b32_dpp v[[SECOND_MOV:[0-9]+]], v{{[0-9]+}} row_bcast:31 row_mask:0xc bank_mask:0xf
; GFX9-O3: v_add_u32_e32 v[[SECOND_ADD:[0-9]+]], v{{[0-9]+}}, v[[SECOND_MOV]]
; GFX9-O0: v_add_u32_e64 v[[SECOND_ADD:[0-9]+]], v{{[0-9]+}}, v[[SECOND_MOV]]
; GFX9: v_mov_b32_e32 v[[SECOND:[0-9]+]], v[[SECOND_ADD]]
; GFX9-O0: buffer_store_dword v[[SECOND]], off, s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, 0 offset:[[SECOND_IMM_OFFSET:[0-9]+]]
  %tmp135 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %tmp107, i32 323, i32 12, i32 15, i1 false)
  %tmp136 = add i32 %tmp107, %tmp135
  %tmp137 = tail call i32 @llvm.amdgcn.strict.wwm.i32(i32 %tmp136)
  br label %merge

merge:
  %merge_value = phi i32 [ 0, %entry ], [%tmp137, %if ]
; GFX9-O3: v_cmp_eq_u32_e32 vcc, v[[FIRST]], v[[SECOND]]
; GFX9-O0: buffer_load_dword v[[FIRST:[0-9]+]], off, s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, 0 offset:[[FIRST_IMM_OFFSET]]
; GFX9-O0: buffer_load_dword v[[SECOND:[0-9]+]], off, s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, 0 offset:[[SECOND_IMM_OFFSET]]
; GFX9-O0: v_cmp_eq_u32_e64 s{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, v[[FIRST]], v[[SECOND]]
  %tmp138 = icmp eq i32 %tmp122, %merge_value
  %tmp139 = sext i1 %tmp138 to i32
  %tmp140 = shl nsw i32 %tmp139, 1
  %tmp141 = and i32 %tmp140, 2
  %tmp145 = bitcast i32 %tmp141 to float
  call void @llvm.amdgcn.raw.buffer.store.f32(float %tmp145, <4 x i32> %tmp14, i32 4, i32 0, i32 0)
  ret void
}

; GFX9-LABEL: {{^}}strict_wwm_called:
define hidden i32 @strict_wwm_called(i32 %a) noinline {
; GFX9-O3: v_add_u32_e32 v1, v0, v0
; GFX9-O0: v_add_u32_e64 v1, v0, v0
  %add = add i32 %a, %a
; GFX9: v_mul_lo_u32 v0, v1, v0
  %mul = mul i32 %add, %a
; GFX9-O3: v_sub_u32_e32 v0, v0, v1
; GFX9-O0: v_sub_u32_e64 v0, v0, v1
  %sub = sub i32 %mul, %add
  ret i32 %sub
}

; GFX9-LABEL: {{^}}strict_wwm_call:
define amdgpu_kernel void @strict_wwm_call(<4 x i32> inreg %tmp14, i32 inreg %arg) {
; GFX9-DAG: s_load_dword [[ARG:s[0-9]+]]
; GFX9-O0-DAG: s_mov_b32 s4, 0{{$}}
; GFX9-O0-DAG: v_mov_b32_e32 v0, [[ARG]]
; GFX9-O0-DAG: v_mov_b32_e32 v2, v0

; GFX9-O3: v_mov_b32_e32 v2, [[ARG]]

; GFX9-NEXT: s_not_b64 exec, exec
; GFX9-O0-NEXT: v_mov_b32_e32 v2, s4
; GFX9-O3-NEXT: v_mov_b32_e32 v2, 0
; GFX9-NEXT: s_not_b64 exec, exec
  %tmp107 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 %arg, i32 0)
; GFX9: v_mov_b32_e32 v0, v2
; GFX9: s_swappc_b64
  %tmp134 = call i32 @strict_wwm_called(i32 %tmp107)
; GFX9: v_mov_b32_e32 v1, v0
; GFX9-O3: v_add_u32_e32 v1, v1, v2
; GFX9-O0: v_add_u32_e64 v1, v1, v2
  %tmp136 = add i32 %tmp134, %tmp107
  %tmp137 = tail call i32 @llvm.amdgcn.strict.wwm.i32(i32 %tmp136)
; GFX9: buffer_store_dword v0
  call void @llvm.amdgcn.raw.buffer.store.i32(i32 %tmp137, <4 x i32> %tmp14, i32 4, i32 0, i32 0)
  ret void
}

; GFX9-LABEL: {{^}}strict_wwm_called_i64:
define i64 @strict_wwm_called_i64(i64 %a) noinline {
  %add = add i64 %a, %a
  %mul = mul i64 %add, %a
  %sub = sub i64 %mul, %add
  ret i64 %sub
}

; GFX9-LABEL: {{^}}strict_wwm_call_i64:
define amdgpu_kernel void @strict_wwm_call_i64(<4 x i32> inreg %tmp14, i64 inreg %arg) {
; GFX9: s_load_dwordx2 s{{\[}}[[ARG_LO:[0-9]+]]:[[ARG_HI:[0-9]+]]{{\]}}

; GFX9-O0: s_mov_b64 s{{\[}}[[ZERO_LO:[0-9]+]]:[[ZERO_HI:[0-9]+]]{{\]}}, 0{{$}}
; GFX9-O0: v_mov_b32_e32 v0, s[[ARG_LO]]
; GFX9-O0: v_mov_b32_e32 v1, s[[ARG_HI]]
; GFX9-O0-DAG: v_mov_b32_e32 v9, v1
; GFX9-O0-DAG: v_mov_b32_e32 v8, v0

; GFX9-O3-DAG: v_mov_b32_e32 v7, s[[ARG_HI]]
; GFX9-O3-DAG: v_mov_b32_e32 v6, s[[ARG_LO]]

; GFX9: s_not_b64 exec, exec
; GFX9-O0-NEXT: v_mov_b32_e32 v8, s[[ZERO_LO]]
; GFX9-O0-NEXT: v_mov_b32_e32 v9, s[[ZERO_HI]]
; GFX9-O3-NEXT: v_mov_b32_e32 v6, 0
; GFX9-O3-NEXT: v_mov_b32_e32 v7, 0
; GFX9-NEXT: s_not_b64 exec, exec
  %tmp107 = tail call i64 @llvm.amdgcn.set.inactive.i64(i64 %arg, i64 0)
; GFX9: s_swappc_b64
  %tmp134 = call i64 @strict_wwm_called_i64(i64 %tmp107)
  %tmp136 = add i64 %tmp134, %tmp107
  %tmp137 = tail call i64 @llvm.amdgcn.strict.wwm.i64(i64 %tmp136)
  %tmp138 = bitcast i64 %tmp137 to <2 x i32>
; GFX9: buffer_store_dwordx2
  call void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32> %tmp138, <4 x i32> %tmp14, i32 4, i32 0, i32 0)
  ret void
}

; GFX9-LABEL: {{^}}strict_wwm_amdgpu_cs_main:
define amdgpu_cs void @strict_wwm_amdgpu_cs_main(<4 x i32> inreg %desc, i32 %index) {
  %tmp17 = shl i32 %index, 5
; GFX9: buffer_load_dwordx4
  %tmp18 = tail call <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32> %desc, i32 %tmp17, i32 0)
  %.i0.upto1.bc = bitcast <4 x i32> %tmp18 to <2 x i64>
  %tmp19 = or i32 %tmp17, 16
; GFX9: buffer_load_dwordx2
  %tmp20 = tail call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> %desc, i32 %tmp19, i32 0)
  %.i0.upto1.extract = extractelement <2 x i64> %.i0.upto1.bc, i32 0
  %tmp22 = tail call i64 @llvm.amdgcn.set.inactive.i64(i64 %.i0.upto1.extract, i64 9223372036854775807)
  %tmp97 = tail call i64 @llvm.amdgcn.strict.wwm.i64(i64 %tmp22)
  %.i1.upto1.extract = extractelement <2 x i64> %.i0.upto1.bc, i32 1
  %tmp99 = tail call i64 @llvm.amdgcn.set.inactive.i64(i64 %.i1.upto1.extract, i64 9223372036854775807)
  %tmp174 = tail call i64 @llvm.amdgcn.strict.wwm.i64(i64 %tmp99)
  %.i25 = bitcast <2 x i32> %tmp20 to i64
  %tmp176 = tail call i64 @llvm.amdgcn.set.inactive.i64(i64 %.i25, i64 9223372036854775807)
  %tmp251 = tail call i64 @llvm.amdgcn.strict.wwm.i64(i64 %tmp176)
  %.cast = bitcast i64 %tmp97 to <2 x float>
  %.cast6 = bitcast i64 %tmp174 to <2 x float>
  %.cast7 = bitcast i64 %tmp251 to <2 x float>
  %tmp254 = shufflevector <2 x float> %.cast, <2 x float> %.cast6, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; GFX9: buffer_store_dwordx4
  tail call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> %tmp254, <4 x i32> %desc, i32 %tmp17, i32 0, i32 0)
  ; GFX9: buffer_store_dwordx2
  tail call void @llvm.amdgcn.raw.buffer.store.v2f32(<2 x float> %.cast7, <4 x i32> %desc, i32 %tmp19, i32 0, i32 0)
  ret void
}

declare i32 @llvm.amdgcn.strict.wwm.i32(i32)
declare i64 @llvm.amdgcn.strict.wwm.i64(i64)
declare i32 @llvm.amdgcn.wwm.i32(i32)
declare i64 @llvm.amdgcn.wwm.i64(i64)
declare i32 @llvm.amdgcn.set.inactive.i32(i32, i32)
declare i64 @llvm.amdgcn.set.inactive.i64(i64, i64)
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1)
declare <2 x float> @llvm.amdgcn.raw.buffer.load.v2f32(<4 x i32>, i32, i32, i32)
declare void @llvm.amdgcn.raw.buffer.store.f32(float, <4 x i32>, i32, i32, i32)
declare void @llvm.amdgcn.raw.buffer.store.i32(i32, <4 x i32>, i32, i32, i32)
declare void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32>, <4 x i32>, i32, i32, i32)
declare void @llvm.amdgcn.raw.buffer.store.v2f32(<2 x float>, <4 x i32>, i32, i32, i32)
declare void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float>, <4 x i32>, i32, i32, i32)
declare <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32>, i32, i32)
declare <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32>, i32, i32)
