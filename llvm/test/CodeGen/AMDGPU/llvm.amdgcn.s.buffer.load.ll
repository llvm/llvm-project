; RUN: llc < %s -march=amdgcn -mcpu=tahiti -verify-machineinstrs | FileCheck %s -check-prefixes=GCN,SI,SICI
; RUN: llc < %s -march=amdgcn -mcpu=hawaii -verify-machineinstrs | FileCheck %s -check-prefixes=GCN,CI,SICI
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s -check-prefixes=GCN,VI

;GCN-LABEL: {{^}}s_buffer_load_imm:
;GCN-NOT: s_waitcnt;
;SI: s_buffer_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0x1
;CI: s_buffer_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0x1
;VI: s_buffer_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0x4
define amdgpu_ps void @s_buffer_load_imm(<4 x i32> inreg %desc) {
main_body:
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 4, i32 0)
  %bitcast = bitcast i32 %load to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %bitcast, float undef, float undef, float undef, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_load_index:
;GCN-NOT: s_waitcnt;
;GCN: s_buffer_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
define amdgpu_ps void @s_buffer_load_index(<4 x i32> inreg %desc, i32 inreg %index) {
main_body:
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 %index, i32 0)
  %bitcast = bitcast i32 %load to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %bitcast, float undef, float undef, float undef, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_load_index_divergent:
;GCN-NOT: s_waitcnt;
;GCN: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
define amdgpu_ps void @s_buffer_load_index_divergent(<4 x i32> inreg %desc, i32 %index) {
main_body:
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 %index, i32 0)
  %bitcast = bitcast i32 %load to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %bitcast, float undef, float undef, float undef, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_loadx2_imm:
;GCN-NOT: s_waitcnt;
;SI: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x10
;CI: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x10
;VI: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x40
define amdgpu_ps void @s_buffer_loadx2_imm(<4 x i32> inreg %desc) {
main_body:
  %load = call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> %desc, i32 64, i32 0)
  %bitcast = bitcast <2 x i32> %load to <2 x float>
  %x = extractelement <2 x float> %bitcast, i32 0
  %y = extractelement <2 x float> %bitcast, i32 1
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float undef, float undef, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_loadx2_index:
;GCN-NOT: s_waitcnt;
;GCN: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
define amdgpu_ps void @s_buffer_loadx2_index(<4 x i32> inreg %desc, i32 inreg %index) {
main_body:
  %load = call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> %desc, i32 %index, i32 0)
  %bitcast = bitcast <2 x i32> %load to <2 x float>
  %x = extractelement <2 x float> %bitcast, i32 0
  %y = extractelement <2 x float> %bitcast, i32 1
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float undef, float undef, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_loadx2_index_divergent:
;GCN-NOT: s_waitcnt;
;GCN: buffer_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
define amdgpu_ps void @s_buffer_loadx2_index_divergent(<4 x i32> inreg %desc, i32 %index) {
main_body:
  %load = call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> %desc, i32 %index, i32 0)
  %bitcast = bitcast <2 x i32> %load to <2 x float>
  %x = extractelement <2 x float> %bitcast, i32 0
  %y = extractelement <2 x float> %bitcast, i32 1
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float undef, float undef, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_loadx3_imm:
;GCN-NOT: s_waitcnt;
;SI: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x10
;CI: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x10
;VI: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x40
define amdgpu_ps void @s_buffer_loadx3_imm(<4 x i32> inreg %desc) {
main_body:
  %load = call <3 x i32> @llvm.amdgcn.s.buffer.load.v3i32(<4 x i32> %desc, i32 64, i32 0)
  %bitcast = bitcast <3 x i32> %load to <3 x float>
  %x = extractelement <3 x float> %bitcast, i32 0
  %y = extractelement <3 x float> %bitcast, i32 1
  %z = extractelement <3 x float> %bitcast, i32 2
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float undef, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_loadx3_index:
;GCN-NOT: s_waitcnt;
;GCN: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
define amdgpu_ps void @s_buffer_loadx3_index(<4 x i32> inreg %desc, i32 inreg %index) {
main_body:
  %load = call <3 x i32> @llvm.amdgcn.s.buffer.load.v3i32(<4 x i32> %desc, i32 %index, i32 0)
  %bitcast = bitcast <3 x i32> %load to <3 x float>
  %x = extractelement <3 x float> %bitcast, i32 0
  %y = extractelement <3 x float> %bitcast, i32 1
  %z = extractelement <3 x float> %bitcast, i32 2
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float undef, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_loadx3_index_divergent:
;GCN-NOT: s_waitcnt;
;SI: buffer_load_dwordx4 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
;CI: buffer_load_dwordx3 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
;VI: buffer_load_dwordx3 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
define amdgpu_ps void @s_buffer_loadx3_index_divergent(<4 x i32> inreg %desc, i32 %index) {
main_body:
  %load = call <3 x i32> @llvm.amdgcn.s.buffer.load.v3i32(<4 x i32> %desc, i32 %index, i32 0)
  %bitcast = bitcast <3 x i32> %load to <3 x float>
  %x = extractelement <3 x float> %bitcast, i32 0
  %y = extractelement <3 x float> %bitcast, i32 1
  %z = extractelement <3 x float> %bitcast, i32 2
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float undef, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_loadx4_imm:
;GCN-NOT: s_waitcnt;
;SI: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x32
;CI: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x32
;VI: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0xc8
define amdgpu_ps void @s_buffer_loadx4_imm(<4 x i32> inreg %desc) {
main_body:
  %load = call <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32> %desc, i32 200, i32 0)
  %bitcast = bitcast <4 x i32> %load to <4 x float>
  %x = extractelement <4 x float> %bitcast, i32 0
  %y = extractelement <4 x float> %bitcast, i32 1
  %z = extractelement <4 x float> %bitcast, i32 2
  %w = extractelement <4 x float> %bitcast, i32 3
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float %w, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_loadx4_index:
;GCN-NOT: s_waitcnt;
;GCN: buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
define amdgpu_ps void @s_buffer_loadx4_index(<4 x i32> inreg %desc, i32 inreg %index) {
main_body:
  %load = call <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32> %desc, i32 %index, i32 0)
  %bitcast = bitcast <4 x i32> %load to <4 x float>
  %x = extractelement <4 x float> %bitcast, i32 0
  %y = extractelement <4 x float> %bitcast, i32 1
  %z = extractelement <4 x float> %bitcast, i32 2
  %w = extractelement <4 x float> %bitcast, i32 3
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float %w, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_loadx4_index_divergent:
;GCN-NOT: s_waitcnt;
;GCN: buffer_load_dwordx4 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
define amdgpu_ps void @s_buffer_loadx4_index_divergent(<4 x i32> inreg %desc, i32 %index) {
main_body:
  %load = call <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32> %desc, i32 %index, i32 0)
  %bitcast = bitcast <4 x i32> %load to <4 x float>
  %x = extractelement <4 x float> %bitcast, i32 0
  %y = extractelement <4 x float> %bitcast, i32 1
  %z = extractelement <4 x float> %bitcast, i32 2
  %w = extractelement <4 x float> %bitcast, i32 3
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float %w, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_load_imm_mergex2:
;GCN-NOT: s_waitcnt;
;SI: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x1
;CI: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x1
;VI: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x4
define amdgpu_ps void @s_buffer_load_imm_mergex2(<4 x i32> inreg %desc) {
main_body:
  %load0 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 4, i32 0)
  %load1 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 8, i32 0)
  %x = bitcast i32 %load0 to float
  %y = bitcast i32 %load1 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float undef, float undef, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_load_imm_mergex4:
;GCN-NOT: s_waitcnt;
;SI: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x2
;CI: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x2
;VI: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x8
define amdgpu_ps void @s_buffer_load_imm_mergex4(<4 x i32> inreg %desc) {
main_body:
  %load0 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 8, i32 0)
  %load1 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 12, i32 0)
  %load2 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 16, i32 0)
  %load3 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 20, i32 0)
  %x = bitcast i32 %load0 to float
  %y = bitcast i32 %load1 to float
  %z = bitcast i32 %load2 to float
  %w = bitcast i32 %load3 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float %w, i1 true, i1 true)
  ret void
}

@gv = external addrspace(1) global i32

;GCN-LABEL: {{^}}s_buffer_load_index_across_bb:
;GCN-NOT: s_waitcnt;
;GCN: v_or_b32
;GCN: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
define amdgpu_ps void @s_buffer_load_index_across_bb(<4 x i32> inreg %desc, i32 %index) {
main_body:
  %tmp = shl i32 %index, 4
  store i32 %tmp, ptr addrspace(1) @gv
  br label %bb1

bb1:                                              ; preds = %main_body
  %tmp1 = or i32 %tmp, 8
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 %tmp1, i32 0)
  %bitcast = bitcast i32 %load to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %bitcast, float undef, float undef, float undef, i1 true, i1 true)
  ret void
}

;GCN-LABEL: {{^}}s_buffer_load_index_across_bb_merged:
;GCN-NOT: s_waitcnt;
;GCN: buffer_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen offset:8
define amdgpu_ps void @s_buffer_load_index_across_bb_merged(<4 x i32> inreg %desc, i32 %index) {
main_body:
  %tmp = shl i32 %index, 4
  br label %bb1

bb1:                                              ; preds = %main_body
  %tmp1 = or i32 %tmp, 8
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 %tmp1, i32 0)
  %tmp2 = or i32 %tmp1, 4
  %load2 = tail call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 %tmp2, i32 0)
  %bitcast = bitcast i32 %load to float
  %bitcast2 = bitcast i32 %load2 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %bitcast, float %bitcast2, float undef, float undef, i1 true, i1 true)
  ret void
}

; GCN-LABEL: {{^}}s_buffer_load_imm_neg1:
; GCN: s_mov_b32 [[K:s[0-9]+]], -1{{$}}
; GCN: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_neg1(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 -1, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_neg4:
; SI: s_mov_b32 [[K:s[0-9]+]], -4{{$}}
; SI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; CI: s_buffer_load_dword s0, s[0:3], 0x3fffffff{{$}}

; VI: s_mov_b32 [[K:s[0-9]+]], -4{{$}}
; VI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_neg4(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 -4, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_neg8:
; SI: s_mov_b32 [[K:s[0-9]+]], -8{{$}}
; SI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; CI: s_buffer_load_dword s0, s[0:3], 0x3ffffffe{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_neg8(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 -8, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_bit31:
; SI: s_brev_b32 [[K:s[0-9]+]], 1{{$}}
; SI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; CI: s_buffer_load_dword s0, s[0:3], 0x20000000{{$}}

; VI: s_brev_b32 [[K:s[0-9]+]], 1{{$}}
; VI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_bit31(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 -2147483648, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_bit30:
; SI: s_mov_b32 [[K:s[0-9]+]], 2.0{{$}}
; SI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; CI: s_buffer_load_dword s0, s[0:3], 0x10000000{{$}}

; VI: s_mov_b32 [[K:s[0-9]+]], 2.0{{$}}
; VI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_bit30(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 1073741824, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_bit29:
; SI: s_brev_b32 [[K:s[0-9]+]], 4{{$}}
; SI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; CI: s_buffer_load_dword s0, s[0:3], 0x8000000{{$}}

; VI: s_brev_b32 [[K:s[0-9]+]], 4{{$}}
; VI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_bit29(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 536870912, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_bit21:
; SI: s_mov_b32 [[K:s[0-9]+]], 0x200000{{$}}
; SI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; CI: s_buffer_load_dword s0, s[0:3], 0x80000{{$}}

; VI: s_mov_b32 [[K:s[0-9]+]], 0x200000{{$}}
; VI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_bit21(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 2097152, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_bit20:
; SI: s_mov_b32 [[K:s[0-9]+]], 0x100000{{$}}
; SI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; CI: s_buffer_load_dword s0, s[0:3], 0x40000{{$}}

; VI: s_mov_b32 [[K:s[0-9]+]], 0x100000{{$}}
; VI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_bit20(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 1048576, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_neg_bit20:
; SI: s_mov_b32 [[K:s[0-9]+]], 0xfff00000{{$}}
; SI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; CI: s_buffer_load_dword s0, s[0:3], 0x3ffc0000{{$}}

; VI: s_mov_b32 [[K:s[0-9]+]], 0xfff00000{{$}}
; VI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_neg_bit20(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32  -1048576, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_bit19:
; SI: s_mov_b32 [[K:s[0-9]+]], 0x80000{{$}}
; SI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; CI: s_buffer_load_dword s0, s[0:3], 0x20000{{$}}

; VI: s_buffer_load_dword s0, s[0:3], 0x80000{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_bit19(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 524288, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_neg_bit19:
; SI: s_mov_b32 [[K:s[0-9]+]], 0xfff80000{{$}}
; SI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; CI: s_buffer_load_dword s0, s[0:3], 0x3ffe0000{{$}}

; VI: s_mov_b32 [[K:s[0-9]+]], 0xfff80000{{$}}
; VI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_neg_bit19(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 -524288, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_255:
; SICI: s_movk_i32 [[K:s[0-9]+]], 0xff{{$}}
; SICI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; VI: s_buffer_load_dword s0, s[0:3], 0xff{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_255(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 255, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_256:
; SICI: s_buffer_load_dword s0, s[0:3], 0x40{{$}}
; VI: s_buffer_load_dword s0, s[0:3], 0x100{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_256(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 256, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_1016:
; SICI: s_buffer_load_dword s0, s[0:3], 0xfe{{$}}
; VI: s_buffer_load_dword s0, s[0:3], 0x3f8{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_1016(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 1016, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_1020:
; SICI: s_buffer_load_dword s0, s[0:3], 0xff{{$}}
; VI: s_buffer_load_dword s0, s[0:3], 0x3fc{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_1020(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 1020, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_1021:
; SICI: s_movk_i32 [[K:s[0-9]+]], 0x3fd{{$}}
; SICI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_1021(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 1021, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_1024:
; SI: s_movk_i32 [[K:s[0-9]+]], 0x400{{$}}
; SI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; CI: s_buffer_load_dword s0, s[0:3], 0x100{{$}}

; VI: s_buffer_load_dword s0, s[0:3], 0x400{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_1024(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 1024, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_1025:
; SICI: s_movk_i32 [[K:s[0-9]+]], 0x401{{$}}
; SICI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; VI: s_buffer_load_dword s0, s[0:3], 0x401{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_1025(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 1025, i32 0)
  ret i32 %load
}

; GCN-LABEL: {{^}}s_buffer_load_imm_1028:
; SI: s_movk_i32 [[K:s[0-9]+]], 0x400{{$}}
; SI: s_buffer_load_dword s0, s[0:3], [[K]]{{$}}

; CI: s_buffer_load_dword s0, s[0:3], 0x100{{$}}
; VI: s_buffer_load_dword s0, s[0:3], 0x400{{$}}
define amdgpu_ps i32 @s_buffer_load_imm_1028(<4 x i32> inreg %desc) {
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 1024, i32 0)
  ret i32 %load
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1)
declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32>, i32, i32)
declare <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32>, i32, i32)
declare <3 x i32> @llvm.amdgcn.s.buffer.load.v3i32(<4 x i32>, i32, i32)
declare <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32>, i32, i32)
