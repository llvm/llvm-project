; RUN: llc -march=amdgcn -mcpu=gfx900 -amdgpu-sroa=0 -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=GCN,GFX9,GFX9-MUBUF %s
; RxN: llc -march=amdgcn -mcpu=gfx906 -amdgpu-sroa=0 -mattr=-promote-alloca,+sram-ecc -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=fiji -amdgpu-sroa=0 -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=GCN,GFX803,NO-D16-HI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -amdgpu-sroa=0 -mattr=-promote-alloca -mattr=+enable-flat-scratch -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=GCN,GFX9,GFX9-FLATSCR %s

; GCN-LABEL: {{^}}store_global_hi_v2i16:
; GCN: s_waitcnt

; GFX9-NEXT: global_store_short_d16_hi v[0:1], v2, off

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; GFX803-NEXT: flat_store_short v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2i16(ptr addrspace(1) %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  store i16 %hi, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_v2f16:
; GCN: s_waitcnt

; GFX9-NEXT: global_store_short_d16_hi v[0:1], v2, off

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; GFX803-NEXT: flat_store_short v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2f16(ptr addrspace(1) %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x half>
  %hi = extractelement <2 x half> %value, i32 1
  store half %hi, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_i32_shift:
; GCN: s_waitcnt

; GFX9-NEXT: global_store_short_d16_hi v[0:1], v2, off

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; GFX803-NEXT: flat_store_short v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_i32_shift(ptr addrspace(1) %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i16
  store i16 %hi, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_v2i16_i8:
; GCN: s_waitcnt

; GFX9-NEXT: global_store_byte_d16_hi v[0:1], v2, off

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; GFX803-NEXT: flat_store_byte v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2i16_i8(ptr addrspace(1) %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  store i8 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_i8_shift:
; GCN: s_waitcnt

; GFX9-NEXT: global_store_byte_d16_hi v[0:1], v2, off

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; GFX803-NEXT: flat_store_byte v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_i8_shift(ptr addrspace(1) %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i8
  store i8 %hi, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_v2i16_max_offset:
; GCN: s_waitcnt
; GFX9-NEXT: global_store_short_d16_hi v[0:1], v2, off offset:4094

; GFX803-DAG: v_add_u32_e32
; GFX803-DAG: v_addc_u32_e32
; GFX803-DAG: v_lshrrev_b32_e32 v2, 16, v2
; GFX803: flat_store_short v[0:1], v2{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2i16_max_offset(ptr addrspace(1) %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds i16, ptr addrspace(1) %out, i64 2047
  store i16 %hi, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_v2i16_min_offset:
; GCN: s_waitcnt
; GFX9-NEXT: global_store_short_d16_hi v[0:1], v2, off offset:-4096{{$}}

; GFX803-DAG: v_add_u32_e32
; GFX803-DAG: v_addc_u32_e32
; GFX803-DAG: v_lshrrev_b32_e32 v2, 16, v2
; GFX803: flat_store_short v[0:1], v{{[0-9]$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2i16_min_offset(ptr addrspace(1) %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds i16, ptr addrspace(1) %out, i64 -2048
  store i16 %hi, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_v2i16_i8_max_offset:
; GCN: s_waitcnt
; GFX9-NEXT: global_store_byte_d16_hi v[0:1], v2, off offset:4095

; GFX803-DAG: v_add_u32_e32
; GFX803-DAG: v_addc_u32_e32
; GFX803-DAG: v_lshrrev_b32_e32 v2, 16, v2
; GFX803: flat_store_byte v[0:1], v{{[0-9]$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2i16_i8_max_offset(ptr addrspace(1) %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  %gep = getelementptr inbounds i8, ptr addrspace(1) %out, i64 4095
  store i8 %trunc, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_v2i16_i8_min_offset:
; GCN: s_waitcnt
; GFX9-NEXT: global_store_byte_d16_hi v[0:1], v2, off offset:-4095

; GFX803-DAG: v_add_u32_e32
; GFX803-DAG: v_addc_u32_e32
; GFX803-DAG: v_lshrrev_b32_e32 v2, 16, v2
; GFX803: flat_store_byte v[0:1], v{{[0-9]$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2i16_i8_min_offset(ptr addrspace(1) %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  %gep = getelementptr inbounds i8, ptr addrspace(1) %out, i64 -4095
  store i8 %trunc, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2i16:
; GCN: s_waitcnt

; GFX9-NEXT: flat_store_short_d16_hi v[0:1], v2{{$}}

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; NO-D16-HI-NEXT: flat_store_short v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2i16(ptr %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  store i16 %hi, ptr %out
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2f16:
; GCN: s_waitcnt

; GFX9-NEXT: flat_store_short_d16_hi v[0:1], v2{{$}}

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; NO-D16-HI-NEXT: flat_store_short v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2f16(ptr %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x half>
  %hi = extractelement <2 x half> %value, i32 1
  store half %hi, ptr %out
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_i32_shift:
; GCN: s_waitcnt

; GFX9-NEXT: flat_store_short_d16_hi v[0:1], v2{{$}}

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; NO-D16-HI-NEXT: flat_store_short v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_i32_shift(ptr %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i16
  store i16 %hi, ptr %out
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2i16_i8:
; GCN: s_waitcnt

; GFX9-NEXT: flat_store_byte_d16_hi v[0:1], v2{{$}}

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; NO-D16-HI-NEXT: flat_store_byte v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2i16_i8(ptr %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  store i8 %trunc, ptr %out
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_i8_shift:
; GCN: s_waitcnt

; GFX9-NEXT: flat_store_byte_d16_hi v[0:1], v2{{$}}

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; NO-D16-HI-NEXT: flat_store_byte v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_i8_shift(ptr %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i8
  store i8 %hi, ptr %out
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2i16_max_offset:
; GCN: s_waitcnt
; GFX9-NEXT: flat_store_short_d16_hi v[0:1], v2 offset:4094{{$}}

; GFX803-DAG: v_add_u32_e32
; GFX803-DAG: v_addc_u32_e32
; GFX803-DAG: v_lshrrev_b32_e32 v2, 16, v2
; GFX803: flat_store_short v[0:1], v2{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2i16_max_offset(ptr %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds i16, ptr %out, i64 2047
  store i16 %hi, ptr %gep
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2i16_neg_offset:
; GCN: s_waitcnt
; GFX803: v_add{{(_co)?}}_{{i|u}}32_e32
; GFX803: v_addc_u32_e32

; GFX9-DAG: v_add_co_u32_e32 v{{[0-9]+}}, vcc, 0xfffff802, v
; GFX9-DAG: v_addc_co_u32_e32 v{{[0-9]+}}, vcc, -1, v

; GFX9-NEXT: flat_store_short_d16_hi v[0:1], v2{{$}}
; GFX803: flat_store_short v[0:1], v2{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2i16_neg_offset(ptr %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds i16, ptr %out, i64 -1023
  store i16 %hi, ptr %gep
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2i16_i8_max_offset:
; GCN: s_waitcnt
; GFX9-NEXT: flat_store_byte_d16_hi v[0:1], v2 offset:4095{{$}}

; GFX803-DAG: v_lshrrev_b32_e32 v2, 16, v2
; GFX803-DAG: v_add_u32_e32
; GFX803-DAG: v_addc_u32_e32
; GFX803: flat_store_byte v[0:1], v2{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2i16_i8_max_offset(ptr %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  %gep = getelementptr inbounds i8, ptr %out, i64 4095
  store i8 %trunc, ptr %gep
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2i16_i8_neg_offset:
; GCN: s_waitcnt

; GFX803-DAG: v_add_u32_e32
; GFX803-DAG: v_addc_u32_e32

; GFX9-DAG: v_add_co_u32_e32 v{{[0-9]+}}, vcc, 0xfffff001, v
; GFX9-DAG: v_addc_co_u32_e32 v{{[0-9]+}}, vcc, -1, v{{[0-9]+}}, vcc

; GFX9-NEXT: flat_store_byte_d16_hi v[0:1], v2{{$}}

; GFX803-DAG: v_lshrrev_b32_e32 v2, 16, v2
; GFX803: flat_store_byte v[0:1], v2{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2i16_i8_neg_offset(ptr %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  %gep = getelementptr inbounds i8, ptr %out, i64 -4095
  store i8 %trunc, ptr %gep
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_v2i16:
; GCN: s_waitcnt

; GFX9-MUBUF-NEXT:   buffer_store_short_d16_hi v1, v0, s[0:3], 0 offen{{$}}
; GFX9-FLATSCR-NEXT: scratch_store_short_d16_hi v0, v1, off

; NO-D16-HI: v_lshrrev_b32_e32 v1, 16, v1
; NO-D16-HI: buffer_store_short v1, v0, s[0:3], 0 offen{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_v2i16(ptr addrspace(5) %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  store i16 %hi, ptr addrspace(5) %out
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_v2f16:
; GCN: s_waitcnt

; GFX9-MUBUF-NEXT:   buffer_store_short_d16_hi v1, v0, s[0:3], 0 offen{{$}}
; GFX9-FLATSCR-NEXT: scratch_store_short_d16_hi v0, v1, off{{$}}

; NO-D16-HI: v_lshrrev_b32_e32 v1, 16, v1
; NO-D16-HI: buffer_store_short v1, v0, s[0:3], 0 offen{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_v2f16(ptr addrspace(5) %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x half>
  %hi = extractelement <2 x half> %value, i32 1
  store half %hi, ptr addrspace(5) %out
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_i32_shift:
; GCN: s_waitcnt

; GFX9-MUBUF-NEXT:   buffer_store_short_d16_hi v1, v0, s[0:3], 0 offen{{$}}
; GFX9-FLATSCR-NEXT: scratch_store_short_d16_hi v0, v1, off{{$}}

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v1, 16, v1
; NO-D16-HI-NEXT: buffer_store_short v1, v0, s[0:3], 0 offen{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_i32_shift(ptr addrspace(5) %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i16
  store i16 %hi, ptr addrspace(5) %out
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_v2i16_i8:
; GCN: s_waitcnt

; GFX9-MUBUF-NEXT:   buffer_store_byte_d16_hi v1, v0, s[0:3], 0 offen{{$}}
; GFX9-FLATSCR-NEXT: scratch_store_byte_d16_hi v0, v1, off{{$}}

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v1, 16, v1
; NO-D16-HI-NEXT: buffer_store_byte v1, v0, s[0:3], 0 offen{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_v2i16_i8(ptr addrspace(5) %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  store i8 %trunc, ptr addrspace(5) %out
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_i8_shift:
; GCN: s_waitcnt

; GFX9-MUBUF-NEXT:   buffer_store_byte_d16_hi v1, v0, s[0:3], 0 offen{{$}}
; GFX9-FLATSCR-NEXT: scratch_store_byte_d16_hi v0, v1, off{{$}}

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v1, 16, v1
; NO-D16-HI-NEXT: buffer_store_byte v1, v0, s[0:3], 0 offen{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_i8_shift(ptr addrspace(5) %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i8
  store i8 %hi, ptr addrspace(5) %out
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_v2i16_max_offset:
; GCN: s_waitcnt
; GFX9-MUBUF:   buffer_store_short_d16_hi v0, off, s[0:3], s32 offset:4094{{$}}
; GFX9-FLATSCR: scratch_store_short_d16_hi off, v0, s32 offset:4094{{$}}

; NO-D16-HI: v_lshrrev_b32_e32 v0, 16, v0
; NO-D16-HI-NEXT: buffer_store_short v0, off, s[0:3], s32 offset:4094{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_v2i16_max_offset(ptr addrspace(5) byval(i16) %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds i16, ptr addrspace(5) %out, i64 2047
  store i16 %hi, ptr addrspace(5) %gep
  ret void
}



; GCN-LABEL: {{^}}store_private_hi_v2i16_nooff:
; GCN: s_waitcnt

; GFX9-MUBUF-NEXT:   buffer_store_short_d16_hi v0, off, s[0:3], 0{{$}}
; GFX9-FLATSCR-NEXT: s_mov_b32 [[SOFF:s[0-9]+]], 0
; GFX9-FLATSCR-NEXT: scratch_store_short_d16_hi off, v0, [[SOFF]]{{$}}

; NO-D16-HI-NEXT: v_lshrrev_b32_e32 v0, 16, v0
; NO-D16-HI-NEXT: buffer_store_short v0, off, s[0:3], 0{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_v2i16_nooff(i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  store volatile i16 %hi, ptr addrspace(5) null
  ret void
}


; GCN-LABEL: {{^}}store_private_hi_v2i16_i8_nooff:
; GCN: s_waitcnt

; GFX9-MUBUF-NEXT:   buffer_store_byte_d16_hi v0, off, s[0:3], 0{{$}}
; GFX9-FLATSCR-NEXT: s_mov_b32 [[SOFF:s[0-9]+]], 0
; GFX9-FLATSCR-NEXT: scratch_store_byte_d16_hi off, v0, [[SOFF]]{{$}}

; NO-D16-HI: v_lshrrev_b32_e32 v0, 16, v0
; NO-D16-HI: buffer_store_byte v0, off, s[0:3], 0{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_v2i16_i8_nooff(i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  store volatile i8 %trunc, ptr addrspace(5) null
  ret void
}

; GCN-LABEL: {{^}}store_local_hi_v2i16:
; GCN: s_waitcnt

; GFX9-NEXT: ds_write_b16_d16_hi v0, v1{{$}}

; NO-D16-HI: v_lshrrev_b32_e32 v1, 16, v1
; NO-D16-HI: ds_write_b16 v0, v1

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_local_hi_v2i16(ptr addrspace(3) %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  store i16 %hi, ptr addrspace(3) %out
  ret void
}

; GCN-LABEL: {{^}}store_local_hi_v2f16:
; GCN: s_waitcnt

; GFX9-NEXT: ds_write_b16_d16_hi v0, v1{{$}}

; NO-D16-HI: v_lshrrev_b32_e32 v1, 16, v1
; NO-D16-HI: ds_write_b16 v0, v1

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_local_hi_v2f16(ptr addrspace(3) %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x half>
  %hi = extractelement <2 x half> %value, i32 1
  store half %hi, ptr addrspace(3) %out
  ret void
}

; GCN-LABEL: {{^}}store_local_hi_i32_shift:
; GCN: s_waitcnt

; GFX9-NEXT: ds_write_b16_d16_hi v0, v1{{$}}

; NO-D16-HI: v_lshrrev_b32_e32 v1, 16, v1
; NO-D16-HI: ds_write_b16 v0, v1

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_local_hi_i32_shift(ptr addrspace(3) %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i16
  store i16 %hi, ptr addrspace(3) %out
  ret void
}

; GCN-LABEL: {{^}}store_local_hi_v2i16_i8:
; GCN: s_waitcnt

; GFX9-NEXT: ds_write_b8_d16_hi v0, v1{{$}}

; NO-D16-HI: v_lshrrev_b32_e32 v1, 16, v1
; NO-D16-HI: ds_write_b8 v0, v1

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_local_hi_v2i16_i8(ptr addrspace(3) %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  store i8 %trunc, ptr addrspace(3) %out
  ret void
}

; GCN-LABEL: {{^}}store_local_hi_v2i16_max_offset:
; GCN: s_waitcnt
; GFX9-NEXT: ds_write_b16_d16_hi v0, v1 offset:65534{{$}}

; NO-D16-HI: v_lshrrev_b32_e32 v1, 16, v1
; NO-D16-HI: ds_write_b16 v0, v1 offset:65534{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_local_hi_v2i16_max_offset(ptr addrspace(3) %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds i16, ptr addrspace(3) %out, i64 32767
  store i16 %hi, ptr addrspace(3) %gep
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_v2i16_to_offset:
; GCN: s_waitcnt
; GFX9-MUBUF:        buffer_store_dword
; GFX9-MUBUF-NEXT:   s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:   buffer_store_short_d16_hi v0, off, s[0:3], s32 offset:4058
; GFX9-MUBUF-NEXT:   s_waitcnt vmcnt(0)
; GFX9-FLATSCR:      scratch_store_dword
; GFX9-FLATSCR-NEXT: s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT: scratch_store_short_d16_hi off, v0, s32 offset:4058
; GFX9-FLATSCR-NEXT: s_waitcnt vmcnt(0)
define void @store_private_hi_v2i16_to_offset(i32 %arg, ptr addrspace(5) %obj0) #0 {
entry:
  %obj1 = alloca [4096 x i16], align 2, addrspace(5)
  store volatile i32 123, ptr addrspace(5) %obj0
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds [4096 x i16], ptr addrspace(5) %obj1, i32 0, i32 2027
  store i16 %hi, ptr addrspace(5) %gep
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_v2i16_i8_to_offset:
; GCN: s_waitcnt
; GFX9-MUBUF:        buffer_store_dword
; GFX9-MUBUF-NEXT:   s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:   buffer_store_byte_d16_hi v0, off, s[0:3], s32 offset:4059
; GFX9-FLATSCR:      scratch_store_dword
; GFX9-FLATSCR-NEXT: s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT: scratch_store_byte_d16_hi off, v0, s32 offset:4059
; GFX9-FLATSCR-NEXT: s_waitcnt vmcnt(0)
define void @store_private_hi_v2i16_i8_to_offset(i32 %arg, ptr addrspace(5) %obj0) #0 {
entry:
  %obj1 = alloca [4096 x i8], align 2, addrspace(5)
  store volatile i32 123, ptr addrspace(5) %obj0
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds [4096 x i8], ptr addrspace(5) %obj1, i32 0, i32 4055
  %trunc = trunc i16 %hi to i8
  store i8 %trunc, ptr addrspace(5) %gep
  ret void
}

attributes #0 = { nounwind }
