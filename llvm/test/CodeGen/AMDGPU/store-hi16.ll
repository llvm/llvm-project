; RUN: llc -march=amdgcn -mcpu=gfx900 -amdgpu-sroa=0 -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=fiji -amdgpu-sroa=0 -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}store_global_hi_v2i16:
; GCN: s_waitcnt

; GFX9-NEXT: global_store_short_d16_hi v[0:1], v2, off

; VI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; VI-NEXT: flat_store_short v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2i16(i16 addrspace(1)* %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  store i16 %hi, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_v2f16:
; GCN: s_waitcnt

; GFX9-NEXT: global_store_short_d16_hi v[0:1], v2, off

; VI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; VI-NEXT: flat_store_short v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2f16(half addrspace(1)* %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x half>
  %hi = extractelement <2 x half> %value, i32 1
  store half %hi, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_i32_shift:
; GCN: s_waitcnt

; GFX9-NEXT: global_store_short_d16_hi v[0:1], v2, off

; VI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; VI-NEXT: flat_store_short v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_i32_shift(i16 addrspace(1)* %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i16
  store i16 %hi, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_v2i16_i8:
; GCN: s_waitcnt

; GFX9-NEXT: global_store_byte_d16_hi v[0:1], v2, off

; VI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; VI-NEXT: flat_store_byte v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2i16_i8(i8 addrspace(1)* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  store i8 %trunc, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_i8_shift:
; GCN: s_waitcnt

; GFX9-NEXT: global_store_byte_d16_hi v[0:1], v2, off

; VI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; VI-NEXT: flat_store_byte v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_i8_shift(i8 addrspace(1)* %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i8
  store i8 %hi, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_v2i16_max_offset:
; GCN: s_waitcnt
; GFX9-NEXT: global_store_short_d16_hi v[0:1], v2, off offset:4094

; VI-DAG: v_add_u32_e32
; VI-DAG: v_addc_u32_e32
; VI-DAG: v_lshrrev_b32_e32 v2, 16, v2

; VI: flat_store_short v[0:1], v2{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2i16_max_offset(i16 addrspace(1)* %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds i16, i16 addrspace(1)* %out, i64 2047
  store i16 %hi, i16 addrspace(1)* %gep
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_v2i16_min_offset:
; GCN: s_waitcnt
; GFX9-NEXT: global_store_short_d16_hi v[0:1], v2, off offset:-4096{{$}}

; VI-DAG: v_add_u32_e32
; VI-DAG: v_addc_u32_e32
; VI-DAG: v_lshrrev_b32_e32 v2, 16, v2

; VI: flat_store_short v[0:1], v{{[0-9]$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2i16_min_offset(i16 addrspace(1)* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds i16, i16 addrspace(1)* %out, i64 -2048
  store i16 %hi, i16 addrspace(1)* %gep
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_v2i16_i8_max_offset:
; GCN: s_waitcnt
; GFX9-NEXT: global_store_byte_d16_hi v[0:1], v2, off offset:4095

; VI-DAG: v_add_u32_e32
; VI-DAG: v_addc_u32_e32
; VI-DAG: v_lshrrev_b32_e32 v2, 16, v2
; VI: flat_store_byte v[0:1], v{{[0-9]$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2i16_i8_max_offset(i8 addrspace(1)* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  %gep = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 4095
  store i8 %trunc, i8 addrspace(1)* %gep
  ret void
}

; GCN-LABEL: {{^}}store_global_hi_v2i16_i8_min_offset:
; GCN: s_waitcnt
; GFX9-NEXT: global_store_byte_d16_hi v[0:1], v2, off offset:-4095

; VI-DAG: v_add_u32_e32
; VI-DAG: v_addc_u32_e32
; VI-DAG: v_lshrrev_b32_e32 v2, 16, v2

; VI: flat_store_byte v[0:1], v{{[0-9]$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_global_hi_v2i16_i8_min_offset(i8 addrspace(1)* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  %gep = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 -4095
  store i8 %trunc, i8 addrspace(1)* %gep
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2i16:
; GCN: s_waitcnt

; GFX9-NEXT: flat_store_short_d16_hi v[0:1], v2{{$}}

; VI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; VI-NEXT: flat_store_short v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2i16(i16* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  store i16 %hi, i16* %out
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2f16:
; GCN: s_waitcnt

; GFX9-NEXT: flat_store_short_d16_hi v[0:1], v2{{$}}

; VI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; VI-NEXT: flat_store_short v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2f16(half* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x half>
  %hi = extractelement <2 x half> %value, i32 1
  store half %hi, half* %out
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_i32_shift:
; GCN: s_waitcnt

; GFX9-NEXT: flat_store_short_d16_hi v[0:1], v2{{$}}

; VI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; VI-NEXT: flat_store_short v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_i32_shift(i16* %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i16
  store i16 %hi, i16* %out
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2i16_i8:
; GCN: s_waitcnt

; GFX9-NEXT: flat_store_byte_d16_hi v[0:1], v2{{$}}

; VI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; VI-NEXT: flat_store_byte v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2i16_i8(i8* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  store i8 %trunc, i8* %out
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_i8_shift:
; GCN: s_waitcnt

; GFX9-NEXT: flat_store_byte_d16_hi v[0:1], v2{{$}}

; VI-NEXT: v_lshrrev_b32_e32 v2, 16, v2
; VI-NEXT: flat_store_byte v[0:1], v2

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_i8_shift(i8* %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i8
  store i8 %hi, i8* %out
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2i16_max_offset:
; GCN: s_waitcnt
; GFX9-NEXT: flat_store_short_d16_hi v[0:1], v2 offset:4094{{$}}

; VI-DAG: v_add_u32_e32
; VI-DAG: v_addc_u32_e32
; VI-DAG: v_lshrrev_b32_e32 v2, 16, v2
; VI: flat_store_short v[0:1], v2{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2i16_max_offset(i16* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds i16, i16* %out, i64 2047
  store i16 %hi, i16* %gep
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2i16_neg_offset:
; GCN: s_waitcnt
; GCN: v_add{{(_co)?}}_{{i|u}}32_e32
; VI: v_addc_u32_e32
; GFX9: v_addc_co_u32_e32

; GFX9-NEXT: flat_store_short_d16_hi v[0:1], v2{{$}}
; VI: flat_store_short v[0:1], v2{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2i16_neg_offset(i16* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds i16, i16* %out, i64 -1023
  store i16 %hi, i16* %gep
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2i16_i8_max_offset:
; GCN: s_waitcnt
; GFX9-NEXT: flat_store_byte_d16_hi v[0:1], v2 offset:4095{{$}}

; VI-DAG: v_lshrrev_b32_e32 v2, 16, v2
; VI-DAG: v_add_u32_e32
; VI-DAG: v_addc_u32_e32
; VI: flat_store_byte v[0:1], v2{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2i16_i8_max_offset(i8* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  %gep = getelementptr inbounds i8, i8* %out, i64 4095
  store i8 %trunc, i8* %gep
  ret void
}

; GCN-LABEL: {{^}}store_flat_hi_v2i16_i8_neg_offset:
; GCN: s_waitcnt
; GCN-DAG: v_add{{(_co)?}}_{{i|u}}32_e32
; VI-DAG: v_addc_u32_e32
; GFX9-DAG: v_addc_co_u32_e32

; GFX9-NEXT: flat_store_byte_d16_hi v[0:1], v2{{$}}
; VI-DAG: v_lshrrev_b32_e32 v2, 16, v2
; VI: flat_store_byte v[0:1], v2{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_flat_hi_v2i16_i8_neg_offset(i8* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  %gep = getelementptr inbounds i8, i8* %out, i64 -4095
  store i8 %trunc, i8* %gep
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_v2i16:
; GCN: s_waitcnt

; GFX9-NEXT: buffer_store_short_d16_hi v1, v0, s[0:3], s4 offen{{$}}

; VI: v_lshrrev_b32_e32 v1, 16, v1
; VI: buffer_store_short v1, v0, s[0:3], s4 offen{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_v2i16(i16 addrspace(5)* %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  store i16 %hi, i16 addrspace(5)* %out
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_v2f16:
; GCN: s_waitcnt

; GFX9-NEXT: buffer_store_short_d16_hi v1, v0, s[0:3], s4 offen{{$}}

; VI: v_lshrrev_b32_e32 v1, 16, v1
; VI: buffer_store_short v1, v0, s[0:3], s4 offen{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_v2f16(half addrspace(5)* %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x half>
  %hi = extractelement <2 x half> %value, i32 1
  store half %hi, half addrspace(5)* %out
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_i32_shift:
; GCN: s_waitcnt

; GFX9-NEXT: buffer_store_short_d16_hi v1, v0, s[0:3], s4 offen{{$}}

; VI-NEXT: v_lshrrev_b32_e32 v1, 16, v1
; VI-NEXT: buffer_store_short v1, v0, s[0:3], s4 offen{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_i32_shift(i16 addrspace(5)* %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i16
  store i16 %hi, i16 addrspace(5)* %out
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_v2i16_i8:
; GCN: s_waitcnt

; GFX9-NEXT: buffer_store_byte_d16_hi v1, v0, s[0:3], s4 offen{{$}}

; VI-NEXT: v_lshrrev_b32_e32 v1, 16, v1
; VI-NEXT: buffer_store_byte v1, v0, s[0:3], s4 offen{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_v2i16_i8(i8 addrspace(5)* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  store i8 %trunc, i8 addrspace(5)* %out
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_i8_shift:
; GCN: s_waitcnt

; GFX9-NEXT: buffer_store_byte_d16_hi v1, v0, s[0:3], s4 offen{{$}}

; VI-NEXT: v_lshrrev_b32_e32 v1, 16, v1
; VI-NEXT: buffer_store_byte v1, v0, s[0:3], s4 offen{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_i8_shift(i8 addrspace(5)* %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i8
  store i8 %hi, i8 addrspace(5)* %out
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_v2i16_max_offset:
; GCN: s_waitcnt
; GFX9: buffer_store_short_d16_hi v0, off, s[0:3], s5 offset:4094{{$}}

; VI: v_lshrrev_b32_e32 v0, 16, v0
; VI-NEXT: buffer_store_short v0, off, s[0:3], s5 offset:4094{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_v2i16_max_offset(i16 addrspace(5)* byval %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds i16, i16 addrspace(5)* %out, i64 2045
  store i16 %hi, i16 addrspace(5)* %gep
  ret void
}



; GCN-LABEL: {{^}}store_private_hi_v2i16_nooff:
; GCN: s_waitcnt

; GFX9-NEXT: buffer_store_short_d16_hi v0, off, s[0:3], s4{{$}}

; VI-NEXT: v_lshrrev_b32_e32 v0, 16, v0
; VI-NEXT: buffer_store_short v0, off, s[0:3], s4{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_v2i16_nooff(i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  store volatile i16 %hi, i16 addrspace(5)* null
  ret void
}


; GCN-LABEL: {{^}}store_private_hi_v2i16_i8_nooff:
; GCN: s_waitcnt

; GFX9-NEXT: buffer_store_byte_d16_hi v0, off, s[0:3], s4{{$}}

; VI: v_lshrrev_b32_e32 v0, 16, v0
; VI: buffer_store_byte v0, off, s[0:3], s4{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_private_hi_v2i16_i8_nooff(i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  store volatile i8 %trunc, i8 addrspace(5)* null
  ret void
}

; GCN-LABEL: {{^}}store_local_hi_v2i16:
; GCN: s_waitcnt

; GFX9-NEXT: ds_write_b16_d16_hi v0, v1{{$}}

; VI: v_lshrrev_b32_e32 v1, 16, v1
; VI: ds_write_b16 v0, v1

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_local_hi_v2i16(i16 addrspace(3)* %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  store i16 %hi, i16 addrspace(3)* %out
  ret void
}

; GCN-LABEL: {{^}}store_local_hi_v2f16:
; GCN: s_waitcnt

; GFX9-NEXT: ds_write_b16_d16_hi v0, v1{{$}}

; VI: v_lshrrev_b32_e32 v1, 16, v1
; VI: ds_write_b16 v0, v1

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_local_hi_v2f16(half addrspace(3)* %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x half>
  %hi = extractelement <2 x half> %value, i32 1
  store half %hi, half addrspace(3)* %out
  ret void
}

; GCN-LABEL: {{^}}store_local_hi_i32_shift:
; GCN: s_waitcnt

; GFX9-NEXT: ds_write_b16_d16_hi v0, v1{{$}}

; VI: v_lshrrev_b32_e32 v1, 16, v1
; VI: ds_write_b16 v0, v1

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_local_hi_i32_shift(i16 addrspace(3)* %out, i32 %value) #0 {
entry:
  %hi32 = lshr i32 %value, 16
  %hi = trunc i32 %hi32 to i16
  store i16 %hi, i16 addrspace(3)* %out
  ret void
}

; GCN-LABEL: {{^}}store_local_hi_v2i16_i8:
; GCN: s_waitcnt

; GFX9-NEXT: ds_write_b8_d16_hi v0, v1{{$}}

; VI: v_lshrrev_b32_e32 v1, 16, v1
; VI: ds_write_b8 v0, v1

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_local_hi_v2i16_i8(i8 addrspace(3)* %out, i32 %arg) #0 {
entry:
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %trunc = trunc i16 %hi to i8
  store i8 %trunc, i8 addrspace(3)* %out
  ret void
}

; GCN-LABEL: {{^}}store_local_hi_v2i16_max_offset:
; GCN: s_waitcnt
; GFX9-NEXT: ds_write_b16_d16_hi v0, v1 offset:65534{{$}}

; VI: v_lshrrev_b32_e32 v1, 16, v1
; VI: ds_write_b16 v0, v1 offset:65534{{$}}

; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @store_local_hi_v2i16_max_offset(i16 addrspace(3)* %out, i32 %arg) #0 {
entry:
  ; FIXME: ABI for pre-gfx9
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds i16, i16 addrspace(3)* %out, i64 32767
  store i16 %hi, i16 addrspace(3)* %gep
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_v2i16_to_offset:
; GCN: s_waitcnt
; GFX9: buffer_store_dword
; GFX9-NEXT: buffer_store_short_d16_hi v0, off, s[0:3], s5 offset:4094
define void @store_private_hi_v2i16_to_offset(i32 %arg) #0 {
entry:
  %obj0 = alloca [10 x i32], align 4, addrspace(5)
  %obj1 = alloca [4096 x i16], align 2, addrspace(5)
  %bc = bitcast [10 x i32] addrspace(5)* %obj0 to i32 addrspace(5)*
  store volatile i32 123, i32 addrspace(5)* %bc
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds [4096 x i16], [4096 x i16] addrspace(5)* %obj1, i32 0, i32 2025
  store i16 %hi, i16 addrspace(5)* %gep
  ret void
}

; GCN-LABEL: {{^}}store_private_hi_v2i16_i8_to_offset:
; GCN: s_waitcnt
; GFX9: buffer_store_dword
; GFX9-NEXT: buffer_store_byte_d16_hi v0, off, s[0:3], s5 offset:4095
define void @store_private_hi_v2i16_i8_to_offset(i32 %arg) #0 {
entry:
  %obj0 = alloca [10 x i32], align 4, addrspace(5)
  %obj1 = alloca [4096 x i8], align 2, addrspace(5)
  %bc = bitcast [10 x i32] addrspace(5)* %obj0 to i32 addrspace(5)*
  store volatile i32 123, i32 addrspace(5)* %bc
  %value = bitcast i32 %arg to <2 x i16>
  %hi = extractelement <2 x i16> %value, i32 1
  %gep = getelementptr inbounds [4096 x i8], [4096 x i8] addrspace(5)* %obj1, i32 0, i32 4051
  %trunc = trunc i16 %hi to i8
  store i8 %trunc, i8 addrspace(5)* %gep
  ret void
}

attributes #0 = { nounwind }
