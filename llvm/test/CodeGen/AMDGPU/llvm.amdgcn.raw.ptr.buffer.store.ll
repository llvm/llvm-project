;RUN: llc < %s -mtriple=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck -check-prefix=VERDE %s
;RUN: llc < %s -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}buffer_store:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[0:3], off, s[0:3], 0
;CHECK: buffer_store_dwordx4 v[4:7], off, s[0:3], 0 glc
;CHECK: buffer_store_dwordx4 v[8:11], off, s[0:3], 0 slc
define amdgpu_ps void @buffer_store(ptr addrspace(8) inreg, <4 x float>, <4 x float>, <4 x float>) {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %1, ptr addrspace(8) %0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %2, ptr addrspace(8) %0, i32 0, i32 0, i32 1)
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %3, ptr addrspace(8) %0, i32 0, i32 0, i32 2)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_immoffs:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[0:3], off, s[0:3], 0 offset:42
define amdgpu_ps void @buffer_store_immoffs(ptr addrspace(8) inreg, <4 x float>) {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %1, ptr addrspace(8) %0, i32 42, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_ofs:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_ofs(ptr addrspace(8) inreg, <4 x float>, i32) {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %1, ptr addrspace(8) %0, i32 %2, i32 0, i32 0)
  ret void
}

; Ideally, the register allocator would avoid the wait here
;
;CHECK-LABEL: {{^}}buffer_store_wait:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
;VERDE: s_waitcnt expcnt(0)
;CHECK: buffer_load_dwordx4 v[0:3], v5, s[0:3], 0 offen
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_store_dwordx4 v[0:3], v6, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_wait(ptr addrspace(8) inreg, <4 x float>, i32, i32, i32) {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %1, ptr addrspace(8) %0, i32 %2, i32 0, i32 0)
  %data = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %0, i32 %3, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %data, ptr addrspace(8) %0, i32 %4, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x1:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dword v0, v1, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_x1(ptr addrspace(8) inreg %rsrc, float %data, i32 %offset) {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %data, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x2:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_x2(ptr addrspace(8) inreg %rsrc, <2 x float> %data, i32 %offset) #0 {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2f32(<2 x float> %data, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x1_offen_merged_and:
;CHECK-NOT: s_waitcnt
;CHECK-DAG: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:4
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:28
define amdgpu_ps void @buffer_store_x1_offen_merged_and(ptr addrspace(8) inreg %rsrc, i32 %a, float %v1, float %v2, float %v3, float %v4, float %v5, float %v6) {
  %a1 = add i32 %a, 4
  %a2 = add i32 %a, 8
  %a3 = add i32 %a, 12
  %a4 = add i32 %a, 16
  %a5 = add i32 %a, 28
  %a6 = add i32 %a, 32
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v1, ptr addrspace(8) %rsrc, i32 %a1, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v2, ptr addrspace(8) %rsrc, i32 %a2, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v3, ptr addrspace(8) %rsrc, i32 %a3, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v4, ptr addrspace(8) %rsrc, i32 %a4, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v5, ptr addrspace(8) %rsrc, i32 %a5, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v6, ptr addrspace(8) %rsrc, i32 %a6, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x1_offen_merged_or:
;CHECK-NOT: s_waitcnt
;CHECK-DAG: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], v{{[0-9]}}, s[0:3], 0 offen offset:4
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], v{{[0-9]}}, s[0:3], 0 offen offset:28
define amdgpu_ps void @buffer_store_x1_offen_merged_or(ptr addrspace(8) inreg %rsrc, i32 %inp, float %v1, float %v2, float %v3, float %v4, float %v5, float %v6) {
  %a = shl i32 %inp, 6
  %a1 = add i32 %a, 4
  %a2 = add i32 %a, 8
  %a3 = add i32 %a, 12
  %a4 = add i32 %a, 16
  %a5 = add i32 %a, 28
  %a6 = add i32 %a, 32
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v1, ptr addrspace(8) %rsrc, i32 %a1, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v2, ptr addrspace(8) %rsrc, i32 %a2, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v3, ptr addrspace(8) %rsrc, i32 %a3, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v4, ptr addrspace(8) %rsrc, i32 %a4, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v5, ptr addrspace(8) %rsrc, i32 %a5, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v6, ptr addrspace(8) %rsrc, i32 %a6, i32 0, i32 0)
  ret void
}


;CHECK-LABEL: {{^}}buffer_store_x1_offen_merged_glc_slc:
;CHECK-NOT: s_waitcnt
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:4{{$}}
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:12 glc{{$}}
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:28 glc slc{{$}}
define amdgpu_ps void @buffer_store_x1_offen_merged_glc_slc(ptr addrspace(8) inreg %rsrc, i32 %a, float %v1, float %v2, float %v3, float %v4, float %v5, float %v6) {
  %a1 = add i32 %a, 4
  %a2 = add i32 %a, 8
  %a3 = add i32 %a, 12
  %a4 = add i32 %a, 16
  %a5 = add i32 %a, 28
  %a6 = add i32 %a, 32
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v1, ptr addrspace(8) %rsrc, i32 %a1, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v2, ptr addrspace(8) %rsrc, i32 %a2, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v3, ptr addrspace(8) %rsrc, i32 %a3, i32 0, i32 1)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v4, ptr addrspace(8) %rsrc, i32 %a4, i32 0, i32 1)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v5, ptr addrspace(8) %rsrc, i32 %a5, i32 0, i32 3)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v6, ptr addrspace(8) %rsrc, i32 %a6, i32 0, i32 3)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x2_offen_merged_and:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:4
define amdgpu_ps void @buffer_store_x2_offen_merged_and(ptr addrspace(8) inreg %rsrc, i32 %a, <2 x float> %v1, <2 x float> %v2) {
  %a1 = add i32 %a, 4
  %a2 = add i32 %a, 12
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2f32(<2 x float> %v1, ptr addrspace(8) %rsrc, i32 %a1, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2f32(<2 x float> %v2, ptr addrspace(8) %rsrc, i32 %a2, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x2_offen_merged_or:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], v{{[0-9]}}, s[0:3], 0 offen offset:4
define amdgpu_ps void @buffer_store_x2_offen_merged_or(ptr addrspace(8) inreg %rsrc, i32 %inp, <2 x float> %v1, <2 x float> %v2) {
  %a = shl i32 %inp, 4
  %a1 = add i32 %a, 4
  %a2 = add i32 %a, 12
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2f32(<2 x float> %v1, ptr addrspace(8) %rsrc, i32 %a1, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2f32(<2 x float> %v2, ptr addrspace(8) %rsrc, i32 %a2, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x1_offset_merged:
;CHECK-NOT: s_waitcnt
;CHECK-DAG: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], off, s[0:3], 0 offset:4
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], off, s[0:3], 0 offset:28
define amdgpu_ps void @buffer_store_x1_offset_merged(ptr addrspace(8) inreg %rsrc, float %v1, float %v2, float %v3, float %v4, float %v5, float %v6) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v1, ptr addrspace(8) %rsrc, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v2, ptr addrspace(8) %rsrc, i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v3, ptr addrspace(8) %rsrc, i32 12, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v4, ptr addrspace(8) %rsrc, i32 16, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v5, ptr addrspace(8) %rsrc, i32 28, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v6, ptr addrspace(8) %rsrc, i32 32, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x2_offset_merged:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], off, s[0:3], 0 offset:4
define amdgpu_ps void @buffer_store_x2_offset_merged(ptr addrspace(8) inreg %rsrc, <2 x float> %v1,<2 x float> %v2) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2f32(<2 x float> %v1, ptr addrspace(8) %rsrc, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2f32(<2 x float> %v2, ptr addrspace(8) %rsrc, i32 12, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_int:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[0:3], off, s[0:3], 0
;CHECK: buffer_store_dwordx2 v[4:5], off, s[0:3], 0 glc
;CHECK: buffer_store_dword v6, off, s[0:3], 0 slc
define amdgpu_ps void @buffer_store_int(ptr addrspace(8) inreg, <4 x i32>, <2 x i32>, i32) {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %1, ptr addrspace(8) %0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %2, ptr addrspace(8) %0, i32 0, i32 0, i32 1)
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3, ptr addrspace(8) %0, i32 0, i32 0, i32 2)
  ret void
}

;CHECK-LABEL: {{^}}raw_ptr_buffer_store_byte:
;CHECK-NEXT: %bb.
;CHECK-NEXT: v_cvt_u32_f32_e32 v{{[0-9]}}, v{{[0-9]}}
;CHECK-NEXT: buffer_store_byte v{{[0-9]}}, off, s[0:3], 0
;CHECK-NEXT: s_endpgm
define amdgpu_ps void @raw_ptr_buffer_store_byte(ptr addrspace(8) inreg %rsrc, float %v1) {
main_body:
  %v2 = fptoui float %v1 to i32
  %v3 = trunc i32 %v2 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %v3, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}raw_ptr_buffer_store_short:
;CHECK-NEXT: %bb.
;CHECK-NEXT: v_cvt_u32_f32_e32 v{{[0-9]}}, v{{[0-9]}}
;CHECK-NEXT: buffer_store_short v{{[0-9]}}, off, s[0:3], 0
;CHECK-NEXT: s_endpgm
define amdgpu_ps void @raw_ptr_buffer_store_short(ptr addrspace(8) inreg %rsrc, float %v1) {
main_body:
  %v2 = fptoui float %v1 to i32
  %v3 = trunc i32 %v2 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %v3, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}raw_ptr_buffer_store_f16:
;CHECK-NEXT: %bb.
;CHECK-NOT: v0
;CHECK-NEXT: buffer_store_short v0, off, s[0:3], 0
;CHECK-NEXT: s_endpgm
define amdgpu_ps void @raw_ptr_buffer_store_f16(ptr addrspace(8) inreg %rsrc, i32 %v1) {
main_body:
  %trunc = trunc i32 %v1 to i16
  %cast = bitcast i16 %trunc to half
  call void @llvm.amdgcn.raw.ptr.buffer.store.f16(half %cast, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_v2f16:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dword v0, v1, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_v2f16(ptr addrspace(8) inreg %rsrc, <2 x half> %data, i32 %offset) {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2f16(<2 x half> %data, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_v4f16:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_v4f16(ptr addrspace(8) inreg %rsrc, <4 x half> %data, i32 %offset) #0 {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4f16(<4 x half> %data, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_v8f16:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_v8f16(ptr addrspace(8) inreg %rsrc, <8 x half> %data, i32 %offset) #0 {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v8f16(<8 x half> %data, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_v2bf16:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dword v0, v1, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_v2bf16(ptr addrspace(8) inreg %rsrc, <2 x bfloat> %data, i32 %offset) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2bf16(<2 x bfloat> %data, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_v4bf16:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_v4bf16(ptr addrspace(8) inreg %rsrc, <4 x bfloat> %data, i32 %offset) #0 {
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4bf16(<4 x bfloat> %data, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}raw_ptr_buffer_store_i16:
;CHECK-NEXT: %bb.
;CHECK-NOT: v0
;CHECK-NEXT: buffer_store_short v0, off, s[0:3], 0
;CHECK-NEXT: s_endpgm
define amdgpu_ps void @raw_ptr_buffer_store_i16(ptr addrspace(8) inreg %rsrc, i32 %v1) {
main_body:
  %trunc = trunc i32 %v1 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %trunc, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_v2i16:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dword v0, v1, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_v2i16(ptr addrspace(8) inreg %rsrc, <2 x i16> %data, i32 %offset) {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i16(<2 x i16> %data, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_v4i16:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_v4i16(ptr addrspace(8) inreg %rsrc, <4 x i16> %data, i32 %offset) #0 {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4i16(<4 x i16> %data, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}

; FIXME:
; define amdgpu_ps void @buffer_store_v6i16(ptr addrspace(8) inreg %rsrc, <6 x i16> %data, i32 %offset) #0 {
; main_body:
;   call void @llvm.amdgcn.raw.ptr.buffer.store.v6i16(<6 x i16> %data, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
;   ret void
; }

;CHECK-LABEL: {{^}}buffer_store_v8i16:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_v8i16(ptr addrspace(8) inreg %rsrc, <8 x i16> %data, i32 %offset) #0 {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v8i16(<8 x i16> %data, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}raw_ptr_buffer_store_x1_offset_merged:
;CHECK-NOT: s_waitcnt
;CHECK-DAG: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], off, s[0:3], 0 offset:4
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], off, s[0:3], 0 offset:28
define amdgpu_ps void @raw_ptr_buffer_store_x1_offset_merged(ptr addrspace(8) inreg %rsrc, float %v1, float %v2, float %v3, float %v4, float %v5, float %v6) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v1, ptr addrspace(8) %rsrc, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v2, ptr addrspace(8) %rsrc, i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v3, ptr addrspace(8) %rsrc, i32 12, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v4, ptr addrspace(8) %rsrc, i32 16, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v5, ptr addrspace(8) %rsrc, i32 28, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v6, ptr addrspace(8) %rsrc, i32 32, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}raw_ptr_buffer_store_x1_offset_swizzled_not_merged:
;CHECK-DAG: buffer_store_dword v{{[0-9]}}, off, s[0:3], 0 offset:4
;CHECK-DAG: buffer_store_dword v{{[0-9]}}, off, s[0:3], 0 offset:8
;CHECK-DAG: buffer_store_dword v{{[0-9]}}, off, s[0:3], 0 offset:12
;CHECK-DAG: buffer_store_dword v{{[0-9]}}, off, s[0:3], 0 offset:16
;CHECK-DAG: buffer_store_dword v{{[0-9]}}, off, s[0:3], 0 offset:28
;CHECK-DAG: buffer_store_dword v{{[0-9]}}, off, s[0:3], 0 offset:32
define amdgpu_ps void @raw_ptr_buffer_store_x1_offset_swizzled_not_merged(ptr addrspace(8) inreg %rsrc, float %v1, float %v2, float %v3, float %v4, float %v5, float %v6) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v1, ptr addrspace(8) %rsrc, i32 4, i32 0, i32 8)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v2, ptr addrspace(8) %rsrc, i32 8, i32 0, i32 8)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v3, ptr addrspace(8) %rsrc, i32 12, i32 0, i32 8)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v4, ptr addrspace(8) %rsrc, i32 16, i32 0, i32 8)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v5, ptr addrspace(8) %rsrc, i32 28, i32 0, i32 8)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %v6, ptr addrspace(8) %rsrc, i32 32, i32 0, i32 8)
  ret void
}

define void @buffer_store_f64__voffset_add(ptr addrspace(8) inreg %rsrc, double %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_f64__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_f64__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.f64(double %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v2f64__voffset_add(ptr addrspace(8) inreg %rsrc, <2 x double> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v2f64__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v2f64__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2f64(<2 x double> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_i64__voffset_add(ptr addrspace(8) inreg %rsrc, i64 %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_i64__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_i64__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.i64(i64 %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v2i64__voffset_add(ptr addrspace(8) inreg %rsrc, <2 x i64> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v2i64__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v2i64__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i64(<2 x i64> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_p0__voffset_add(ptr addrspace(8) inreg %rsrc, ptr %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_p0__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_p0__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.p0(ptr %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v2p0__voffset_add(ptr addrspace(8) inreg %rsrc, <2 x ptr> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v2p0__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v2p0__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2p0(<2 x ptr> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_p1__voffset_add(ptr addrspace(8) inreg %rsrc, ptr addrspace(1) %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_p1__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_p1__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.p1(ptr addrspace(1) %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v2p1__voffset_add(ptr addrspace(8) inreg %rsrc, <2 x ptr addrspace(1)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v2p1__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v2p1__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2p1(<2 x ptr addrspace(1)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_p4__voffset_add(ptr addrspace(8) inreg %rsrc, ptr addrspace(4) %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_p4__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_p4__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.p4(ptr addrspace(4) %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v2p4__voffset_add(ptr addrspace(8) inreg %rsrc, <2 x ptr addrspace(4)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v2p4__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v2p4__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2p4(<2 x ptr addrspace(4)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_p999__voffset_add(ptr addrspace(8) inreg %rsrc, ptr addrspace(999) %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_p999__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_p999__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.p999(ptr addrspace(999) %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v2p999__voffset_add(ptr addrspace(8) inreg %rsrc, <2 x ptr addrspace(999)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v2p999__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v2p999__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2p999(<2 x ptr addrspace(999)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_p2__voffset_add(ptr addrspace(8) inreg %rsrc, ptr addrspace(2) %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_p2__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dword v0, v1, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_p2__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dword v0, v1, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.p2(ptr addrspace(2) %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v2p2__voffset_add(ptr addrspace(8) inreg %rsrc, <2 x ptr addrspace(2)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v2p2__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v2p2__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2p2(<2 x ptr addrspace(2)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v3p2__voffset_add(ptr addrspace(8) inreg %rsrc, <3 x ptr addrspace(2)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v3p2__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx3 v[0:2], v3, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v3p2__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx3 v[0:2], v3, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v3p2(<3 x ptr addrspace(2)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v4p2__voffset_add(ptr addrspace(8) inreg %rsrc, <4 x ptr addrspace(2)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v4p2__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v4p2__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4p2(<4 x ptr addrspace(2)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_p3__voffset_add(ptr addrspace(8) inreg %rsrc, ptr addrspace(3) %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_p3__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dword v0, v1, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_p3__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dword v0, v1, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.p3(ptr addrspace(3) %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v2p3__voffset_add(ptr addrspace(8) inreg %rsrc, <2 x ptr addrspace(3)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v2p3__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v2p3__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2p3(<2 x ptr addrspace(3)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v3p3__voffset_add(ptr addrspace(8) inreg %rsrc, <3 x ptr addrspace(3)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v3p3__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx3 v[0:2], v3, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v3p3__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx3 v[0:2], v3, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v3p3(<3 x ptr addrspace(3)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v4p3__voffset_add(ptr addrspace(8) inreg %rsrc, <4 x ptr addrspace(3)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v4p3__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v4p3__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4p3(<4 x ptr addrspace(3)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_p5__voffset_add(ptr addrspace(8) inreg %rsrc, ptr addrspace(5) %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_p5__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dword v0, v1, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_p5__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dword v0, v1, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.p5(ptr addrspace(5) %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v2p5__voffset_add(ptr addrspace(8) inreg %rsrc, <2 x ptr addrspace(5)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v2p5__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v2p5__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2p5(<2 x ptr addrspace(5)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v3p5__voffset_add(ptr addrspace(8) inreg %rsrc, <3 x ptr addrspace(5)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v3p5__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx3 v[0:2], v3, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v3p5__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx3 v[0:2], v3, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v3p5(<3 x ptr addrspace(5)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v4p5__voffset_add(ptr addrspace(8) inreg %rsrc, <4 x ptr addrspace(5)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v4p5__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v4p5__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4p5(<4 x ptr addrspace(5)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_p6__voffset_add(ptr addrspace(8) inreg %rsrc, ptr addrspace(6) %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_p6__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dword v0, v1, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_p6__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dword v0, v1, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.p6(ptr addrspace(6) %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v2p6__voffset_add(ptr addrspace(8) inreg %rsrc, <2 x ptr addrspace(6)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v2p6__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v2p6__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2p6(<2 x ptr addrspace(6)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v3p6__voffset_add(ptr addrspace(8) inreg %rsrc, <3 x ptr addrspace(6)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v3p6__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx3 v[0:2], v3, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v3p6__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx3 v[0:2], v3, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v3p6(<3 x ptr addrspace(6)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

define void @buffer_store_v4p6__voffset_add(ptr addrspace(8) inreg %rsrc, <4 x ptr addrspace(6)> %data, i32 %voffset) #0 {
; VERDE-LABEL: buffer_store_v4p6__voffset_add:
; VERDE:       ; %bb.0:
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VERDE-NEXT:    s_mov_b32 s11, s17
; VERDE-NEXT:    s_mov_b32 s10, s16
; VERDE-NEXT:    s_mov_b32 s9, s7
; VERDE-NEXT:    s_mov_b32 s8, s6
; VERDE-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; VERDE-NEXT:    s_waitcnt vmcnt(0) expcnt(0)
; VERDE-NEXT:    s_setpc_b64 s[30:31]
;
; CHECK-LABEL: buffer_store_v4p6__voffset_add:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_mov_b32 s11, s17
; CHECK-NEXT:    s_mov_b32 s10, s16
; CHECK-NEXT:    s_mov_b32 s9, s7
; CHECK-NEXT:    s_mov_b32 s8, s6
; CHECK-NEXT:    buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:60
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %voffset.add = add i32 %voffset, 60
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4p6(<4 x ptr addrspace(6)> %data, ptr addrspace(8) %rsrc, i32 %voffset.add, i32 0, i32 0)
  ret void
}

declare void @llvm.amdgcn.raw.ptr.buffer.store.f32(float, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.v2f32(<2 x float>, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float>, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32>, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32>, ptr addrspace(8), i32, i32, i32) #0
declare <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8), i32, i32, i32) #1
declare void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.f16(half, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.v2f16(<2 x half>, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4f16(<4 x half>, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.v2i16(<2 x i16>, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i16(<4 x i16>, ptr addrspace(8), i32, i32, i32) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
