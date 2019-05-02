;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck -check-prefix=VERDE %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}buffer_store:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[0:3], off, s[0:3], 0
;CHECK: buffer_store_dwordx4 v[4:7], off, s[0:3], 0 glc
;CHECK: buffer_store_dwordx4 v[8:11], off, s[0:3], 0 slc
define amdgpu_ps void @buffer_store(<4 x i32> inreg, <4 x float>, <4 x float>, <4 x float>) {
main_body:
  call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> %1, <4 x i32> %0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> %2, <4 x i32> %0, i32 0, i32 0, i32 1)
  call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> %3, <4 x i32> %0, i32 0, i32 0, i32 2)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_immoffs:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[0:3], off, s[0:3], 0 offset:42
define amdgpu_ps void @buffer_store_immoffs(<4 x i32> inreg, <4 x float>) {
main_body:
  call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> %1, <4 x i32> %0, i32 42, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_ofs:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_ofs(<4 x i32> inreg, <4 x float>, i32) {
main_body:
  call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> %1, <4 x i32> %0, i32 %2, i32 0, i32 0)
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
define amdgpu_ps void @buffer_store_wait(<4 x i32> inreg, <4 x float>, i32, i32, i32) {
main_body:
  call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> %1, <4 x i32> %0, i32 %2, i32 0, i32 0)
  %data = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %0, i32 %3, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> %data, <4 x i32> %0, i32 %4, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x1:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dword v0, v1, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_x1(<4 x i32> inreg %rsrc, float %data, i32 %offset) {
main_body:
  call void @llvm.amdgcn.raw.buffer.store.f32(float %data, <4 x i32> %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x2:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
define amdgpu_ps void @buffer_store_x2(<4 x i32> inreg %rsrc, <2 x float> %data, i32 %offset) #0 {
main_body:
  call void @llvm.amdgcn.raw.buffer.store.v2f32(<2 x float> %data, <4 x i32> %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x1_offen_merged_and:
;CHECK-NOT: s_waitcnt
;CHECK-DAG: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:4
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:28
define amdgpu_ps void @buffer_store_x1_offen_merged_and(<4 x i32> inreg %rsrc, i32 %a, float %v1, float %v2, float %v3, float %v4, float %v5, float %v6) {
  %a1 = add i32 %a, 4
  %a2 = add i32 %a, 8
  %a3 = add i32 %a, 12
  %a4 = add i32 %a, 16
  %a5 = add i32 %a, 28
  %a6 = add i32 %a, 32
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v1, <4 x i32> %rsrc, i32 %a1, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v2, <4 x i32> %rsrc, i32 %a2, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v3, <4 x i32> %rsrc, i32 %a3, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v4, <4 x i32> %rsrc, i32 %a4, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v5, <4 x i32> %rsrc, i32 %a5, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v6, <4 x i32> %rsrc, i32 %a6, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x1_offen_merged_or:
;CHECK-NOT: s_waitcnt
;CHECK-DAG: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], v{{[0-9]}}, s[0:3], 0 offen offset:4
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], v{{[0-9]}}, s[0:3], 0 offen offset:28
define amdgpu_ps void @buffer_store_x1_offen_merged_or(<4 x i32> inreg %rsrc, i32 %inp, float %v1, float %v2, float %v3, float %v4, float %v5, float %v6) {
  %a = shl i32 %inp, 6
  %a1 = add i32 %a, 4
  %a2 = add i32 %a, 8
  %a3 = add i32 %a, 12
  %a4 = add i32 %a, 16
  %a5 = add i32 %a, 28
  %a6 = add i32 %a, 32
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v1, <4 x i32> %rsrc, i32 %a1, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v2, <4 x i32> %rsrc, i32 %a2, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v3, <4 x i32> %rsrc, i32 %a3, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v4, <4 x i32> %rsrc, i32 %a4, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v5, <4 x i32> %rsrc, i32 %a5, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v6, <4 x i32> %rsrc, i32 %a6, i32 0, i32 0)
  ret void
}


;CHECK-LABEL: {{^}}buffer_store_x1_offen_merged_glc_slc:
;CHECK-NOT: s_waitcnt
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:4{{$}}
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:12 glc{{$}}
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:28 glc slc{{$}}
define amdgpu_ps void @buffer_store_x1_offen_merged_glc_slc(<4 x i32> inreg %rsrc, i32 %a, float %v1, float %v2, float %v3, float %v4, float %v5, float %v6) {
  %a1 = add i32 %a, 4
  %a2 = add i32 %a, 8
  %a3 = add i32 %a, 12
  %a4 = add i32 %a, 16
  %a5 = add i32 %a, 28
  %a6 = add i32 %a, 32
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v1, <4 x i32> %rsrc, i32 %a1, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v2, <4 x i32> %rsrc, i32 %a2, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v3, <4 x i32> %rsrc, i32 %a3, i32 0, i32 1)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v4, <4 x i32> %rsrc, i32 %a4, i32 0, i32 1)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v5, <4 x i32> %rsrc, i32 %a5, i32 0, i32 3)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v6, <4 x i32> %rsrc, i32 %a6, i32 0, i32 3)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x2_offen_merged_and:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:4
define amdgpu_ps void @buffer_store_x2_offen_merged_and(<4 x i32> inreg %rsrc, i32 %a, <2 x float> %v1, <2 x float> %v2) {
  %a1 = add i32 %a, 4
  %a2 = add i32 %a, 12
  call void @llvm.amdgcn.raw.buffer.store.v2f32(<2 x float> %v1, <4 x i32> %rsrc, i32 %a1, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.v2f32(<2 x float> %v2, <4 x i32> %rsrc, i32 %a2, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x2_offen_merged_or:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], v{{[0-9]}}, s[0:3], 0 offen offset:4
define amdgpu_ps void @buffer_store_x2_offen_merged_or(<4 x i32> inreg %rsrc, i32 %inp, <2 x float> %v1, <2 x float> %v2) {
  %a = shl i32 %inp, 4
  %a1 = add i32 %a, 4
  %a2 = add i32 %a, 12
  call void @llvm.amdgcn.raw.buffer.store.v2f32(<2 x float> %v1, <4 x i32> %rsrc, i32 %a1, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.v2f32(<2 x float> %v2, <4 x i32> %rsrc, i32 %a2, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x1_offset_merged:
;CHECK-NOT: s_waitcnt
;CHECK-DAG: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], off, s[0:3], 0 offset:4
;CHECK-DAG: buffer_store_dwordx2 v[{{[0-9]}}:{{[0-9]}}], off, s[0:3], 0 offset:28
define amdgpu_ps void @buffer_store_x1_offset_merged(<4 x i32> inreg %rsrc, float %v1, float %v2, float %v3, float %v4, float %v5, float %v6) {
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v1, <4 x i32> %rsrc, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v2, <4 x i32> %rsrc, i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v3, <4 x i32> %rsrc, i32 12, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v4, <4 x i32> %rsrc, i32 16, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v5, <4 x i32> %rsrc, i32 28, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %v6, <4 x i32> %rsrc, i32 32, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_x2_offset_merged:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[{{[0-9]}}:{{[0-9]}}], off, s[0:3], 0 offset:4
define amdgpu_ps void @buffer_store_x2_offset_merged(<4 x i32> inreg %rsrc, <2 x float> %v1,<2 x float> %v2) {
  call void @llvm.amdgcn.raw.buffer.store.v2f32(<2 x float> %v1, <4 x i32> %rsrc, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.v2f32(<2 x float> %v2, <4 x i32> %rsrc, i32 12, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_int:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx4 v[0:3], off, s[0:3], 0
;CHECK: buffer_store_dwordx2 v[4:5], off, s[0:3], 0 glc
;CHECK: buffer_store_dword v6, off, s[0:3], 0 slc
define amdgpu_ps void @buffer_store_int(<4 x i32> inreg, <4 x i32>, <2 x i32>, i32) {
main_body:
  call void @llvm.amdgcn.raw.buffer.store.v4i32(<4 x i32> %1, <4 x i32> %0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32> %2, <4 x i32> %0, i32 0, i32 0, i32 1)
  call void @llvm.amdgcn.raw.buffer.store.i32(i32 %3, <4 x i32> %0, i32 0, i32 0, i32 2)
  ret void
}

;CHECK-LABEL: {{^}}raw_buffer_store_byte:
;CHECK-NEXT: %bb.
;CHECK-NEXT: v_cvt_u32_f32_e32 v{{[0-9]}}, v{{[0-9]}}
;CHECK-NEXT: buffer_store_byte v{{[0-9]}}, off, s[0:3], 0
;CHECK-NEXT: s_endpgm
define amdgpu_ps void @raw_buffer_store_byte(<4 x i32> inreg %rsrc, float %v1) {
main_body:
  %v2 = fptoui float %v1 to i32
  %v3 = trunc i32 %v2 to i8
  call void @llvm.amdgcn.raw.buffer.store.i8(i8 %v3, <4 x i32> %rsrc, i32 0, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}raw_buffer_store_short:
;CHECK-NEXT: %bb.
;CHECK-NEXT: v_cvt_u32_f32_e32 v{{[0-9]}}, v{{[0-9]}}
;CHECK-NEXT: buffer_store_short v{{[0-9]}}, off, s[0:3], 0
;CHECK-NEXT: s_endpgm
define amdgpu_ps void @raw_buffer_store_short(<4 x i32> inreg %rsrc, float %v1) {
main_body:
  %v2 = fptoui float %v1 to i32
  %v3 = trunc i32 %v2 to i16
  call void @llvm.amdgcn.raw.buffer.store.i16(i16 %v3, <4 x i32> %rsrc, i32 0, i32 0, i32 0)
  ret void
}

declare void @llvm.amdgcn.raw.buffer.store.f32(float, <4 x i32>, i32, i32, i32) #0
declare void @llvm.amdgcn.raw.buffer.store.v2f32(<2 x float>, <4 x i32>, i32, i32, i32) #0
declare void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float>, <4 x i32>, i32, i32, i32) #0
declare void @llvm.amdgcn.raw.buffer.store.i32(i32, <4 x i32>, i32, i32, i32) #0
declare void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32>, <4 x i32>, i32, i32, i32) #0
declare void @llvm.amdgcn.raw.buffer.store.v4i32(<4 x i32>, <4 x i32>, i32, i32, i32) #0
declare <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32>, i32, i32, i32) #1
declare void @llvm.amdgcn.raw.buffer.store.i8(i8, <4 x i32>, i32, i32, i32) #0
declare void @llvm.amdgcn.raw.buffer.store.i16(i16, <4 x i32>, i32, i32, i32) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
