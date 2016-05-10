; RUN: llc -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI %s

declare float @llvm.AMDGPU.cvt.f32.ubyte0(i32) nounwind readnone
declare float @llvm.AMDGPU.cvt.f32.ubyte1(i32) nounwind readnone
declare float @llvm.AMDGPU.cvt.f32.ubyte2(i32) nounwind readnone
declare float @llvm.AMDGPU.cvt.f32.ubyte3(i32) nounwind readnone

; SI-LABEL: {{^}}test_unpack_byte0_to_float:
; SI: v_cvt_f32_ubyte0
define void @test_unpack_byte0_to_float(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte0(i32 %val) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_unpack_byte1_to_float:
; SI: v_cvt_f32_ubyte1
define void @test_unpack_byte1_to_float(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte1(i32 %val) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_unpack_byte2_to_float:
; SI: v_cvt_f32_ubyte2
define void @test_unpack_byte2_to_float(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte2(i32 %val) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_unpack_byte3_to_float:
; SI: v_cvt_f32_ubyte3
define void @test_unpack_byte3_to_float(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte3(i32 %val) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}byte1_shift8:
; SI: buffer_load_dword [[VAL:v[0-9]+]]
; SI-NOT: [[VAL]]
; SI: v_cvt_f32_ubyte2_e32 [[CONV:v[0-9]+]], [[VAL]]
; SI: buffer_store_dword [[CONV]]
define void @byte1_shift8(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %shift = lshr i32 %val, 8
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte1(i32 %shift) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}byte1_shift7:
; SI: buffer_load_dword [[VAL:v[0-9]+]]
; SI: v_lshrrev_b32_e32 [[SRL:v[0-9]+]], 7, [[VAL]]
; SI: v_cvt_f32_ubyte1_e32 [[CONV:v[0-9]+]], [[SRL]]
; SI: buffer_store_dword [[CONV]]
define void @byte1_shift7(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %shift = lshr i32 %val, 7
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte1(i32 %shift) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}byte1_shift16:
; SI: buffer_load_dword [[VAL:v[0-9]+]]
; SI-NOT: [[VAL]]
; SI: v_cvt_f32_ubyte3_e32 [[CONV:v[0-9]+]], [[VAL]]
; SI: buffer_store_dword [[CONV]]
define void @byte1_shift16(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %shift = lshr i32 %val, 16
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte1(i32 %shift) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}byte2_shift8:
; SI: buffer_load_dword [[VAL:v[0-9]+]]
; SI-NOT: [[VAL]]
; SI: v_cvt_f32_ubyte3_e32 [[CONV:v[0-9]+]], [[VAL]]
; SI: buffer_store_dword [[CONV]]
define void @byte2_shift8(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %shift = lshr i32 %val, 8
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte2(i32 %shift) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; XXX - undef
; SI-LABEL: {{^}}byte1_shift24:
; SI: v_cvt_f32_ubyte1_e32 [[CONV:v[0-9]+]], 0
; SI: buffer_store_dword [[CONV]]
define void @byte1_shift24(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %shift = lshr i32 %val, 24
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte1(i32 %shift) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}
