; RUN: llc -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI %s

declare float @llvm.AMDGPU.cvt.f32.ubyte0(i32) nounwind readnone
declare float @llvm.AMDGPU.cvt.f32.ubyte1(i32) nounwind readnone
declare float @llvm.AMDGPU.cvt.f32.ubyte2(i32) nounwind readnone
declare float @llvm.AMDGPU.cvt.f32.ubyte3(i32) nounwind readnone

; SI-LABEL: {{^}}test_unpack_byte0_to_float:
; SI: v_cvt_f32_ubyte0
define void @test_unpack_byte0_to_float(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32 addrspace(1)* %in, align 4
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte0(i32 %val) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_unpack_byte1_to_float:
; SI: v_cvt_f32_ubyte1
define void @test_unpack_byte1_to_float(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32 addrspace(1)* %in, align 4
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte1(i32 %val) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_unpack_byte2_to_float:
; SI: v_cvt_f32_ubyte2
define void @test_unpack_byte2_to_float(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32 addrspace(1)* %in, align 4
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte2(i32 %val) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_unpack_byte3_to_float:
; SI: v_cvt_f32_ubyte3
define void @test_unpack_byte3_to_float(float addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32 addrspace(1)* %in, align 4
  %cvt = call float @llvm.AMDGPU.cvt.f32.ubyte3(i32 %val) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}
