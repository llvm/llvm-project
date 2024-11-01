; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s -check-prefix=CHECK

; Test that buffer_load_format with VGPR resource descriptor is properly
; legalized.

; CHECK-LABEL: {{^}}test_none:
; CHECK: buffer_load_format_x v0, off, {{s\[[0-9]+:[0-9]+\]}}, 0{{$}}
define amdgpu_vs float @test_none(ptr addrspace(4) inreg %base, i32 %i) {
main_body:
  %ptr = getelementptr <4 x i32>, ptr addrspace(4) %base, i32 %i
  %tmp2 = load <4 x i32>, ptr addrspace(4) %ptr, align 32
  %tmp7 = call float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32> %tmp2, i32 0, i32 0, i32 0)
  ret float %tmp7
}

; CHECK-LABEL: {{^}}test_idxen:
; CHECK: buffer_load_format_x v0, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 idxen{{$}}
define amdgpu_vs float @test_idxen(ptr addrspace(4) inreg %base, i32 %i) {
main_body:
  %ptr = getelementptr <4 x i32>, ptr addrspace(4) %base, i32 %i
  %tmp2 = load <4 x i32>, ptr addrspace(4) %ptr, align 32
  %tmp7 = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %tmp2, i32 undef, i32 0, i32 0, i32 0)
  ret float %tmp7
}

; CHECK-LABEL: {{^}}test_offen:
; CHECK: buffer_load_format_x v0, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen{{$}}
define amdgpu_vs float @test_offen(ptr addrspace(4) inreg %base, i32 %i) {
main_body:
  %ptr = getelementptr <4 x i32>, ptr addrspace(4) %base, i32 %i
  %tmp2 = load <4 x i32>, ptr addrspace(4) %ptr, align 32
  %tmp7 = call float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32> %tmp2, i32 undef, i32 0, i32 0)
  ret float %tmp7
}

; CHECK-LABEL: {{^}}test_both:
; CHECK: buffer_load_format_x v0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 idxen offen{{$}}
define amdgpu_vs float @test_both(ptr addrspace(4) inreg %base, i32 %i) {
main_body:
  %ptr = getelementptr <4 x i32>, ptr addrspace(4) %base, i32 %i
  %tmp2 = load <4 x i32>, ptr addrspace(4) %ptr, align 32
  %tmp7 = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %tmp2, i32 undef, i32 undef, i32 0, i32 0)
  ret float %tmp7
}

declare float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32>, i32, i32, i32, i32 immarg) #1
declare float @llvm.amdgcn.raw.buffer.load.format.f32(<4 x i32>, i32, i32, i32 immarg) #1

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readonly }
