; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define amdgpu_ps void @test_wmma_f32_16x16x32_bf16(<16 x bfloat> %A, <16 x bfloat> %B, <8 x float> %C, ptr addrspace(1) %out) {
; CHECK-LABEL: define amdgpu_ps void @test_wmma_f32_16x16x32_bf16(<16 x bfloat> %A, <16 x bfloat> %B, <8 x float> %C, ptr addrspace(1) %out) {
; CHECK-NEXT: bb:
; CHECK-NEXT:   %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %A, <16 x bfloat> %B, i16 0, <8 x float> %C, i1 false, i1 true)
; CHECK-NEXT:   store <8 x float> %res, ptr addrspace(1) %out, align 32
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(i1 0, <16 x bfloat> %A, i1 0, <16 x bfloat> %B, i16 0, <8 x float> %C, i1 false, i1 true)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

define amdgpu_ps void @test_wmma_f32_16x16x4_f32(<2 x float> %A, <2 x float> %B, <8 x float> %C, ptr addrspace(1) %out) {
; CHECK-LABEL: define amdgpu_ps void @test_wmma_f32_16x16x4_f32(<2 x float> %A, <2 x float> %B, <8 x float> %C, ptr addrspace(1) %out) {
; CHECK-NEXT: bb:
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x4.f32.v8f32.v2f32(<2 x float> %A, <2 x float> %B, i16 0, <8 x float> %C, i1 false, i1 true)
; CHECK-NEXT: store <8 x float> %res, ptr addrspace(1) %out
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x4.f32.v8f32.v2f32(i1 0, <2 x float> %A, i1 0, <2 x float> %B, i16 0, <8 x float> %C, i1 false, i1 true)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

define amdgpu_ps void @test_wmma_bf16_16x16x32_bf16(<16 x bfloat> %A, <16 x bfloat> %B, <8 x bfloat> %C, ptr addrspace(1) %out) {
; CHECK-LABEL: define amdgpu_ps void @test_wmma_bf16_16x16x32_bf16(<16 x bfloat> %A, <16 x bfloat> %B, <8 x bfloat> %C, ptr addrspace(1) %out) {
; CHECK-NEXT: bb:
; CHECK-NEXT: %res = call <8 x bfloat> @llvm.amdgcn.wmma.bf16.16x16x32.bf16.v8bf16.v16bf16(<16 x bfloat> %A, <16 x bfloat> %B, i16 0, <8 x bfloat> %C, i1 false, i1 true)
; CHECK-NEXT: store <8 x bfloat> %res, ptr addrspace(1) %out
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
bb:
  %res = call <8 x bfloat> @llvm.amdgcn.wmma.bf16.16x16x32.bf16.v8bf16.v16bf16(i1 0, <16 x bfloat> %A, i1 0, <16 x bfloat> %B, i16 0, <8 x bfloat> %C, i1 false, i1 true)
  store <8 x bfloat> %res, ptr addrspace(1) %out
  ret void
}

define amdgpu_ps void @test_wmma_f32_16x16x32_f16(<16 x half> %A, <16 x half> %B, <8 x float> %C, ptr addrspace(1) %out) {
; CHECK-LABEL: define amdgpu_ps void @test_wmma_f32_16x16x32_f16(<16 x half> %A, <16 x half> %B, <8 x float> %C, ptr addrspace(1) %out) {
; CHECK-NEXT: bb:
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16(<16 x half> %A, <16 x half> %B, i16 0, <8 x float> %C, i1 false, i1 true)
; CHECK-NEXT: store <8 x float> %res, ptr addrspace(1) %out
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16(i1 0, <16 x half> %A, i1 0, <16 x half> %B, i16 0, <8 x float> %C, i1 false, i1 true)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

define amdgpu_ps void @test_wmma_f16_16x16x32_f16(<16 x half> %A, <16 x half> %B, <8 x half> %C, ptr addrspace(1) %out) {
; CHECK-LABEL: define amdgpu_ps void @test_wmma_f16_16x16x32_f16(<16 x half> %A, <16 x half> %B, <8 x half> %C, ptr addrspace(1) %out) {
; CHECK-NEXT: bb:
; CHECK-NEXT: %res = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x32.f16.v8f16.v16f16(<16 x half> %A, <16 x half> %B, i16 0, <8 x half> %C, i1 false, i1 true)
; CHECK-NEXT: store <8 x half> %res, ptr addrspace(1) %out
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
bb:
  %res = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x32.f16.v8f16.v16f16(i1 0, <16 x half> %A, i1 0, <16 x half> %B, i16 0, <8 x half> %C, i1 false, i1 true)
  store <8 x half> %res, ptr addrspace(1) %out
  ret void
}

define amdgpu_ps void @test_wmma_bf16f32_16x16x32_bf16(<16 x bfloat> %A, <16 x bfloat> %B, <8 x float> %C, ptr addrspace(1) %out) {
; CHECK-LABEL: define amdgpu_ps void @test_wmma_bf16f32_16x16x32_bf16(<16 x bfloat> %A, <16 x bfloat> %B, <8 x float> %C, ptr addrspace(1) %out) {
; CHECK-NEXT: bb:
; CHECK-NEXT: %res = call <8 x bfloat> @llvm.amdgcn.wmma.bf16f32.16x16x32.bf16.v8bf16.v16bf16.v8f32(<16 x bfloat> %A, <16 x bfloat> %B, i16 0, <8 x float> %C, i1 false, i1 true)
; CHECK-NEXT: store <8 x bfloat> %res, ptr addrspace(1) %out
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
bb:
  %res = call <8 x bfloat> @llvm.amdgcn.wmma.bf16f32.16x16x32.bf16.v8bf16.v16bf16(i1 0, <16 x bfloat> %A, i1 0, <16 x bfloat> %B, i16 0, <8 x float> %C, i1 false, i1 true)
  store <8 x bfloat> %res, ptr addrspace(1) %out
  ret void
}
