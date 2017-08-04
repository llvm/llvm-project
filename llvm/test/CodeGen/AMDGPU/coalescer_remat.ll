; RUN: llc -march=amdgcn -verify-machineinstrs -mtriple=amdgcn-- -o - %s | FileCheck %s

declare float @llvm.fma.f32(float, float, float)

; This checks that rematerialization support of the coalescer does not
; unnecessarily widen the register class. Without those fixes > 20 VGprs
; are used here
; Also check that some rematerialization of the 0 constant happened.
; CHECK-LABEL: foobar
; CHECK:  v_mov_b32_e32 v{{[0-9]+}}, 0
; CHECK:  v_mov_b32_e32 v{{[0-9]+}}, 0
; CHECK:  v_mov_b32_e32 v{{[0-9]+}}, 0
; CHECK:  v_mov_b32_e32 v{{[0-9]+}}, 0
; It's probably OK if this is slightly higher:
; CHECK: ; NumVgprs: 4
define amdgpu_kernel void @foobar(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in, i32 %flag) {
entry:
  %cmpflag = icmp eq i32 %flag, 1
  br i1 %cmpflag, label %loop, label %exit

loop:
  %c = phi i32 [0, %entry], [%cnext, %loop]
  %v0 = phi float [0.0, %entry], [%fma.0, %loop]
  %v1 = phi float [0.0, %entry], [%fma.1, %loop]
  %v2 = phi float [0.0, %entry], [%fma.2, %loop]
  %v3 = phi float [0.0, %entry], [%fma.3, %loop]

  ; Try to get the 0 constant to get coalesced into a wide register
  %blup = insertelement <4 x float> undef, float %v0, i32 0
  store <4 x float> %blup, <4 x float> addrspace(1)* %out

  %load = load <4 x float>, <4 x float> addrspace(1)* %in
  %load.0 = extractelement <4 x float> %load, i32 0
  %load.1 = extractelement <4 x float> %load, i32 1
  %load.2 = extractelement <4 x float> %load, i32 2
  %load.3 = extractelement <4 x float> %load, i32 3
  %fma.0 = call float @llvm.fma.f32(float %v0, float %load.0, float %v0)
  %fma.1 = call float @llvm.fma.f32(float %v1, float %load.1, float %v1)
  %fma.2 = call float @llvm.fma.f32(float %v2, float %load.2, float %v2)
  %fma.3 = call float @llvm.fma.f32(float %v3, float %load.3, float %v3)

  %cnext = add nsw i32 %c, 1
  %cmp = icmp eq i32 %cnext, 42
  br i1 %cmp, label %exit, label %loop

exit:
  %ev0 = phi float [0.0, %entry], [%fma.0, %loop]
  %ev1 = phi float [0.0, %entry], [%fma.1, %loop]
  %ev2 = phi float [0.0, %entry], [%fma.2, %loop]
  %ev3 = phi float [0.0, %entry], [%fma.3, %loop]
  %dst.0 = insertelement <4 x float> undef,  float %ev0, i32 0
  %dst.1 = insertelement <4 x float> %dst.0, float %ev1, i32 1
  %dst.2 = insertelement <4 x float> %dst.1, float %ev2, i32 2
  %dst.3 = insertelement <4 x float> %dst.2, float %ev3, i32 3
  store <4 x float> %dst.3, <4 x float> addrspace(1)* %out
  ret void
}
