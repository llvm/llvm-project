; RUN: llc -verify-machineinstrs -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a9 -mattr=+neon,+neonfp -float-abi=hard < %s | FileCheck %s

define <2 x float> @test_vmovs_via_vext_lane0to0(float %arg, <2 x float> %in) {
; CHECK: test_vmovs_via_vext_lane0to0:
  %vec = insertelement <2 x float> %in, float %arg, i32 0
  %res = fadd <2 x float> %vec, %vec

; CHECK: vext.32 d1, d1, d0, #1
; CHECK: vext.32 d1, d1, d1, #1
; CHECK: vadd.f32 {{d[0-9]+}}, d1, d1

  ret <2 x float> %res
}

define <2 x float> @test_vmovs_via_vext_lane0to1(float %arg, <2 x float> %in) {
; CHECK: test_vmovs_via_vext_lane0to1:
  %vec = insertelement <2 x float> %in, float %arg, i32 1
  %res = fadd <2 x float> %vec, %vec

; CHECK: vext.32 d1, d1, d1, #1
; CHECK: vext.32 d1, d1, d0, #1
; CHECK: vadd.f32 {{d[0-9]+}}, d1, d1

  ret <2 x float> %res
}

define <2 x float> @test_vmovs_via_vext_lane1to0(float, float %arg, <2 x float> %in) {
; CHECK: test_vmovs_via_vext_lane1to0:
  %vec = insertelement <2 x float> %in, float %arg, i32 0
  %res = fadd <2 x float> %vec, %vec

; CHECK: vext.32 d1, d1, d1, #1
; CHECK: vext.32 d1, d0, d1, #1
; CHECK: vadd.f32 {{d[0-9]+}}, d1, d1

  ret <2 x float> %res
}

define <2 x float> @test_vmovs_via_vext_lane1to1(float, float %arg, <2 x float> %in) {
; CHECK: test_vmovs_via_vext_lane1to1:
  %vec = insertelement <2 x float> %in, float %arg, i32 1
  %res = fadd <2 x float> %vec, %vec

; CHECK: vext.32 d1, d0, d1, #1
; CHECK: vext.32 d1, d1, d1, #1
; CHECK: vadd.f32 {{d[0-9]+}}, d1, d1

  ret <2 x float> %res
}


define float @test_vmovs_via_vdup(float, float %ret, float %lhs, float %rhs) {
; CHECK: test_vmovs_via_vdup:

  ; Do an operation (which will end up NEON because of +neonfp) to convince the
  ; execution-domain pass that NEON is a good thing to use.
  %res = fadd float %ret, %ret
  ;  It makes sense for LLVM to do the addition in d0 here, because it's going
  ;  to be returned. This means it will want a "vmov s0, s1":
; CHECK: vdup.32 d0, d0[1]

  ret float %res
}

declare float @llvm.sqrt.f32(float)

declare void @bar()

; This is a comp
define float @test_ineligible(float, float %in) {
; CHECK: test_ineligible:

  %sqrt = call float @llvm.sqrt.f32(float %in)
  %val = fadd float %sqrt, %sqrt

  ; This call forces a move from a callee-saved register to the return-reg. That
  ; move is not eligible for conversion to a d-register instructions because the
  ; use-def chains would be messed up. Primarily a compile-test (we used to
  ; internal fault).
  call void @bar()
; CHECL: bl bar
; CHECK: vext.32
; CHECK: vext.32
  ret float %val
}

define i32 @test_vmovs_no_sreg(i32 %in) {
; CHECK: test_vmovs_no_sreg:

  ; Check that the movement to and from GPRs takes place in the NEON domain.
; CHECK: vmov.32 d
  %x = bitcast i32 %in to float

  %res = fadd float %x, %x

; CHECK: vmov.32 r{{[0-9]+}}, d
  %resi = bitcast float %res to i32

  ret i32 %resi
}
