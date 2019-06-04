; RUN: llc -mtriple=arm-eabi -mattr=+armv8.2-a,+fullfp16,+neon -float-abi=hard -O1 < %s | FileCheck %s
; RUN: llc -mtriple=arm-eabi -mattr=+armv8.2-a,+fullfp16,+neon -float-abi=soft -O1 < %s | FileCheck %s

define float @test_vget_lane_f16_1(<4 x half> %a) nounwind {
; CHECK-LABEL: test_vget_lane_f16_1:
; CHECK:      vmovx.f16 s0, s0
; CHECK-NEXT: vcvtb.f32.f16 s0, s0
entry:
  %elt = extractelement <4 x half> %a, i32 1
  %conv = fpext half %elt to float
  ret float %conv
}

define float @test_vget_lane_f16_2(<4 x half> %a) nounwind {
; CHECK-LABEL: test_vget_lane_f16_2:
; CHECK-NOT:  vmovx.f16
; CHECK:      vcvtb.f32.f16 s0, s1
entry:
  %elt = extractelement <4 x half> %a, i32 2
  %conv = fpext half %elt to float
  ret float %conv
}

define float @test_vget_laneq_f16_6(<8 x half> %a) nounwind {
; CHECK-LABEL: test_vget_laneq_f16_6:
; CHECK-NOT:  vmovx.f16
; CHECK:      vcvtb.f32.f16 s0, s3
entry:
  %elt = extractelement <8 x half> %a, i32 6
  %conv = fpext half %elt to float
  ret float %conv
}

define float @test_vget_laneq_f16_7(<8 x half> %a) nounwind {
; CHECK-LABEL: test_vget_laneq_f16_7:
; CHECK:      vmovx.f16 s0, s3
; CHECK:      vcvtb.f32.f16 s0, s0
entry:
  %elt = extractelement <8 x half> %a, i32 7
  %conv = fpext half %elt to float
  ret float %conv
}

define <4 x half> @test_vset_lane_f16(<4 x half> %a, float %fb) nounwind {
; CHECK-LABEL: test_vset_lane_f16:
; CHECK: vmov.f16 r[[GPR:[0-9]+]], s{{[0-9]+}}
; CHECK: vmov.16  d{{[0-9]+}}[3], r[[GPR]]
entry:
  %b = fptrunc float %fb to half
  %x = insertelement <4 x half> %a, half %b, i32 3
  ret <4 x half> %x
}

define <8 x half> @test_vset_laneq_f16_1(<8 x half> %a, float %fb) nounwind {
; CHECK-LABEL: test_vset_laneq_f16_1:
; CHECK: vmov.f16 r[[GPR:[0-9]+]], s{{[0-9]+}}
; CHECK: vmov.16  d{{[0-9]+}}[1], r[[GPR]]
entry:
  %b = fptrunc float %fb to half
  %x = insertelement <8 x half> %a, half %b, i32 1
  ret <8 x half> %x
}

define <8 x half> @test_vset_laneq_f16_7(<8 x half> %a, float %fb) nounwind {
; CHECK-LABEL: test_vset_laneq_f16_7:
; CHECK: vmov.f16 r[[GPR:[0-9]+]], s{{[0-9]+}}
; CHECK: vmov.16  d{{[0-9]+}}[3], r[[GPR]]
entry:
  %b = fptrunc float %fb to half
  %x = insertelement <8 x half> %a, half %b, i32 7
  ret <8 x half> %x
}
