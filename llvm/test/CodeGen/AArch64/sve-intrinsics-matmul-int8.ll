; RUN: llc -mattr=+sve,+i8mm < %s -o - | FileCheck %s
; RUN: llc -mattr=+sme,+i8mm -force-streaming < %s | FileCheck %s
; RUN: llc -mattr=+sve,+sme,+i8mm -force-streaming-compatible < %s | FileCheck %s

target triple = "aarch64"

define <vscale x 4 x i32> @usdot(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: usdot:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    usdot z0.s, z1.b, z2.b
; CHECK-NEXT:    ret
entry:
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @usdot_lane_0(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: usdot_lane_0:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    usdot z0.s, z1.b, z2.b[0]
; CHECK-NEXT:    ret
entry:
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 0)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @usdot_lane_1(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: usdot_lane_1:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    usdot z0.s, z1.b, z2.b[1]
; CHECK-NEXT:    ret
entry:
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 1)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @usdot_lane_2(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: usdot_lane_2:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    usdot z0.s, z1.b, z2.b[2]
; CHECK-NEXT:    ret
entry:
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 2)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @usdot_lane_3(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: usdot_lane_3:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    usdot z0.s, z1.b, z2.b[3]
; CHECK-NEXT:    ret
entry:
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 3)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @sudot_lane_0(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: sudot_lane_0:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    sudot z0.s, z1.b, z2.b[0]
; CHECK-NEXT:    ret
entry:
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sudot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 0)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @sudot_lane_1(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: sudot_lane_1:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    sudot z0.s, z1.b, z2.b[1]
; CHECK-NEXT:    ret
entry:
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sudot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 1)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @sudot_lane_2(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: sudot_lane_2:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    sudot z0.s, z1.b, z2.b[2]
; CHECK-NEXT:    ret
entry:
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sudot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 2)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @sudot_lane_3(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: sudot_lane_3:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    sudot z0.s, z1.b, z2.b[3]
; CHECK-NEXT:    ret
entry:
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sudot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 3)
  ret <vscale x 4 x i32> %val
}


declare <vscale x 4 x i32> @llvm.aarch64.sve.usmmla.nxv4i32(<vscale x 4 x i32>, <vscale x 16 x i8>, <vscale x 16 x i8>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.usdot.nxv4i32(<vscale x 4 x i32>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 16 x i8>, <vscale x 16 x i8>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sudot.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 16 x i8>, <vscale x 16 x i8>, i32)
