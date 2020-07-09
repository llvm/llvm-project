; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+sve,+i8mm -asm-verbose=0 < %s -o - 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

define <vscale x 4 x i32> @smmla(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
entry:
; CHECK-LABEL: smmla:
; CHECK-NEXT:  smmla   z0.s, z1.b, z2.b
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.smmla.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @ummla(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
entry:
; CHECK-LABEL: ummla:
; CHECK-NEXT:  ummla   z0.s, z1.b, z2.b
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.ummla.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @usmmla(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
entry:
; CHECK-LABEL: usmmla:
; CHECK-NEXT:  usmmla   z0.s, z1.b, z2.b
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.usmmla.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @usdot(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
entry:
; CHECK-LABEL: usdot:
; CHECK-NEXT:  usdot   z0.s, z1.b, z2.b
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @usdot_lane_0(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
entry:
; CHECK-LABEL: usdot_lane_0:
; CHECK-NEXT:  usdot   z0.s, z1.b, z2.b[0]
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 0)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @usdot_lane_1(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
entry:
; CHECK-LABEL: usdot_lane_1:
; CHECK-NEXT:  usdot   z0.s, z1.b, z2.b[1]
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 1)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @usdot_lane_2(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
entry:
; CHECK-LABEL: usdot_lane_2:
; CHECK-NEXT:  usdot   z0.s, z1.b, z2.b[2]
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 2)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @usdot_lane_3(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
entry:
; CHECK-LABEL: usdot_lane_3:
; CHECK-NEXT:  usdot   z0.s, z1.b, z2.b[3]
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 3)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @sudot_lane_0(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
entry:
; CHECK-LABEL: sudot_lane_0:
; CHECK-NEXT:  sudot   z0.s, z1.b, z2.b[0]
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sudot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 0)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @sudot_lane_1(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
entry:
; CHECK-LABEL: sudot_lane_1:
; CHECK-NEXT:  sudot   z0.s, z1.b, z2.b[1]
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sudot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 1)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @sudot_lane_2(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
entry:
; CHECK-LABEL: sudot_lane_2:
; CHECK-NEXT:  sudot   z0.s, z1.b, z2.b[2]
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sudot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 2)
  ret <vscale x 4 x i32> %val
}

define <vscale x 4 x i32> @sudot_lane_3(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
entry:
; CHECK-LABEL: sudot_lane_3:
; CHECK-NEXT:  sudot   z0.s, z1.b, z2.b[3]
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sudot.lane.nxv4i32(<vscale x 4 x i32> %r, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, i32 3)
  ret <vscale x 4 x i32> %val
}


declare <vscale x 4 x i32> @llvm.aarch64.sve.smmla.nxv4i32(<vscale x 4 x i32>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ummla.nxv4i32(<vscale x 4 x i32>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.usmmla.nxv4i32(<vscale x 4 x i32>, <vscale x 16 x i8>, <vscale x 16 x i8>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.usdot.nxv4i32(<vscale x 4 x i32>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 16 x i8>, <vscale x 16 x i8>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sudot.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 16 x i8>, <vscale x 16 x i8>, i32)

