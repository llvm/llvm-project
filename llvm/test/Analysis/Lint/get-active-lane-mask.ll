; RUN: opt -lint -disable-output < %s 2>&1 | FileCheck %s

define <4 x i1> @t1(i32 %IV) {
;
; CHECK:      get_active_lane_mask: operand #2 must be greater than 0
; CHECK-NEXT: %res = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %IV, i32 0)
;
  %res = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %IV, i32 0)
  ret <4 x i1> %res
}

define <4 x i1> @t2(i32 %IV) {
;
; CHECK-NOT: get_active_lane_mask
; CHECK-NOT: call <4 x i1> @llvm.get.active.lane.mask
;
  %res = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %IV, i32 1)
  ret <4 x i1> %res
}

define <4 x i1> @t3(i32 %IV) {
;
; CHECK-NOT: get_active_lane_mask
; CHECK-NOT: call <4 x i1> @llvm.get.active.lane.mask
;
  %res = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %IV, i32 -1)
  ret <4 x i1> %res
}

define <4 x i1> @t4(i32 %IV, i32 %TC) {
;
; CHECK-NOT: get_active_lane_mask
; CHECK-NOT: call <4 x i1> @llvm.get.active.lane.mask
;
  %res = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %IV, i32 %TC)
  ret <4 x i1> %res
}

declare <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32, i32)
