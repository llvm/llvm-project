; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s


define <4 x i32> @t1(i32 %IV, i32 %TC) {
; CHECK:      get_active_lane_mask: element type is not i1
; CHECK-NEXT: %res = call <4 x i32> @llvm.get.active.lane.mask.v4i32.i32(i32 %IV, i32 %TC)

  %res = call <4 x i32> @llvm.get.active.lane.mask.v4i32.i32(i32 %IV, i32 %TC)
  ret <4 x i32> %res
}

define i32 @t2(i32 %IV, i32 %TC) {
; CHECK:      Intrinsic has incorrect return type!
; CHECK-NEXT: i32 (i32, i32)* @llvm.get.active.lane.mask.i32.i32

  %res = call i32 @llvm.get.active.lane.mask.i32.i32(i32 %IV, i32 %TC)
  ret i32 %res
}

define <4 x i1> @t3(i32 %IV) {
; CHECK:      get_active_lane_mask: operand #2 must be greater than 0
; CHECK-NEXT: %res = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %IV, i32 0)

  %res = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %IV, i32 0)
  ret <4 x i1> %res
}

define <4 x i1> @t4(i32 %IV) {
; CHECK-NOT: get_active_lane_mask
; CHECK-NOT: call <4 x i1> @llvm.get.active.lane.mask

  %res = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %IV, i32 1)
  ret <4 x i1> %res
}

define <4 x i1> @t5(i32 %IV) {
; CHECK-NOT: get_active_lane_mask
; CHECK-NOT: call <4 x i1> @llvm.get.active.lane.mask

  %res = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %IV, i32 -1)
  ret <4 x i1> %res
}

declare <4 x i32> @llvm.get.active.lane.mask.v4i32.i32(i32, i32)
declare i32 @llvm.get.active.lane.mask.i32.i32(i32, i32)
declare <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32, i32)
