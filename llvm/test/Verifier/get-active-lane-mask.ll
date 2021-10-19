; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare <4 x i32> @llvm.get.active.lane.mask.v4i32.i32(i32, i32)

define <4 x i32> @t1(i32 %IV, i32 %TC) {
; CHECK:      get_active_lane_mask: element type is not i1
; CHECK-NEXT: %res = call <4 x i32> @llvm.get.active.lane.mask.v4i32.i32(i32 %IV, i32 %TC)

  %res = call <4 x i32> @llvm.get.active.lane.mask.v4i32.i32(i32 %IV, i32 %TC)
  ret <4 x i32> %res
}

declare i32 @llvm.get.active.lane.mask.i32.i32(i32, i32)

define i32 @t2(i32 %IV, i32 %TC) {
; CHECK:      Intrinsic has incorrect return type!
; CHECK-NEXT: i32 (i32, i32)* @llvm.get.active.lane.mask.i32.i32

  %res = call i32 @llvm.get.active.lane.mask.i32.i32(i32 %IV, i32 %TC)
  ret i32 %res
}
