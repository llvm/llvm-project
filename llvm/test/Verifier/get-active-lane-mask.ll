; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

declare <4 x i32> @llvm.get.active.lane.mask.v4i32.i32(i32, i32)

define <4 x i32> @t1(i32 %IV, i32 %TC) {
; CHECK:      get_active_lane_mask: element type is not i1
; CHECK-NEXT: %res = call <4 x i32> @llvm.get.active.lane.mask.v4i32.i32(i32 %IV, i32 %TC)

  %res = call <4 x i32> @llvm.get.active.lane.mask.v4i32.i32(i32 %IV, i32 %TC)
  ret <4 x i32> %res
}

; CHECK:      intrinsic return type (overload type 0) expected any integer vector, but got i32
; CHECK-NEXT: declare i32 @llvm.get.active.lane.mask.i32.i32(i32, i32)
declare i32 @llvm.get.active.lane.mask.i32.i32(i32, i32)

; CHECK:      intrinsic argument 0 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x i1> @llvm.get.active.lane.mask.v4i1.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i1> @llvm.get.active.lane.mask.v4i1.v4i32(<4 x i32>, <4 x i32>)
