; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <2 x double> @splice_v2f64_idx_neg3(<2 x double> %a, <2 x double> %b, i32 %evl1, i32 %evl2) #0 {
  %res = call <2 x double> @llvm.experimental.vp.splice.v2f64(<2 x double> %a, <2 x double> %b, i32 -3, <2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <2 x double> %res
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <vscale x 2 x double> @splice_nxv2f64_idx_neg3_vscale_min1(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 %evl1, i32 %evl2) #0 {
  %res = call <vscale x 2 x double> @llvm.experimental.vp.splice.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 -3, <vscale x 2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <vscale x 2 x double> %res
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <vscale x 2 x double> @splice_nxv2f64_idx_neg5_vscale_min2(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 %evl1, i32 %evl2) #1 {
  %res = call <vscale x 2 x double> @llvm.experimental.vp.splice.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 -5, <vscale x 2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <vscale x 2 x double> %res
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <2 x double> @splice_v2f64_idx2(<2 x double> %a, <2 x double> %b, i32 %evl1, i32 %evl2) #0 {
  %res = call <2 x double> @llvm.experimental.vp.splice.v2f64(<2 x double> %a, <2 x double> %b, i32 2, <2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <2 x double> %res
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <2 x double> @splice_v2f64_idx3(<2 x double> %a, <2 x double> %b, i32 %evl1, i32 %evl2) #1 {
  %res = call <2 x double> @llvm.experimental.vp.splice.v2f64(<2 x double> %a, <2 x double> %b, i32 4, <2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <2 x double> %res
}

attributes #0 = { vscale_range(1,16) }
attributes #1 = { vscale_range(2,16) }
