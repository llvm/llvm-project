; RUN: llc -march=hexagon -O2 < %s
; REQUIRES: asserts

target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b6, !prof !3

b1:                                               ; preds = %b0
  br i1 undef, label %b2, label %b3, !prof !3

b2:                                               ; preds = %b1
  unreachable

b3:                                               ; preds = %b1
  br label %b4

b4:                                               ; preds = %b4, %b3
  %v0 = load <32 x i32>, ptr undef, align 512, !tbaa !4
  %v1 = shufflevector <32 x i32> %v0, <32 x i32> undef, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %v2 = shufflevector <64 x i32> undef, <64 x i32> %v1, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>
  %v3 = trunc <128 x i32> %v2 to <128 x i16>
  %v4 = mul nsw <128 x i16> undef, %v3
  %v5 = bitcast <128 x i16> %v4 to <64 x i32>
  %v6 = tail call <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32> undef, <64 x i32> %v5)
  %v7 = bitcast <64 x i32> %v6 to <128 x i16>
  %v8 = shufflevector <128 x i16> %v7, <128 x i16> undef, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>
  br i1 undef, label %b5, label %b4

b5:                                               ; preds = %b4
  store <64 x i16> %v8, ptr undef, align 1024, !tbaa !7
  br label %b6

b6:                                               ; preds = %b5, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32>, <64 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }
attributes #1 = { nounwind readnone }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"halide_use_soft_float_abi", i32 0}
!1 = !{i32 2, !"halide_mcpu", !"hexagonv60"}
!2 = !{i32 2, !"halide_mattrs", !"+hvx"}
!3 = !{!"branch_weights", i32 1073741824, i32 0}
!4 = !{!5, !5, i64 0}
!5 = !{!"mask", !6}
!6 = !{!"Halide buffer"}
!7 = !{!8, !8, i64 0}
!8 = !{!"sum.width64.base64", !9}
!9 = !{!"sum.width128.base0", !10}
!10 = !{!"sum.width256.base0", !11}
!11 = !{!"sum.width512.base0", !12}
!12 = !{!"sum.width1024.base0", !13}
!13 = !{!"sum", !6}
