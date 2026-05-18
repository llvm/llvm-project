; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that this doesn't crash.
; CHECK: vmem

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dso_local void @f0(<64 x i16> %a0, <64 x i16> %a1, ptr %a2, ptr %a3) local_unnamed_addr #0 {
b0:
  %v8 = getelementptr inbounds i16, ptr %a2, i32 -1
  %v9 = load i16, ptr %v8, align 2
  %v10 = sext i16 %v9 to i32
  %v145 = xor i32 0, -1
  %tt0 = getelementptr inbounds i16, ptr %a2, i32 1
  %v469 = load i16, ptr %tt0, align 2
  %v533 = insertelement <64 x i16> %a0, i16 %v469, i64 63
  %tt1 = getelementptr inbounds i16, ptr %a2, i32 3
  %v597 = load i16, ptr %tt1, align 2
  %v661 = insertelement <64 x i16> %a1, i16 %v597, i64 63
  %v662 = sext <64 x i16> %v533 to <64 x i32>
  %v663 = sext <64 x i16> %v661 to <64 x i32>
  %v1837 = getelementptr inbounds i16, ptr %a2, i32 %v145
  %v1838 = getelementptr inbounds i16, ptr %v1837, i32 -63
  %v1839 = load <64 x i16>, ptr %v1838, align 2
  %v1840 = shufflevector <64 x i16> %v1839, <64 x i16> poison, <64 x i32> <i32 63, i32 62, i32 61, i32 60, i32 59, i32 58, i32 57, i32 56, i32 55, i32 54, i32 53, i32 52, i32 51, i32 50, i32 49, i32 48, i32 47, i32 46, i32 45, i32 44, i32 43, i32 42, i32 41, i32 40, i32 39, i32 38, i32 37, i32 36, i32 35, i32 34, i32 33, i32 32, i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %v1841 = getelementptr inbounds i16, ptr %v1837, i32 -127
  %v1842 = load <64 x i16>, ptr %v1841, align 2
  %v1843 = shufflevector <64 x i16> %v1842, <64 x i16> poison, <64 x i32> <i32 63, i32 62, i32 61, i32 60, i32 59, i32 58, i32 57, i32 56, i32 55, i32 54, i32 53, i32 52, i32 51, i32 50, i32 49, i32 48, i32 47, i32 46, i32 45, i32 44, i32 43, i32 42, i32 41, i32 40, i32 39, i32 38, i32 37, i32 36, i32 35, i32 34, i32 33, i32 32, i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %v1844 = sext <64 x i16> %v1840 to <64 x i32>
  %v1845 = sext <64 x i16> %v1843 to <64 x i32>
  %v1846 = mul nsw <64 x i32> %v1844, %v662
  %v1847 = mul nsw <64 x i32> %v1845, %v663
  %v1848 = add <64 x i32> %v1846, zeroinitializer
  %v1849 = add <64 x i32> %v1847, zeroinitializer
  %v1930 = add <64 x i32> %v1849, %v1848
  %v1932 = add <64 x i32> %v1930, zeroinitializer
  %v1934 = add <64 x i32> %v1932, zeroinitializer
  %v1936 = add <64 x i32> %v1934, zeroinitializer
  %v1938 = add <64 x i32> %v1936, zeroinitializer
  %v1940 = add <64 x i32> %v1938, zeroinitializer
  %v1942 = add <64 x i32> %v1940, zeroinitializer
  %v1943 = extractelement <64 x i32> %v1942, i32 0
  %v2515 = load i16, ptr %a3, align 2
  %v2516 = sext i16 %v2515 to i32
  %v2524 = mul nsw i32 %v10, %v2516
  %v2525 = sub nsw i32 %v1943, %v2524
  %v2527 = add nsw i32 %v2525, 0
  %v2572 = getelementptr inbounds i32, ptr %a3, i32 0
  store i32 %v2527, ptr %v2572, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #1

attributes #0 = { "target-features"="+hvx-ieee-fp,+hvx-length128b,+hvxv73,-long-calls" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"frame-pointer", i32 2}
