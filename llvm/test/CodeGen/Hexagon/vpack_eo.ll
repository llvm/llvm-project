; RUN: llc -mtriple=hexagon < %s | FileCheck %s
target triple = "hexagon-unknown--elf"

; CHECK-DAG: vpacke
; CHECK-DAG: vpacko

%struct.buffer_t = type { i64, ptr, [4 x i32], [4 x i32], [4 x i32], i32, i8, i8, [6 x i8] }

; Function Attrs: norecurse nounwind
define i32 @__Strided_LoadTest(ptr noalias nocapture readonly %InputOne.buffer, ptr noalias nocapture readonly %InputTwo.buffer, ptr noalias nocapture readonly %Strided_LoadTest.buffer) #0 {
entry:
  %buf_host = getelementptr inbounds %struct.buffer_t, ptr %InputOne.buffer, i32 0, i32 1
  %InputOne.host45 = load ptr, ptr %buf_host, align 4
  %buf_host10 = getelementptr inbounds %struct.buffer_t, ptr %InputTwo.buffer, i32 0, i32 1
  %InputTwo.host46 = load ptr, ptr %buf_host10, align 4
  %buf_host27 = getelementptr inbounds %struct.buffer_t, ptr %Strided_LoadTest.buffer, i32 0, i32 1
  %Strided_LoadTest.host44 = load ptr, ptr %buf_host27, align 4
  %0 = load <32 x i16>, ptr %InputOne.host45, align 2, !tbaa !4
  %1 = getelementptr inbounds i16, ptr %InputOne.host45, i32 32
  %2 = load <32 x i16>, ptr %1, align 2, !tbaa !4
  %3 = shufflevector <32 x i16> %0, <32 x i16> %2, <32 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 33, i32 35, i32 37, i32 39, i32 41, i32 43, i32 45, i32 47, i32 49, i32 51, i32 53, i32 55, i32 57, i32 59, i32 61, i32 63>
  %4 = load <32 x i16>, ptr %InputTwo.host46, align 2, !tbaa !7
  %5 = getelementptr inbounds i16, ptr %InputTwo.host46, i32 32
  %6 = load <32 x i16>, ptr %5, align 2, !tbaa !7
  %7 = shufflevector <32 x i16> %4, <32 x i16> %6, <32 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62>
  %8 = bitcast <32 x i16> %3 to <16 x i32>
  %9 = bitcast <32 x i16> %7 to <16 x i32>
  %10 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %8, <16 x i32> %9)
  store <16 x i32> %10, ptr %Strided_LoadTest.host44, align 2, !tbaa !9
  %.inc = getelementptr i16, ptr %InputOne.host45, i32 64
  %.inc49 = getelementptr i16, ptr %InputTwo.host46, i32 64
  %.inc52 = getelementptr i16, ptr %Strided_LoadTest.host44, i32 32
  %11 = load <32 x i16>, ptr %.inc, align 2, !tbaa !4
  %12 = getelementptr inbounds i16, ptr %InputOne.host45, i32 96
  %13 = load <32 x i16>, ptr %12, align 2, !tbaa !4
  %14 = shufflevector <32 x i16> %11, <32 x i16> %13, <32 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 33, i32 35, i32 37, i32 39, i32 41, i32 43, i32 45, i32 47, i32 49, i32 51, i32 53, i32 55, i32 57, i32 59, i32 61, i32 63>
  %15 = load <32 x i16>, ptr %.inc49, align 2, !tbaa !7
  %16 = getelementptr inbounds i16, ptr %InputTwo.host46, i32 96
  %17 = load <32 x i16>, ptr %16, align 2, !tbaa !7
  %18 = shufflevector <32 x i16> %15, <32 x i16> %17, <32 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62>
  %19 = bitcast <32 x i16> %14 to <16 x i32>
  %20 = bitcast <32 x i16> %18 to <16 x i32>
  %21 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %19, <16 x i32> %20)
  store <16 x i32> %21, ptr %.inc52, align 2, !tbaa !9
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #1

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }

!4 = !{!5, !5, i64 0}
!5 = !{!"InputOne", !6}
!6 = !{!"Halide buffer"}
!7 = !{!8, !8, i64 0}
!8 = !{!"InputTwo", !6}
!9 = !{!10, !10, i64 0}
!10 = !{!"Strided_LoadTest", !6}
