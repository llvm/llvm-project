; RUN: llc -march=hexagon < %s | FileCheck %s
target triple = "hexagon-unknown--elf"

; CHECK-DAG: vpacke
; CHECK-DAG: vpacko

%struct.buffer_t = type { i64, i8*, [4 x i32], [4 x i32], [4 x i32], i32, i8, i8, [6 x i8] }

; Function Attrs: norecurse nounwind
define i32 @__Strided_LoadTest(%struct.buffer_t* noalias nocapture readonly %InputOne.buffer, %struct.buffer_t* noalias nocapture readonly %InputTwo.buffer, %struct.buffer_t* noalias nocapture readonly %Strided_LoadTest.buffer) #0 {
entry:
  %buf_host = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %InputOne.buffer, i32 0, i32 1
  %0 = bitcast i8** %buf_host to i16**
  %InputOne.host45 = load i16*, i16** %0, align 4
  %buf_host10 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %InputTwo.buffer, i32 0, i32 1
  %1 = bitcast i8** %buf_host10 to i16**
  %InputTwo.host46 = load i16*, i16** %1, align 4
  %buf_host27 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %Strided_LoadTest.buffer, i32 0, i32 1
  %2 = bitcast i8** %buf_host27 to i16**
  %Strided_LoadTest.host44 = load i16*, i16** %2, align 4
  %3 = bitcast i16* %InputOne.host45 to <32 x i16>*
  %4 = load <32 x i16>, <32 x i16>* %3, align 2, !tbaa !4
  %5 = getelementptr inbounds i16, i16* %InputOne.host45, i32 32
  %6 = bitcast i16* %5 to <32 x i16>*
  %7 = load <32 x i16>, <32 x i16>* %6, align 2, !tbaa !4
  %8 = shufflevector <32 x i16> %4, <32 x i16> %7, <32 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 33, i32 35, i32 37, i32 39, i32 41, i32 43, i32 45, i32 47, i32 49, i32 51, i32 53, i32 55, i32 57, i32 59, i32 61, i32 63>
  %9 = bitcast i16* %InputTwo.host46 to <32 x i16>*
  %10 = load <32 x i16>, <32 x i16>* %9, align 2, !tbaa !7
  %11 = getelementptr inbounds i16, i16* %InputTwo.host46, i32 32
  %12 = bitcast i16* %11 to <32 x i16>*
  %13 = load <32 x i16>, <32 x i16>* %12, align 2, !tbaa !7
  %14 = shufflevector <32 x i16> %10, <32 x i16> %13, <32 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62>
  %15 = bitcast <32 x i16> %8 to <16 x i32>
  %16 = bitcast <32 x i16> %14 to <16 x i32>
  %17 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %15, <16 x i32> %16)
  %18 = bitcast i16* %Strided_LoadTest.host44 to <16 x i32>*
  store <16 x i32> %17, <16 x i32>* %18, align 2, !tbaa !9
  %.inc = getelementptr i16, i16* %InputOne.host45, i32 64
  %.inc49 = getelementptr i16, i16* %InputTwo.host46, i32 64
  %.inc52 = getelementptr i16, i16* %Strided_LoadTest.host44, i32 32
  %19 = bitcast i16* %.inc to <32 x i16>*
  %20 = load <32 x i16>, <32 x i16>* %19, align 2, !tbaa !4
  %21 = getelementptr inbounds i16, i16* %InputOne.host45, i32 96
  %22 = bitcast i16* %21 to <32 x i16>*
  %23 = load <32 x i16>, <32 x i16>* %22, align 2, !tbaa !4
  %24 = shufflevector <32 x i16> %20, <32 x i16> %23, <32 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 33, i32 35, i32 37, i32 39, i32 41, i32 43, i32 45, i32 47, i32 49, i32 51, i32 53, i32 55, i32 57, i32 59, i32 61, i32 63>
  %25 = bitcast i16* %.inc49 to <32 x i16>*
  %26 = load <32 x i16>, <32 x i16>* %25, align 2, !tbaa !7
  %27 = getelementptr inbounds i16, i16* %InputTwo.host46, i32 96
  %28 = bitcast i16* %27 to <32 x i16>*
  %29 = load <32 x i16>, <32 x i16>* %28, align 2, !tbaa !7
  %30 = shufflevector <32 x i16> %26, <32 x i16> %29, <32 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62>
  %31 = bitcast <32 x i16> %24 to <16 x i32>
  %32 = bitcast <32 x i16> %30 to <16 x i32>
  %33 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %31, <16 x i32> %32)
  %34 = bitcast i16* %.inc52 to <16 x i32>*
  store <16 x i32> %33, <16 x i32>* %34, align 2, !tbaa !9
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #1

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,-hvx-double" }
attributes #1 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,-hvx-double" }

!4 = !{!5, !5, i64 0}
!5 = !{!"InputOne", !6}
!6 = !{!"Halide buffer"}
!7 = !{!8, !8, i64 0}
!8 = !{!"InputTwo", !6}
!9 = !{!10, !10, i64 0}
!10 = !{!"Strided_LoadTest", !6}
