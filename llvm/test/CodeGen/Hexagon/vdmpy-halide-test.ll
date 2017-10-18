; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Thie tests checks a compiler assert. So the test just needs to compile for it to pass
target triple = "hexagon-unknown--elf"

%struct.buffer_t = type { i64, i8*, [4 x i32], [4 x i32], [4 x i32], i32, i8, i8, [6 x i8] }

; Function Attrs: norecurse nounwind
define i32 @__testOne(%struct.buffer_t* noalias nocapture readonly %inputOne.buffer, %struct.buffer_t* noalias nocapture readonly %inputTwo.buffer, %struct.buffer_t* noalias nocapture readonly %testOne.buffer) #0 {
entry:
  %buf_host = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %inputOne.buffer, i32 0, i32 1
  %inputOne.host = load i8*, i8** %buf_host, align 4
  %buf_min = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %inputOne.buffer, i32 0, i32 4, i32 0
  %inputOne.min.0 = load i32, i32* %buf_min, align 4
  %buf_host10 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %inputTwo.buffer, i32 0, i32 1
  %inputTwo.host = load i8*, i8** %buf_host10, align 4
  %buf_min22 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %inputTwo.buffer, i32 0, i32 4, i32 0
  %inputTwo.min.0 = load i32, i32* %buf_min22, align 4
  %buf_host27 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %testOne.buffer, i32 0, i32 1
  %testOne.host = load i8*, i8** %buf_host27, align 4
  %buf_extent31 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %testOne.buffer, i32 0, i32 2, i32 0
  %testOne.extent.0 = load i32, i32* %buf_extent31, align 4
  %buf_min39 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %testOne.buffer, i32 0, i32 4, i32 0
  %testOne.min.0 = load i32, i32* %buf_min39, align 4
  %0 = ashr i32 %testOne.extent.0, 4
  %1 = icmp sgt i32 %0, 0
  br i1 %1, label %"for testOne.s0.x.x.preheader", label %"end for testOne.s0.x.x"

"for testOne.s0.x.x.preheader":                   ; preds = %entry
  %2 = bitcast i8* %inputOne.host to i16*
  %3 = bitcast i8* %inputTwo.host to i16*
  %4 = bitcast i8* %testOne.host to i32*
  br label %"for testOne.s0.x.x"

"for testOne.s0.x.x":                             ; preds = %"for testOne.s0.x.x", %"for testOne.s0.x.x.preheader"
  %.phi = phi i32* [ %4, %"for testOne.s0.x.x.preheader" ], [ %.inc, %"for testOne.s0.x.x" ]
  %testOne.s0.x.x = phi i32 [ 0, %"for testOne.s0.x.x.preheader" ], [ %50, %"for testOne.s0.x.x" ]
  %5 = shl nsw i32 %testOne.s0.x.x, 4
  %6 = add nsw i32 %5, %testOne.min.0
  %7 = shl nsw i32 %6, 1
  %8 = sub nsw i32 %7, %inputOne.min.0
  %9 = getelementptr inbounds i16, i16* %2, i32 %8
  %10 = bitcast i16* %9 to <16 x i16>*
  %11 = load <16 x i16>, <16 x i16>* %10, align 2, !tbaa !5
  %12 = add nsw i32 %8, 15
  %13 = getelementptr inbounds i16, i16* %2, i32 %12
  %14 = bitcast i16* %13 to <16 x i16>*
  %15 = load <16 x i16>, <16 x i16>* %14, align 2, !tbaa !5
  %16 = shufflevector <16 x i16> %11, <16 x i16> %15, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %17 = add nsw i32 %8, 1
  %18 = getelementptr inbounds i16, i16* %2, i32 %17
  %19 = bitcast i16* %18 to <16 x i16>*
  %20 = load <16 x i16>, <16 x i16>* %19, align 2, !tbaa !5
  %21 = add nsw i32 %8, 16
  %22 = getelementptr inbounds i16, i16* %2, i32 %21
  %23 = bitcast i16* %22 to <16 x i16>*
  %24 = load <16 x i16>, <16 x i16>* %23, align 2, !tbaa !5
  %25 = shufflevector <16 x i16> %20, <16 x i16> %24, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %26 = shufflevector <16 x i16> %16, <16 x i16> %25, <32 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  %27 = sub nsw i32 %7, %inputTwo.min.0
  %28 = getelementptr inbounds i16, i16* %3, i32 %27
  %29 = bitcast i16* %28 to <16 x i16>*
  %30 = load <16 x i16>, <16 x i16>* %29, align 2, !tbaa !8
  %31 = add nsw i32 %27, 15
  %32 = getelementptr inbounds i16, i16* %3, i32 %31
  %33 = bitcast i16* %32 to <16 x i16>*
  %34 = load <16 x i16>, <16 x i16>* %33, align 2, !tbaa !8
  %35 = shufflevector <16 x i16> %30, <16 x i16> %34, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %36 = add nsw i32 %27, 1
  %37 = getelementptr inbounds i16, i16* %3, i32 %36
  %38 = bitcast i16* %37 to <16 x i16>*
  %39 = load <16 x i16>, <16 x i16>* %38, align 2, !tbaa !8
  %40 = add nsw i32 %27, 16
  %41 = getelementptr inbounds i16, i16* %3, i32 %40
  %42 = bitcast i16* %41 to <16 x i16>*
  %43 = load <16 x i16>, <16 x i16>* %42, align 2, !tbaa !8
  %44 = shufflevector <16 x i16> %39, <16 x i16> %43, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %45 = shufflevector <16 x i16> %35, <16 x i16> %44, <32 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  %46 = bitcast <32 x i16> %26 to <16 x i32>
  %47 = bitcast <32 x i16> %45 to <16 x i32>
  %48 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhvsat(<16 x i32> %46, <16 x i32> %47)
  %49 = bitcast i32* %.phi to <16 x i32>*
  store <16 x i32> %48, <16 x i32>* %49, align 4, !tbaa !10
  %50 = add nuw nsw i32 %testOne.s0.x.x, 1
  %51 = icmp eq i32 %50, %0
  %.inc = getelementptr i32, i32* %.phi, i32 16
  br i1 %51, label %"end for testOne.s0.x.x", label %"for testOne.s0.x.x"

"end for testOne.s0.x.x":                         ; preds = %"for testOne.s0.x.x", %entry
  %52 = add nsw i32 %testOne.extent.0, 15
  %53 = ashr i32 %52, 4
  %54 = icmp sgt i32 %53, %0
  br i1 %54, label %"for testOne.s0.x.x44.preheader", label %destructor_block

"for testOne.s0.x.x44.preheader":                 ; preds = %"end for testOne.s0.x.x"
  %55 = add nsw i32 %testOne.min.0, %testOne.extent.0
  %56 = shl nsw i32 %55, 1
  %57 = sub nsw i32 %56, %inputOne.min.0
  %58 = add nsw i32 %57, -32
  %59 = bitcast i8* %inputOne.host to i16*
  %60 = getelementptr inbounds i16, i16* %59, i32 %58
  %61 = bitcast i16* %60 to <16 x i16>*
  %62 = load <16 x i16>, <16 x i16>* %61, align 2
  %63 = add nsw i32 %57, -17
  %64 = getelementptr inbounds i16, i16* %59, i32 %63
  %65 = bitcast i16* %64 to <16 x i16>*
  %66 = load <16 x i16>, <16 x i16>* %65, align 2
  %67 = add nsw i32 %57, -31
  %68 = getelementptr inbounds i16, i16* %59, i32 %67
  %69 = bitcast i16* %68 to <16 x i16>*
  %70 = load <16 x i16>, <16 x i16>* %69, align 2
  %71 = add nsw i32 %57, -16
  %72 = getelementptr inbounds i16, i16* %59, i32 %71
  %73 = bitcast i16* %72 to <16 x i16>*
  %74 = load <16 x i16>, <16 x i16>* %73, align 2
  %75 = shufflevector <16 x i16> %70, <16 x i16> %74, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %76 = sub nsw i32 %56, %inputTwo.min.0
  %77 = add nsw i32 %76, -32
  %78 = bitcast i8* %inputTwo.host to i16*
  %79 = getelementptr inbounds i16, i16* %78, i32 %77
  %80 = bitcast i16* %79 to <16 x i16>*
  %81 = load <16 x i16>, <16 x i16>* %80, align 2
  %82 = add nsw i32 %76, -17
  %83 = getelementptr inbounds i16, i16* %78, i32 %82
  %84 = bitcast i16* %83 to <16 x i16>*
  %85 = load <16 x i16>, <16 x i16>* %84, align 2
  %86 = shufflevector <16 x i16> %81, <16 x i16> %85, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %87 = add nsw i32 %76, -31
  %88 = getelementptr inbounds i16, i16* %78, i32 %87
  %89 = bitcast i16* %88 to <16 x i16>*
  %90 = load <16 x i16>, <16 x i16>* %89, align 2
  %91 = add nsw i32 %76, -16
  %92 = getelementptr inbounds i16, i16* %78, i32 %91
  %93 = bitcast i16* %92 to <16 x i16>*
  %94 = load <16 x i16>, <16 x i16>* %93, align 2
  %95 = shufflevector <16 x i16> %90, <16 x i16> %94, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %96 = shufflevector <16 x i16> %86, <16 x i16> %95, <32 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  %97 = bitcast <32 x i16> %96 to <16 x i32>
  %98 = add nsw i32 %testOne.extent.0, -16
  %99 = bitcast i8* %testOne.host to i32*
  %100 = getelementptr inbounds i32, i32* %99, i32 %98
  %101 = shufflevector <16 x i16> %62, <16 x i16> %66, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %102 = shufflevector <16 x i16> %101, <16 x i16> %75, <32 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  %103 = bitcast <32 x i16> %102 to <16 x i32>
  %104 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhvsat(<16 x i32> %103, <16 x i32> %97)
  %105 = bitcast i32* %100 to <16 x i32>*
  store <16 x i32> %104, <16 x i32>* %105, align 4, !tbaa !10
  br label %destructor_block

destructor_block:                                 ; preds = %"for testOne.s0.x.x44.preheader", %"end for testOne.s0.x.x"
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhvsat(<16 x i32>, <16 x i32>) #1

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }

!5 = !{!6, !6, i64 0}
!6 = !{!"inputOne", !7}
!7 = !{!"Halide buffer"}
!8 = !{!9, !9, i64 0}
!9 = !{!"inputTwo", !7}
!10 = !{!11, !11, i64 0}
!11 = !{!"testOne", !7}
