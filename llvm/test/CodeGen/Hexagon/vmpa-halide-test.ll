; RUN: llc -march=hexagon < %s
; Thie tests checks a compiler assert. So the test just needs to compile
; for it to pass

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
  %0 = ashr i32 %testOne.extent.0, 6
  %1 = icmp sgt i32 %0, 0
  br i1 %1, label %"for testOne.s0.x.x.preheader", label %"end for testOne.s0.x.x"

"for testOne.s0.x.x.preheader":                   ; preds = %entry
  %2 = bitcast i8* %testOne.host to i16*
  br label %"for testOne.s0.x.x"

"for testOne.s0.x.x":                             ; preds = %"for testOne.s0.x.x", %"for testOne.s0.x.x.preheader"
  %.phi = phi i16* [ %2, %"for testOne.s0.x.x.preheader" ], [ %.inc, %"for testOne.s0.x.x" ]
  %testOne.s0.x.x = phi i32 [ 0, %"for testOne.s0.x.x.preheader" ], [ %38, %"for testOne.s0.x.x" ]
  %3 = shl nsw i32 %testOne.s0.x.x, 6
  %4 = add nsw i32 %3, %testOne.min.0
  %5 = shl nsw i32 %4, 1
  %6 = sub nsw i32 %5, %inputOne.min.0
  %7 = getelementptr inbounds i8, i8* %inputOne.host, i32 %6
  %8 = bitcast i8* %7 to <64 x i8>*
  %9 = load <64 x i8>, <64 x i8>* %8, align 1, !tbaa !5
  %10 = add nsw i32 %6, 64
  %11 = getelementptr inbounds i8, i8* %inputOne.host, i32 %10
  %12 = bitcast i8* %11 to <64 x i8>*
  %13 = load <64 x i8>, <64 x i8>* %12, align 1, !tbaa !5
  %14 = shufflevector <64 x i8> %9, <64 x i8> %13, <64 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62, i32 64, i32 66, i32 68, i32 70, i32 72, i32 74, i32 76, i32 78, i32 80, i32 82, i32 84, i32 86, i32 88, i32 90, i32 92, i32 94, i32 96, i32 98, i32 100, i32 102, i32 104, i32 106, i32 108, i32 110, i32 112, i32 114, i32 116, i32 118, i32 120, i32 122, i32 124, i32 126>
  %15 = shufflevector <64 x i8> %9, <64 x i8> %13, <64 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 33, i32 35, i32 37, i32 39, i32 41, i32 43, i32 45, i32 47, i32 49, i32 51, i32 53, i32 55, i32 57, i32 59, i32 61, i32 63, i32 65, i32 67, i32 69, i32 71, i32 73, i32 75, i32 77, i32 79, i32 81, i32 83, i32 85, i32 87, i32 89, i32 91, i32 93, i32 95, i32 97, i32 99, i32 101, i32 103, i32 105, i32 107, i32 109, i32 111, i32 113, i32 115, i32 117, i32 119, i32 121, i32 123, i32 125, i32 127>
  %16 = shufflevector <64 x i8> %14, <64 x i8> %15, <128 x i32> <i32 0, i32 64, i32 1, i32 65, i32 2, i32 66, i32 3, i32 67, i32 4, i32 68, i32 5, i32 69, i32 6, i32 70, i32 7, i32 71, i32 8, i32 72, i32 9, i32 73, i32 10, i32 74, i32 11, i32 75, i32 12, i32 76, i32 13, i32 77, i32 14, i32 78, i32 15, i32 79, i32 16, i32 80, i32 17, i32 81, i32 18, i32 82, i32 19, i32 83, i32 20, i32 84, i32 21, i32 85, i32 22, i32 86, i32 23, i32 87, i32 24, i32 88, i32 25, i32 89, i32 26, i32 90, i32 27, i32 91, i32 28, i32 92, i32 29, i32 93, i32 30, i32 94, i32 31, i32 95, i32 32, i32 96, i32 33, i32 97, i32 34, i32 98, i32 35, i32 99, i32 36, i32 100, i32 37, i32 101, i32 38, i32 102, i32 39, i32 103, i32 40, i32 104, i32 41, i32 105, i32 42, i32 106, i32 43, i32 107, i32 44, i32 108, i32 45, i32 109, i32 46, i32 110, i32 47, i32 111, i32 48, i32 112, i32 49, i32 113, i32 50, i32 114, i32 51, i32 115, i32 52, i32 116, i32 53, i32 117, i32 54, i32 118, i32 55, i32 119, i32 56, i32 120, i32 57, i32 121, i32 58, i32 122, i32 59, i32 123, i32 60, i32 124, i32 61, i32 125, i32 62, i32 126, i32 63, i32 127>
  %17 = sub nsw i32 %5, %inputTwo.min.0
  %18 = getelementptr inbounds i8, i8* %inputTwo.host, i32 %17
  %19 = bitcast i8* %18 to <64 x i8>*
  %20 = load <64 x i8>, <64 x i8>* %19, align 1, !tbaa !8
  %21 = add nsw i32 %17, 64
  %22 = getelementptr inbounds i8, i8* %inputTwo.host, i32 %21
  %23 = bitcast i8* %22 to <64 x i8>*
  %24 = load <64 x i8>, <64 x i8>* %23, align 1, !tbaa !8
  %25 = shufflevector <64 x i8> %20, <64 x i8> %24, <64 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62, i32 64, i32 66, i32 68, i32 70, i32 72, i32 74, i32 76, i32 78, i32 80, i32 82, i32 84, i32 86, i32 88, i32 90, i32 92, i32 94, i32 96, i32 98, i32 100, i32 102, i32 104, i32 106, i32 108, i32 110, i32 112, i32 114, i32 116, i32 118, i32 120, i32 122, i32 124, i32 126>
  %26 = shufflevector <64 x i8> %20, <64 x i8> %24, <64 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 33, i32 35, i32 37, i32 39, i32 41, i32 43, i32 45, i32 47, i32 49, i32 51, i32 53, i32 55, i32 57, i32 59, i32 61, i32 63, i32 65, i32 67, i32 69, i32 71, i32 73, i32 75, i32 77, i32 79, i32 81, i32 83, i32 85, i32 87, i32 89, i32 91, i32 93, i32 95, i32 97, i32 99, i32 101, i32 103, i32 105, i32 107, i32 109, i32 111, i32 113, i32 115, i32 117, i32 119, i32 121, i32 123, i32 125, i32 127>
  %27 = shufflevector <64 x i8> %25, <64 x i8> %26, <128 x i32> <i32 0, i32 64, i32 1, i32 65, i32 2, i32 66, i32 3, i32 67, i32 4, i32 68, i32 5, i32 69, i32 6, i32 70, i32 7, i32 71, i32 8, i32 72, i32 9, i32 73, i32 10, i32 74, i32 11, i32 75, i32 12, i32 76, i32 13, i32 77, i32 14, i32 78, i32 15, i32 79, i32 16, i32 80, i32 17, i32 81, i32 18, i32 82, i32 19, i32 83, i32 20, i32 84, i32 21, i32 85, i32 22, i32 86, i32 23, i32 87, i32 24, i32 88, i32 25, i32 89, i32 26, i32 90, i32 27, i32 91, i32 28, i32 92, i32 29, i32 93, i32 30, i32 94, i32 31, i32 95, i32 32, i32 96, i32 33, i32 97, i32 34, i32 98, i32 35, i32 99, i32 36, i32 100, i32 37, i32 101, i32 38, i32 102, i32 39, i32 103, i32 40, i32 104, i32 41, i32 105, i32 42, i32 106, i32 43, i32 107, i32 44, i32 108, i32 45, i32 109, i32 46, i32 110, i32 47, i32 111, i32 48, i32 112, i32 49, i32 113, i32 50, i32 114, i32 51, i32 115, i32 52, i32 116, i32 53, i32 117, i32 54, i32 118, i32 55, i32 119, i32 56, i32 120, i32 57, i32 121, i32 58, i32 122, i32 59, i32 123, i32 60, i32 124, i32 61, i32 125, i32 62, i32 126, i32 63, i32 127>
  %28 = bitcast <128 x i8> %16 to <32 x i32>
  %29 = bitcast <128 x i8> %27 to <32 x i32>
  %30 = tail call <32 x i32> @llvm.hexagon.V6.vmpabuuv(<32 x i32> %28, <32 x i32> %29)
  %31 = bitcast <32 x i32> %30 to <64 x i16>
  %32 = shufflevector <64 x i16> %31, <64 x i16> undef, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %33 = bitcast i16* %.phi to <32 x i16>*
  store <32 x i16> %32, <32 x i16>* %33, align 2, !tbaa !10
  %34 = shufflevector <64 x i16> %31, <64 x i16> undef, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %35 = or i32 %3, 32
  %36 = getelementptr inbounds i16, i16* %2, i32 %35
  %37 = bitcast i16* %36 to <32 x i16>*
  store <32 x i16> %34, <32 x i16>* %37, align 2, !tbaa !10
  %38 = add nuw nsw i32 %testOne.s0.x.x, 1
  %39 = icmp eq i32 %38, %0
  %.inc = getelementptr i16, i16* %.phi, i32 64
  br i1 %39, label %"end for testOne.s0.x.x", label %"for testOne.s0.x.x"

"end for testOne.s0.x.x":                         ; preds = %"for testOne.s0.x.x", %entry
  %40 = add nsw i32 %testOne.extent.0, 63
  %41 = ashr i32 %40, 6
  %42 = icmp sgt i32 %41, %0
  br i1 %42, label %"for testOne.s0.x.x44.preheader", label %destructor_block

"for testOne.s0.x.x44.preheader":                 ; preds = %"end for testOne.s0.x.x"
  %43 = add nsw i32 %testOne.min.0, %testOne.extent.0
  %44 = shl nsw i32 %43, 1
  %45 = sub nsw i32 %44, %inputOne.min.0
  %46 = add nsw i32 %45, -128
  %47 = getelementptr inbounds i8, i8* %inputOne.host, i32 %46
  %48 = bitcast i8* %47 to <64 x i8>*
  %49 = load <64 x i8>, <64 x i8>* %48, align 1
  %50 = add nsw i32 %45, -64
  %51 = getelementptr inbounds i8, i8* %inputOne.host, i32 %50
  %52 = bitcast i8* %51 to <64 x i8>*
  %53 = load <64 x i8>, <64 x i8>* %52, align 1
  %54 = shufflevector <64 x i8> %49, <64 x i8> %53, <64 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62, i32 64, i32 66, i32 68, i32 70, i32 72, i32 74, i32 76, i32 78, i32 80, i32 82, i32 84, i32 86, i32 88, i32 90, i32 92, i32 94, i32 96, i32 98, i32 100, i32 102, i32 104, i32 106, i32 108, i32 110, i32 112, i32 114, i32 116, i32 118, i32 120, i32 122, i32 124, i32 126>
  %55 = shufflevector <64 x i8> %49, <64 x i8> %53, <64 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 33, i32 35, i32 37, i32 39, i32 41, i32 43, i32 45, i32 47, i32 49, i32 51, i32 53, i32 55, i32 57, i32 59, i32 61, i32 63, i32 65, i32 67, i32 69, i32 71, i32 73, i32 75, i32 77, i32 79, i32 81, i32 83, i32 85, i32 87, i32 89, i32 91, i32 93, i32 95, i32 97, i32 99, i32 101, i32 103, i32 105, i32 107, i32 109, i32 111, i32 113, i32 115, i32 117, i32 119, i32 121, i32 123, i32 125, i32 127>
  %56 = shufflevector <64 x i8> %54, <64 x i8> %55, <128 x i32> <i32 0, i32 64, i32 1, i32 65, i32 2, i32 66, i32 3, i32 67, i32 4, i32 68, i32 5, i32 69, i32 6, i32 70, i32 7, i32 71, i32 8, i32 72, i32 9, i32 73, i32 10, i32 74, i32 11, i32 75, i32 12, i32 76, i32 13, i32 77, i32 14, i32 78, i32 15, i32 79, i32 16, i32 80, i32 17, i32 81, i32 18, i32 82, i32 19, i32 83, i32 20, i32 84, i32 21, i32 85, i32 22, i32 86, i32 23, i32 87, i32 24, i32 88, i32 25, i32 89, i32 26, i32 90, i32 27, i32 91, i32 28, i32 92, i32 29, i32 93, i32 30, i32 94, i32 31, i32 95, i32 32, i32 96, i32 33, i32 97, i32 34, i32 98, i32 35, i32 99, i32 36, i32 100, i32 37, i32 101, i32 38, i32 102, i32 39, i32 103, i32 40, i32 104, i32 41, i32 105, i32 42, i32 106, i32 43, i32 107, i32 44, i32 108, i32 45, i32 109, i32 46, i32 110, i32 47, i32 111, i32 48, i32 112, i32 49, i32 113, i32 50, i32 114, i32 51, i32 115, i32 52, i32 116, i32 53, i32 117, i32 54, i32 118, i32 55, i32 119, i32 56, i32 120, i32 57, i32 121, i32 58, i32 122, i32 59, i32 123, i32 60, i32 124, i32 61, i32 125, i32 62, i32 126, i32 63, i32 127>
  %57 = sub nsw i32 %44, %inputTwo.min.0
  %58 = add nsw i32 %57, -128
  %59 = getelementptr inbounds i8, i8* %inputTwo.host, i32 %58
  %60 = bitcast i8* %59 to <64 x i8>*
  %61 = load <64 x i8>, <64 x i8>* %60, align 1
  %62 = add nsw i32 %57, -64
  %63 = getelementptr inbounds i8, i8* %inputTwo.host, i32 %62
  %64 = bitcast i8* %63 to <64 x i8>*
  %65 = load <64 x i8>, <64 x i8>* %64, align 1
  %66 = shufflevector <64 x i8> %61, <64 x i8> %65, <64 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62, i32 64, i32 66, i32 68, i32 70, i32 72, i32 74, i32 76, i32 78, i32 80, i32 82, i32 84, i32 86, i32 88, i32 90, i32 92, i32 94, i32 96, i32 98, i32 100, i32 102, i32 104, i32 106, i32 108, i32 110, i32 112, i32 114, i32 116, i32 118, i32 120, i32 122, i32 124, i32 126>
  %67 = shufflevector <64 x i8> %61, <64 x i8> %65, <64 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 33, i32 35, i32 37, i32 39, i32 41, i32 43, i32 45, i32 47, i32 49, i32 51, i32 53, i32 55, i32 57, i32 59, i32 61, i32 63, i32 65, i32 67, i32 69, i32 71, i32 73, i32 75, i32 77, i32 79, i32 81, i32 83, i32 85, i32 87, i32 89, i32 91, i32 93, i32 95, i32 97, i32 99, i32 101, i32 103, i32 105, i32 107, i32 109, i32 111, i32 113, i32 115, i32 117, i32 119, i32 121, i32 123, i32 125, i32 127>
  %68 = shufflevector <64 x i8> %66, <64 x i8> %67, <128 x i32> <i32 0, i32 64, i32 1, i32 65, i32 2, i32 66, i32 3, i32 67, i32 4, i32 68, i32 5, i32 69, i32 6, i32 70, i32 7, i32 71, i32 8, i32 72, i32 9, i32 73, i32 10, i32 74, i32 11, i32 75, i32 12, i32 76, i32 13, i32 77, i32 14, i32 78, i32 15, i32 79, i32 16, i32 80, i32 17, i32 81, i32 18, i32 82, i32 19, i32 83, i32 20, i32 84, i32 21, i32 85, i32 22, i32 86, i32 23, i32 87, i32 24, i32 88, i32 25, i32 89, i32 26, i32 90, i32 27, i32 91, i32 28, i32 92, i32 29, i32 93, i32 30, i32 94, i32 31, i32 95, i32 32, i32 96, i32 33, i32 97, i32 34, i32 98, i32 35, i32 99, i32 36, i32 100, i32 37, i32 101, i32 38, i32 102, i32 39, i32 103, i32 40, i32 104, i32 41, i32 105, i32 42, i32 106, i32 43, i32 107, i32 44, i32 108, i32 45, i32 109, i32 46, i32 110, i32 47, i32 111, i32 48, i32 112, i32 49, i32 113, i32 50, i32 114, i32 51, i32 115, i32 52, i32 116, i32 53, i32 117, i32 54, i32 118, i32 55, i32 119, i32 56, i32 120, i32 57, i32 121, i32 58, i32 122, i32 59, i32 123, i32 60, i32 124, i32 61, i32 125, i32 62, i32 126, i32 63, i32 127>
  %69 = bitcast <128 x i8> %56 to <32 x i32>
  %70 = bitcast <128 x i8> %68 to <32 x i32>
  %71 = tail call <32 x i32> @llvm.hexagon.V6.vmpabuuv(<32 x i32> %69, <32 x i32> %70)
  %72 = bitcast <32 x i32> %71 to <64 x i16>
  %73 = add nsw i32 %testOne.extent.0, -64
  %74 = bitcast i8* %testOne.host to i16*
  %75 = getelementptr inbounds i16, i16* %74, i32 %73
  %76 = bitcast i16* %75 to <32 x i16>*
  %77 = add nsw i32 %testOne.extent.0, -32
  %78 = getelementptr inbounds i16, i16* %74, i32 %77
  %79 = shufflevector <64 x i16> %72, <64 x i16> undef, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %80 = shufflevector <64 x i16> %72, <64 x i16> undef, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %81 = bitcast i16* %78 to <32 x i16>*
  store <32 x i16> %79, <32 x i16>* %76, align 2, !tbaa !10
  store <32 x i16> %80, <32 x i16>* %81, align 2, !tbaa !10
  br label %destructor_block

destructor_block:                                 ; preds = %"for testOne.s0.x.x44.preheader", %"end for testOne.s0.x.x"
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpabuuv(<32 x i32>, <32 x i32>) #1

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }

!5 = !{!6, !6, i64 0}
!6 = !{!"inputOne", !7}
!7 = !{!"Halide buffer"}
!8 = !{!9, !9, i64 0}
!9 = !{!"inputTwo", !7}
!10 = !{!11, !11, i64 0}
!11 = !{!"testOne", !7}
