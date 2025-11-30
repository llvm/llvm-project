; RUN: llc -march=hexagon -hexagon-opt-shuffvec -hexagon-widen-short-vector -hexagon-hvx-widen=32 -mv73 -mhvx -mattr=+hvx-length128b < %s
; REQUIRES: asserts

define dllexport i32 @test(ptr noalias align 128 %0, ptr noalias align 128 %1, ptr noalias align 128 %2) local_unnamed_addr {
entry:
  call void @llvm.assume(i1 true) [ "align"(ptr %0, i32 128) ]
  call void @llvm.assume(i1 true) [ "align"(ptr %1, i32 128) ]
  call void @llvm.assume(i1 true) [ "align"(ptr %2, i32 128) ]
  %3 = load <32 x i8>, ptr %2, align 128
  %4 = zext <32 x i8> %3 to <32 x i32>
  %5 = mul nuw nsw <32 x i32> %4, <i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21>
  %scevgep = getelementptr i32, ptr %0, i32 128
  %scevgep13 = getelementptr i8, ptr %1, i32 128
  br label %for_begin1.preheader

for_begin1.preheader:                             ; preds = %for_end3, %entry
  %lsr.iv14 = phi ptr [ %scevgep15, %for_end3 ], [ %scevgep13, %entry ]
  %lsr.iv1 = phi ptr [ %scevgep2, %for_end3 ], [ %scevgep, %entry ]
  %6 = phi i32 [ 0, %entry ], [ %47, %for_end3 ]
  br label %for_body2

for_end:                                          ; preds = %for_end3
  ret i32 0

for_body2:                                        ; preds = %for_body2, %for_begin1.preheader
  %lsr.iv16 = phi ptr [ %scevgep17, %for_body2 ], [ %lsr.iv14, %for_begin1.preheader ]
  %lsr.iv3 = phi ptr [ %scevgep4, %for_body2 ], [ %lsr.iv1, %for_begin1.preheader ]
  %lsr.iv = phi i32 [ %lsr.iv.next, %for_body2 ], [ 128, %for_begin1.preheader ]
  %scevgep20 = getelementptr <32 x i8>, ptr %lsr.iv16, i32 -4
  %7 = load <32 x i8>, ptr %scevgep20, align 128
  %8 = zext <32 x i8> %7 to <32 x i32>
  %9 = mul nuw nsw <32 x i32> %8, <i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96>
  %10 = add nsw <32 x i32> %9, <i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303>
  %11 = add nsw <32 x i32> %10, %5
  %scevgep6 = getelementptr <32 x i32>, ptr %lsr.iv3, i32 -4
  store <32 x i32> %11, ptr %scevgep6, align 128
  %scevgep21 = getelementptr <32 x i8>, ptr %lsr.iv16, i32 -3
  %12 = load <32 x i8>, ptr %scevgep21, align 32
  %13 = zext <32 x i8> %12 to <32 x i32>
  %14 = mul nuw nsw <32 x i32> %13, <i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96>
  %15 = add nsw <32 x i32> %14, <i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303>
  %16 = add nsw <32 x i32> %15, %5
  %scevgep8 = getelementptr <32 x i32>, ptr %lsr.iv3, i32 -3
  store <32 x i32> %16, ptr %scevgep8, align 128
  %scevgep22 = getelementptr <32 x i8>, ptr %lsr.iv16, i32 -2
  %17 = load <32 x i8>, ptr %scevgep22, align 64
  %18 = zext <32 x i8> %17 to <32 x i32>
  %19 = mul nuw nsw <32 x i32> %18, <i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96>
  %20 = add nsw <32 x i32> %19, <i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303>
  %21 = add nsw <32 x i32> %20, %5
  %scevgep9 = getelementptr <32 x i32>, ptr %lsr.iv3, i32 -2
  store <32 x i32> %21, ptr %scevgep9, align 128
  %scevgep23 = getelementptr <32 x i8>, ptr %lsr.iv16, i32 -1
  %22 = load <32 x i8>, ptr %scevgep23, align 32
  %23 = zext <32 x i8> %22 to <32 x i32>
  %24 = mul nuw nsw <32 x i32> %23, <i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96>
  %25 = add nsw <32 x i32> %24, <i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303>
  %26 = add nsw <32 x i32> %25, %5
  %scevgep10 = getelementptr <32 x i32>, ptr %lsr.iv3, i32 -1
  store <32 x i32> %26, ptr %scevgep10, align 128
  %27 = load <32 x i8>, ptr %lsr.iv16, align 128
  %28 = zext <32 x i8> %27 to <32 x i32>
  %29 = mul nuw nsw <32 x i32> %28, <i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96>
  %30 = add nsw <32 x i32> %29, <i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303>
  %31 = add nsw <32 x i32> %30, %5
  store <32 x i32> %31, ptr %lsr.iv3, align 128
  %scevgep24 = getelementptr <32 x i8>, ptr %lsr.iv16, i32 1
  %32 = load <32 x i8>, ptr %scevgep24, align 32
  %33 = zext <32 x i8> %32 to <32 x i32>
  %34 = mul nuw nsw <32 x i32> %33, <i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96>
  %35 = add nsw <32 x i32> %34, <i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303>
  %36 = add nsw <32 x i32> %35, %5
  %scevgep12 = getelementptr <32 x i32>, ptr %lsr.iv3, i32 1
  store <32 x i32> %36, ptr %scevgep12, align 128
  %scevgep25 = getelementptr <32 x i8>, ptr %lsr.iv16, i32 2
  %37 = load <32 x i8>, ptr %scevgep25, align 64
  %38 = zext <32 x i8> %37 to <32 x i32>
  %39 = mul nuw nsw <32 x i32> %38, <i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96>
  %40 = add nsw <32 x i32> %39, <i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303>
  %41 = add nsw <32 x i32> %40, %5
  %scevgep11 = getelementptr <32 x i32>, ptr %lsr.iv3, i32 2
  store <32 x i32> %41, ptr %scevgep11, align 128
  %scevgep19 = getelementptr <32 x i8>, ptr %lsr.iv16, i32 3
  %42 = load <32 x i8>, ptr %scevgep19, align 32
  %43 = zext <32 x i8> %42 to <32 x i32>
  %44 = mul nuw nsw <32 x i32> %43, <i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96>
  %45 = add nsw <32 x i32> %44, <i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303, i32 -9303>
  %46 = add nsw <32 x i32> %45, %5
  %scevgep7 = getelementptr <32 x i32>, ptr %lsr.iv3, i32 3
  store <32 x i32> %46, ptr %scevgep7, align 128
  %lsr.iv.next = add nsw i32 %lsr.iv, -8
  %scevgep4 = getelementptr i32, ptr %lsr.iv3, i32 256
  %scevgep17 = getelementptr i8, ptr %lsr.iv16, i32 256
  %exitcond.not.7 = icmp eq i32 %lsr.iv.next, 0
  br i1 %exitcond.not.7, label %for_end3, label %for_body2

for_end3:                                         ; preds = %for_body2
  %47 = add nuw nsw i32 %6, 1
  %scevgep2 = getelementptr i32, ptr %lsr.iv1, i32 4096
  %scevgep15 = getelementptr i8, ptr %lsr.iv14, i32 4096
  %exitcond4.not = icmp eq i32 %47, 128
  br i1 %exitcond4.not, label %for_end, label %for_begin1.preheader
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef)
