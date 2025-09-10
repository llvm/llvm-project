; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr51933.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr51933.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@v1 = internal global i8 0, align 4
@v2 = internal global [256 x i8] zeroinitializer, align 16
@v3 = internal global [256 x i8] zeroinitializer, align 16
@.str = private unnamed_addr constant [18 x i8] c"abcdeghhijkmmnoqq\00", align 1

; Function Attrs: noinline nounwind uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  tail call void asm sideeffect "", "imr,imr,imr,~{memory}"(ptr nonnull @v1, ptr nonnull @v2, ptr nonnull @v3) #5, !srcloc !6
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local noundef i32 @bar(i32 noundef returned %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) local_unnamed_addr #0 {
  %4 = load i8, ptr @v1, align 4, !tbaa !7
  %5 = icmp eq i8 %4, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %3
  tail call void @foo()
  br label %7

7:                                                ; preds = %6, %3
  %8 = icmp sgt i32 %0, 0
  br i1 %8, label %9, label %26

9:                                                ; preds = %7
  %10 = zext nneg i32 %0 to i64
  br label %11

11:                                               ; preds = %9, %11
  %12 = phi i64 [ 0, %9 ], [ %24, %11 ]
  %13 = getelementptr inbounds nuw i16, ptr %1, i64 %12
  %14 = load i16, ptr %13, align 2, !tbaa !10
  %15 = icmp ult i16 %14, 256
  %16 = zext nneg i16 %14 to i64
  %17 = getelementptr inbounds nuw i8, ptr @v2, i64 %16
  %18 = and i16 %14, 255
  %19 = zext nneg i16 %18 to i64
  %20 = getelementptr inbounds nuw i8, ptr @v3, i64 %19
  %21 = select i1 %15, ptr %17, ptr %20
  %22 = load i8, ptr %21, align 1, !tbaa !7
  %23 = getelementptr inbounds nuw i8, ptr %2, i64 %12
  store i8 %22, ptr %23, align 1, !tbaa !7
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !12

26:                                               ; preds = %11, %7
  %27 = sext i32 %0 to i64
  %28 = getelementptr inbounds i8, ptr %2, i64 %27
  store i8 0, ptr %28, align 1, !tbaa !7
  ret i32 %0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca [18 x i8], align 1
  %2 = alloca [18 x i16], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  store <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, ptr @v2, align 16, !tbaa !7
  store <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16>, ptr @v3, align 16, !tbaa !7
  store <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 16), align 16, !tbaa !7
  store <16 x i8> <i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31, i8 32>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 16), align 16, !tbaa !7
  store <16 x i8> <i8 32, i8 33, i8 34, i8 35, i8 36, i8 37, i8 38, i8 39, i8 40, i8 41, i8 42, i8 43, i8 44, i8 45, i8 46, i8 47>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 32), align 16, !tbaa !7
  store <16 x i8> <i8 33, i8 34, i8 35, i8 36, i8 37, i8 38, i8 39, i8 40, i8 41, i8 42, i8 43, i8 44, i8 45, i8 46, i8 47, i8 48>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 32), align 16, !tbaa !7
  store <16 x i8> <i8 48, i8 49, i8 50, i8 51, i8 52, i8 53, i8 54, i8 55, i8 56, i8 57, i8 58, i8 59, i8 60, i8 61, i8 62, i8 63>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 48), align 16, !tbaa !7
  store <16 x i8> <i8 49, i8 50, i8 51, i8 52, i8 53, i8 54, i8 55, i8 56, i8 57, i8 58, i8 59, i8 60, i8 61, i8 62, i8 63, i8 64>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 48), align 16, !tbaa !7
  store <16 x i8> <i8 64, i8 65, i8 66, i8 67, i8 68, i8 69, i8 70, i8 71, i8 72, i8 73, i8 74, i8 75, i8 76, i8 77, i8 78, i8 79>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 64), align 16, !tbaa !7
  store <16 x i8> <i8 65, i8 66, i8 67, i8 68, i8 69, i8 70, i8 71, i8 72, i8 73, i8 74, i8 75, i8 76, i8 77, i8 78, i8 79, i8 80>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 64), align 16, !tbaa !7
  store <16 x i8> <i8 80, i8 81, i8 82, i8 83, i8 84, i8 85, i8 86, i8 87, i8 88, i8 89, i8 90, i8 91, i8 92, i8 93, i8 94, i8 95>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 80), align 16, !tbaa !7
  store <16 x i8> <i8 81, i8 82, i8 83, i8 84, i8 85, i8 86, i8 87, i8 88, i8 89, i8 90, i8 91, i8 92, i8 93, i8 94, i8 95, i8 96>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 80), align 16, !tbaa !7
  store <16 x i8> <i8 96, i8 97, i8 98, i8 99, i8 100, i8 101, i8 102, i8 103, i8 104, i8 105, i8 106, i8 107, i8 108, i8 109, i8 110, i8 111>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 96), align 16, !tbaa !7
  store <16 x i8> <i8 97, i8 98, i8 99, i8 100, i8 101, i8 102, i8 103, i8 104, i8 105, i8 106, i8 107, i8 108, i8 109, i8 110, i8 111, i8 112>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 96), align 16, !tbaa !7
  store <16 x i8> <i8 112, i8 113, i8 114, i8 115, i8 116, i8 117, i8 118, i8 119, i8 120, i8 121, i8 122, i8 123, i8 124, i8 125, i8 126, i8 127>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 112), align 16, !tbaa !7
  store <16 x i8> <i8 113, i8 114, i8 115, i8 116, i8 117, i8 118, i8 119, i8 120, i8 121, i8 122, i8 123, i8 124, i8 125, i8 126, i8 127, i8 -128>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 112), align 16, !tbaa !7
  store <16 x i8> <i8 -128, i8 -127, i8 -126, i8 -125, i8 -124, i8 -123, i8 -122, i8 -121, i8 -120, i8 -119, i8 -118, i8 -117, i8 -116, i8 -115, i8 -114, i8 -113>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 128), align 16, !tbaa !7
  store <16 x i8> <i8 -127, i8 -126, i8 -125, i8 -124, i8 -123, i8 -122, i8 -121, i8 -120, i8 -119, i8 -118, i8 -117, i8 -116, i8 -115, i8 -114, i8 -113, i8 -112>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 128), align 16, !tbaa !7
  store <16 x i8> <i8 -112, i8 -111, i8 -110, i8 -109, i8 -108, i8 -107, i8 -106, i8 -105, i8 -104, i8 -103, i8 -102, i8 -101, i8 -100, i8 -99, i8 -98, i8 -97>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 144), align 16, !tbaa !7
  store <16 x i8> <i8 -111, i8 -110, i8 -109, i8 -108, i8 -107, i8 -106, i8 -105, i8 -104, i8 -103, i8 -102, i8 -101, i8 -100, i8 -99, i8 -98, i8 -97, i8 -96>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 144), align 16, !tbaa !7
  store <16 x i8> <i8 -96, i8 -95, i8 -94, i8 -93, i8 -92, i8 -91, i8 -90, i8 -89, i8 -88, i8 -87, i8 -86, i8 -85, i8 -84, i8 -83, i8 -82, i8 -81>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 160), align 16, !tbaa !7
  store <16 x i8> <i8 -95, i8 -94, i8 -93, i8 -92, i8 -91, i8 -90, i8 -89, i8 -88, i8 -87, i8 -86, i8 -85, i8 -84, i8 -83, i8 -82, i8 -81, i8 -80>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 160), align 16, !tbaa !7
  store <16 x i8> <i8 -80, i8 -79, i8 -78, i8 -77, i8 -76, i8 -75, i8 -74, i8 -73, i8 -72, i8 -71, i8 -70, i8 -69, i8 -68, i8 -67, i8 -66, i8 -65>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 176), align 16, !tbaa !7
  store <16 x i8> <i8 -79, i8 -78, i8 -77, i8 -76, i8 -75, i8 -74, i8 -73, i8 -72, i8 -71, i8 -70, i8 -69, i8 -68, i8 -67, i8 -66, i8 -65, i8 -64>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 176), align 16, !tbaa !7
  store <16 x i8> <i8 -64, i8 -63, i8 -62, i8 -61, i8 -60, i8 -59, i8 -58, i8 -57, i8 -56, i8 -55, i8 -54, i8 -53, i8 -52, i8 -51, i8 -50, i8 -49>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 192), align 16, !tbaa !7
  store <16 x i8> <i8 -63, i8 -62, i8 -61, i8 -60, i8 -59, i8 -58, i8 -57, i8 -56, i8 -55, i8 -54, i8 -53, i8 -52, i8 -51, i8 -50, i8 -49, i8 -48>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 192), align 16, !tbaa !7
  store <16 x i8> <i8 -48, i8 -47, i8 -46, i8 -45, i8 -44, i8 -43, i8 -42, i8 -41, i8 -40, i8 -39, i8 -38, i8 -37, i8 -36, i8 -35, i8 -34, i8 -33>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 208), align 16, !tbaa !7
  store <16 x i8> <i8 -47, i8 -46, i8 -45, i8 -44, i8 -43, i8 -42, i8 -41, i8 -40, i8 -39, i8 -38, i8 -37, i8 -36, i8 -35, i8 -34, i8 -33, i8 -32>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 208), align 16, !tbaa !7
  store <16 x i8> <i8 -32, i8 -31, i8 -30, i8 -29, i8 -28, i8 -27, i8 -26, i8 -25, i8 -24, i8 -23, i8 -22, i8 -21, i8 -20, i8 -19, i8 -18, i8 -17>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 224), align 16, !tbaa !7
  store <16 x i8> <i8 -31, i8 -30, i8 -29, i8 -28, i8 -27, i8 -26, i8 -25, i8 -24, i8 -23, i8 -22, i8 -21, i8 -20, i8 -19, i8 -18, i8 -17, i8 -16>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 224), align 16, !tbaa !7
  store <16 x i8> <i8 -16, i8 -15, i8 -14, i8 -13, i8 -12, i8 -11, i8 -10, i8 -9, i8 -8, i8 -7, i8 -6, i8 -5, i8 -4, i8 -3, i8 -2, i8 -1>, ptr getelementptr inbounds nuw (i8, ptr @v2, i64 240), align 16, !tbaa !7
  store <16 x i8> <i8 -15, i8 -14, i8 -13, i8 -12, i8 -11, i8 -10, i8 -9, i8 -8, i8 -7, i8 -6, i8 -5, i8 -4, i8 -3, i8 -2, i8 -1, i8 0>, ptr getelementptr inbounds nuw (i8, ptr @v3, i64 240), align 16, !tbaa !7
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 32
  store i16 113, ptr %4, align 16, !tbaa !10
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 34
  store i16 0, ptr %5, align 2, !tbaa !10
  store <8 x i16> <i16 97, i16 98, i16 99, i16 100, i16 101, i16 1638, i16 1383, i16 104>, ptr %2, align 16, !tbaa !10
  store <8 x i16> <i16 105, i16 106, i16 107, i16 8300, i16 109, i16 110, i16 111, i16 1392>, ptr %3, align 16, !tbaa !10
  tail call void @foo()
  %6 = call i32 @bar(i32 noundef 17, ptr noundef nonnull %2, ptr noundef nonnull %1)
  %7 = call i32 @bcmp(ptr noundef nonnull dereferenceable(18) %1, ptr noundef nonnull dereferenceable(18) @.str, i64 18)
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %10, label %9

9:                                                ; preds = %0
  tail call void @abort() #6
  unreachable

10:                                               ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #4

attributes #0 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #5 = { nounwind }
attributes #6 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 431}
!7 = !{!8, !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"short", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
