; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr27260.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr27260.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@buf = dso_local local_unnamed_addr global [65 x i8] zeroinitializer, align 16

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ne i32 %0, 2
  %3 = zext i1 %2 to i8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(64) @buf, i8 %3, i64 64, i1 false)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  store i8 2, ptr getelementptr inbounds nuw (i8, ptr @buf, i64 64), align 16, !tbaa !6
  %1 = load <16 x i8>, ptr @buf, align 16, !tbaa !6
  %2 = freeze <16 x i8> %1
  %3 = icmp ne <16 x i8> %2, zeroinitializer
  %4 = bitcast <16 x i1> %3 to i16
  %5 = icmp eq i16 %4, 0
  br i1 %5, label %6, label %24, !llvm.loop !9

6:                                                ; preds = %0
  %7 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @buf, i64 16), align 16, !tbaa !6
  %8 = freeze <16 x i8> %7
  %9 = icmp ne <16 x i8> %8, zeroinitializer
  %10 = bitcast <16 x i1> %9 to i16
  %11 = icmp eq i16 %10, 0
  br i1 %11, label %12, label %24, !llvm.loop !9

12:                                               ; preds = %6
  %13 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @buf, i64 32), align 16, !tbaa !6
  %14 = freeze <16 x i8> %13
  %15 = icmp ne <16 x i8> %14, zeroinitializer
  %16 = bitcast <16 x i1> %15 to i16
  %17 = icmp eq i16 %16, 0
  br i1 %17, label %18, label %24, !llvm.loop !9

18:                                               ; preds = %12
  %19 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @buf, i64 48), align 16, !tbaa !6
  %20 = freeze <16 x i8> %19
  %21 = icmp ne <16 x i8> %20, zeroinitializer
  %22 = bitcast <16 x i1> %21 to i16
  %23 = icmp ne i16 %22, 0
  br i1 %23, label %24, label %25

24:                                               ; preds = %0, %6, %12, %18
  tail call void @abort() #4
  unreachable

25:                                               ; preds = %18
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(64) @buf, i8 1, i64 64, i1 false)
  %26 = load <16 x i8>, ptr @buf, align 16, !tbaa !6
  %27 = freeze <16 x i8> %26
  %28 = icmp ne <16 x i8> %27, splat (i8 1)
  %29 = bitcast <16 x i1> %28 to i16
  %30 = icmp eq i16 %29, 0
  br i1 %30, label %31, label %49, !llvm.loop !13

31:                                               ; preds = %25
  %32 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @buf, i64 16), align 16, !tbaa !6
  %33 = freeze <16 x i8> %32
  %34 = icmp ne <16 x i8> %33, splat (i8 1)
  %35 = bitcast <16 x i1> %34 to i16
  %36 = icmp eq i16 %35, 0
  br i1 %36, label %37, label %49, !llvm.loop !13

37:                                               ; preds = %31
  %38 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @buf, i64 32), align 16, !tbaa !6
  %39 = freeze <16 x i8> %38
  %40 = icmp ne <16 x i8> %39, splat (i8 1)
  %41 = bitcast <16 x i1> %40 to i16
  %42 = icmp eq i16 %41, 0
  br i1 %42, label %43, label %49, !llvm.loop !13

43:                                               ; preds = %37
  %44 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @buf, i64 48), align 16, !tbaa !6
  %45 = freeze <16 x i8> %44
  %46 = icmp ne <16 x i8> %45, splat (i8 1)
  %47 = bitcast <16 x i1> %46 to i16
  %48 = icmp ne i16 %47, 0
  br i1 %48, label %49, label %50

49:                                               ; preds = %25, %31, %37, %43
  tail call void @abort() #4
  unreachable

50:                                               ; preds = %43
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(64) @buf, i8 0, i64 64, i1 false)
  %51 = load <16 x i8>, ptr @buf, align 16, !tbaa !6
  %52 = freeze <16 x i8> %51
  %53 = icmp ne <16 x i8> %52, zeroinitializer
  %54 = bitcast <16 x i1> %53 to i16
  %55 = icmp eq i16 %54, 0
  br i1 %55, label %56, label %74, !llvm.loop !14

56:                                               ; preds = %50
  %57 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @buf, i64 16), align 16, !tbaa !6
  %58 = freeze <16 x i8> %57
  %59 = icmp ne <16 x i8> %58, zeroinitializer
  %60 = bitcast <16 x i1> %59 to i16
  %61 = icmp eq i16 %60, 0
  br i1 %61, label %62, label %74, !llvm.loop !14

62:                                               ; preds = %56
  %63 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @buf, i64 32), align 16, !tbaa !6
  %64 = freeze <16 x i8> %63
  %65 = icmp ne <16 x i8> %64, zeroinitializer
  %66 = bitcast <16 x i1> %65 to i16
  %67 = icmp eq i16 %66, 0
  br i1 %67, label %68, label %74, !llvm.loop !14

68:                                               ; preds = %62
  %69 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @buf, i64 48), align 16, !tbaa !6
  %70 = freeze <16 x i8> %69
  %71 = icmp ne <16 x i8> %70, zeroinitializer
  %72 = bitcast <16 x i1> %71 to i16
  %73 = icmp ne i16 %72, 0
  br i1 %73, label %74, label %75

74:                                               ; preds = %50, %56, %62, %68
  tail call void @abort() #4
  unreachable

75:                                               ; preds = %68
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11, !12}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.isvectorized", i32 1}
!12 = !{!"llvm.loop.unroll.runtime.disable"}
!13 = distinct !{!13, !10, !11, !12}
!14 = distinct !{!14, !10, !11, !12}
