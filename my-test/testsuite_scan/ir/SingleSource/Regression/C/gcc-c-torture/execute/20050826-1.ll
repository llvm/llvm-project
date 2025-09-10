; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050826-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050826-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.A = type { [1 x i8], [5 x i8], [1 x i8], [2041 x i8] }

@.str = private unnamed_addr constant [8 x i8] c"\01HELLO\01\00", align 1
@a = dso_local global %struct.A zeroinitializer, align 4
@.str.1 = private unnamed_addr constant [6 x i8] c"HELLO\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @bar(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(8) %0, ptr noundef nonnull dereferenceable(8) @.str, i64 8)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 7
  br label %10

6:                                                ; preds = %1
  tail call void @abort() #5
  unreachable

7:                                                ; preds = %10
  %8 = add nuw nsw i64 %11, 1
  %9 = icmp eq i64 %8, 2041
  br i1 %9, label %16, label %10, !llvm.loop !6

10:                                               ; preds = %4, %7
  %11 = phi i64 [ 0, %4 ], [ %8, %7 ]
  %12 = getelementptr inbounds nuw i8, ptr %5, i64 %11
  %13 = load i8, ptr %12, align 1, !tbaa !8
  %14 = icmp eq i8 %13, 0
  br i1 %14, label %7, label %15

15:                                               ; preds = %10
  tail call void @abort() #5
  unreachable

16:                                               ; preds = %7
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @foo() local_unnamed_addr #0 {
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(2041) getelementptr inbounds nuw (i8, ptr @a, i64 7), i8 0, i64 2041, i1 false)
  store i8 1, ptr @a, align 4, !tbaa !8
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) getelementptr inbounds nuw (i8, ptr @a, i64 1), ptr noundef nonnull align 1 dereferenceable(6) @.str.1, i64 5, i1 false)
  store i8 1, ptr getelementptr inbounds nuw (i8, ptr @a, i64 6), align 2, !tbaa !8
  %1 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(8) @a, ptr noundef nonnull dereferenceable(8) @.str, i64 8)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %15

3:                                                ; preds = %0, %3
  %4 = phi i64 [ %9, %3 ], [ 0, %0 ]
  %5 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 7), i64 %4
  %6 = load <16 x i8>, ptr %5, align 1, !tbaa !8
  %7 = freeze <16 x i8> %6
  %8 = icmp ne <16 x i8> %7, zeroinitializer
  %9 = add nuw i64 %4, 16
  %10 = bitcast <16 x i1> %8 to i16
  %11 = icmp ne i16 %10, 0
  %12 = icmp eq i64 %9, 2032
  %13 = or i1 %11, %12
  br i1 %13, label %14, label %3, !llvm.loop !11

14:                                               ; preds = %3
  br i1 %11, label %44, label %17

15:                                               ; preds = %0
  tail call void @abort() #5
  unreachable

16:                                               ; preds = %17
  ret i32 0

17:                                               ; preds = %14
  %18 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2039), align 1, !tbaa !8
  %19 = icmp eq i8 %18, 0
  %20 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2040), align 4
  %21 = icmp eq i8 %20, 0
  %22 = select i1 %19, i1 %21, i1 false
  %23 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2041), align 1
  %24 = icmp eq i8 %23, 0
  %25 = select i1 %22, i1 %24, i1 false
  %26 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2042), align 2
  %27 = icmp eq i8 %26, 0
  %28 = select i1 %25, i1 %27, i1 false
  %29 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2043), align 1
  %30 = icmp eq i8 %29, 0
  %31 = select i1 %28, i1 %30, i1 false
  %32 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2044), align 4
  %33 = icmp eq i8 %32, 0
  %34 = select i1 %31, i1 %33, i1 false
  %35 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2045), align 1
  %36 = icmp eq i8 %35, 0
  %37 = select i1 %34, i1 %36, i1 false
  %38 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2046), align 2
  %39 = icmp eq i8 %38, 0
  %40 = select i1 %37, i1 %39, i1 false
  %41 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2047), align 1
  %42 = icmp eq i8 %41, 0
  %43 = select i1 %40, i1 %42, i1 false
  br i1 %43, label %16, label %44

44:                                               ; preds = %17, %14
  tail call void @abort() #5
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(2041) getelementptr inbounds nuw (i8, ptr @a, i64 7), i8 0, i64 2041, i1 false)
  store i8 1, ptr @a, align 4, !tbaa !8
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) getelementptr inbounds nuw (i8, ptr @a, i64 1), ptr noundef nonnull align 1 dereferenceable(6) @.str.1, i64 5, i1 false)
  store i8 1, ptr getelementptr inbounds nuw (i8, ptr @a, i64 6), align 2, !tbaa !8
  %1 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(8) @a, ptr noundef nonnull dereferenceable(8) @.str, i64 8)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %15

3:                                                ; preds = %0, %3
  %4 = phi i64 [ %9, %3 ], [ 0, %0 ]
  %5 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 7), i64 %4
  %6 = load <16 x i8>, ptr %5, align 1, !tbaa !8
  %7 = freeze <16 x i8> %6
  %8 = icmp ne <16 x i8> %7, zeroinitializer
  %9 = add nuw i64 %4, 16
  %10 = bitcast <16 x i1> %8 to i16
  %11 = icmp ne i16 %10, 0
  %12 = icmp eq i64 %9, 2032
  %13 = or i1 %11, %12
  br i1 %13, label %14, label %3, !llvm.loop !14

14:                                               ; preds = %3
  br i1 %11, label %44, label %17

15:                                               ; preds = %0
  tail call void @abort() #5
  unreachable

16:                                               ; preds = %17
  ret i32 0

17:                                               ; preds = %14
  %18 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2039), align 1, !tbaa !8
  %19 = icmp eq i8 %18, 0
  %20 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2040), align 4
  %21 = icmp eq i8 %20, 0
  %22 = select i1 %19, i1 %21, i1 false
  %23 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2041), align 1
  %24 = icmp eq i8 %23, 0
  %25 = select i1 %22, i1 %24, i1 false
  %26 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2042), align 2
  %27 = icmp eq i8 %26, 0
  %28 = select i1 %25, i1 %27, i1 false
  %29 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2043), align 1
  %30 = icmp eq i8 %29, 0
  %31 = select i1 %28, i1 %30, i1 false
  %32 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2044), align 4
  %33 = icmp eq i8 %32, 0
  %34 = select i1 %31, i1 %33, i1 false
  %35 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2045), align 1
  %36 = icmp eq i8 %35, 0
  %37 = select i1 %34, i1 %36, i1 false
  %38 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2046), align 2
  %39 = icmp eq i8 %38, 0
  %40 = select i1 %37, i1 %39, i1 false
  %41 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 2047), align 1
  %42 = icmp eq i8 %41, 0
  %43 = select i1 %40, i1 %42, i1 false
  br i1 %43, label %16, label %44

44:                                               ; preds = %17, %14
  tail call void @abort() #5
  unreachable
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #4

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!9, !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = distinct !{!11, !7, !12, !13}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !7, !12, !13}
