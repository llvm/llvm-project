; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20170401-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20170401-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.source = type { i32, i32 }

@flag = internal unnamed_addr global i1 false, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca %struct.source, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  store <2 x i32> zeroinitializer, ptr %1, align 8, !tbaa !6
  store i1 false, ptr @flag, align 4
  %2 = call fastcc i16 @next(ptr noundef %1)
  %3 = icmp eq i16 %2, -1
  br i1 %3, label %4, label %50

4:                                                ; preds = %0
  %5 = call fastcc i16 @next(ptr noundef %1)
  %6 = icmp eq i16 %5, 0
  br i1 %6, label %7, label %50

7:                                                ; preds = %4
  %8 = call fastcc i16 @next(ptr noundef %1)
  %9 = icmp eq i16 %8, 0
  br i1 %9, label %10, label %50

10:                                               ; preds = %7
  %11 = call fastcc i16 @next(ptr noundef %1)
  %12 = icmp eq i16 %11, 0
  br i1 %12, label %13, label %50

13:                                               ; preds = %10
  %14 = call fastcc i16 @next(ptr noundef %1)
  %15 = icmp eq i16 %14, 0
  br i1 %15, label %16, label %50

16:                                               ; preds = %13
  %17 = call fastcc i16 @next(ptr noundef %1)
  %18 = icmp eq i16 %17, 0
  br i1 %18, label %19, label %50

19:                                               ; preds = %16
  %20 = call fastcc i16 @next(ptr noundef %1)
  %21 = icmp eq i16 %20, 0
  br i1 %21, label %22, label %50

22:                                               ; preds = %19
  %23 = call fastcc i16 @next(ptr noundef %1)
  %24 = icmp eq i16 %23, 0
  br i1 %24, label %25, label %50

25:                                               ; preds = %22
  %26 = call fastcc i16 @next(ptr noundef %1)
  %27 = icmp eq i16 %26, 0
  br i1 %27, label %28, label %50

28:                                               ; preds = %25
  %29 = call fastcc i16 @next(ptr noundef %1)
  %30 = icmp eq i16 %29, 0
  br i1 %30, label %31, label %50

31:                                               ; preds = %28
  %32 = call fastcc i16 @next(ptr noundef %1)
  %33 = icmp eq i16 %32, 0
  br i1 %33, label %34, label %50

34:                                               ; preds = %31
  %35 = call fastcc i16 @next(ptr noundef %1)
  %36 = icmp eq i16 %35, 0
  br i1 %36, label %37, label %50

37:                                               ; preds = %34
  %38 = call fastcc i16 @next(ptr noundef %1)
  %39 = icmp eq i16 %38, 0
  br i1 %39, label %40, label %50

40:                                               ; preds = %37
  %41 = call fastcc i16 @next(ptr noundef %1)
  %42 = icmp eq i16 %41, 0
  br i1 %42, label %43, label %50

43:                                               ; preds = %40
  %44 = call fastcc i16 @next(ptr noundef %1)
  %45 = icmp eq i16 %44, 0
  br i1 %45, label %46, label %50

46:                                               ; preds = %43
  %47 = call fastcc i16 @next(ptr noundef %1)
  %48 = icmp eq i16 %47, 0
  br i1 %48, label %49, label %50

49:                                               ; preds = %46
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0

50:                                               ; preds = %46, %43, %40, %37, %34, %31, %28, %25, %22, %19, %16, %13, %10, %7, %4, %0
  tail call void @abort() #6
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable
define internal fastcc range(i16 -1, 1) i16 @next(ptr noundef nonnull captures(none) %0) unnamed_addr #2 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %3 = load i32, ptr %0, align 4, !tbaa !10
  %4 = load i32, ptr %2, align 4, !tbaa !12
  %5 = icmp slt i32 %3, %4
  br i1 %5, label %13, label %6

6:                                                ; preds = %1
  %7 = load i1, ptr @flag, align 4
  br i1 %7, label %8, label %12

8:                                                ; preds = %6
  store i1 false, ptr @flag, align 4
  tail call fastcc void @fetch(ptr noundef %0)
  %9 = load i32, ptr %0, align 4, !tbaa !10
  %10 = load i32, ptr %2, align 4, !tbaa !12
  %11 = icmp slt i32 %9, %10
  br i1 %11, label %13, label %12

12:                                               ; preds = %8, %6
  store i1 true, ptr @flag, align 4
  br label %13

13:                                               ; preds = %8, %1, %12
  %14 = phi i16 [ -1, %12 ], [ 0, %1 ], [ 0, %8 ]
  ret i16 %14
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal fastcc void @fetch(ptr noundef nonnull writeonly captures(none) initializes((4, 8)) %0) unnamed_addr #4 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 128, ptr %2, align 4, !tbaa !12
  ret void
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !7, i64 0}
!11 = !{!"source", !7, i64 0, !7, i64 4}
!12 = !{!11, !7, i64 4}
