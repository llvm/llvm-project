; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20010129-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20010129-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@baz1.l = internal unnamed_addr global i64 0, align 8
@bar = dso_local local_unnamed_addr global ptr null, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local i64 @baz1(ptr noundef readnone captures(none) %0) local_unnamed_addr #0 {
  %2 = load i64, ptr @baz1.l, align 8, !tbaa !6
  %3 = add nsw i64 %2, 1
  store i64 %3, ptr @baz1.l, align 8, !tbaa !6
  ret i64 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @baz2(ptr noundef readnone captures(none) %0) local_unnamed_addr #1 {
  ret i32 0
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @baz3(i32 noundef %0) local_unnamed_addr #2 {
  %2 = icmp eq i32 %0, 0
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void @abort() #8
  unreachable

4:                                                ; preds = %1
  ret i32 1
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @foo(ptr noundef readnone captures(none) %0, i64 noundef %1, i32 noundef %2) local_unnamed_addr #2 {
  %4 = load i64, ptr @baz1.l, align 8, !tbaa !6
  %5 = add i64 %4, 1
  %6 = icmp slt i64 %4, %1
  br i1 %6, label %7, label %38

7:                                                ; preds = %3
  %8 = and i32 %2, 16
  %9 = icmp eq i32 %8, 0
  %10 = and i32 %2, 16384
  %11 = icmp eq i32 %10, 0
  %12 = and i32 %2, 2
  %13 = icmp eq i32 %12, 0
  %14 = and i32 %2, 128
  %15 = icmp eq i32 %14, 0
  %16 = load ptr, ptr @bar, align 8
  %17 = freeze ptr %16
  %18 = icmp eq ptr %17, null
  %19 = or i1 %15, %18
  br i1 %9, label %20, label %22

20:                                               ; preds = %7
  %21 = add i64 %1, 1
  br label %38

22:                                               ; preds = %7
  %23 = and i32 %2, 13832
  %24 = icmp eq i32 %23, 0
  br i1 %24, label %25, label %35

25:                                               ; preds = %22
  br i1 %13, label %26, label %29

26:                                               ; preds = %25
  br i1 %19, label %27, label %37

27:                                               ; preds = %26
  %28 = add i64 %1, 1
  br label %38

29:                                               ; preds = %25
  br i1 %11, label %30, label %32

30:                                               ; preds = %29
  %31 = add i64 %1, 1
  br label %38

32:                                               ; preds = %29
  br i1 %19, label %33, label %37

33:                                               ; preds = %32
  %34 = add i64 %1, 1
  br label %38

35:                                               ; preds = %22
  %36 = add i64 %1, 1
  br label %38

37:                                               ; preds = %32, %26
  store i64 %5, ptr @baz1.l, align 8, !tbaa !6
  tail call void @abort() #8
  unreachable

38:                                               ; preds = %20, %35, %33, %30, %27, %3
  %39 = phi i64 [ %5, %3 ], [ %28, %27 ], [ %31, %30 ], [ %34, %33 ], [ %36, %35 ], [ %21, %20 ]
  store i64 %39, ptr @baz1.l, align 8, !tbaa !6
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  %1 = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #9
  store ptr null, ptr %1, align 8, !tbaa !10
  store ptr %1, ptr @bar, align 8, !tbaa !12
  %2 = load i64, ptr @baz1.l, align 8, !tbaa !6
  %3 = call i64 @llvm.smax.i64(i64 %2, i64 1)
  %4 = add nuw i64 %3, 1
  store i64 %4, ptr @baz1.l, align 8, !tbaa !6
  call void @exit(i32 noundef 0) #8
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #6

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #7

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { noreturn nounwind }
attributes #9 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"any pointer", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"any p2 pointer", !11, i64 0}
