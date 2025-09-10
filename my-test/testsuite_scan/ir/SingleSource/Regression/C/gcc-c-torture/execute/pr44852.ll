; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr44852.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr44852.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@__const.main.s = private unnamed_addr constant [7 x i8] c"999999\00", align 1
@.str = private unnamed_addr constant [7 x i8] c"199999\00", align 1

; Function Attrs: noinline nounwind uwtable
define dso_local noundef ptr @sf(ptr noundef captures(address, ret: address, provenance) %0, ptr noundef readnone captures(address) %1) local_unnamed_addr #0 {
  %3 = ptrtoint ptr %0 to i64
  %4 = ptrtoint ptr %1 to i64
  tail call void asm sideeffect "", ""() #6, !srcloc !6
  %5 = add i64 %4, 1
  %6 = sub i64 %5, %3
  %7 = getelementptr i8, ptr %0, i64 %6
  %8 = sub i64 %4, %3
  %9 = getelementptr i8, ptr %0, i64 %8
  %10 = sub i64 %3, %4
  %11 = and i64 %10, 3
  %12 = icmp eq i64 %11, 0
  br i1 %12, label %22, label %13

13:                                               ; preds = %2, %19
  %14 = phi ptr [ %16, %19 ], [ %0, %2 ]
  %15 = phi i64 [ %20, %19 ], [ 0, %2 ]
  %16 = getelementptr inbounds i8, ptr %14, i64 -1
  %17 = load i8, ptr %16, align 1, !tbaa !7
  %18 = icmp eq i8 %17, 57
  br i1 %18, label %19, label %56

19:                                               ; preds = %13
  %20 = add i64 %15, 1
  %21 = icmp eq i64 %20, %11
  br i1 %21, label %22, label %13, !llvm.loop !10

22:                                               ; preds = %19, %2
  %23 = phi ptr [ %0, %2 ], [ %16, %19 ]
  %24 = sub i64 %4, %3
  %25 = icmp ugt i64 %24, -4
  br i1 %25, label %45, label %26

26:                                               ; preds = %22, %43
  %27 = phi ptr [ %40, %43 ], [ %23, %22 ]
  %28 = getelementptr inbounds i8, ptr %27, i64 -1
  %29 = load i8, ptr %28, align 1, !tbaa !7
  %30 = icmp eq i8 %29, 57
  br i1 %30, label %31, label %54

31:                                               ; preds = %26
  %32 = getelementptr inbounds i8, ptr %27, i64 -2
  %33 = load i8, ptr %32, align 1, !tbaa !7
  %34 = icmp eq i8 %33, 57
  br i1 %34, label %35, label %51

35:                                               ; preds = %31
  %36 = getelementptr inbounds i8, ptr %27, i64 -3
  %37 = load i8, ptr %36, align 1, !tbaa !7
  %38 = icmp eq i8 %37, 57
  br i1 %38, label %39, label %48

39:                                               ; preds = %35
  %40 = getelementptr inbounds i8, ptr %27, i64 -4
  %41 = load i8, ptr %40, align 1, !tbaa !7
  %42 = icmp eq i8 %41, 57
  br i1 %42, label %43, label %46

43:                                               ; preds = %39
  %44 = icmp eq ptr %40, %1
  br i1 %44, label %45, label %26, !llvm.loop !12

45:                                               ; preds = %43, %22
  store i8 48, ptr %9, align 1, !tbaa !7
  br label %61

46:                                               ; preds = %39
  %47 = getelementptr inbounds i8, ptr %27, i64 -3
  br label %56

48:                                               ; preds = %35
  %49 = getelementptr inbounds i8, ptr %27, i64 -2
  %50 = getelementptr inbounds i8, ptr %27, i64 -3
  br label %56

51:                                               ; preds = %31
  %52 = getelementptr inbounds i8, ptr %27, i64 -1
  %53 = getelementptr inbounds i8, ptr %27, i64 -2
  br label %56

54:                                               ; preds = %26
  %55 = getelementptr inbounds i8, ptr %27, i64 -1
  br label %56

56:                                               ; preds = %13, %46, %48, %51, %54
  %57 = phi ptr [ %47, %46 ], [ %49, %48 ], [ %52, %51 ], [ %27, %54 ], [ %14, %13 ]
  %58 = phi ptr [ %40, %46 ], [ %50, %48 ], [ %53, %51 ], [ %55, %54 ], [ %16, %13 ]
  %59 = phi i8 [ %41, %46 ], [ %37, %48 ], [ %33, %51 ], [ %29, %54 ], [ %17, %13 ]
  %60 = add i8 %59, 1
  br label %61

61:                                               ; preds = %56, %45
  %62 = phi i8 [ %60, %56 ], [ 49, %45 ]
  %63 = phi ptr [ %57, %56 ], [ %7, %45 ]
  %64 = phi ptr [ %58, %56 ], [ %9, %45 ]
  store i8 %62, ptr %64, align 1, !tbaa !7
  ret ptr %63
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = alloca [7 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %1, ptr noundef nonnull align 1 dereferenceable(7) @__const.main.s, i64 7, i1 false)
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 2
  %3 = call ptr @sf(ptr noundef nonnull %2, ptr noundef nonnull %1)
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %5 = icmp eq ptr %3, %4
  br i1 %5, label %6, label %9

6:                                                ; preds = %0
  %7 = call i32 @bcmp(ptr noundef nonnull dereferenceable(7) %1, ptr noundef nonnull dereferenceable(7) @.str, i64 7)
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %10, label %9

9:                                                ; preds = %6, %0
  call void @abort() #7
  unreachable

10:                                               ; preds = %6
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #5

attributes #0 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 70}
!7 = !{!8, !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.unroll.disable"}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
