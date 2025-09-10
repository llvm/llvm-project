; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr38151.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr38151.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S2848 = type { i32, { i32, i32 }, [4 x i8] }
%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@s2848 = dso_local local_unnamed_addr global %struct.S2848 zeroinitializer, align 16
@fails = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn uwtable
define dso_local void @check2848va(i32 %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %4 = load i32, ptr %3, align 8
  %5 = icmp sgt i32 %4, -1
  br i1 %5, label %16, label %6

6:                                                ; preds = %1
  %7 = add nsw i32 %4, 15
  %8 = and i32 %7, -16
  %9 = add i32 %8, 16
  store i32 %9, ptr %3, align 8
  %10 = icmp slt i32 %9, 1
  br i1 %10, label %11, label %16

11:                                               ; preds = %6
  %12 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %13 = load ptr, ptr %12, align 8
  %14 = sext i32 %8 to i64
  %15 = getelementptr inbounds i8, ptr %13, i64 %14
  br label %21

16:                                               ; preds = %6, %1
  %17 = load ptr, ptr %2, align 8
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 15
  %19 = call align 16 ptr @llvm.ptrmask.p0.i64(ptr nonnull %18, i64 -16)
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 16
  store ptr %20, ptr %2, align 8
  br label %21

21:                                               ; preds = %16, %11
  %22 = phi ptr [ %15, %11 ], [ %19, %16 ]
  %23 = load i32, ptr %22, align 8, !tbaa !6
  %24 = getelementptr inbounds nuw i8, ptr %22, i64 4
  %25 = load i32, ptr %24, align 4
  %26 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %27 = load i32, ptr %26, align 8, !tbaa !10
  %28 = load i32, ptr @s2848, align 16, !tbaa !11
  %29 = icmp eq i32 %28, %23
  br i1 %29, label %33, label %30

30:                                               ; preds = %21
  %31 = load i32, ptr @fails, align 4, !tbaa !6
  %32 = add nsw i32 %31, 1
  store i32 %32, ptr @fails, align 4, !tbaa !6
  br label %33

33:                                               ; preds = %30, %21
  %34 = load i32, ptr getelementptr inbounds nuw (i8, ptr @s2848, i64 4), align 4
  %35 = load i32, ptr getelementptr inbounds nuw (i8, ptr @s2848, i64 8), align 8
  %36 = icmp ne i32 %34, %25
  %37 = icmp ne i32 %35, %27
  %38 = or i1 %36, %37
  br i1 %38, label %39, label %42

39:                                               ; preds = %33
  %40 = load i32, ptr @fails, align 4, !tbaa !6
  %41 = add nsw i32 %40, 1
  store i32 %41, ptr @fails, align 4, !tbaa !6
  br label %42

42:                                               ; preds = %39, %33
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.ptrmask.p0.i64(ptr, i64) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  store <2 x i32> <i32 -267489557, i32 723419448>, ptr @s2848, align 16
  store i32 -218144346, ptr getelementptr inbounds nuw (i8, ptr @s2848, i64 8), align 8
  %1 = load i128, ptr @s2848, align 16
  tail call void (i32, ...) @check2848va(i32 poison, i128 %1)
  %2 = load i32, ptr @fails, align 4, !tbaa !6
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort() #7
  unreachable

5:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #5

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!8, !8, i64 0}
!11 = !{!12, !7, i64 0}
!12 = !{!"S2848", !7, i64 0, !8, i64 4, !13, i64 16}
!13 = !{!""}
