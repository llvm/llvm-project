; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ffs-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ffs-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.anon = type { i32, i32 }

@ffstesttab = dso_local local_unnamed_addr global [8 x %struct.anon] [%struct.anon { i32 -2147483648, i32 32 }, %struct.anon { i32 -1515870811, i32 1 }, %struct.anon { i32 1515870810, i32 2 }, %struct.anon { i32 -889323520, i32 18 }, %struct.anon { i32 32768, i32 16 }, %struct.anon { i32 42405, i32 1 }, %struct.anon { i32 23130, i32 2 }, %struct.anon { i32 3232, i32 6 }], align 4

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @ffstesttab, align 4, !tbaa !6
  %2 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %1, i1 true)
  %3 = add nuw nsw i32 %2, 1
  %4 = icmp eq i32 %1, 0
  %5 = select i1 %4, i32 0, i32 %3
  %6 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 4), align 4, !tbaa !11
  %7 = icmp eq i32 %5, %6
  br i1 %7, label %8, label %65

8:                                                ; preds = %0
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 8), align 4, !tbaa !6
  %10 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %9, i1 true)
  %11 = add nuw nsw i32 %10, 1
  %12 = icmp eq i32 %9, 0
  %13 = select i1 %12, i32 0, i32 %11
  %14 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 12), align 4, !tbaa !11
  %15 = icmp eq i32 %13, %14
  br i1 %15, label %16, label %65

16:                                               ; preds = %8
  %17 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 16), align 4, !tbaa !6
  %18 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %17, i1 true)
  %19 = add nuw nsw i32 %18, 1
  %20 = icmp eq i32 %17, 0
  %21 = select i1 %20, i32 0, i32 %19
  %22 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 20), align 4, !tbaa !11
  %23 = icmp eq i32 %21, %22
  br i1 %23, label %24, label %65

24:                                               ; preds = %16
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 24), align 4, !tbaa !6
  %26 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %25, i1 true)
  %27 = add nuw nsw i32 %26, 1
  %28 = icmp eq i32 %25, 0
  %29 = select i1 %28, i32 0, i32 %27
  %30 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 28), align 4, !tbaa !11
  %31 = icmp eq i32 %29, %30
  br i1 %31, label %32, label %65

32:                                               ; preds = %24
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 32), align 4, !tbaa !6
  %34 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %33, i1 true)
  %35 = add nuw nsw i32 %34, 1
  %36 = icmp eq i32 %33, 0
  %37 = select i1 %36, i32 0, i32 %35
  %38 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 36), align 4, !tbaa !11
  %39 = icmp eq i32 %37, %38
  br i1 %39, label %40, label %65

40:                                               ; preds = %32
  %41 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 40), align 4, !tbaa !6
  %42 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %41, i1 true)
  %43 = add nuw nsw i32 %42, 1
  %44 = icmp eq i32 %41, 0
  %45 = select i1 %44, i32 0, i32 %43
  %46 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 44), align 4, !tbaa !11
  %47 = icmp eq i32 %45, %46
  br i1 %47, label %48, label %65

48:                                               ; preds = %40
  %49 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 48), align 4, !tbaa !6
  %50 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %49, i1 true)
  %51 = add nuw nsw i32 %50, 1
  %52 = icmp eq i32 %49, 0
  %53 = select i1 %52, i32 0, i32 %51
  %54 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 52), align 4, !tbaa !11
  %55 = icmp eq i32 %53, %54
  br i1 %55, label %56, label %65

56:                                               ; preds = %48
  %57 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 56), align 4, !tbaa !6
  %58 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %57, i1 true)
  %59 = add nuw nsw i32 %58, 1
  %60 = icmp eq i32 %57, 0
  %61 = select i1 %60, i32 0, i32 %59
  %62 = load i32, ptr getelementptr inbounds nuw (i8, ptr @ffstesttab, i64 60), align 4, !tbaa !11
  %63 = icmp eq i32 %61, %62
  br i1 %63, label %64, label %65

64:                                               ; preds = %56
  tail call void @exit(i32 noundef 0) #4
  unreachable

65:                                               ; preds = %56, %48, %40, %32, %24, %16, %8, %0
  tail call void @abort() #4
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.cttz.i32(i32, i1 immarg) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #3

attributes #0 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"", !8, i64 0, !8, i64 4}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!7, !8, i64 4}
