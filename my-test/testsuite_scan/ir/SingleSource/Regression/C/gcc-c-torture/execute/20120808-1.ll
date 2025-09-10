; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20120808-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20120808-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@d = dso_local global [32 x i8] zeroinitializer, align 1
@i = dso_local global i32 0, align 4
@cp = dso_local global ptr null, align 8

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load volatile i32, ptr @i, align 4, !tbaa !6
  %2 = sext i32 %1 to i64
  %3 = getelementptr inbounds i8, ptr @d, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 1
  store volatile ptr %4, ptr @cp, align 8, !tbaa !10
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 2
  %6 = load i8, ptr %5, align 1, !tbaa !13
  store volatile ptr %5, ptr @cp, align 8, !tbaa !10
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 3
  %8 = load i8, ptr %7, align 1, !tbaa !13
  store volatile ptr %7, ptr @cp, align 8, !tbaa !10
  %9 = getelementptr inbounds nuw i8, ptr %3, i64 4
  store volatile ptr %9, ptr @cp, align 8, !tbaa !10
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 5
  store volatile ptr %10, ptr @cp, align 8, !tbaa !10
  %11 = getelementptr inbounds nuw i8, ptr %3, i64 6
  store volatile ptr %11, ptr @cp, align 8, !tbaa !10
  %12 = getelementptr inbounds nuw i8, ptr %3, i64 7
  store volatile ptr %12, ptr @cp, align 8, !tbaa !10
  %13 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store volatile ptr %13, ptr @cp, align 8, !tbaa !10
  %14 = getelementptr inbounds nuw i8, ptr %3, i64 9
  store volatile ptr %14, ptr @cp, align 8, !tbaa !10
  %15 = getelementptr inbounds nuw i8, ptr %3, i64 10
  store volatile ptr %15, ptr @cp, align 8, !tbaa !10
  %16 = getelementptr inbounds nuw i8, ptr %3, i64 11
  store volatile ptr %16, ptr @cp, align 8, !tbaa !10
  %17 = getelementptr inbounds nuw i8, ptr %3, i64 12
  store volatile ptr %17, ptr @cp, align 8, !tbaa !10
  %18 = getelementptr inbounds nuw i8, ptr %3, i64 13
  store volatile ptr %18, ptr @cp, align 8, !tbaa !10
  %19 = getelementptr inbounds nuw i8, ptr %3, i64 14
  store volatile ptr %19, ptr @cp, align 8, !tbaa !10
  %20 = getelementptr inbounds nuw i8, ptr %3, i64 15
  store volatile ptr %20, ptr @cp, align 8, !tbaa !10
  %21 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store volatile ptr %21, ptr @cp, align 8, !tbaa !10
  %22 = getelementptr inbounds nuw i8, ptr %3, i64 17
  store volatile ptr %22, ptr @cp, align 8, !tbaa !10
  %23 = getelementptr inbounds nuw i8, ptr %3, i64 18
  store volatile ptr %23, ptr @cp, align 8, !tbaa !10
  %24 = getelementptr inbounds nuw i8, ptr %3, i64 19
  store volatile ptr %24, ptr @cp, align 8, !tbaa !10
  %25 = getelementptr inbounds nuw i8, ptr %3, i64 20
  store volatile ptr %25, ptr @cp, align 8, !tbaa !10
  %26 = getelementptr inbounds nuw i8, ptr %3, i64 21
  store volatile ptr %26, ptr @cp, align 8, !tbaa !10
  %27 = getelementptr inbounds nuw i8, ptr %3, i64 22
  store volatile ptr %27, ptr @cp, align 8, !tbaa !10
  %28 = getelementptr inbounds nuw i8, ptr %3, i64 23
  store volatile ptr %28, ptr @cp, align 8, !tbaa !10
  %29 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store volatile ptr %29, ptr @cp, align 8, !tbaa !10
  %30 = getelementptr inbounds nuw i8, ptr %3, i64 25
  store volatile ptr %30, ptr @cp, align 8, !tbaa !10
  %31 = getelementptr inbounds nuw i8, ptr %3, i64 26
  %32 = load i8, ptr %31, align 1, !tbaa !13
  store volatile ptr %31, ptr @cp, align 8, !tbaa !10
  %33 = getelementptr inbounds nuw i8, ptr %3, i64 27
  store volatile ptr %33, ptr @cp, align 8, !tbaa !10
  %34 = getelementptr inbounds nuw i8, ptr %3, i64 28
  store volatile ptr %34, ptr @cp, align 8, !tbaa !10
  %35 = getelementptr inbounds nuw i8, ptr %3, i64 29
  store volatile ptr %35, ptr @cp, align 8, !tbaa !10
  %36 = getelementptr inbounds nuw i8, ptr %3, i64 30
  store volatile ptr %36, ptr @cp, align 8, !tbaa !10
  %37 = and i8 %6, 2
  %38 = icmp ne i8 %37, 0
  %39 = and i8 %8, 4
  %40 = icmp ne i8 %39, 0
  %41 = select i1 %38, i1 true, i1 %40
  %42 = and i8 %32, 1
  %43 = icmp ne i8 %42, 0
  %44 = select i1 %41, i1 true, i1 %43
  br i1 %44, label %48, label %45

45:                                               ; preds = %0
  %46 = load volatile ptr, ptr @cp, align 8, !tbaa !10
  %47 = icmp eq ptr %46, getelementptr inbounds nuw (i8, ptr @d, i64 30)
  br i1 %47, label %49, label %48

48:                                               ; preds = %45, %0
  tail call void @abort() #3
  unreachable

49:                                               ; preds = %45
  tail call void @exit(i32 noundef 0) #3
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #2

attributes #0 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 omnipotent char", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
!13 = !{!8, !8, i64 0}
