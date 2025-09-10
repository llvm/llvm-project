; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20020118-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20020118-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@q = dso_local local_unnamed_addr global ptr null, align 8
@n = dso_local global i32 0, align 4

; Function Attrs: nofree norecurse noreturn nounwind memory(readwrite, argmem: read) uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  %1 = load ptr, ptr @q, align 8, !tbaa !6
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 2
  br label %3

3:                                                ; preds = %3, %0
  %4 = load i8, ptr %2, align 1, !tbaa !11
  %5 = sext i8 %4 to i32
  store volatile i32 %5, ptr @n, align 4, !tbaa !12
  %6 = load i8, ptr %2, align 1, !tbaa !11
  %7 = sext i8 %6 to i32
  store volatile i32 %7, ptr @n, align 4, !tbaa !12
  %8 = load i8, ptr %2, align 1, !tbaa !11
  %9 = sext i8 %8 to i32
  store volatile i32 %9, ptr @n, align 4, !tbaa !12
  %10 = load i8, ptr %2, align 1, !tbaa !11
  %11 = sext i8 %10 to i32
  store volatile i32 %11, ptr @n, align 4, !tbaa !12
  %12 = load i8, ptr %2, align 1, !tbaa !11
  %13 = sext i8 %12 to i32
  store volatile i32 %13, ptr @n, align 4, !tbaa !12
  %14 = load i8, ptr %2, align 1, !tbaa !11
  %15 = sext i8 %14 to i32
  store volatile i32 %15, ptr @n, align 4, !tbaa !12
  %16 = load i8, ptr %2, align 1, !tbaa !11
  %17 = sext i8 %16 to i32
  store volatile i32 %17, ptr @n, align 4, !tbaa !12
  %18 = load i8, ptr %2, align 1, !tbaa !11
  %19 = sext i8 %18 to i32
  store volatile i32 %19, ptr @n, align 4, !tbaa !12
  %20 = load i8, ptr %2, align 1, !tbaa !11
  %21 = sext i8 %20 to i32
  store volatile i32 %21, ptr @n, align 4, !tbaa !12
  %22 = load i8, ptr %2, align 1, !tbaa !11
  %23 = sext i8 %22 to i32
  store volatile i32 %23, ptr @n, align 4, !tbaa !12
  %24 = load i8, ptr %2, align 1, !tbaa !11
  %25 = sext i8 %24 to i32
  store volatile i32 %25, ptr @n, align 4, !tbaa !12
  br label %3
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  tail call void @exit(i32 noundef 0) #3
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #2

attributes #0 = { nofree norecurse noreturn nounwind memory(readwrite, argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!9, !9, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"int", !9, i64 0}
