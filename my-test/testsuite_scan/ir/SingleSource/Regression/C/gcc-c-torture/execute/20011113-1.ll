; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20011113-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20011113-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@t = dso_local global { i8, i8, i8, i8, [4 x i8], [4 x i64] } { i8 26, i8 0, i8 0, i8 0, [4 x i8] zeroinitializer, [4 x i64] [i64 0, i64 21, i64 22, i64 23] }, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local noalias noundef ptr @foo(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %3 = load i64, ptr %2, align 8
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %5 = load i64, ptr %4, align 8
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %7 = load i64, ptr %6, align 8
  %8 = icmp eq i64 %3, 21
  %9 = icmp eq i64 %5, 22
  %10 = select i1 %8, i1 %9, i1 false
  %11 = icmp eq i64 %7, 23
  %12 = select i1 %10, i1 %11, i1 false
  br i1 %12, label %14, label %13

13:                                               ; preds = %1
  tail call void @abort() #4
  unreachable

14:                                               ; preds = %1
  ret ptr null
}

; Function Attrs: nofree nounwind uwtable
define dso_local noalias noundef ptr @bar(ptr dead_on_return noundef readonly captures(none) %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = load i64, ptr %0, align 8, !tbaa !6
  %4 = icmp eq i64 %3, 21
  br i1 %4, label %5, label %13

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %7 = load i64, ptr %6, align 8, !tbaa !6
  %8 = icmp eq i64 %7, 22
  br i1 %8, label %9, label %13

9:                                                ; preds = %5
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %11 = load i64, ptr %10, align 8, !tbaa !6
  %12 = icmp eq i64 %11, 23
  br i1 %12, label %14, label %13

13:                                               ; preds = %9, %5, %2
  tail call void @abort() #4
  unreachable

14:                                               ; preds = %9
  ret ptr null
}

; Function Attrs: nofree nounwind uwtable
define dso_local noalias noundef ptr @baz(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %3 = load i64, ptr %2, align 8
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %5 = load i64, ptr %4, align 8
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %7 = load i64, ptr %6, align 8
  %8 = icmp eq i64 %3, 21
  %9 = icmp eq i64 %5, 22
  %10 = select i1 %8, i1 %9, i1 false
  %11 = icmp eq i64 %7, 23
  %12 = select i1 %10, i1 %11, i1 false
  br i1 %12, label %14, label %13

13:                                               ; preds = %1
  tail call void @abort() #4
  unreachable

14:                                               ; preds = %1
  ret ptr null
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = tail call ptr @baz(ptr noundef nonnull @t)
  %2 = tail call ptr @foo(ptr noundef nonnull @t)
  tail call void @exit(i32 noundef 0) #4
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #3

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
