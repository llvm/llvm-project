; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strpbrk.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strpbrk.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree nounwind uwtable
define dso_local void @fn(ptr noundef readonly %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds i8, ptr %1, i64 -8
  %4 = load ptr, ptr %3, align 8, !tbaa !6
  %5 = tail call ptr @strchr(ptr noundef nonnull dereferenceable(1) %4, i32 104)
  %6 = icmp eq ptr %5, %0
  br i1 %6, label %8, label %7

7:                                                ; preds = %2
  tail call void @abort() #4
  unreachable

8:                                                ; preds = %2
  %9 = load i8, ptr %0, align 1
  %10 = icmp eq i8 %9, 104
  br i1 %10, label %12, label %11

11:                                               ; preds = %8
  tail call void @abort() #4
  unreachable

12:                                               ; preds = %8
  %13 = tail call ptr @strchr(ptr noundef nonnull dereferenceable(1) %0, i32 119)
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 6
  %15 = icmp eq ptr %13, %14
  br i1 %15, label %17, label %16

16:                                               ; preds = %12
  tail call void @abort() #4
  unreachable

17:                                               ; preds = %12
  %18 = tail call ptr @strchr(ptr noundef nonnull dereferenceable(1) %14, i32 111)
  %19 = getelementptr inbounds nuw i8, ptr %0, i64 7
  %20 = icmp eq ptr %18, %19
  br i1 %20, label %22, label %21

21:                                               ; preds = %17
  tail call void @abort() #4
  unreachable

22:                                               ; preds = %17
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @main_test() local_unnamed_addr #2 {
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare ptr @strchr(ptr, i32) local_unnamed_addr #3

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: read) }
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
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
