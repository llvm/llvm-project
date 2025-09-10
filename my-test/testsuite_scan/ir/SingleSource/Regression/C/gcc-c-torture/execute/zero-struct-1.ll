; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/zero-struct-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/zero-struct-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@y = dso_local global [3 x i8] zeroinitializer, align 1
@f = dso_local local_unnamed_addr global ptr @y, align 8
@ff = dso_local local_unnamed_addr global ptr @y, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @h() local_unnamed_addr #0 {
  %1 = load ptr, ptr @f, align 8, !tbaa !6
  %2 = load ptr, ptr @ff, align 8, !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 2
  store ptr %3, ptr @f, align 8, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 2
  store ptr %4, ptr @ff, align 8, !tbaa !6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load ptr, ptr @f, align 8, !tbaa !6
  %2 = load ptr, ptr @ff, align 8, !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 2
  store ptr %3, ptr @f, align 8, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 2
  store ptr %4, ptr @ff, align 8, !tbaa !6
  %5 = icmp eq ptr %1, @y
  br i1 %5, label %7, label %6

6:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

7:                                                ; preds = %0
  %8 = icmp eq ptr %2, @y
  br i1 %8, label %10, label %9

9:                                                ; preds = %7
  tail call void @abort() #3
  unreachable

10:                                               ; preds = %7
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
