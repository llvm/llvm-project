; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr44468.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr44468.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.Q = type { float, %struct.S }
%struct.S = type { i32, i32 }

@s = dso_local global %struct.Q zeroinitializer, align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable
define dso_local i32 @test1(ptr noundef writeonly captures(none) initializes((4, 8)) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @s, i64 4), align 4, !tbaa !6
  store i32 3, ptr %2, align 4, !tbaa !13
  %3 = load i32, ptr getelementptr inbounds nuw (i8, ptr @s, i64 4), align 4, !tbaa !6
  ret i32 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable
define dso_local i32 @test2(ptr noundef writeonly captures(none) initializes((4, 8)) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @s, i64 4), align 4, !tbaa !6
  store i32 3, ptr %2, align 4, !tbaa !13
  %3 = load i32, ptr getelementptr inbounds nuw (i8, ptr @s, i64 4), align 4, !tbaa !6
  ret i32 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable
define dso_local i32 @test3(ptr noundef writeonly captures(none) initializes((4, 8)) %0) local_unnamed_addr #0 {
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @s, i64 4), align 4, !tbaa !6
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 3, ptr %2, align 4, !tbaa !13
  %3 = load i32, ptr getelementptr inbounds nuw (i8, ptr @s, i64 4), align 4, !tbaa !6
  ret i32 %3
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store <2 x i32> <i32 1, i32 2>, ptr getelementptr inbounds nuw (i8, ptr @s, i64 4), align 4, !tbaa !14
  %1 = tail call i32 @test1(ptr noundef nonnull @s)
  %2 = icmp eq i32 %1, 3
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

4:                                                ; preds = %0
  store <2 x i32> <i32 1, i32 2>, ptr getelementptr inbounds nuw (i8, ptr @s, i64 4), align 4, !tbaa !14
  %5 = tail call i32 @test2(ptr noundef nonnull @s)
  %6 = icmp eq i32 %5, 3
  br i1 %6, label %8, label %7

7:                                                ; preds = %4
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %4
  store <2 x i32> <i32 1, i32 2>, ptr getelementptr inbounds nuw (i8, ptr @s, i64 4), align 4, !tbaa !14
  %9 = tail call i32 @test3(ptr noundef nonnull @s)
  %10 = icmp eq i32 %9, 3
  br i1 %10, label %12, label %11

11:                                               ; preds = %8
  tail call void @abort() #3
  unreachable

12:                                               ; preds = %8
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !12, i64 4}
!7 = !{!"Q", !8, i64 0, !11, i64 4}
!8 = !{!"float", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"S", !12, i64 0, !12, i64 4}
!12 = !{!"int", !9, i64 0}
!13 = !{!11, !12, i64 0}
!14 = !{!12, !12, i64 0}
