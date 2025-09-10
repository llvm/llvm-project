; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/static.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/static.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@test.x = internal unnamed_addr global i32 0, align 4
@str.1 = private unnamed_addr constant [6 x i8] c"error\00", align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable
define dso_local noundef i32 @test(i32 noundef returned %0, ptr noundef writeonly captures(none) initializes((0, 4)) %1) local_unnamed_addr #0 {
  %3 = load i32, ptr @test.x, align 4, !tbaa !6
  store i32 %3, ptr %1, align 4, !tbaa !6
  store i32 %0, ptr @test.x, align 4, !tbaa !6
  ret i32 %0
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i32, ptr @test.x, align 4, !tbaa !6
  store i32 1, ptr @test.x, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %4

3:                                                ; preds = %0
  store i32 -1, ptr @test.x, align 4, !tbaa !6
  br label %10

4:                                                ; preds = %0
  %5 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.1)
  %6 = load i32, ptr @test.x, align 4, !tbaa !6
  store i32 -1, ptr @test.x, align 4, !tbaa !6
  %7 = icmp eq i32 %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %4
  %9 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.1)
  br label %10

10:                                               ; preds = %3, %4, %8
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind }

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
