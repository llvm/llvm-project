; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SignlessTypes/factor.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SignlessTypes/factor.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@plane = internal unnamed_addr global [50 x i8] zeroinitializer, align 16
@.str = private unnamed_addr constant [4 x i8] c"%d \00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fill() local_unnamed_addr #0 {
  store <16 x i8> <i8 -1, i8 1, i8 3, i8 5, i8 3, i8 9, i8 7, i8 17, i8 3, i8 5, i8 11, i8 33, i8 7, i8 65, i8 19, i8 13>, ptr @plane, align 16, !tbaa !6
  store <16 x i8> <i8 3, i8 -127, i8 7, i8 1, i8 11, i8 21, i8 35, i8 1, i8 7, i8 9, i8 67, i8 5, i8 19, i8 1, i8 15, i8 1>, ptr getelementptr inbounds nuw (i8, ptr @plane, i64 16), align 16, !tbaa !6
  store <16 x i8> <i8 3, i8 37, i8 -125, i8 25, i8 7, i8 1, i8 3, i8 69, i8 11, i8 1, i8 23, i8 1, i8 35, i8 13, i8 3, i8 1>, ptr getelementptr inbounds nuw (i8, ptr @plane, i64 32), align 16, !tbaa !6
  store <2 x i8> <i8 7, i8 17>, ptr getelementptr inbounds nuw (i8, ptr @plane, i64 48), align 16, !tbaa !6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #1 {
  store <16 x i8> <i8 -1, i8 1, i8 3, i8 5, i8 3, i8 9, i8 7, i8 17, i8 3, i8 5, i8 11, i8 33, i8 7, i8 65, i8 19, i8 13>, ptr @plane, align 16, !tbaa !6
  store <16 x i8> <i8 3, i8 -127, i8 7, i8 1, i8 11, i8 21, i8 35, i8 1, i8 7, i8 9, i8 67, i8 5, i8 19, i8 1, i8 15, i8 1>, ptr getelementptr inbounds nuw (i8, ptr @plane, i64 16), align 16, !tbaa !6
  store <16 x i8> <i8 3, i8 37, i8 -125, i8 25, i8 7, i8 1, i8 3, i8 69, i8 11, i8 1, i8 23, i8 1, i8 35, i8 13, i8 3, i8 1>, ptr getelementptr inbounds nuw (i8, ptr @plane, i64 32), align 16, !tbaa !6
  store <2 x i8> <i8 7, i8 17>, ptr getelementptr inbounds nuw (i8, ptr @plane, i64 48), align 16, !tbaa !6
  br label %3

3:                                                ; preds = %2, %12
  %4 = phi i64 [ %13, %12 ], [ 0, %2 ]
  %5 = getelementptr inbounds nuw i8, ptr @plane, i64 %4
  %6 = load i8, ptr %5, align 1, !tbaa !6
  %7 = and i8 %6, 16
  %8 = icmp eq i8 %7, 0
  br i1 %8, label %12, label %9

9:                                                ; preds = %3
  %10 = trunc nuw nsw i64 %4 to i32
  %11 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %10)
  br label %12

12:                                               ; preds = %3, %9
  %13 = add nuw nsw i64 %4, 1
  %14 = icmp eq i64 %13, 50
  br i1 %14, label %15, label %3, !llvm.loop !9

15:                                               ; preds = %12
  %16 = tail call i32 @putchar(i32 10)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #3

attributes #0 = { nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
