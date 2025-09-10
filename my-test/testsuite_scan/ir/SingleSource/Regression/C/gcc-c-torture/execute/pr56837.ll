; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr56837.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr56837.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global [1024 x { i32, i32 }] zeroinitializer, align 4

; Function Attrs: nofree noinline norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %6, %1 ]
  %3 = getelementptr inbounds nuw { i32, i32 }, ptr @a, i64 %2
  %4 = getelementptr inbounds nuw { i32, i32 }, ptr @a, i64 %2
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 32
  store <8 x i32> <i32 -1, i32 0, i32 -1, i32 0, i32 -1, i32 0, i32 -1, i32 0>, ptr %3, align 4
  store <8 x i32> <i32 -1, i32 0, i32 -1, i32 0, i32 -1, i32 0, i32 -1, i32 0>, ptr %5, align 4
  %6 = add nuw i64 %2, 8
  %7 = icmp eq i64 %6, 1024
  br i1 %7, label %8, label %1, !llvm.loop !6

8:                                                ; preds = %1
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  tail call void @foo()
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %11, %1 ]
  %3 = getelementptr inbounds nuw { i32, i32 }, ptr @a, i64 %2
  %4 = load <8 x i32>, ptr %3, align 4
  %5 = freeze <8 x i32> %4
  %6 = shufflevector <8 x i32> %5, <8 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %7 = shufflevector <8 x i32> %5, <8 x i32> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %8 = icmp ne <4 x i32> %6, splat (i32 -1)
  %9 = icmp ne <4 x i32> %7, zeroinitializer
  %10 = or <4 x i1> %8, %9
  %11 = add nuw i64 %2, 4
  %12 = bitcast <4 x i1> %10 to i4
  %13 = icmp ne i4 %12, 0
  %14 = icmp eq i64 %11, 1024
  %15 = or i1 %13, %14
  br i1 %15, label %16, label %1, !llvm.loop !10

16:                                               ; preds = %1
  br i1 %13, label %17, label %18

17:                                               ; preds = %16
  tail call void @abort() #3
  unreachable

18:                                               ; preds = %16
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { nofree noinline norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = distinct !{!6, !7, !8, !9}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!"llvm.loop.isvectorized", i32 1}
!9 = !{!"llvm.loop.unroll.runtime.disable"}
!10 = distinct !{!10, !7, !8, !9}
