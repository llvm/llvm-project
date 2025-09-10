; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr20527-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr20527-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@b = dso_local global [4 x i64] [i64 1, i64 5, i64 11, i64 23], align 8

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @f(ptr noundef writeonly captures(none) %0, ptr noundef readonly captures(none) %1, i64 noundef %2, i64 noundef %3) local_unnamed_addr #0 {
  %5 = icmp sgt i64 %2, %3
  br i1 %5, label %19, label %6

6:                                                ; preds = %4, %6
  %7 = phi i64 [ %15, %6 ], [ 0, %4 ]
  %8 = phi i64 [ %9, %6 ], [ %2, %4 ]
  %9 = add nsw i64 %8, 1
  %10 = getelementptr inbounds i64, ptr %1, i64 %9
  %11 = load i64, ptr %10, align 8, !tbaa !6
  %12 = getelementptr inbounds i64, ptr %1, i64 %8
  %13 = load i64, ptr %12, align 8, !tbaa !6
  %14 = sub nsw i64 %11, %13
  %15 = add nsw i64 %14, %7
  %16 = add nsw i64 %15, -1
  %17 = getelementptr inbounds i64, ptr %0, i64 %8
  store i64 %16, ptr %17, align 8, !tbaa !6
  %18 = icmp eq i64 %8, %3
  br i1 %18, label %19, label %6, !llvm.loop !10

19:                                               ; preds = %6, %4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca [3 x i64], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @f(ptr noundef nonnull %1, ptr noundef nonnull @b, i64 noundef 0, i64 noundef 2)
  %2 = load i64, ptr %1, align 8, !tbaa !6
  %3 = icmp ne i64 %2, 3
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load i64, ptr %4, align 8
  %6 = icmp ne i64 %5, 9
  %7 = select i1 %3, i1 true, i1 %6
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %9 = load i64, ptr %8, align 8
  %10 = icmp ne i64 %9, 21
  %11 = select i1 %7, i1 true, i1 %10
  br i1 %11, label %12, label %13

12:                                               ; preds = %0
  tail call void @abort() #6
  unreachable

13:                                               ; preds = %0
  tail call void @exit(i32 noundef 0) #6
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #4

attributes #0 = { nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }
attributes #6 = { noreturn nounwind }

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
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
