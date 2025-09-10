; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr88714.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr88714.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.T = type { ptr, ptr, ptr }

@t = dso_local local_unnamed_addr global ptr null, align 8
@o = dso_local local_unnamed_addr global ptr null, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @bar(ptr noundef readnone captures(address) %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  switch i32 %3, label %20 [
    i32 -1, label %5
    i32 0, label %11
  ]

5:                                                ; preds = %4
  %6 = icmp ne ptr %0, null
  %7 = or i32 %2, %1
  %8 = icmp ne i32 %7, 0
  %9 = or i1 %6, %8
  br i1 %9, label %10, label %21

10:                                               ; preds = %5
  tail call void @abort() #4
  unreachable

11:                                               ; preds = %4
  %12 = load ptr, ptr @t, align 8, !tbaa !6
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %14 = load ptr, ptr %13, align 8, !tbaa !11
  %15 = icmp ne ptr %0, %14
  %16 = icmp ne i32 %1, 0
  %17 = or i1 %16, %15
  %18 = icmp ne i32 %2, 12
  %19 = or i1 %18, %17
  br i1 %19, label %20, label %21

20:                                               ; preds = %4, %11
  tail call void @abort() #4
  unreachable

21:                                               ; preds = %11, %5
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(ptr noundef writeonly captures(none) initializes((16, 24)) %0, ptr noundef captures(none) %1, ptr noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = load ptr, ptr @o, align 8, !tbaa !14
  store i32 %3, ptr %5, align 4, !tbaa !15
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %2, ptr %6, align 8, !tbaa !17
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %8 = load ptr, ptr %7, align 8, !tbaa !17
  %9 = icmp eq ptr %8, null
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %11 = load i32, ptr %10, align 8, !tbaa !19
  br i1 %9, label %15, label %12

12:                                               ; preds = %4
  %13 = load i32, ptr %8, align 4, !tbaa !15
  %14 = add nsw i32 %13, %11
  store i32 %14, ptr %10, align 8, !tbaa !19
  br label %15

15:                                               ; preds = %4, %12
  %16 = phi i32 [ %14, %12 ], [ %11, %4 ]
  %17 = icmp eq i32 %16, 12
  br i1 %17, label %19, label %18

18:                                               ; preds = %15
  tail call void @abort() #4
  unreachable

19:                                               ; preds = %15
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca [4 x i32], align 4
  %2 = alloca %struct.T, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store ptr %3, ptr @o, align 8, !tbaa !14
  store ptr %2, ptr @t, align 8, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { noreturn nounwind }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 _ZTS1T", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !13, i64 16}
!12 = !{!"T", !13, i64 0, !13, i64 8, !13, i64 16}
!13 = !{!"p1 int", !8, i64 0}
!14 = !{!13, !13, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !9, i64 0}
!17 = !{!18, !13, i64 16}
!18 = !{!"S", !16, i64 0, !16, i64 4, !16, i64 8, !13, i64 16}
!19 = !{!18, !16, i64 8}
