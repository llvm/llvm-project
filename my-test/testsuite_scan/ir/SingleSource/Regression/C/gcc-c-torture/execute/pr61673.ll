; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr61673.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr61673.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@e = dso_local local_unnamed_addr global i8 0, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @bar(i8 noundef %0) local_unnamed_addr #0 {
  switch i8 %0, label %2 [
    i8 -121, label %3
    i8 84, label %3
  ]

2:                                                ; preds = %1
  tail call void @abort() #5
  unreachable

3:                                                ; preds = %1, %1
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @foo(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = load i8, ptr %0, align 1, !tbaa !6
  %3 = icmp slt i8 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  store i8 %2, ptr @e, align 4, !tbaa !6
  br label %5

5:                                                ; preds = %4, %1
  tail call void @bar(i8 noundef %2)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: read, inaccessiblemem: none) uwtable
define dso_local void @baz(ptr noundef readonly captures(none) %0) local_unnamed_addr #3 {
  %2 = load i8, ptr %0, align 1, !tbaa !6
  %3 = icmp slt i8 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  store i8 %2, ptr @e, align 4, !tbaa !6
  br label %5

5:                                                ; preds = %4, %1
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  %1 = alloca [2 x i8], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  store i16 -30892, ptr %1, align 4
  store i8 33, ptr @e, align 4, !tbaa !6
  call void @foo(ptr noundef nonnull %1)
  %2 = load i8, ptr @e, align 4, !tbaa !6
  %3 = icmp eq i8 %2, 33
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort() #5
  unreachable

5:                                                ; preds = %0
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 1
  call void @foo(ptr noundef nonnull %6)
  %7 = load i8, ptr @e, align 4, !tbaa !6
  %8 = icmp eq i8 %7, -121
  br i1 %8, label %10, label %9

9:                                                ; preds = %5
  tail call void @abort() #5
  unreachable

10:                                               ; preds = %5
  store i8 33, ptr @e, align 4, !tbaa !6
  call void @baz(ptr noundef nonnull %1)
  %11 = load i8, ptr @e, align 4, !tbaa !6
  %12 = icmp eq i8 %11, 33
  br i1 %12, label %14, label %13

13:                                               ; preds = %10
  tail call void @abort() #5
  unreachable

14:                                               ; preds = %10
  call void @baz(ptr noundef nonnull %6)
  %15 = load i8, ptr @e, align 4, !tbaa !6
  %16 = icmp eq i8 %15, -121
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  tail call void @abort() #5
  unreachable

18:                                               ; preds = %14
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  ret i32 0
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { noreturn nounwind }
attributes #6 = { nounwind }

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
