; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr51877.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr51877.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.A = type { i32, [32 x i8] }

@bar.n = internal unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global %struct.A zeroinitializer, align 4
@b = dso_local global %struct.A zeroinitializer, align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable
define dso_local void @bar(ptr dead_on_unwind noalias writable writeonly sret(%struct.A) align 4 captures(none) initializes((0, 36)) %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = load i32, ptr @bar.n, align 4, !tbaa !6
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr @bar.n, align 4, !tbaa !6
  store i32 %4, ptr %0, align 4, !tbaa !10
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 4
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(32) %5, i8 0, i64 32, i1 false)
  %6 = trunc i32 %1 to i8
  store i8 %6, ptr %5, align 4, !tbaa !12
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #1

; Function Attrs: noinline nounwind uwtable
define dso_local void @baz() local_unnamed_addr #2 {
  tail call void asm sideeffect "", "~{memory}"() #7, !srcloc !13
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @foo(ptr noundef writeonly captures(none) %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = alloca %struct.A, align 4
  %4 = alloca %struct.A, align 4
  %5 = icmp eq i32 %1, 6
  br i1 %5, label %6, label %7

6:                                                ; preds = %2
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #7
  call void @bar(ptr dead_on_unwind nonnull writable sret(%struct.A) align 4 %3, i32 noundef 7)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(36) @a, ptr noundef nonnull align 4 dereferenceable(36) %3, i64 36, i1 false), !tbaa.struct !14
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #7
  br label %8

7:                                                ; preds = %2
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #7
  call void @bar(ptr dead_on_unwind nonnull writable sret(%struct.A) align 4 %4, i32 noundef 7)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(36) %0, ptr noundef nonnull align 4 dereferenceable(36) %4, i64 36, i1 false), !tbaa.struct !14
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #7
  br label %8

8:                                                ; preds = %7, %6
  tail call void @baz()
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  %1 = alloca %struct.A, align 4
  %2 = alloca %struct.A, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  call void @bar(ptr dead_on_unwind nonnull writable sret(%struct.A) align 4 %1, i32 noundef 3)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(36) @a, ptr noundef nonnull align 4 dereferenceable(36) %1, i64 36, i1 false), !tbaa.struct !14
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  call void @bar(ptr dead_on_unwind nonnull writable sret(%struct.A) align 4 %2, i32 noundef 4)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(36) @b, ptr noundef nonnull align 4 dereferenceable(36) %2, i64 36, i1 false), !tbaa.struct !14
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  %3 = load i32, ptr @a, align 4, !tbaa !10
  %4 = icmp ne i32 %3, 1
  %5 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 4), align 4
  %6 = icmp ne i8 %5, 3
  %7 = select i1 %4, i1 true, i1 %6
  %8 = load i32, ptr @b, align 4
  %9 = icmp ne i32 %8, 2
  %10 = select i1 %7, i1 true, i1 %9
  %11 = load i8, ptr getelementptr inbounds nuw (i8, ptr @b, i64 4), align 4
  %12 = icmp ne i8 %11, 4
  %13 = select i1 %10, i1 true, i1 %12
  br i1 %13, label %14, label %15

14:                                               ; preds = %0
  tail call void @abort() #8
  unreachable

15:                                               ; preds = %0
  tail call void @foo(ptr noundef nonnull @b, i32 noundef 0)
  %16 = load i32, ptr @a, align 4, !tbaa !10
  %17 = icmp ne i32 %16, 1
  %18 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 4), align 4
  %19 = icmp ne i8 %18, 3
  %20 = select i1 %17, i1 true, i1 %19
  %21 = load i32, ptr @b, align 4
  %22 = icmp ne i32 %21, 3
  %23 = select i1 %20, i1 true, i1 %22
  %24 = load i8, ptr getelementptr inbounds nuw (i8, ptr @b, i64 4), align 4
  %25 = icmp ne i8 %24, 7
  %26 = select i1 %23, i1 true, i1 %25
  br i1 %26, label %27, label %28

27:                                               ; preds = %15
  tail call void @abort() #8
  unreachable

28:                                               ; preds = %15
  tail call void @foo(ptr noundef nonnull @b, i32 noundef 6)
  %29 = load i32, ptr @a, align 4, !tbaa !10
  %30 = icmp ne i32 %29, 4
  %31 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 4), align 4
  %32 = icmp ne i8 %31, 7
  %33 = select i1 %30, i1 true, i1 %32
  %34 = load i32, ptr @b, align 4
  %35 = icmp ne i32 %34, 3
  %36 = select i1 %33, i1 true, i1 %35
  %37 = load i8, ptr getelementptr inbounds nuw (i8, ptr @b, i64 4), align 4
  %38 = icmp ne i8 %37, 7
  %39 = select i1 %36, i1 true, i1 %38
  br i1 %39, label %40, label %41

40:                                               ; preds = %28
  tail call void @abort() #8
  unreachable

41:                                               ; preds = %28
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #6

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nounwind }
attributes #8 = { noreturn nounwind }

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
!10 = !{!11, !7, i64 0}
!11 = !{!"A", !7, i64 0, !8, i64 4}
!12 = !{!8, !8, i64 0}
!13 = !{i64 343}
!14 = !{i64 0, i64 4, !6, i64 4, i64 32, !12}
