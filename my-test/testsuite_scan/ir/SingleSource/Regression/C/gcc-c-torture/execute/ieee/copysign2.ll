; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ieee/copysign2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ieee/copysign2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@Zl = internal constant [8 x fp128] [fp128 0xL00000000000000003FFF000000000000, fp128 0xL0000000000000000BFFF000000000000, fp128 0xL0000000000000000BFFF000000000000, fp128 0xL00000000000000008000000000000000, fp128 0xL00000000000000008000000000000000, fp128 0xL00000000000000000000000000000000, fp128 0xL0000000000000000FFFF000000000000, fp128 0xL00000000000000007FFF800000000000], align 16

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @testf() local_unnamed_addr #0 {
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @test() local_unnamed_addr #0 {
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testl() local_unnamed_addr #3 {
  %1 = alloca [8 x fp128], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  store fp128 0xL00000000000000003FFF000000000000, ptr %1, align 16, !tbaa !6
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store fp128 0xL0000000000000000BFFF000000000000, ptr %2, align 16, !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 32
  store fp128 0xL0000000000000000BFFF000000000000, ptr %3, align 16, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 48
  store fp128 0xL00000000000000008000000000000000, ptr %4, align 16, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 64
  store fp128 0xL00000000000000008000000000000000, ptr %5, align 16, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 80
  store fp128 0xL00000000000000000000000000000000, ptr %6, align 16, !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 96
  store fp128 0xL0000000000000000FFFF000000000000, ptr %7, align 16, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 112
  store fp128 0xL00000000000000007FFF800000000000, ptr %8, align 16, !tbaa !6
  %9 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %1, ptr noundef nonnull dereferenceable(16) @Zl, i64 16)
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %11, label %33

11:                                               ; preds = %0
  %12 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %2, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Zl, i64 16), i64 16)
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %14, label %33

14:                                               ; preds = %11
  %15 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %3, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Zl, i64 32), i64 16)
  %16 = icmp eq i32 %15, 0
  br i1 %16, label %17, label %33

17:                                               ; preds = %14
  %18 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %4, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Zl, i64 48), i64 16)
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %20, label %33

20:                                               ; preds = %17
  %21 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %5, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Zl, i64 64), i64 16)
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %23, label %33

23:                                               ; preds = %20
  %24 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %6, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Zl, i64 80), i64 16)
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %26, label %33

26:                                               ; preds = %23
  %27 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %7, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Zl, i64 96), i64 16)
  %28 = icmp eq i32 %27, 0
  br i1 %28, label %29, label %33

29:                                               ; preds = %26
  %30 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %8, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Zl, i64 112), i64 16)
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %32, label %33

32:                                               ; preds = %29
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret void

33:                                               ; preds = %29, %26, %23, %20, %17, %14, %11, %0
  tail call void @abort() #6
  unreachable
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  tail call void @testl()
  ret i32 0
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #5 = { nounwind }
attributes #6 = { cold noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long double", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
