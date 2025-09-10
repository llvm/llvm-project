; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ieee/copysign1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ieee/copysign1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.Dl = type { fp128, fp128, fp128 }

@Tl = internal constant [8 x %struct.Dl] [%struct.Dl { fp128 0xL00000000000000003FFF000000000000, fp128 0xL00000000000000004000000000000000, fp128 0xL00000000000000003FFF000000000000 }, %struct.Dl { fp128 0xL00000000000000003FFF000000000000, fp128 0xL0000000000000000C000000000000000, fp128 0xL0000000000000000BFFF000000000000 }, %struct.Dl { fp128 0xL0000000000000000BFFF000000000000, fp128 0xL0000000000000000C000000000000000, fp128 0xL0000000000000000BFFF000000000000 }, %struct.Dl { fp128 0xL00000000000000000000000000000000, fp128 0xL0000000000000000C000000000000000, fp128 0xL00000000000000008000000000000000 }, %struct.Dl { fp128 0xL00000000000000008000000000000000, fp128 0xL0000000000000000C000000000000000, fp128 0xL00000000000000008000000000000000 }, %struct.Dl { fp128 0xL00000000000000008000000000000000, fp128 0xL00000000000000004000000000000000, fp128 0xL00000000000000000000000000000000 }, %struct.Dl { fp128 0xL00000000000000007FFF000000000000, fp128 0xL00000000000000008000000000000000, fp128 0xL0000000000000000FFFF000000000000 }, %struct.Dl { fp128 0xL0000000000000000FFFF800000000000, fp128 0xL00000000000000007FFF000000000000, fp128 0xL00000000000000007FFF800000000000 }], align 16

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef float @cf(float noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = tail call float @llvm.copysign.f32(float %0, float %1)
  ret float %3
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.copysign.f32(float, float) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @testf() local_unnamed_addr #0 {
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef double @c(double noundef %0, double noundef %1) local_unnamed_addr #0 {
  %3 = tail call double @llvm.copysign.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.copysign.f64(double, double) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @test() local_unnamed_addr #0 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef fp128 @cl(fp128 noundef %0, fp128 noundef %1) local_unnamed_addr #0 {
  %3 = tail call fp128 @llvm.copysign.f128(fp128 %0, fp128 %1)
  ret fp128 %3
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare fp128 @llvm.copysign.f128(fp128, fp128) #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @testl() local_unnamed_addr #4 {
  %1 = alloca fp128, align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  store fp128 0xL00000000000000003FFF000000000000, ptr %1, align 16, !tbaa !6
  %2 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %1, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Tl, i64 32), i64 16)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %26

4:                                                ; preds = %0
  store fp128 0xL0000000000000000BFFF000000000000, ptr %1, align 16, !tbaa !6
  %5 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %1, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Tl, i64 80), i64 16)
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %7, label %26

7:                                                ; preds = %4
  %8 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %1, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Tl, i64 128), i64 16)
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %10, label %26

10:                                               ; preds = %7
  store fp128 0xL00000000000000008000000000000000, ptr %1, align 16, !tbaa !6
  %11 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %1, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Tl, i64 176), i64 16)
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %13, label %26

13:                                               ; preds = %10
  %14 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %1, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Tl, i64 224), i64 16)
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %16, label %26

16:                                               ; preds = %13
  store fp128 0xL00000000000000000000000000000000, ptr %1, align 16, !tbaa !6
  %17 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %1, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Tl, i64 272), i64 16)
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %26

19:                                               ; preds = %16
  store fp128 0xL0000000000000000FFFF000000000000, ptr %1, align 16, !tbaa !6
  %20 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %1, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Tl, i64 320), i64 16)
  %21 = icmp eq i32 %20, 0
  br i1 %21, label %22, label %26

22:                                               ; preds = %19
  store fp128 0xL00000000000000007FFF800000000000, ptr %1, align 16, !tbaa !6
  %23 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %1, ptr noundef nonnull dereferenceable(16) getelementptr inbounds nuw (i8, ptr @Tl, i64 368), i64 16)
  %24 = icmp eq i32 %23, 0
  br i1 %24, label %25, label %26

25:                                               ; preds = %22
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  ret void

26:                                               ; preds = %22, %19, %16, %13, %10, %7, %4, %0
  tail call void @abort() #7
  unreachable
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  tail call void @testl()
  ret i32 0
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #5

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #6 = { nounwind }
attributes #7 = { cold noreturn nounwind }

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
