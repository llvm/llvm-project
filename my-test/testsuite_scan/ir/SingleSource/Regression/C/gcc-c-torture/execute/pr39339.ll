; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr39339.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr39339.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.C = type { i32, %struct.D }
%struct.D = type { i32 }
%struct.A = type { ptr, i32 }
%struct.B = type { ptr, i8 }

@__const.main.e = private unnamed_addr constant { i64, i64, { i32, { i8, i8, i8, i8 } } } { i64 5, i64 0, { i32, { i8, i8, i8, i8 } } { i32 6, { i8, i8, i8, i8 } { i8 -1, i8 -1, i8 127, i8 85 } } }, align 8

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @foo(ptr noundef readonly captures(none) %0, i32 noundef %1, i32 noundef %2, ptr noundef readonly captures(none) %3) local_unnamed_addr #0 {
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = load i64, ptr %5, align 8, !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %8 = load i32, ptr %7, align 4, !tbaa !14
  %9 = and i32 %2, 15
  %10 = and i32 %8, -16
  %11 = or disjoint i32 %10, %9
  %12 = load ptr, ptr %3, align 8, !tbaa !15
  %13 = load ptr, ptr %12, align 8, !tbaa !19
  %14 = getelementptr inbounds %struct.C, ptr %13, i64 %6
  store i32 %1, ptr %14, align 4, !tbaa !22
  %15 = getelementptr inbounds %struct.C, ptr %13, i64 %6, i32 1
  store i32 %11, ptr %15, align 4, !tbaa !14
  %16 = or i32 %11, 4194304
  %17 = icmp sgt i32 %2, 1
  br i1 %17, label %18, label %28

18:                                               ; preds = %4, %18
  %19 = phi i32 [ %26, %18 ], [ 1, %4 ]
  %20 = phi i64 [ %21, %18 ], [ %6, %4 ]
  %21 = add nsw i64 %20, 1
  %22 = load ptr, ptr %3, align 8, !tbaa !15
  %23 = load ptr, ptr %22, align 8, !tbaa !19
  %24 = getelementptr inbounds %struct.C, ptr %23, i64 %21
  store i32 %1, ptr %24, align 4, !tbaa !22
  %25 = getelementptr inbounds %struct.C, ptr %23, i64 %21, i32 1
  store i32 %16, ptr %25, align 4, !tbaa !14
  %26 = add nuw nsw i32 %19, 1
  %27 = icmp eq i32 %26, %2
  br i1 %27, label %28, label %18, !llvm.loop !23

28:                                               ; preds = %18, %4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca [4 x %struct.C], align 4
  %2 = alloca %struct.A, align 8
  %3 = alloca %struct.B, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  store ptr %1, ptr %2, align 8, !tbaa !19
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store <2 x i32> <i32 4, i32 0>, ptr %4, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  store ptr %2, ptr %3, align 8, !tbaa !15
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i8 1, ptr %5, align 8
  %6 = getelementptr inbounds nuw i8, ptr %3, i64 9
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %6, i8 0, i64 7, i1 false)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(32) %1, i8 0, i64 32, i1 false)
  call void @foo(ptr noundef nonnull @__const.main.e, i32 noundef 65, i32 noundef 2, ptr noundef nonnull %3)
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %8 = load i32, ptr %7, align 4
  %9 = icmp eq i32 %8, 1434451954
  br i1 %9, label %11, label %10

10:                                               ; preds = %0
  call void @abort() #6
  unreachable

11:                                               ; preds = %0
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %13 = load i32, ptr %12, align 4
  %14 = icmp eq i32 %13, 1434451954
  br i1 %14, label %16, label %15

15:                                               ; preds = %11
  call void @abort() #6
  unreachable

16:                                               ; preds = %11
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

attributes #0 = { nofree noinline norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !8, i64 8}
!7 = !{!"E", !8, i64 0, !8, i64 8, !11, i64 16}
!8 = !{!"long", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"C", !12, i64 0, !13, i64 4}
!12 = !{!"int", !9, i64 0}
!13 = !{!"D", !12, i64 0, !12, i64 0, !12, i64 2, !12, i64 2, !12, i64 2, !12, i64 3, !12, i64 3, !12, i64 3, !12, i64 3, !12, i64 3, !12, i64 3, !12, i64 3, !12, i64 3}
!14 = !{!9, !9, i64 0}
!15 = !{!16, !17, i64 0}
!16 = !{!"B", !17, i64 0, !9, i64 8}
!17 = !{!"p1 _ZTS1A", !18, i64 0}
!18 = !{!"any pointer", !9, i64 0}
!19 = !{!20, !21, i64 0}
!20 = !{!"A", !21, i64 0, !12, i64 8}
!21 = !{!"p1 _ZTS1C", !18, i64 0}
!22 = !{!11, !12, i64 0}
!23 = distinct !{!23, !24}
!24 = !{!"llvm.loop.mustprogress"}
