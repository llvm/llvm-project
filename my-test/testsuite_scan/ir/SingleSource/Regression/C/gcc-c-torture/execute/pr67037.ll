; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr67037.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr67037.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@extfunc = dso_local local_unnamed_addr global ptr null, align 8

; Function Attrs: noinline nounwind uwtable
define dso_local range(i32 0, 2) i32 @badfunc(i32 %0, i32 %1, i32 %2, i32 %3, ptr noundef writeonly captures(none) %4, i32 noundef %5) local_unnamed_addr #0 {
  %7 = alloca [5348 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #4
  %8 = load ptr, ptr @extfunc, align 8, !tbaa !6
  %9 = tail call i64 %8() #4
  %10 = icmp eq i64 %9, 0
  br i1 %10, label %11, label %37

11:                                               ; preds = %6
  %12 = load ptr, ptr @extfunc, align 8, !tbaa !6
  %13 = tail call i64 %12() #4
  %14 = icmp eq i64 %13, 0
  br i1 %14, label %15, label %37

15:                                               ; preds = %11
  %16 = load ptr, ptr @extfunc, align 8, !tbaa !6
  %17 = call i64 %16(ptr noundef nonnull %7) #4
  %18 = icmp ugt i32 %5, 1
  br i1 %18, label %19, label %33

19:                                               ; preds = %15
  %20 = getelementptr inbounds nuw i8, ptr %4, i64 2
  store i16 78, ptr %4, align 2, !tbaa !10
  %21 = add i32 %5, -3
  %22 = icmp ult i32 %21, -2
  br i1 %22, label %23, label %35, !llvm.loop !12

23:                                               ; preds = %19
  %24 = getelementptr inbounds nuw i8, ptr %4, i64 4
  store i16 84, ptr %20, align 2, !tbaa !10
  %25 = and i32 %5, -2
  %26 = icmp eq i32 %25, 2
  br i1 %26, label %35, label %27, !llvm.loop !12

27:                                               ; preds = %23
  %28 = getelementptr inbounds nuw i8, ptr %4, i64 6
  store i16 70, ptr %24, align 2, !tbaa !10
  %29 = add i32 %5, -5
  %30 = icmp ult i32 %29, -2
  br i1 %30, label %31, label %35, !llvm.loop !12

31:                                               ; preds = %27
  %32 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i16 83, ptr %28, align 2, !tbaa !10
  br label %35

33:                                               ; preds = %15
  %34 = icmp eq i32 %5, 0
  br i1 %34, label %37, label %35

35:                                               ; preds = %33, %31, %27, %23, %19
  %36 = phi ptr [ %4, %33 ], [ %32, %31 ], [ %28, %27 ], [ %24, %23 ], [ %20, %19 ]
  store i16 0, ptr %36, align 2, !tbaa !10
  br label %37

37:                                               ; preds = %35, %33, %11, %6
  %38 = phi i32 [ 0, %6 ], [ 0, %11 ], [ 1, %33 ], [ 1, %35 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #4
  ret i32 %38
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #2 {
  %1 = alloca [6 x i16], align 2
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  store ptr @f, ptr @extfunc, align 8, !tbaa !6
  %2 = call i32 @badfunc(i32 poison, i32 poison, i32 poison, i32 poison, ptr noundef nonnull %1, i32 noundef 6)
  %3 = xor i32 %2, 1
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define internal noundef i64 @f() #3 {
  ret i64 0
}

attributes #0 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"short", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
