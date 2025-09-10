; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20041214-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20041214-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@.str = private unnamed_addr constant [3 x i8] c"%s\00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"asdf\00", align 1

; Function Attrs: nofree norecurse nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local noundef i32 @g(ptr noundef %0, ptr noundef readonly captures(none) %1, ptr dead_on_return noundef captures(none) %2) local_unnamed_addr #0 {
  %4 = load i8, ptr %1, align 1, !tbaa !6
  %5 = icmp eq i8 %4, 0
  br i1 %5, label %30, label %6

6:                                                ; preds = %3
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %8 = getelementptr inbounds nuw i8, ptr %2, i64 8
  br label %9

9:                                                ; preds = %6, %23
  %10 = phi ptr [ %27, %23 ], [ %1, %6 ]
  %11 = load i32, ptr %7, align 8
  %12 = icmp sgt i32 %11, -1
  br i1 %12, label %20, label %13

13:                                               ; preds = %9
  %14 = add nsw i32 %11, 8
  store i32 %14, ptr %7, align 8
  %15 = icmp samesign ult i32 %11, -7
  br i1 %15, label %16, label %20

16:                                               ; preds = %13
  %17 = load ptr, ptr %8, align 8
  %18 = sext i32 %11 to i64
  %19 = getelementptr inbounds i8, ptr %17, i64 %18
  br label %23

20:                                               ; preds = %13, %9
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 8
  store ptr %22, ptr %2, align 8
  br label %23

23:                                               ; preds = %20, %16
  %24 = phi ptr [ %19, %16 ], [ %21, %20 ]
  %25 = load ptr, ptr %24, align 8, !tbaa !9
  %26 = tail call ptr @strcpy(ptr noundef nonnull dereferenceable(1) %0, ptr noundef nonnull dereferenceable(1) %25) #8
  %27 = getelementptr inbounds nuw i8, ptr %10, i64 2
  %28 = load i8, ptr %27, align 1, !tbaa !6
  %29 = icmp eq i8 %28, 0
  br i1 %29, label %30, label %9, !llvm.loop !12

30:                                               ; preds = %23, %3
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare ptr @strcpy(ptr noalias noundef returned writeonly, ptr noalias noundef readonly captures(none)) local_unnamed_addr #1

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local void @f(ptr noundef %0, ptr noundef readonly captures(none) %1, ...) local_unnamed_addr #2 {
  %3 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #8
  call void @llvm.va_start.p0(ptr nonnull %3)
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %5 = load ptr, ptr %4, align 8, !tbaa !14
  %6 = load i8, ptr %1, align 1, !tbaa !6
  %7 = icmp eq i8 %6, 0
  br i1 %7, label %35, label %8

8:                                                ; preds = %2
  %9 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %10 = load i32, ptr %9, align 8, !tbaa !15
  %11 = load ptr, ptr %3, align 8, !tbaa !14
  br label %12

12:                                               ; preds = %8, %26
  %13 = phi i32 [ %27, %26 ], [ %10, %8 ]
  %14 = phi ptr [ %28, %26 ], [ %11, %8 ]
  %15 = phi ptr [ %32, %26 ], [ %1, %8 ]
  %16 = icmp sgt i32 %13, -1
  br i1 %16, label %23, label %17

17:                                               ; preds = %12
  %18 = add nsw i32 %13, 8
  %19 = icmp samesign ult i32 %13, -7
  br i1 %19, label %20, label %23

20:                                               ; preds = %17
  %21 = sext i32 %13 to i64
  %22 = getelementptr inbounds i8, ptr %5, i64 %21
  br label %26

23:                                               ; preds = %17, %12
  %24 = phi i32 [ %13, %12 ], [ %18, %17 ]
  %25 = getelementptr inbounds nuw i8, ptr %14, i64 8
  br label %26

26:                                               ; preds = %23, %20
  %27 = phi i32 [ %24, %23 ], [ %18, %20 ]
  %28 = phi ptr [ %25, %23 ], [ %14, %20 ]
  %29 = phi ptr [ %14, %23 ], [ %22, %20 ]
  %30 = load ptr, ptr %29, align 8, !tbaa !9
  %31 = call ptr @strcpy(ptr noundef nonnull dereferenceable(1) %0, ptr noundef nonnull dereferenceable(1) %30) #8
  %32 = getelementptr inbounds nuw i8, ptr %15, i64 2
  %33 = load i8, ptr %32, align 1, !tbaa !6
  %34 = icmp eq i8 %33, 0
  br i1 %34, label %35, label %12, !llvm.loop !12

35:                                               ; preds = %26, %2
  call void @llvm.va_end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  %1 = alloca [10 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #8
  call void (ptr, ptr, ...) @f(ptr noundef nonnull %1, ptr noundef nonnull @.str, ptr noundef nonnull @.str.1, i32 noundef 0)
  %2 = call i32 @bcmp(ptr noundef nonnull dereferenceable(5) %1, ptr noundef nonnull dereferenceable(5) @.str.1, i64 5)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  call void @abort() #9
  unreachable

5:                                                ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #8
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #6

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #7

attributes #0 = { nofree norecurse nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #5 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #8 = { nounwind }
attributes #9 = { noreturn nounwind }

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
!9 = !{!10, !10, i64 0}
!10 = !{!"p1 omnipotent char", !11, i64 0}
!11 = !{!"any pointer", !7, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!11, !11, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !7, i64 0}
