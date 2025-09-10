; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/methcall.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/methcall.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [6 x i8] c"true\0A\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"false\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local i8 @toggle_value(ptr noundef readonly captures(none) %0) #0 {
  %2 = load i8, ptr %0, align 8, !tbaa !6
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local noundef ptr @toggle_activate(ptr noundef returned captures(ret: address, provenance) %0) #1 {
  %2 = load i8, ptr %0, align 8, !tbaa !6
  %3 = icmp eq i8 %2, 0
  %4 = zext i1 %3 to i8
  store i8 %4, ptr %0, align 8, !tbaa !6
  ret ptr %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local noundef ptr @init_Toggle(ptr noundef returned writeonly captures(ret: address, provenance) initializes((0, 1), (8, 24)) %0, i8 noundef %1) local_unnamed_addr #2 {
  store i8 %1, ptr %0, align 8, !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr @toggle_value, ptr %3, align 8, !tbaa !11
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr @toggle_activate, ptr %4, align 8, !tbaa !12
  ret ptr %0
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(write, argmem: none, inaccessiblemem: readwrite) uwtable
define dso_local noalias noundef ptr @new_Toggle(i8 noundef %0) local_unnamed_addr #3 {
  %2 = tail call noalias dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #9
  store i8 %0, ptr %2, align 8, !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store ptr @toggle_value, ptr %3, align 8, !tbaa !11
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store ptr @toggle_activate, ptr %4, align 8, !tbaa !12
  ret ptr %2
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local noundef ptr @nth_toggle_activate(ptr noundef returned captures(ret: address, provenance) %0) #1 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %3 = load i32, ptr %2, align 4, !tbaa !13
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr %2, align 4, !tbaa !13
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %6 = load i32, ptr %5, align 8, !tbaa !16
  %7 = icmp slt i32 %4, %6
  br i1 %7, label %12, label %8

8:                                                ; preds = %1
  %9 = load i8, ptr %0, align 8, !tbaa !17
  %10 = icmp eq i8 %9, 0
  %11 = zext i1 %10 to i8
  store i8 %11, ptr %0, align 8, !tbaa !17
  store i32 0, ptr %2, align 4, !tbaa !13
  br label %12

12:                                               ; preds = %8, %1
  ret ptr %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local noundef ptr @init_NthToggle(ptr noundef returned writeonly captures(ret: address, provenance) initializes((16, 32)) %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store i32 %1, ptr %3, align 8, !tbaa !16
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 28
  store i32 0, ptr %4, align 4, !tbaa !13
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr @nth_toggle_activate, ptr %5, align 8, !tbaa !18
  ret ptr %0
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(write, argmem: none, inaccessiblemem: readwrite) uwtable
define dso_local noalias noundef ptr @new_NthToggle(i8 noundef %0, i32 noundef %1) local_unnamed_addr #3 {
  %3 = tail call noalias dereferenceable_or_null(32) ptr @malloc(i64 noundef 32) #9
  store i8 %0, ptr %3, align 8, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr @toggle_value, ptr %4, align 8, !tbaa !11
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store i32 %1, ptr %6, align 8, !tbaa !16
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 28
  store i32 0, ptr %7, align 4, !tbaa !13
  store ptr @nth_toggle_activate, ptr %5, align 8, !tbaa !18
  ret ptr %3
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #5 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %9

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !19
  %7 = tail call i64 @strtol(ptr noundef nonnull captures(none) %6, ptr noundef null, i32 noundef 10) #10
  %8 = trunc i64 %7 to i32
  br label %9

9:                                                ; preds = %2, %4
  %10 = phi i32 [ %8, %4 ], [ 500000000, %2 ]
  %11 = tail call noalias dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #9
  store i8 1, ptr %11, align 8, !tbaa !6
  %12 = getelementptr inbounds nuw i8, ptr %11, i64 8
  store ptr @toggle_value, ptr %12, align 8, !tbaa !11
  %13 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store ptr @toggle_activate, ptr %13, align 8, !tbaa !12
  %14 = icmp sgt i32 %10, 0
  br i1 %14, label %15, label %27

15:                                               ; preds = %9, %15
  %16 = phi i32 [ %22, %15 ], [ 0, %9 ]
  %17 = load ptr, ptr %13, align 8, !tbaa !12
  %18 = tail call ptr %17(ptr noundef nonnull %11) #10
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 8
  %20 = load ptr, ptr %19, align 8, !tbaa !11
  %21 = tail call i8 %20(ptr noundef nonnull %11) #10
  %22 = add nuw nsw i32 %16, 1
  %23 = icmp eq i32 %22, %10
  br i1 %23, label %24, label %15, !llvm.loop !21

24:                                               ; preds = %15
  %25 = icmp eq i8 %21, 0
  %26 = select i1 %25, ptr @.str.1, ptr @.str
  br label %27

27:                                               ; preds = %24, %9
  %28 = phi ptr [ @.str, %9 ], [ %26, %24 ]
  %29 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %28)
  tail call void @free(ptr noundef nonnull %11) #10
  %30 = tail call noalias dereferenceable_or_null(32) ptr @malloc(i64 noundef 32) #9
  store i8 1, ptr %30, align 8, !tbaa !6
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 8
  store ptr @toggle_value, ptr %31, align 8, !tbaa !11
  %32 = getelementptr inbounds nuw i8, ptr %30, i64 16
  %33 = getelementptr inbounds nuw i8, ptr %30, i64 24
  store <2 x i32> <i32 3, i32 0>, ptr %33, align 8, !tbaa !23
  store ptr @nth_toggle_activate, ptr %32, align 8, !tbaa !18
  br i1 %14, label %34, label %46

34:                                               ; preds = %27, %34
  %35 = phi i32 [ %41, %34 ], [ 0, %27 ]
  %36 = load ptr, ptr %32, align 8, !tbaa !18
  %37 = tail call ptr %36(ptr noundef nonnull %30) #10
  %38 = getelementptr inbounds nuw i8, ptr %37, i64 8
  %39 = load ptr, ptr %38, align 8, !tbaa !11
  %40 = tail call i8 %39(ptr noundef nonnull %30) #10
  %41 = add nuw nsw i32 %35, 1
  %42 = icmp eq i32 %41, %10
  br i1 %42, label %43, label %34, !llvm.loop !24

43:                                               ; preds = %34
  %44 = icmp eq i8 %40, 0
  %45 = select i1 %44, ptr @.str.1, ptr @.str
  br label %46

46:                                               ; preds = %43, %27
  %47 = phi ptr [ @.str, %27 ], [ %45, %43 ]
  %48 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %47)
  tail call void @free(ptr noundef nonnull %30) #10
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #6

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #7

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #8

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree nounwind willreturn memory(write, argmem: none, inaccessiblemem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nounwind allocsize(0) }
attributes #10 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"Toggle", !8, i64 0, !10, i64 8, !10, i64 16}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!"any pointer", !8, i64 0}
!11 = !{!7, !10, i64 8}
!12 = !{!7, !10, i64 16}
!13 = !{!14, !15, i64 28}
!14 = !{!"NthToggle", !8, i64 0, !10, i64 8, !10, i64 16, !15, i64 24, !15, i64 28}
!15 = !{!"int", !8, i64 0}
!16 = !{!14, !15, i64 24}
!17 = !{!14, !8, i64 0}
!18 = !{!14, !10, i64 16}
!19 = !{!20, !20, i64 0}
!20 = !{!"p1 omnipotent char", !10, i64 0}
!21 = distinct !{!21, !22}
!22 = !{!"llvm.loop.mustprogress"}
!23 = !{!15, !15, i64 0}
!24 = distinct !{!24, !22}
