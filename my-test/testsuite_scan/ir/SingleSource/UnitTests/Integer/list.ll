; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/list.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/list.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@array = dso_local local_unnamed_addr global [192 x i32] [i32 103, i32 198, i32 105, i32 115, i32 81, i32 255, i32 74, i32 236, i32 41, i32 205, i32 186, i32 171, i32 242, i32 251, i32 227, i32 70, i32 124, i32 194, i32 84, i32 248, i32 27, i32 232, i32 231, i32 141, i32 118, i32 90, i32 46, i32 99, i32 51, i32 159, i32 201, i32 154, i32 102, i32 50, i32 13, i32 183, i32 49, i32 88, i32 163, i32 90, i32 37, i32 93, i32 5, i32 23, i32 88, i32 233, i32 94, i32 212, i32 171, i32 178, i32 205, i32 198, i32 155, i32 180, i32 84, i32 17, i32 14, i32 130, i32 116, i32 65, i32 33, i32 61, i32 220, i32 135, i32 112, i32 233, i32 62, i32 161, i32 65, i32 225, i32 252, i32 103, i32 62, i32 1, i32 126, i32 151, i32 234, i32 220, i32 107, i32 150, i32 143, i32 56, i32 92, i32 42, i32 236, i32 176, i32 59, i32 251, i32 50, i32 175, i32 60, i32 84, i32 236, i32 24, i32 219, i32 92, i32 2, i32 26, i32 254, i32 67, i32 251, i32 250, i32 170, i32 58, i32 251, i32 41, i32 209, i32 230, i32 5, i32 60, i32 124, i32 148, i32 117, i32 216, i32 190, i32 97, i32 137, i32 249, i32 92, i32 187, i32 168, i32 153, i32 15, i32 149, i32 177, i32 235, i32 241, i32 179, i32 5, i32 239, i32 247, i32 0, i32 233, i32 161, i32 58, i32 229, i32 202, i32 11, i32 203, i32 208, i32 72, i32 71, i32 100, i32 189, i32 31, i32 35, i32 30, i32 168, i32 28, i32 123, i32 100, i32 197, i32 20, i32 115, i32 90, i32 197, i32 94, i32 75, i32 121, i32 99, i32 59, i32 112, i32 100, i32 36, i32 17, i32 158, i32 9, i32 220, i32 170, i32 212, i32 172, i32 242, i32 27, i32 16, i32 175, i32 59, i32 51, i32 205, i32 227, i32 80, i32 72, i32 71, i32 21, i32 92, i32 187, i32 111, i32 34, i32 25, i32 186, i32 155, i32 125, i32 245], align 4
@.str = private unnamed_addr constant [35 x i8] c"error: i = %d, x = %d, array = %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [42 x i8] c"error: i = %d, y = %hhd, expected = %hhd\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local void @test() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %10, %1 ]
  %3 = phi ptr [ null, %0 ], [ %4, %1 ]
  %4 = tail call noalias dereferenceable_or_null(16) ptr @malloc(i64 noundef 16) #4
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr %3, ptr %5, align 8, !tbaa !6
  %6 = getelementptr inbounds nuw i32, ptr @array, i64 %2
  %7 = load i32, ptr %6, align 4, !tbaa !14
  store i32 %7, ptr %4, align 8, !tbaa !15
  %8 = add nsw i32 %7, -1
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 4
  store i32 %8, ptr %9, align 4, !tbaa !16
  %10 = add nuw nsw i64 %2, 1
  %11 = icmp eq i64 %10, 192
  br i1 %11, label %12, label %1, !llvm.loop !17

12:                                               ; preds = %1, %35
  %13 = phi i32 [ %15, %35 ], [ 0, %1 ]
  %14 = phi ptr [ %37, %35 ], [ %4, %1 ]
  %15 = add i32 %13, 1
  %16 = load i32, ptr %14, align 8, !tbaa !15
  %17 = sub i32 191, %13
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds nuw i32, ptr @array, i64 %18
  %20 = load i32, ptr %19, align 4, !tbaa !14
  %21 = icmp eq i32 %16, %20
  br i1 %21, label %25, label %22

22:                                               ; preds = %12
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %15, i32 noundef %16, i32 noundef %20)
  %24 = load i32, ptr %19, align 4, !tbaa !14
  br label %25

25:                                               ; preds = %22, %12
  %26 = phi i32 [ %24, %22 ], [ %16, %12 ]
  %27 = getelementptr inbounds nuw i8, ptr %14, i64 4
  %28 = load i32, ptr %27, align 4, !tbaa !16
  %29 = add i32 %26, 127
  %30 = and i32 %29, 127
  %31 = icmp eq i32 %28, %30
  br i1 %31, label %35, label %32

32:                                               ; preds = %25
  %33 = and i32 %28, 255
  %34 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, i32 noundef %33, i32 noundef %30)
  br label %35

35:                                               ; preds = %32, %25
  %36 = getelementptr inbounds nuw i8, ptr %14, i64 8
  %37 = load ptr, ptr %36, align 8, !tbaa !6
  tail call void @free(ptr noundef nonnull %14) #5
  %38 = icmp eq ptr %37, null
  br i1 %38, label %39, label %12, !llvm.loop !19

39:                                               ; preds = %35
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call void @test()
  ret i32 0
}

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind allocsize(0) }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !12, i64 8}
!7 = !{!"myList", !8, i64 0, !12, i64 8}
!8 = !{!"myStruct", !9, i64 0, !9, i64 4}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!"p1 _ZTS6myList", !13, i64 0}
!13 = !{!"any pointer", !10, i64 0}
!14 = !{!9, !9, i64 0}
!15 = !{!7, !9, i64 0}
!16 = !{!7, !9, i64 4}
!17 = distinct !{!17, !18}
!18 = !{!"llvm.loop.mustprogress"}
!19 = distinct !{!19, !18}
