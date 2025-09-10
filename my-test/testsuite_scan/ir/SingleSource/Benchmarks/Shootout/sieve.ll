; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/sieve.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/sieve.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@main.flags = internal unnamed_addr global [8193 x i8] zeroinitializer, align 1
@.str = private unnamed_addr constant [11 x i8] c"Count: %d\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %10

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !6
  %7 = tail call i64 @strtol(ptr noundef nonnull captures(none) %6, ptr noundef null, i32 noundef 10) #5
  %8 = trunc i64 %7 to i32
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %68, label %10

10:                                               ; preds = %2, %4
  %11 = phi i32 [ 170000, %2 ], [ %8, %4 ]
  br label %15

12:                                               ; preds = %63
  %13 = add nsw i32 %16, -1
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %68, label %15, !llvm.loop !11

15:                                               ; preds = %10, %12
  %16 = phi i32 [ %13, %12 ], [ %11, %10 ]
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(8191) getelementptr inbounds nuw (i8, ptr @main.flags, i64 2), i8 1, i64 8191, i1 false), !tbaa !13
  br label %17

17:                                               ; preds = %15, %63
  %18 = phi i64 [ 0, %15 ], [ %67, %63 ]
  %19 = phi i32 [ 0, %15 ], [ %64, %63 ]
  %20 = phi i64 [ 2, %15 ], [ %65, %63 ]
  %21 = mul nuw nsw i64 %18, 3
  %22 = tail call i64 @llvm.umax.i64(i64 %21, i64 8187)
  %23 = mul nsw i64 %18, -3
  %24 = add i64 %22, %23
  %25 = icmp ne i64 %24, 0
  %26 = sext i1 %25 to i64
  %27 = add i64 %24, %26
  %28 = getelementptr inbounds nuw i8, ptr @main.flags, i64 %20
  %29 = load i8, ptr %28, align 1, !tbaa !13
  %30 = icmp eq i8 %29, 0
  br i1 %30, label %63, label %31

31:                                               ; preds = %17
  %32 = icmp samesign ult i64 %20, 4097
  br i1 %32, label %33, label %61

33:                                               ; preds = %31
  %34 = shl nuw nsw i64 %20, 1
  %35 = select i1 %25, i64 2, i64 1
  %36 = udiv i64 %27, %20
  %37 = add i64 %35, %36
  %38 = icmp ult i64 %37, 2
  br i1 %38, label %54, label %39

39:                                               ; preds = %33
  %40 = and i64 %37, -2
  %41 = add i64 %40, 2
  %42 = mul i64 %20, %41
  %43 = getelementptr i8, ptr @main.flags, i64 %20
  br label %44

44:                                               ; preds = %44, %39
  %45 = phi i64 [ 0, %39 ], [ %50, %44 ]
  %46 = add i64 %45, 2
  %47 = mul i64 %20, %46
  %48 = getelementptr inbounds nuw i8, ptr @main.flags, i64 %47
  %49 = getelementptr i8, ptr %43, i64 %47
  store i8 0, ptr %48, align 1, !tbaa !13
  store i8 0, ptr %49, align 1, !tbaa !13
  %50 = add nuw i64 %45, 2
  %51 = icmp eq i64 %50, %40
  br i1 %51, label %52, label %44, !llvm.loop !14

52:                                               ; preds = %44
  %53 = icmp eq i64 %37, %40
  br i1 %53, label %61, label %54

54:                                               ; preds = %33, %52
  %55 = phi i64 [ %34, %33 ], [ %42, %52 ]
  br label %56

56:                                               ; preds = %54, %56
  %57 = phi i64 [ %59, %56 ], [ %55, %54 ]
  %58 = getelementptr inbounds nuw i8, ptr @main.flags, i64 %57
  store i8 0, ptr %58, align 1, !tbaa !13
  %59 = add nuw nsw i64 %57, %20
  %60 = icmp samesign ult i64 %59, 8193
  br i1 %60, label %56, label %61, !llvm.loop !17

61:                                               ; preds = %56, %52, %31
  %62 = add nsw i32 %19, 1
  br label %63

63:                                               ; preds = %17, %61
  %64 = phi i32 [ %62, %61 ], [ %19, %17 ]
  %65 = add nuw nsw i64 %20, 1
  %66 = icmp eq i64 %65, 8193
  %67 = add i64 %18, 1
  br i1 %66, label %12, label %17, !llvm.loop !18

68:                                               ; preds = %12, %4
  %69 = phi i32 [ 0, %4 ], [ %64, %12 ]
  %70 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %69)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #4

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
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
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!9, !9, i64 0}
!14 = distinct !{!14, !12, !15, !16}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !12, !15}
!18 = distinct !{!18, !12}
