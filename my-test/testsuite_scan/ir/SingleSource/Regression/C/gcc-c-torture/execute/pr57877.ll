; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr57877.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr57877.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@b = dso_local global i32 0, align 4
@c = dso_local local_unnamed_addr global ptr @b, align 8
@f = dso_local local_unnamed_addr global i32 6, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i16 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @f, align 4, !tbaa !6
  %2 = sext i32 %1 to i64
  %3 = load i32, ptr @g, align 4, !tbaa !6
  %4 = icmp slt i32 %3, 1
  br i1 %4, label %8, label %5

5:                                                ; preds = %0
  %6 = load i32, ptr @e, align 4, !tbaa !6
  %7 = icmp eq i32 %6, 1
  br i1 %7, label %82, label %81

8:                                                ; preds = %0
  %9 = load ptr, ptr @c, align 8, !tbaa !10
  %10 = load i32, ptr @a, align 4, !tbaa !6
  %11 = sub i32 1, %3
  %12 = icmp ult i32 %11, 16
  br i1 %12, label %64, label %13

13:                                               ; preds = %8
  %14 = getelementptr i8, ptr %9, i64 4
  %15 = icmp ult ptr @h, getelementptr inbounds nuw (i8, ptr @e, i64 4)
  %16 = icmp ult ptr @e, getelementptr inbounds nuw (i8, ptr @h, i64 4)
  %17 = and i1 %15, %16
  %18 = icmp ult ptr @h, getelementptr inbounds nuw (i8, ptr @g, i64 4)
  %19 = icmp ult ptr @g, getelementptr inbounds nuw (i8, ptr @h, i64 4)
  %20 = and i1 %18, %19
  %21 = or i1 %17, %20
  %22 = icmp ugt ptr %14, @h
  %23 = icmp ult ptr %9, getelementptr inbounds nuw (i8, ptr @h, i64 4)
  %24 = and i1 %22, %23
  %25 = or i1 %21, %24
  %26 = icmp ult ptr @e, getelementptr inbounds nuw (i8, ptr @g, i64 4)
  %27 = icmp ult ptr @g, getelementptr inbounds nuw (i8, ptr @e, i64 4)
  %28 = and i1 %26, %27
  %29 = or i1 %25, %28
  %30 = icmp ugt ptr %14, @e
  %31 = icmp ult ptr %9, getelementptr inbounds nuw (i8, ptr @e, i64 4)
  %32 = and i1 %30, %31
  %33 = or i1 %29, %32
  %34 = icmp ugt ptr %14, @g
  %35 = icmp ult ptr %9, getelementptr inbounds nuw (i8, ptr @g, i64 4)
  %36 = and i1 %34, %35
  %37 = or i1 %33, %36
  br i1 %37, label %64, label %38

38:                                               ; preds = %13
  %39 = and i32 %11, -4
  %40 = add i32 %3, %39
  %41 = insertelement <4 x i64> poison, i64 %2, i64 0
  %42 = shufflevector <4 x i64> %41, <4 x i64> poison, <4 x i32> zeroinitializer
  %43 = add i32 %3, 3
  %44 = load i32, ptr %9, align 4, !tbaa !6, !alias.scope !13
  store i32 %44, ptr @h, align 4, !tbaa !6, !alias.scope !16, !noalias !18
  %45 = shl i32 %44, 16
  %46 = ashr exact i32 %45, 16
  %47 = icmp eq i32 %46, %10
  %48 = insertelement <4 x i1> poison, i1 %47, i64 0
  %49 = shufflevector <4 x i1> %48, <4 x i1> poison, <4 x i32> zeroinitializer
  %50 = zext <4 x i1> %49 to <4 x i64>
  %51 = icmp ugt <4 x i64> %42, %50
  %52 = extractelement <4 x i1> %51, i64 3
  %53 = zext i1 %52 to i32
  store i32 %53, ptr @e, align 4, !tbaa !6, !alias.scope !21, !noalias !22
  br label %54

54:                                               ; preds = %54, %38
  %55 = phi i32 [ 0, %38 ], [ %58, %54 ]
  %56 = phi i32 [ %43, %38 ], [ %59, %54 ]
  %57 = add i32 %56, 1
  %58 = add nuw i32 %55, 4
  %59 = add i32 %56, 4
  %60 = icmp eq i32 %58, %39
  br i1 %60, label %61, label %54, !llvm.loop !23

61:                                               ; preds = %54
  store i32 %57, ptr @g, align 4, !tbaa !6, !alias.scope !27, !noalias !13
  %62 = extractelement <4 x i1> %51, i64 3
  %63 = icmp eq i32 %11, %39
  br i1 %63, label %77, label %64

64:                                               ; preds = %13, %8, %61
  %65 = phi i32 [ %3, %13 ], [ %3, %8 ], [ %40, %61 ]
  br label %66

66:                                               ; preds = %64, %66
  %67 = phi i32 [ %75, %66 ], [ %65, %64 ]
  %68 = load i32, ptr %9, align 4, !tbaa !6
  store i32 %68, ptr @h, align 4, !tbaa !6
  %69 = shl i32 %68, 16
  %70 = ashr exact i32 %69, 16
  %71 = icmp eq i32 %70, %10
  %72 = zext i1 %71 to i64
  %73 = icmp ugt i64 %2, %72
  %74 = zext i1 %73 to i32
  store i32 %74, ptr @e, align 4, !tbaa !6
  %75 = add i32 %67, 1
  store i32 %75, ptr @g, align 4, !tbaa !6
  %76 = icmp eq i32 %67, 0
  br i1 %76, label %77, label %66, !llvm.loop !28

77:                                               ; preds = %66, %61
  %78 = phi i32 [ %44, %61 ], [ %68, %66 ]
  %79 = phi i1 [ %62, %61 ], [ %73, %66 ]
  %80 = trunc i32 %78 to i16
  store i16 %80, ptr @d, align 4, !tbaa !29
  br i1 %79, label %82, label %81

81:                                               ; preds = %5, %77
  tail call void @abort() #2
  unreachable

82:                                               ; preds = %5, %77
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { noreturn nounwind }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 int", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
!13 = !{!14}
!14 = distinct !{!14, !15}
!15 = distinct !{!15, !"LVerDomain"}
!16 = !{!17}
!17 = distinct !{!17, !15}
!18 = !{!19, !20, !14}
!19 = distinct !{!19, !15}
!20 = distinct !{!20, !15}
!21 = !{!19}
!22 = !{!20, !14}
!23 = distinct !{!23, !24, !25, !26}
!24 = !{!"llvm.loop.mustprogress"}
!25 = !{!"llvm.loop.isvectorized", i32 1}
!26 = !{!"llvm.loop.unroll.runtime.disable"}
!27 = !{!20}
!28 = distinct !{!28, !24, !25}
!29 = !{!30, !30, i64 0}
!30 = !{!"short", !8, i64 0}
