; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20180921-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20180921-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@ss = dso_local local_unnamed_addr global ptr null, align 8
@j = internal unnamed_addr global i32 0, align 4
@i = dso_local local_unnamed_addr global [6 x i32] zeroinitializer, align 4
@an = dso_local local_unnamed_addr global i32 0, align 4
@h = internal unnamed_addr global i1 false, align 4
@c = dso_local local_unnamed_addr global i8 0, align 4
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@am = internal unnamed_addr global i16 0, align 4
@__const.aw.bf = private unnamed_addr constant { i32, i8, i8, [2 x i8], i32 } { i32 908, i8 5, i8 0, [2 x i8] zeroinitializer, i32 3 }, align 4
@ag = internal unnamed_addr global i32 8, align 4
@f = internal unnamed_addr global i32 0, align 4
@af = internal unnamed_addr global i32 0, align 4
@ao = dso_local local_unnamed_addr global i32 0, align 4
@ap = dso_local local_unnamed_addr global i32 0, align 4
@ab = internal unnamed_addr global i32 0, align 4
@g = internal unnamed_addr global { i32, i8, i8, [2 x i8], i32 } { i32 9, i8 5, i8 0, [2 x i8] zeroinitializer, i32 0 }, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local i32 @dummy(ptr noundef %0, ...) local_unnamed_addr #0 {
  store ptr %0, ptr @ss, align 8, !tbaa !6
  ret i32 undef
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @aq(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr @j, align 4, !tbaa !11
  %3 = sext i32 %2 to i64
  %4 = getelementptr inbounds i32, ptr @i, i64 %3
  %5 = load i32, ptr %4, align 4, !tbaa !11
  %6 = xor i32 %2, %5
  %7 = and i32 %6, 5
  %8 = zext nneg i32 %7 to i64
  %9 = getelementptr inbounds nuw i32, ptr @i, i64 %8
  %10 = load i32, ptr %9, align 4, !tbaa !11
  %11 = and i32 %10, 4090
  store i32 %11, ptr @j, align 4, !tbaa !11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @as(i32 noundef %0) local_unnamed_addr #2 {
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @aw(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i1, ptr @h, align 4
  %3 = select i1 %2, i8 9, i8 5
  %4 = icmp eq i32 %0, 0
  %5 = load i16, ptr @am, align 4, !tbaa !13
  br i1 %4, label %82, label %6

6:                                                ; preds = %1
  %7 = load i32, ptr @f, align 4
  %8 = freeze i32 %7
  %9 = load i32, ptr @af, align 4
  %10 = load i32, ptr @ag, align 4
  %11 = load i32, ptr @j, align 4
  %12 = load i32, ptr @ab, align 4
  br label %13

13:                                               ; preds = %74, %6
  %14 = phi i32 [ %12, %6 ], [ %65, %74 ]
  %15 = phi i32 [ %11, %6 ], [ %66, %74 ]
  %16 = phi i32 [ %8, %6 ], [ %67, %74 ]
  %17 = phi i32 [ %12, %6 ], [ %75, %74 ]
  %18 = phi i32 [ %11, %6 ], [ %76, %74 ]
  %19 = phi i32 [ %8, %6 ], [ %77, %74 ]
  %20 = phi i32 [ %9, %6 ], [ %71, %74 ]
  %21 = phi i32 [ %10, %6 ], [ %72, %74 ]
  %22 = phi i16 [ %5, %6 ], [ %23, %74 ]
  %23 = add i16 %22, -95
  %24 = and i32 %21, 95
  %25 = sub nsw i32 0, %24
  %26 = xor i32 %20, %25
  %27 = icmp sgt i32 %26, 8
  br i1 %27, label %85, label %78

28:                                               ; preds = %52
  %29 = icmp eq i32 %36, 0
  br i1 %29, label %81, label %30

30:                                               ; preds = %78, %28
  %31 = phi i32 [ %53, %28 ], [ %14, %78 ]
  %32 = phi i32 [ %54, %28 ], [ %15, %78 ]
  %33 = phi i32 [ %59, %28 ], [ %19, %78 ]
  %34 = phi i32 [ %58, %28 ], [ %18, %78 ]
  %35 = phi i32 [ %57, %28 ], [ %17, %78 ]
  %36 = phi i32 [ %61, %28 ], [ %25, %78 ]
  %37 = phi i32 [ %62, %28 ], [ %26, %78 ]
  %38 = phi i32 [ %56, %28 ], [ %18, %78 ]
  %39 = phi i32 [ %55, %28 ], [ %17, %78 ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(12) @g, ptr noundef nonnull align 4 dereferenceable(12) @__const.aw.bf, i64 12, i1 false), !tbaa.struct !15
  %40 = icmp eq i32 %33, 0
  br i1 %40, label %52, label %41

41:                                               ; preds = %30
  %42 = sext i32 %34 to i64
  %43 = getelementptr inbounds i32, ptr @i, i64 %42
  %44 = load i32, ptr %43, align 4, !tbaa !11
  %45 = xor i32 %44, %34
  %46 = and i32 %45, 5
  %47 = zext nneg i32 %46 to i64
  %48 = getelementptr inbounds nuw i32, ptr @i, i64 %47
  %49 = load i32, ptr %48, align 4, !tbaa !11
  %50 = and i32 %49, 4090
  store i32 %50, ptr @j, align 4, !tbaa !11
  %51 = add nsw i32 %35, -1
  store i32 %51, ptr @ab, align 4, !tbaa !11
  br label %52

52:                                               ; preds = %41, %30
  %53 = phi i32 [ %51, %41 ], [ %31, %30 ]
  %54 = phi i32 [ %50, %41 ], [ %32, %30 ]
  %55 = phi i32 [ %51, %41 ], [ %39, %30 ]
  %56 = phi i32 [ %50, %41 ], [ %38, %30 ]
  %57 = phi i32 [ %51, %41 ], [ %35, %30 ]
  %58 = phi i32 [ %50, %41 ], [ %34, %30 ]
  %59 = add nsw i32 %33, 1
  %60 = and i32 %36, 95
  %61 = sub nsw i32 0, %60
  %62 = xor i32 %37, %61
  %63 = icmp sgt i32 %62, 8
  br i1 %63, label %84, label %28

64:                                               ; preds = %81, %78
  %65 = phi i32 [ %53, %81 ], [ %14, %78 ]
  %66 = phi i32 [ %54, %81 ], [ %15, %78 ]
  %67 = phi i32 [ %59, %81 ], [ %16, %78 ]
  %68 = phi i32 [ %55, %81 ], [ %17, %78 ]
  %69 = phi i32 [ %56, %81 ], [ %18, %78 ]
  %70 = phi i32 [ %59, %81 ], [ %19, %78 ]
  %71 = phi i32 [ %62, %81 ], [ %26, %78 ]
  %72 = phi i32 [ %61, %81 ], [ %25, %78 ]
  store i32 %72, ptr @ag, align 4, !tbaa !11
  store i32 %71, ptr @af, align 4, !tbaa !11
  %73 = icmp eq i32 %70, 0
  br i1 %73, label %80, label %74

74:                                               ; preds = %64, %80
  %75 = phi i32 [ %68, %64 ], [ %65, %80 ]
  %76 = phi i32 [ %69, %64 ], [ %66, %80 ]
  %77 = phi i32 [ %70, %64 ], [ %67, %80 ]
  br label %13

78:                                               ; preds = %13
  %79 = icmp eq i32 %21, 0
  br i1 %79, label %64, label %30

80:                                               ; preds = %64
  store i8 %3, ptr @c, align 4, !tbaa !16
  store ptr @.str, ptr @ss, align 8, !tbaa !6
  store i16 %23, ptr @am, align 4, !tbaa !13
  store i32 0, ptr @ao, align 4, !tbaa !11
  br label %74

81:                                               ; preds = %28
  store i32 %59, ptr @f, align 4, !tbaa !11
  br label %64

82:                                               ; preds = %1
  store i32 5, ptr @an, align 4, !tbaa !11
  %83 = add i16 %5, -95
  store i8 %3, ptr @c, align 4, !tbaa !16
  store ptr @.str, ptr @ss, align 8, !tbaa !6
  store i16 %83, ptr @am, align 4, !tbaa !13
  br label %86

84:                                               ; preds = %52
  store i32 5, ptr @an, align 4, !tbaa !11
  store i32 %59, ptr @f, align 4, !tbaa !11
  br label %87

85:                                               ; preds = %13
  store i32 5, ptr @an, align 4, !tbaa !11
  br label %87

86:                                               ; preds = %82, %86
  br label %86

87:                                               ; preds = %84, %85
  %88 = phi i32 [ %62, %84 ], [ %26, %85 ]
  %89 = phi i32 [ %61, %84 ], [ %25, %85 ]
  store i8 %3, ptr @c, align 4, !tbaa !16
  store ptr @.str, ptr @ss, align 8, !tbaa !6
  store i16 %23, ptr @am, align 4, !tbaa !13
  store i32 %89, ptr @ag, align 4, !tbaa !11
  store i32 %88, ptr @af, align 4, !tbaa !11
  store i1 true, ptr @h, align 4
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  %1 = tail call i32 @aw(i32 noundef 1)
  %2 = load i16, ptr getelementptr inbounds nuw (i8, ptr @g, i64 4), align 4
  %3 = and i16 %2, 511
  %4 = icmp eq i16 %3, 5
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  tail call void @abort() #7
  unreachable

6:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { noreturn nounwind }

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
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"short", !9, i64 0}
!15 = !{i64 0, i64 4, !11, i64 4, i64 2, !16, i64 8, i64 4, !11}
!16 = !{!9, !9, i64 0}
