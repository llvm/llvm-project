; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr78170.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr78170.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S0 = type <{ i32, i32, i32, i32, i32, i64 }>

@b = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@__const.fn1.e = private unnamed_addr constant { i32, i32, i32, i32, i32, i8, i8, i8, i8, i8, i8, i8, i8 } { i32 0, i32 0, i32 0, i32 0, i32 0, i8 0, i8 -128, i8 0, i8 0, i8 4, i8 0, i8 0, i8 0 }, align 4
@c = dso_local local_unnamed_addr global %struct.S0 zeroinitializer, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1() local_unnamed_addr #0 {
  store i32 1, ptr @b, align 4, !tbaa !6
  store i32 1, ptr @d, align 4, !tbaa !6
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3, !llvm.loop !10

3:                                                ; preds = %0
  store i64 21474803712, ptr getelementptr inbounds nuw (i8, ptr @c, i64 20), align 4
  br label %5

4:                                                ; preds = %0
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(28) @c, ptr noundef nonnull align 4 dereferenceable(28) @__const.fn1.e, i64 20, i1 false), !tbaa.struct !12
  store i64 21474803712, ptr getelementptr inbounds nuw (i8, ptr @c, i64 20), align 4
  store i32 0, ptr @b, align 4, !tbaa !6
  ret void

5:                                                ; preds = %3, %5
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(28) @c, ptr noundef nonnull align 4 dereferenceable(28) @__const.fn1.e, i64 20, i1 false), !tbaa.struct !12
  br label %5
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  store i32 1, ptr @b, align 4, !tbaa !6
  store i32 1, ptr @d, align 4, !tbaa !6
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %5, label %3, !llvm.loop !10

3:                                                ; preds = %0
  store i64 21474803712, ptr getelementptr inbounds nuw (i8, ptr @c, i64 20), align 4
  br label %4

4:                                                ; preds = %4, %3
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(28) @c, ptr noundef nonnull align 4 dereferenceable(28) @__const.fn1.e, i64 20, i1 false), !tbaa.struct !12
  br label %4

5:                                                ; preds = %0
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(28) @c, ptr noundef nonnull align 4 dereferenceable(28) @__const.fn1.e, i64 20, i1 false), !tbaa.struct !12
  store i64 21474803712, ptr getelementptr inbounds nuw (i8, ptr @c, i64 20), align 4
  store i32 0, ptr @b, align 4, !tbaa !6
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }

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
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{i64 0, i64 4, !6, i64 4, i64 4, !6, i64 8, i64 4, !6, i64 12, i64 4, !6, i64 16, i64 4, !6, i64 20, i64 8, !13}
!13 = !{!8, !8, i64 0}
