; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20060910-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20060910-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.input_ty = type { ptr, ptr }

@b = dso_local global [6 x i8] zeroinitializer, align 1
@s = dso_local local_unnamed_addr global %struct.input_ty zeroinitializer, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @input_getc_complicated(ptr noundef readnone captures(none) %0) local_unnamed_addr #0 {
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local noundef i32 @check_header(ptr noundef captures(none) %0) local_unnamed_addr #1 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load ptr, ptr %2, align 8, !tbaa !6
  %4 = load ptr, ptr %0, align 8, !tbaa !12
  %5 = icmp ult ptr %4, %3
  %6 = zext i1 %5 to i64
  %7 = getelementptr inbounds nuw i8, ptr %4, i64 %6
  %8 = icmp ult ptr %7, %3
  %9 = zext i1 %8 to i64
  %10 = getelementptr inbounds nuw i8, ptr %7, i64 %9
  %11 = or i1 %5, %8
  %12 = icmp ult ptr %10, %3
  %13 = zext i1 %12 to i64
  %14 = getelementptr inbounds nuw i8, ptr %10, i64 %13
  %15 = or i1 %11, %12
  %16 = icmp ult ptr %14, %3
  %17 = zext i1 %16 to i64
  %18 = getelementptr inbounds nuw i8, ptr %14, i64 %17
  %19 = or i1 %15, %16
  %20 = icmp ult ptr %18, %3
  %21 = zext i1 %20 to i64
  %22 = getelementptr inbounds nuw i8, ptr %18, i64 %21
  %23 = or i1 %19, %20
  %24 = icmp ult ptr %22, %3
  %25 = or i1 %23, %24
  br i1 %25, label %26, label %29

26:                                               ; preds = %1
  %27 = zext i1 %24 to i64
  %28 = getelementptr inbounds nuw i8, ptr %22, i64 %27
  store ptr %28, ptr %0, align 8, !tbaa !12
  br label %29

29:                                               ; preds = %1, %26
  ret i32 1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  store ptr getelementptr inbounds nuw (i8, ptr @b, i64 6), ptr getelementptr inbounds nuw (i8, ptr @s, i64 8), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @b, i64 6), ptr @s, align 8, !tbaa !12
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 8}
!7 = !{!"input_ty", !8, i64 0, !8, i64 8}
!8 = !{!"p1 omnipotent char", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!7, !8, i64 0}
