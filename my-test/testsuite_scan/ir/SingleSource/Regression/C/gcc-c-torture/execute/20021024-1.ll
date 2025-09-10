; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20021024-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20021024-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@cp = dso_local local_unnamed_addr global ptr null, align 8
@m = dso_local local_unnamed_addr global i64 0, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @bar(i32 noundef %0, ptr noundef captures(none) %1) local_unnamed_addr #1 {
  %3 = lshr i32 %0, 23
  %4 = lshr i32 %0, 9
  %5 = and i32 %4, 511
  %6 = and i32 %0, 511
  %7 = load ptr, ptr @cp, align 8, !tbaa !6
  %8 = zext nneg i32 %5 to i64
  %9 = getelementptr inbounds nuw i64, ptr %1, i64 %8
  %10 = zext nneg i32 %3 to i64
  %11 = getelementptr inbounds nuw i64, ptr %1, i64 %10
  %12 = icmp eq i32 %6, 0
  br i1 %12, label %13, label %17

13:                                               ; preds = %2, %13
  store i64 1, ptr %7, align 8, !tbaa !11
  %14 = load i64, ptr %9, align 8, !tbaa !11
  %15 = load i64, ptr %11, align 8, !tbaa !11
  %16 = add i64 %15, %14
  store i64 %16, ptr @m, align 8, !tbaa !11
  store i64 2, ptr %7, align 8, !tbaa !11
  br label %13

17:                                               ; preds = %2
  store i64 1, ptr %7, align 8, !tbaa !11
  %18 = load i64, ptr %9, align 8, !tbaa !11
  %19 = load i64, ptr %11, align 8, !tbaa !11
  %20 = add i64 %19, %18
  store i64 %20, ptr @m, align 8, !tbaa !11
  store i64 2, ptr %7, align 8, !tbaa !11
  %21 = zext nneg i32 %6 to i64
  %22 = getelementptr inbounds nuw i64, ptr %1, i64 %21
  store i64 1, ptr %22, align 8, !tbaa !11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = alloca i64, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  store ptr %1, ptr @cp, align 8, !tbaa !6
  store i64 58, ptr @m, align 8, !tbaa !11
  store i64 2, ptr %1, align 8, !tbaa !11
  call void @exit(i32 noundef 0) #6
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 long long", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"long long", !9, i64 0}
