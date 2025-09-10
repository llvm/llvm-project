; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20111208-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20111208-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree norecurse nounwind memory(write, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 256) i32 @pack_unpack(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(address) %1) local_unnamed_addr #0 {
  %3 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %1) #4
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 %3
  %5 = icmp eq i64 %3, 0
  br i1 %5, label %24, label %6

6:                                                ; preds = %2, %21
  %7 = phi ptr [ %22, %21 ], [ %0, %2 ]
  %8 = phi ptr [ %9, %21 ], [ %1, %2 ]
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 1
  %10 = load i8, ptr %8, align 1, !tbaa !6
  switch i8 %10, label %21 [
    i8 115, label %11
    i8 108, label %15
  ]

11:                                               ; preds = %6
  %12 = load i16, ptr %7, align 1
  %13 = getelementptr inbounds nuw i8, ptr %7, i64 2
  %14 = sext i16 %12 to i32
  br label %18

15:                                               ; preds = %6
  %16 = load i32, ptr %7, align 1
  %17 = getelementptr inbounds nuw i8, ptr %7, i64 4
  br label %18

18:                                               ; preds = %15, %11
  %19 = phi i32 [ %14, %11 ], [ %16, %15 ]
  %20 = phi ptr [ %13, %11 ], [ %17, %15 ]
  tail call fastcc void @do_something(i32 noundef %19)
  br label %21

21:                                               ; preds = %18, %6
  %22 = phi ptr [ %7, %6 ], [ %20, %18 ]
  %23 = icmp ult ptr %9, %4
  br i1 %23, label %6, label %24, !llvm.loop !9

24:                                               ; preds = %21, %2
  %25 = phi ptr [ %0, %2 ], [ %22, %21 ]
  %26 = load i8, ptr %25, align 1, !tbaa !6
  %27 = zext i8 %26 to i32
  ret i32 %27
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define internal fastcc void @do_something(i32 noundef %0) unnamed_addr #2 {
  store i32 %0, ptr @a, align 4, !tbaa !11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  tail call fastcc void @do_something(i32 noundef 384)
  tail call fastcc void @do_something(i32 noundef -1071776001)
  ret i32 0
}

attributes #0 = { nofree norecurse nounwind memory(write, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind willreturn memory(read) }

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
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !7, i64 0}
