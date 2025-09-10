; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20040302-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20040302-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@code = dso_local global [5 x i32] [i32 0, i32 0, i32 0, i32 0, i32 1], align 4
@bar.l = internal unnamed_addr constant [2 x ptr] [ptr blockaddress(@bar, %4), ptr blockaddress(@bar, %6)], align 8

; Function Attrs: nofree norecurse nounwind memory(inaccessiblemem: readwrite) uwtable
define dso_local void @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %2)
  store volatile i32 -1, ptr %2, align 4, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %2)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree norecurse nounwind memory(argmem: read, inaccessiblemem: readwrite) uwtable
define dso_local void @bar(ptr noundef readonly captures(none) %0) local_unnamed_addr #2 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %3)
  store volatile i32 -1, ptr %3, align 4, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %3)
  br label %7

4:                                                ; preds = %7
  call void @llvm.lifetime.start.p0(ptr nonnull %2)
  store volatile i32 -1, ptr %2, align 4, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %2)
  %5 = getelementptr inbounds nuw i8, ptr %8, i64 4
  br label %7

6:                                                ; preds = %7
  ret void

7:                                                ; preds = %4, %1
  %8 = phi ptr [ %0, %1 ], [ %5, %4 ]
  %9 = load i32, ptr %8, align 4, !tbaa !6
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds ptr, ptr @bar.l, i64 %10
  %12 = load ptr, ptr %11, align 8, !tbaa !10
  indirectbr ptr %12, [label %4, label %6]
}

; Function Attrs: nofree norecurse nounwind memory(read, argmem: none, inaccessiblemem: readwrite) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  tail call void @bar(ptr noundef nonnull @code)
  ret i32 0
}

attributes #0 = { nofree norecurse nounwind memory(inaccessiblemem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree norecurse nounwind memory(argmem: read, inaccessiblemem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree norecurse nounwind memory(read, argmem: none, inaccessiblemem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
!11 = !{!"any pointer", !8, i64 0}
