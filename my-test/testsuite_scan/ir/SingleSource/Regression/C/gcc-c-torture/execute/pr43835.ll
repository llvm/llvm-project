; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr43835.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr43835.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.Pcc_cell = type { ptr, i64, i64 }

; Function Attrs: cold nofree noinline noreturn nounwind uwtable
define dso_local void @Parrot_gc_mark_PMC_alive_fun(ptr readnone captures(none) %0, ptr readnone captures(none) %1) local_unnamed_addr #0 {
  tail call void @abort() #5
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(ptr noundef nonnull readnone captures(none) %0, ptr noundef nonnull readonly captures(none) %1) local_unnamed_addr #2 {
  tail call fastcc void @mark_cell(ptr noundef %0, ptr noundef %1)
  ret void
}

; Function Attrs: nofree noinline nounwind uwtable
define internal fastcc void @mark_cell(ptr noundef nonnull readnone captures(none) %0, ptr noundef nonnull readonly captures(none) %1) unnamed_addr #3 {
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %4 = load i64, ptr %3, align 8, !tbaa !6
  %5 = icmp eq i64 %4, 4
  br i1 %5, label %6, label %14

6:                                                ; preds = %2
  %7 = load ptr, ptr %1, align 8, !tbaa !13
  %8 = icmp eq ptr %7, null
  br i1 %8, label %14, label %9

9:                                                ; preds = %6
  %10 = load i32, ptr %7, align 4, !tbaa !14
  %11 = and i32 %10, 262144
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %13, label %14

13:                                               ; preds = %9
  tail call void @Parrot_gc_mark_PMC_alive_fun(ptr nonnull poison, ptr nonnull poison)
  unreachable

14:                                               ; preds = %9, %6, %2
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.Pcc_cell, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  store ptr null, ptr %2, align 8, !tbaa !13
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store <2 x i64> <i64 42, i64 4>, ptr %3, align 8, !tbaa !17
  call fastcc void @mark_cell(ptr noundef nonnull readnone %1, ptr noundef nonnull readonly %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #4

attributes #0 = { cold nofree noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { noreturn nounwind }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !12, i64 16}
!7 = !{!"Pcc_cell", !8, i64 0, !12, i64 8, !12, i64 16}
!8 = !{!"p1 _ZTS3PMC", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!"long", !10, i64 0}
!13 = !{!7, !8, i64 0}
!14 = !{!15, !16, i64 0}
!15 = !{!"PMC", !16, i64 0}
!16 = !{!"int", !10, i64 0}
!17 = !{!12, !12, i64 0}
