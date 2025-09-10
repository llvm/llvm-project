; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20081218-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20081218-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.A = type { i32, i32, [512 x i8] }

@a = dso_local local_unnamed_addr global %struct.A zeroinitializer, align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @foo() local_unnamed_addr #0 {
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(520) @a, i8 38, i64 520, i1 false)
  ret i32 640034342
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @bar() local_unnamed_addr #0 {
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(520) @a, i8 54, i64 520, i1 false)
  store i32 909588022, ptr getelementptr inbounds nuw (i8, ptr @a, i64 4), align 4, !tbaa !6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = tail call i32 @foo()
  %2 = icmp eq i32 %1, 640034342
  br i1 %2, label %3, label %15

3:                                                ; preds = %0, %3
  %4 = phi i64 [ %9, %3 ], [ 0, %0 ]
  %5 = getelementptr inbounds nuw i8, ptr @a, i64 %4
  %6 = load <16 x i8>, ptr %5, align 4, !tbaa !11
  %7 = freeze <16 x i8> %6
  %8 = icmp ne <16 x i8> %7, splat (i8 38)
  %9 = add nuw i64 %4, 16
  %10 = bitcast <16 x i1> %8 to i16
  %11 = icmp ne i16 %10, 0
  %12 = icmp eq i64 %9, 512
  %13 = or i1 %11, %12
  br i1 %13, label %14, label %3, !llvm.loop !12

14:                                               ; preds = %3
  br i1 %11, label %43, label %19

15:                                               ; preds = %0
  tail call void @abort() #4
  unreachable

16:                                               ; preds = %19
  tail call void @bar()
  %17 = load i32, ptr getelementptr inbounds nuw (i8, ptr @a, i64 4), align 4, !tbaa !6
  %18 = icmp eq i32 %17, 909588022
  br i1 %18, label %45, label %44

19:                                               ; preds = %14
  %20 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 512), align 4, !tbaa !11
  %21 = icmp eq i8 %20, 38
  %22 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 513), align 1
  %23 = icmp eq i8 %22, 38
  %24 = select i1 %21, i1 %23, i1 false
  %25 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 514), align 2
  %26 = icmp eq i8 %25, 38
  %27 = select i1 %24, i1 %26, i1 false
  %28 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 515), align 1
  %29 = icmp eq i8 %28, 38
  %30 = select i1 %27, i1 %29, i1 false
  %31 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 516), align 4
  %32 = icmp eq i8 %31, 38
  %33 = select i1 %30, i1 %32, i1 false
  %34 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 517), align 1
  %35 = icmp eq i8 %34, 38
  %36 = select i1 %33, i1 %35, i1 false
  %37 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 518), align 2
  %38 = icmp eq i8 %37, 38
  %39 = select i1 %36, i1 %38, i1 false
  %40 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 519), align 1
  %41 = icmp eq i8 %40, 38
  %42 = select i1 %39, i1 %41, i1 false
  br i1 %42, label %16, label %43

43:                                               ; preds = %19, %14
  tail call void @abort() #4
  unreachable

44:                                               ; preds = %16
  tail call void @abort() #4
  unreachable

45:                                               ; preds = %16
  store i32 909522486, ptr getelementptr inbounds nuw (i8, ptr @a, i64 4), align 4, !tbaa !6
  br label %46

46:                                               ; preds = %46, %45
  %47 = phi i64 [ 0, %45 ], [ %52, %46 ]
  %48 = getelementptr inbounds nuw i8, ptr @a, i64 %47
  %49 = load <16 x i8>, ptr %48, align 4, !tbaa !11
  %50 = freeze <16 x i8> %49
  %51 = icmp ne <16 x i8> %50, splat (i8 54)
  %52 = add nuw i64 %47, 16
  %53 = bitcast <16 x i1> %51 to i16
  %54 = icmp ne i16 %53, 0
  %55 = icmp eq i64 %52, 512
  %56 = or i1 %54, %55
  br i1 %56, label %57, label %46, !llvm.loop !16

57:                                               ; preds = %46
  br i1 %54, label %83, label %59

58:                                               ; preds = %59
  ret i32 0

59:                                               ; preds = %57
  %60 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 512), align 4, !tbaa !11
  %61 = icmp eq i8 %60, 54
  %62 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 513), align 1
  %63 = icmp eq i8 %62, 54
  %64 = select i1 %61, i1 %63, i1 false
  %65 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 514), align 2
  %66 = icmp eq i8 %65, 54
  %67 = select i1 %64, i1 %66, i1 false
  %68 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 515), align 1
  %69 = icmp eq i8 %68, 54
  %70 = select i1 %67, i1 %69, i1 false
  %71 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 516), align 4
  %72 = icmp eq i8 %71, 54
  %73 = select i1 %70, i1 %72, i1 false
  %74 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 517), align 1
  %75 = icmp eq i8 %74, 54
  %76 = select i1 %73, i1 %75, i1 false
  %77 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 518), align 2
  %78 = icmp eq i8 %77, 54
  %79 = select i1 %76, i1 %78, i1 false
  %80 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 519), align 1
  %81 = icmp eq i8 %80, 54
  %82 = select i1 %79, i1 %81, i1 false
  br i1 %82, label %58, label %83

83:                                               ; preds = %59, %57
  tail call void @abort() #4
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 4}
!7 = !{!"A", !8, i64 0, !8, i64 4, !9, i64 8}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!9, !9, i64 0}
!12 = distinct !{!12, !13, !14, !15}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!"llvm.loop.isvectorized", i32 1}
!15 = !{!"llvm.loop.unroll.runtime.disable"}
!16 = distinct !{!16, !13, !14, !15}
