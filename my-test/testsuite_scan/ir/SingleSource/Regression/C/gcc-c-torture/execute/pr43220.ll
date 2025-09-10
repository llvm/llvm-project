; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr43220.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr43220.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@p = dso_local global ptr null, align 8

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i32 [ 0, %0 ], [ %18, %1 ]
  %3 = urem i32 %2, 1000
  %4 = or disjoint i32 %3, 1
  %5 = zext nneg i32 %4 to i64
  %6 = call ptr @llvm.stacksave.p0()
  %7 = alloca i32, i64 %5, align 4
  store i32 1, ptr %7, align 4, !tbaa !6
  %8 = zext nneg i32 %3 to i64
  %9 = getelementptr inbounds nuw i32, ptr %7, i64 %8
  store i32 2, ptr %9, align 4, !tbaa !6
  store volatile ptr %7, ptr @p, align 8, !tbaa !10
  %10 = or disjoint i32 %2, 1
  call void @llvm.stackrestore.p0(ptr %6)
  %11 = urem i32 %10, 1000
  %12 = add nuw nsw i32 %11, 1
  %13 = zext nneg i32 %12 to i64
  %14 = call ptr @llvm.stacksave.p0()
  %15 = alloca i32, i64 %13, align 4
  store i32 1, ptr %15, align 4, !tbaa !6
  %16 = zext nneg i32 %11 to i64
  %17 = getelementptr inbounds nuw i32, ptr %15, i64 %16
  store i32 2, ptr %17, align 4, !tbaa !6
  store volatile ptr %15, ptr @p, align 8, !tbaa !10
  %18 = add nuw nsw i32 %2, 2
  call void @llvm.stackrestore.p0(ptr %14)
  %19 = icmp samesign ult i32 %2, 999998
  br i1 %19, label %1, label %20

20:                                               ; preds = %1
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #1

attributes #0 = { nofree norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn }

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
