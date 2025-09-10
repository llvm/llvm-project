; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr47337.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr47337.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [2 x i8] c"2\00", align 1
@w = dso_local global ptr @.str, align 8
@a = internal unnamed_addr global [256 x i32] zeroinitializer, align 16
@b = internal unnamed_addr global i32 0, align 4

; Function Attrs: nofree norecurse nounwind memory(readwrite, argmem: read) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load volatile ptr, ptr @w, align 8, !tbaa !6
  %2 = load i8, ptr %1, align 1
  %3 = icmp eq i8 %2, 49
  br i1 %3, label %4, label %8

4:                                                ; preds = %0
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %6 = load i8, ptr %5, align 1
  %7 = icmp eq i8 %6, 0
  br label %8

8:                                                ; preds = %0, %4
  %9 = phi i1 [ false, %0 ], [ %7, %4 ]
  store <4 x i32> splat (i32 1), ptr @a, align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 16), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 32), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 48), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 64), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 80), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 96), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 112), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 128), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 144), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 160), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 176), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 192), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 208), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 224), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 240), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 256), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 272), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 288), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 304), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 320), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 336), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 352), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 368), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 384), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 400), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 416), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 432), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 448), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 464), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 480), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 496), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 512), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 528), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 544), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 560), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 576), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 592), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 608), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 624), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 640), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 656), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 672), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 688), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 704), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 720), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 736), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 752), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 768), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 784), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 800), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 816), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 832), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 848), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 864), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 880), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 896), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 912), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 928), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 944), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 960), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 976), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 992), align 16, !tbaa !11
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @a, i64 1008), align 16, !tbaa !11
  br i1 %9, label %44, label %10

10:                                               ; preds = %8
  %11 = load i32, ptr @b, align 4, !tbaa !11
  %12 = and i32 %11, 1
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds nuw i32, ptr @a, i64 %13
  %15 = load i32, ptr %14, align 4, !tbaa !11
  %16 = and i32 %15, 1
  %17 = zext nneg i32 %16 to i64
  %18 = getelementptr inbounds nuw i32, ptr @a, i64 %17
  %19 = load i32, ptr %18, align 4, !tbaa !11
  %20 = and i32 %19, 1
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds nuw i32, ptr @a, i64 %21
  %23 = load i32, ptr %22, align 4, !tbaa !11
  %24 = and i32 %23, 1
  %25 = zext nneg i32 %24 to i64
  %26 = getelementptr inbounds nuw i32, ptr @a, i64 %25
  %27 = load i32, ptr %26, align 4, !tbaa !11
  %28 = and i32 %27, 1
  %29 = zext nneg i32 %28 to i64
  %30 = getelementptr inbounds nuw i32, ptr @a, i64 %29
  %31 = load i32, ptr %30, align 4, !tbaa !11
  %32 = and i32 %31, 1
  %33 = zext nneg i32 %32 to i64
  %34 = getelementptr inbounds nuw i32, ptr @a, i64 %33
  %35 = load i32, ptr %34, align 4, !tbaa !11
  %36 = and i32 %35, 1
  %37 = zext nneg i32 %36 to i64
  %38 = getelementptr inbounds nuw i32, ptr @a, i64 %37
  %39 = load i32, ptr %38, align 4, !tbaa !11
  %40 = and i32 %39, 1
  %41 = zext nneg i32 %40 to i64
  %42 = getelementptr inbounds nuw i32, ptr @a, i64 %41
  %43 = load i32, ptr %42, align 4, !tbaa !11
  store i32 %43, ptr @b, align 4, !tbaa !11
  br label %44

44:                                               ; preds = %10, %8
  ret i32 0
}

attributes #0 = { nofree norecurse nounwind memory(readwrite, argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
