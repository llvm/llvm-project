; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr84524.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr84524.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: write) uwtable
define dso_local void @foo(ptr noundef writeonly captures(none) %0) local_unnamed_addr #0 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi i64 [ 0, %1 ], [ %39, %2 ]
  %4 = phi <8 x i16> [ <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, %1 ], [ %40, %2 ]
  %5 = shl <8 x i16> %4, splat (i16 9)
  %6 = xor <8 x i16> %5, splat (i16 4129)
  %7 = and <8 x i16> %4, splat (i16 128)
  %8 = icmp eq <8 x i16> %7, zeroinitializer
  %9 = select <8 x i1> %8, <8 x i16> %5, <8 x i16> %6
  %10 = shl <8 x i16> %9, splat (i16 1)
  %11 = xor <8 x i16> %10, splat (i16 4129)
  %12 = icmp slt <8 x i16> %9, zeroinitializer
  %13 = select <8 x i1> %12, <8 x i16> %11, <8 x i16> %10
  %14 = shl <8 x i16> %13, splat (i16 1)
  %15 = xor <8 x i16> %14, splat (i16 4129)
  %16 = icmp slt <8 x i16> %13, zeroinitializer
  %17 = select <8 x i1> %16, <8 x i16> %15, <8 x i16> %14
  %18 = shl <8 x i16> %17, splat (i16 1)
  %19 = xor <8 x i16> %18, splat (i16 4129)
  %20 = icmp slt <8 x i16> %17, zeroinitializer
  %21 = select <8 x i1> %20, <8 x i16> %19, <8 x i16> %18
  %22 = shl <8 x i16> %21, splat (i16 1)
  %23 = xor <8 x i16> %22, splat (i16 4129)
  %24 = icmp slt <8 x i16> %21, zeroinitializer
  %25 = select <8 x i1> %24, <8 x i16> %23, <8 x i16> %22
  %26 = shl <8 x i16> %25, splat (i16 1)
  %27 = xor <8 x i16> %26, splat (i16 4129)
  %28 = icmp slt <8 x i16> %25, zeroinitializer
  %29 = select <8 x i1> %28, <8 x i16> %27, <8 x i16> %26
  %30 = shl <8 x i16> %29, splat (i16 1)
  %31 = xor <8 x i16> %30, splat (i16 4129)
  %32 = icmp slt <8 x i16> %29, zeroinitializer
  %33 = select <8 x i1> %32, <8 x i16> %31, <8 x i16> %30
  %34 = shl <8 x i16> %33, splat (i16 1)
  %35 = xor <8 x i16> %34, splat (i16 4129)
  %36 = icmp slt <8 x i16> %33, zeroinitializer
  %37 = select <8 x i1> %36, <8 x i16> %35, <8 x i16> %34
  %38 = getelementptr inbounds nuw i16, ptr %0, i64 %3
  store <8 x i16> %37, ptr %38, align 2, !tbaa !6
  %39 = add nuw i64 %3, 8
  %40 = add <8 x i16> %4, splat (i16 8)
  %41 = icmp eq i64 %39, 256
  br i1 %41, label %42, label %2, !llvm.loop !10

42:                                               ; preds = %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca [256 x i16], align 2
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  br label %2

2:                                                ; preds = %2, %0
  %3 = phi i64 [ 0, %0 ], [ %39, %2 ]
  %4 = phi <8 x i16> [ <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, %0 ], [ %40, %2 ]
  %5 = shl <8 x i16> %4, splat (i16 9)
  %6 = xor <8 x i16> %5, splat (i16 4129)
  %7 = and <8 x i16> %4, splat (i16 128)
  %8 = icmp eq <8 x i16> %7, zeroinitializer
  %9 = select <8 x i1> %8, <8 x i16> %5, <8 x i16> %6
  %10 = shl <8 x i16> %9, splat (i16 1)
  %11 = xor <8 x i16> %10, splat (i16 4129)
  %12 = icmp slt <8 x i16> %9, zeroinitializer
  %13 = select <8 x i1> %12, <8 x i16> %11, <8 x i16> %10
  %14 = shl <8 x i16> %13, splat (i16 1)
  %15 = xor <8 x i16> %14, splat (i16 4129)
  %16 = icmp slt <8 x i16> %13, zeroinitializer
  %17 = select <8 x i1> %16, <8 x i16> %15, <8 x i16> %14
  %18 = shl <8 x i16> %17, splat (i16 1)
  %19 = xor <8 x i16> %18, splat (i16 4129)
  %20 = icmp slt <8 x i16> %17, zeroinitializer
  %21 = select <8 x i1> %20, <8 x i16> %19, <8 x i16> %18
  %22 = shl <8 x i16> %21, splat (i16 1)
  %23 = xor <8 x i16> %22, splat (i16 4129)
  %24 = icmp slt <8 x i16> %21, zeroinitializer
  %25 = select <8 x i1> %24, <8 x i16> %23, <8 x i16> %22
  %26 = shl <8 x i16> %25, splat (i16 1)
  %27 = xor <8 x i16> %26, splat (i16 4129)
  %28 = icmp slt <8 x i16> %25, zeroinitializer
  %29 = select <8 x i1> %28, <8 x i16> %27, <8 x i16> %26
  %30 = shl <8 x i16> %29, splat (i16 1)
  %31 = xor <8 x i16> %30, splat (i16 4129)
  %32 = icmp slt <8 x i16> %29, zeroinitializer
  %33 = select <8 x i1> %32, <8 x i16> %31, <8 x i16> %30
  %34 = shl <8 x i16> %33, splat (i16 1)
  %35 = xor <8 x i16> %34, splat (i16 4129)
  %36 = icmp slt <8 x i16> %33, zeroinitializer
  %37 = select <8 x i1> %36, <8 x i16> %35, <8 x i16> %34
  %38 = getelementptr inbounds nuw i16, ptr %1, i64 %3
  store <8 x i16> %37, ptr %38, align 2, !tbaa !6
  %39 = add nuw i64 %3, 8
  %40 = add <8 x i16> %4, splat (i16 8)
  %41 = icmp eq i64 %39, 256
  br i1 %41, label %46, label %2, !llvm.loop !14

42:                                               ; preds = %46
  %43 = add nuw nsw i64 %47, 1
  %44 = icmp eq i64 %43, 256
  br i1 %44, label %45, label %46, !llvm.loop !15

45:                                               ; preds = %42
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 0

46:                                               ; preds = %2, %42
  %47 = phi i64 [ %43, %42 ], [ 0, %2 ]
  %48 = trunc i64 %47 to i16
  %49 = shl nuw i16 %48, 8
  %50 = tail call i16 asm sideeffect "", "=r,0"(i16 %49) #4, !srcloc !16
  %51 = shl i16 %50, 1
  %52 = xor i16 %51, 4129
  %53 = icmp slt i16 %50, 0
  %54 = select i1 %53, i16 %52, i16 %51
  %55 = tail call i16 asm sideeffect "", "=r,0"(i16 %54) #4, !srcloc !16
  %56 = shl i16 %55, 1
  %57 = xor i16 %56, 4129
  %58 = icmp slt i16 %55, 0
  %59 = select i1 %58, i16 %57, i16 %56
  %60 = tail call i16 asm sideeffect "", "=r,0"(i16 %59) #4, !srcloc !16
  %61 = shl i16 %60, 1
  %62 = xor i16 %61, 4129
  %63 = icmp slt i16 %60, 0
  %64 = select i1 %63, i16 %62, i16 %61
  %65 = tail call i16 asm sideeffect "", "=r,0"(i16 %64) #4, !srcloc !16
  %66 = shl i16 %65, 1
  %67 = xor i16 %66, 4129
  %68 = icmp slt i16 %65, 0
  %69 = select i1 %68, i16 %67, i16 %66
  %70 = tail call i16 asm sideeffect "", "=r,0"(i16 %69) #4, !srcloc !16
  %71 = shl i16 %70, 1
  %72 = xor i16 %71, 4129
  %73 = icmp slt i16 %70, 0
  %74 = select i1 %73, i16 %72, i16 %71
  %75 = tail call i16 asm sideeffect "", "=r,0"(i16 %74) #4, !srcloc !16
  %76 = shl i16 %75, 1
  %77 = xor i16 %76, 4129
  %78 = icmp slt i16 %75, 0
  %79 = select i1 %78, i16 %77, i16 %76
  %80 = tail call i16 asm sideeffect "", "=r,0"(i16 %79) #4, !srcloc !16
  %81 = shl i16 %80, 1
  %82 = xor i16 %81, 4129
  %83 = icmp slt i16 %80, 0
  %84 = select i1 %83, i16 %82, i16 %81
  %85 = tail call i16 asm sideeffect "", "=r,0"(i16 %84) #4, !srcloc !16
  %86 = shl i16 %85, 1
  %87 = xor i16 %86, 4129
  %88 = icmp slt i16 %85, 0
  %89 = select i1 %88, i16 %87, i16 %86
  %90 = getelementptr inbounds nuw i16, ptr %1, i64 %47
  %91 = load i16, ptr %90, align 2, !tbaa !6
  %92 = icmp eq i16 %91, %89
  br i1 %92, label %42, label %93

93:                                               ; preds = %46
  tail call void @abort() #5
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"short", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11, !12, !13}
!15 = distinct !{!15, !11}
!16 = !{i64 468}
