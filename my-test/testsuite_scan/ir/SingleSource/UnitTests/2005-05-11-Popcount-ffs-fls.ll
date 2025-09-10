; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2005-05-11-Popcount-ffs-fls.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2005-05-11-Popcount-ffs-fls.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@nlz10b.table = internal unnamed_addr constant [64 x i8] c" \14\13cc\12c\07\0A\11cc\0Ec\06cc\09c\10cc\01\1Ac\0Dcc\18\05ccc\15c\08\0Bc\0Fcccc\02\1B\00\19c\16c\0Ccc\03\1Cc\17c\04\1Dcc\1E\1F", align 1
@i = dso_local local_unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [54 x i8] c"LLVM: n: %d, clz(n): %d, popcount(n): %d, ctz(n): %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [54 x i8] c"REF : n: %d, clz(n): %d, popcount(n): %d, ctz(n): %d\0A\00", align 1
@.str.3 = private unnamed_addr constant [56 x i8] c"LLVM: n: %lld, clz(n): %d, popcount(n): %d, ctz(n): %d\0A\00", align 1
@.str.4 = private unnamed_addr constant [64 x i8] c"REF LO BITS : n: %lld, clz(n): %d, popcount(n): %d, ctz(n): %d\0A\00", align 1
@.str.5 = private unnamed_addr constant [58 x i8] c"FFS: 0:%d, 1:%d, 2:%d, 7:%d, 1024:%d, 1234:%d i:%d, l:%d\0A\00", align 1
@.str.6 = private unnamed_addr constant [67 x i8] c"__builtin_ffs: 0:%d, 1:%d, 2:%d, 7:%d, 1024:%d, 1234:%d i:%d l:%d\0A\00", align 1
@str.7 = private unnamed_addr constant [8 x i8] c"  ***  \00", align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 256) i32 @nlz10b(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 1
  %3 = or i32 %2, %0
  %4 = lshr i32 %3, 2
  %5 = or i32 %4, %3
  %6 = lshr i32 %5, 4
  %7 = or i32 %6, %5
  %8 = lshr i32 %7, 8
  %9 = or i32 %8, %7
  %10 = lshr i32 %9, 16
  %11 = xor i32 %10, -1
  %12 = and i32 %9, %11
  %13 = mul i32 %12, -42972673
  %14 = lshr i32 %13, 26
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds nuw i8, ptr @nlz10b.table, i64 %15
  %17 = load i8, ptr %16, align 1, !tbaa !6
  %18 = zext i8 %17 to i32
  ret i32 %18
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 288) i32 @nlzll(i64 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i64 %0, 4294967296
  br i1 %2, label %3, label %23

3:                                                ; preds = %1
  %4 = trunc nuw i64 %0 to i32
  %5 = lshr i32 %4, 1
  %6 = or i32 %5, %4
  %7 = lshr i32 %6, 2
  %8 = or i32 %7, %6
  %9 = lshr i32 %8, 4
  %10 = or i32 %9, %8
  %11 = lshr i32 %10, 8
  %12 = or i32 %11, %10
  %13 = lshr i32 %12, 16
  %14 = xor i32 %13, -1
  %15 = and i32 %12, %14
  %16 = mul i32 %15, -42972673
  %17 = lshr i32 %16, 26
  %18 = zext nneg i32 %17 to i64
  %19 = getelementptr inbounds nuw i8, ptr @nlz10b.table, i64 %18
  %20 = load i8, ptr %19, align 1, !tbaa !6
  %21 = zext i8 %20 to i32
  %22 = add nuw nsw i32 %21, 32
  br label %43

23:                                               ; preds = %1
  %24 = lshr i64 %0, 32
  %25 = trunc nuw i64 %24 to i32
  %26 = lshr i32 %25, 1
  %27 = or i32 %26, %25
  %28 = lshr i32 %27, 2
  %29 = or i32 %28, %27
  %30 = lshr i32 %29, 4
  %31 = or i32 %30, %29
  %32 = lshr i32 %31, 8
  %33 = or i32 %32, %31
  %34 = lshr i32 %33, 16
  %35 = xor i32 %34, -1
  %36 = and i32 %33, %35
  %37 = mul i32 %36, -42972673
  %38 = lshr i32 %37, 26
  %39 = zext nneg i32 %38 to i64
  %40 = getelementptr inbounds nuw i8, ptr @nlz10b.table, i64 %39
  %41 = load i8, ptr %40, align 1, !tbaa !6
  %42 = zext i8 %41 to i32
  br label %43

43:                                               ; preds = %23, %3
  %44 = phi i32 [ %22, %3 ], [ %42, %23 ]
  ret i32 %44
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 33) i32 @pop(i32 noundef %0) local_unnamed_addr #0 {
  %2 = tail call range(i32 0, 33) i32 @llvm.ctpop.i32(i32 %0)
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 65) i32 @popll(i64 noundef %0) local_unnamed_addr #0 {
  %2 = insertelement <2 x i64> poison, i64 %0, i64 0
  %3 = shufflevector <2 x i64> %2, <2 x i64> poison, <2 x i32> zeroinitializer
  %4 = lshr <2 x i64> %3, <i64 32, i64 0>
  %5 = trunc <2 x i64> %4 to <2 x i32>
  %6 = tail call range(i32 0, 33) <2 x i32> @llvm.ctpop.v2i32(<2 x i32> %5)
  %7 = shufflevector <2 x i32> %6, <2 x i32> poison, <2 x i32> <i32 1, i32 poison>
  %8 = add nuw nsw <2 x i32> %6, %7
  %9 = extractelement <2 x i32> %8, i64 0
  ret i32 %9
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 33) i32 @ntz8(i32 noundef %0) local_unnamed_addr #0 {
  %2 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %0, i1 false)
  ret i32 %2
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store i32 10, ptr @i, align 4, !tbaa !9
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i32 [ 10, %0 ], [ %31, %1 ]
  %3 = tail call range(i32 0, 33) i32 @llvm.ctlz.i32(i32 %2, i1 false)
  %4 = tail call range(i32 0, 33) i32 @llvm.ctpop.i32(i32 %2)
  %5 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %2, i1 false)
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5)
  %7 = load i32, ptr @i, align 4, !tbaa !9
  %8 = lshr i32 %7, 1
  %9 = or i32 %8, %7
  %10 = lshr i32 %9, 2
  %11 = or i32 %10, %9
  %12 = lshr i32 %11, 4
  %13 = or i32 %12, %11
  %14 = lshr i32 %13, 8
  %15 = or i32 %14, %13
  %16 = lshr i32 %15, 16
  %17 = xor i32 %16, -1
  %18 = and i32 %15, %17
  %19 = mul i32 %18, -42972673
  %20 = lshr i32 %19, 26
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds nuw i8, ptr @nlz10b.table, i64 %21
  %23 = load i8, ptr %22, align 1, !tbaa !6
  %24 = zext i8 %23 to i32
  %25 = tail call range(i32 0, 33) i32 @llvm.ctpop.i32(i32 %7)
  %26 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %7, i1 false)
  %27 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %7, i32 noundef %24, i32 noundef %25, i32 noundef %26)
  %28 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %29 = load i32, ptr @i, align 4, !tbaa !9
  %30 = mul i32 %29, -3
  %31 = add i32 %30, -3
  store i32 %31, ptr @i, align 4, !tbaa !9
  %32 = icmp slt i32 %31, 139045193
  br i1 %32, label %1, label %33, !llvm.loop !11

33:                                               ; preds = %1, %83
  %34 = phi i64 [ %91, %83 ], [ -10000, %1 ]
  %35 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %34, i1 false)
  %36 = trunc nuw nsw i64 %35 to i32
  %37 = tail call range(i64 0, 65) i64 @llvm.ctpop.i64(i64 %34)
  %38 = trunc nuw nsw i64 %37 to i32
  %39 = trunc i64 %34 to i32
  %40 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %39, i1 false)
  %41 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %34, i32 noundef %36, i32 noundef %38, i32 noundef %40)
  %42 = icmp ult i64 %34, 4294967296
  br i1 %42, label %43, label %62

43:                                               ; preds = %33
  %44 = lshr i32 %39, 1
  %45 = or i32 %44, %39
  %46 = lshr i32 %45, 2
  %47 = or i32 %46, %45
  %48 = lshr i32 %47, 4
  %49 = or i32 %48, %47
  %50 = lshr i32 %49, 8
  %51 = or i32 %50, %49
  %52 = lshr i32 %51, 16
  %53 = xor i32 %52, -1
  %54 = and i32 %51, %53
  %55 = mul i32 %54, -42972673
  %56 = lshr i32 %55, 26
  %57 = zext nneg i32 %56 to i64
  %58 = getelementptr inbounds nuw i8, ptr @nlz10b.table, i64 %57
  %59 = load i8, ptr %58, align 1, !tbaa !6
  %60 = zext i8 %59 to i32
  %61 = add nuw nsw i32 %60, 32
  br label %83

62:                                               ; preds = %33
  %63 = lshr i64 %34, 32
  %64 = trunc nuw i64 %63 to i32
  %65 = lshr i32 %64, 1
  %66 = or i32 %65, %64
  %67 = lshr i32 %66, 2
  %68 = or i32 %67, %66
  %69 = lshr i32 %68, 4
  %70 = or i32 %69, %68
  %71 = lshr i32 %70, 8
  %72 = or i32 %71, %70
  %73 = lshr i32 %72, 16
  %74 = xor i32 %73, -1
  %75 = and i32 %72, %74
  %76 = mul i32 %75, -42972673
  %77 = lshr i32 %76, 26
  %78 = zext nneg i32 %77 to i64
  %79 = getelementptr inbounds nuw i8, ptr @nlz10b.table, i64 %78
  %80 = load i8, ptr %79, align 1, !tbaa !6
  %81 = zext i8 %80 to i32
  %82 = tail call range(i32 0, 33) i32 @llvm.ctpop.i32(i32 %64)
  br label %83

83:                                               ; preds = %43, %62
  %84 = phi i32 [ 0, %43 ], [ %82, %62 ]
  %85 = phi i32 [ %61, %43 ], [ %81, %62 ]
  %86 = tail call range(i32 0, 33) i32 @llvm.ctpop.i32(i32 %39)
  %87 = add nuw nsw i32 %84, %86
  %88 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i64 noundef %34, i32 noundef %85, i32 noundef %87, i32 noundef %40)
  %89 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %90 = mul i64 %34, -3
  %91 = add i64 %90, -3
  %92 = icmp slt i64 %91, 1390451930000
  br i1 %92, label %33, label %93, !llvm.loop !13

93:                                               ; preds = %83
  %94 = load i32, ptr @i, align 4, !tbaa !9
  %95 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %94, i1 true)
  %96 = add nuw nsw i32 %95, 1
  %97 = icmp eq i32 %94, 0
  %98 = select i1 %97, i32 0, i32 %96
  %99 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 1, i32 noundef 11, i32 noundef 2, i32 noundef %98, i32 noundef 1)
  %100 = load i32, ptr @i, align 4, !tbaa !9
  %101 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %100, i1 true)
  %102 = add nuw nsw i32 %101, 1
  %103 = icmp eq i32 %100, 0
  %104 = select i1 %103, i32 0, i32 %102
  %105 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 1, i32 noundef 11, i32 noundef 2, i32 noundef %104, i32 noundef 1)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.ctlz.i32(i32, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.ctpop.i32(i32) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.cttz.i32(i32, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.ctlz.i64(i64, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.ctpop.i64(i64) #3

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x i32> @llvm.ctpop.v2i32(<2 x i32>) #5

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nofree nounwind }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = distinct !{!13, !12}
