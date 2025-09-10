; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68532.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68532.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@in = dso_local global [128 x i16] zeroinitializer, align 16

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local range(i32 0, 65536) i32 @test(i16 noundef %0, ptr noundef readonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = trunc i32 %2 to i16
  %5 = load i16, ptr %1, align 2, !tbaa !6
  %6 = mul i16 %5, %4
  %7 = add i16 %6, %0
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %9 = load i16, ptr %8, align 2, !tbaa !6
  %10 = mul i16 %9, %4
  %11 = add i16 %10, %7
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %13 = load i16, ptr %12, align 2, !tbaa !6
  %14 = mul i16 %13, %4
  %15 = add i16 %14, %11
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %17 = load i16, ptr %16, align 2, !tbaa !6
  %18 = mul i16 %17, %4
  %19 = add i16 %18, %15
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %21 = load i16, ptr %20, align 2, !tbaa !6
  %22 = mul i16 %21, %4
  %23 = add i16 %22, %19
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %25 = load i16, ptr %24, align 2, !tbaa !6
  %26 = mul i16 %25, %4
  %27 = add i16 %26, %23
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %29 = load i16, ptr %28, align 2, !tbaa !6
  %30 = mul i16 %29, %4
  %31 = add i16 %30, %27
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %33 = load i16, ptr %32, align 2, !tbaa !6
  %34 = mul i16 %33, %4
  %35 = add i16 %34, %31
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %37 = load i16, ptr %36, align 2, !tbaa !6
  %38 = mul i16 %37, %4
  %39 = add i16 %38, %35
  %40 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %41 = load i16, ptr %40, align 2, !tbaa !6
  %42 = mul i16 %41, %4
  %43 = add i16 %42, %39
  %44 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %45 = load i16, ptr %44, align 2, !tbaa !6
  %46 = mul i16 %45, %4
  %47 = add i16 %46, %43
  %48 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %49 = load i16, ptr %48, align 2, !tbaa !6
  %50 = mul i16 %49, %4
  %51 = add i16 %50, %47
  %52 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %53 = load i16, ptr %52, align 2, !tbaa !6
  %54 = mul i16 %53, %4
  %55 = add i16 %54, %51
  %56 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %57 = load i16, ptr %56, align 2, !tbaa !6
  %58 = mul i16 %57, %4
  %59 = add i16 %58, %55
  %60 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %61 = load i16, ptr %60, align 2, !tbaa !6
  %62 = mul i16 %61, %4
  %63 = add i16 %62, %59
  %64 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %65 = load i16, ptr %64, align 2, !tbaa !6
  %66 = mul i16 %65, %4
  %67 = add i16 %66, %63
  %68 = zext i16 %67 to i32
  ret i32 %68
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, ptr @in, align 16, !tbaa !6
  store <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 16), align 16, !tbaa !6
  store <8 x i16> <i16 16, i16 17, i16 18, i16 19, i16 20, i16 21, i16 22, i16 23>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 32), align 16, !tbaa !6
  store <8 x i16> <i16 24, i16 25, i16 26, i16 27, i16 28, i16 29, i16 30, i16 31>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 48), align 16, !tbaa !6
  store <8 x i16> <i16 32, i16 33, i16 34, i16 35, i16 36, i16 37, i16 38, i16 39>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 64), align 16, !tbaa !6
  store <8 x i16> <i16 40, i16 41, i16 42, i16 43, i16 44, i16 45, i16 46, i16 47>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 80), align 16, !tbaa !6
  store <8 x i16> <i16 48, i16 49, i16 50, i16 51, i16 52, i16 53, i16 54, i16 55>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 96), align 16, !tbaa !6
  store <8 x i16> <i16 56, i16 57, i16 58, i16 59, i16 60, i16 61, i16 62, i16 63>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 112), align 16, !tbaa !6
  store <8 x i16> <i16 64, i16 65, i16 66, i16 67, i16 68, i16 69, i16 70, i16 71>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 128), align 16, !tbaa !6
  store <8 x i16> <i16 72, i16 73, i16 74, i16 75, i16 76, i16 77, i16 78, i16 79>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 144), align 16, !tbaa !6
  store <8 x i16> <i16 80, i16 81, i16 82, i16 83, i16 84, i16 85, i16 86, i16 87>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 160), align 16, !tbaa !6
  store <8 x i16> <i16 88, i16 89, i16 90, i16 91, i16 92, i16 93, i16 94, i16 95>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 176), align 16, !tbaa !6
  store <8 x i16> <i16 96, i16 97, i16 98, i16 99, i16 100, i16 101, i16 102, i16 103>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 192), align 16, !tbaa !6
  store <8 x i16> <i16 104, i16 105, i16 106, i16 107, i16 108, i16 109, i16 110, i16 111>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 208), align 16, !tbaa !6
  store <8 x i16> <i16 112, i16 113, i16 114, i16 115, i16 116, i16 117, i16 118, i16 119>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 224), align 16, !tbaa !6
  store <8 x i16> <i16 120, i16 121, i16 122, i16 123, i16 124, i16 125, i16 126, i16 127>, ptr getelementptr inbounds nuw (i8, ptr @in, i64 240), align 16, !tbaa !6
  %1 = tail call i32 @test(i16 noundef 0, ptr noundef nonnull @in, i32 noundef 1)
  %2 = icmp eq i32 %1, 960
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

4:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

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
