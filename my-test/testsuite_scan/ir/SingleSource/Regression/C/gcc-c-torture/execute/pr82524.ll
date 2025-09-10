; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr82524.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr82524.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%union.U = type { i32 }

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local range(i32 0, 16777216) i32 @bar(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 3
  %4 = load i8, ptr %3, align 1, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 3
  %6 = load i8, ptr %5, align 1, !tbaa !6
  %7 = xor i8 %4, -1
  %8 = zext i8 %6 to i16
  %9 = add nuw nsw i16 %8, 1
  %10 = zext i8 %7 to i16
  %11 = mul nuw i16 %9, %10
  %12 = lshr i16 %11, 8
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 2
  %14 = load i8, ptr %13, align 2, !tbaa !6
  %15 = zext i8 %14 to i16
  %16 = add nuw nsw i16 %15, 1
  %17 = zext i8 %4 to i16
  %18 = mul nuw i16 %16, %17
  %19 = lshr i16 %18, 8
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 2
  %21 = load i8, ptr %20, align 2, !tbaa !6
  %22 = zext i8 %21 to i16
  %23 = add nuw nsw i16 %22, 1
  %24 = mul nuw i16 %23, %12
  %25 = lshr i16 %24, 8
  %26 = add nuw nsw i16 %25, %19
  %27 = getelementptr inbounds nuw i8, ptr %0, i64 1
  %28 = load i8, ptr %27, align 1, !tbaa !6
  %29 = zext i8 %28 to i16
  %30 = add nuw nsw i16 %29, 1
  %31 = mul nuw i16 %30, %17
  %32 = and i16 %31, -256
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %34 = load i8, ptr %33, align 1, !tbaa !6
  %35 = zext i8 %34 to i16
  %36 = add nuw nsw i16 %35, 1
  %37 = mul nuw i16 %36, %12
  %38 = load i8, ptr %0, align 4, !tbaa !6
  %39 = zext i8 %38 to i16
  %40 = add nuw nsw i16 %39, 1
  %41 = mul nuw i16 %40, %17
  %42 = lshr i16 %41, 8
  %43 = load i8, ptr %1, align 4, !tbaa !6
  %44 = zext i8 %43 to i16
  %45 = add nuw nsw i16 %44, 1
  %46 = mul nuw i16 %45, %12
  %47 = lshr i16 %46, 8
  %48 = add nuw nsw i16 %47, %42
  %49 = and i16 %26, 255
  %50 = zext nneg i16 %49 to i32
  %51 = shl nuw nsw i32 %50, 16
  %52 = add i16 %37, %32
  %53 = and i16 %52, -256
  %54 = zext i16 %53 to i32
  %55 = or disjoint i32 %51, %54
  %56 = and i16 %48, 255
  %57 = zext nneg i16 %56 to i32
  %58 = or disjoint i32 %55, %57
  ret i32 %58
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca %union.U, align 4
  %2 = alloca %union.U, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #4
  store <4 x i8> <i8 -1, i8 -1, i8 -1, i8 0>, ptr %1, align 4, !tbaa !6
  store i32 -1, ptr %2, align 4
  %3 = call i32 @bar(ptr noundef nonnull %1, ptr noundef nonnull %2)
  %4 = icmp eq i32 %3, 16777215
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  tail call void @abort() #5
  unreachable

6:                                                ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #4
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
