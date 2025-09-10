; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/bswap-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/bswap-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2139062144) i32 @partial_read_le32(i64 %0) local_unnamed_addr #0 {
  %2 = trunc i64 %0 to i32
  %3 = and i32 %2, 2139062143
  ret i32 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2139095040) i32 @partial_read_be32(i64 %0) local_unnamed_addr #0 {
  %2 = trunc i64 %0 to i32
  %3 = insertelement <4 x i32> poison, i32 %2, i64 0
  %4 = shufflevector <4 x i32> %3, <4 x i32> poison, <4 x i32> zeroinitializer
  %5 = lshr <4 x i32> %4, <i32 24, i32 8, i32 8, i32 24>
  %6 = shl <4 x i32> %4, <i32 24, i32 8, i32 8, i32 24>
  %7 = shufflevector <4 x i32> %5, <4 x i32> %6, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %8 = and <4 x i32> %7, <i32 127, i32 32512, i32 8323072, i32 2130706432>
  %9 = tail call i32 @llvm.vector.reduce.or.v4i32(<4 x i32> %8)
  ret i32 %9
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local i32 @fake_read_le32(ptr noundef readonly captures(none) %0, ptr noundef writeonly captures(none) initializes((0, 1)) %1) local_unnamed_addr #1 {
  %3 = load i8, ptr %0, align 1, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 1
  %5 = load i8, ptr %4, align 1, !tbaa !6
  store i8 1, ptr %1, align 1, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 2
  %7 = load i8, ptr %6, align 1, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 3
  %9 = load i8, ptr %8, align 1, !tbaa !6
  %10 = zext i8 %3 to i32
  %11 = zext i8 %5 to i32
  %12 = shl nuw nsw i32 %11, 8
  %13 = or disjoint i32 %12, %10
  %14 = zext i8 %7 to i32
  %15 = shl nuw nsw i32 %14, 16
  %16 = or disjoint i32 %13, %15
  %17 = zext i8 %9 to i32
  %18 = shl nuw i32 %17, 24
  %19 = or disjoint i32 %16, %18
  ret i32 %19
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local i32 @fake_read_be32(ptr noundef readonly captures(none) %0, ptr noundef writeonly captures(none) initializes((0, 1)) %1) local_unnamed_addr #1 {
  %3 = load i8, ptr %0, align 1, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 1
  %5 = load i8, ptr %4, align 1, !tbaa !6
  store i8 1, ptr %1, align 1, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 2
  %7 = load i8, ptr %6, align 1, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 3
  %9 = load i8, ptr %8, align 1, !tbaa !6
  %10 = zext i8 %9 to i32
  %11 = zext i8 %7 to i32
  %12 = shl nuw nsw i32 %11, 8
  %13 = zext i8 %5 to i32
  %14 = shl nuw nsw i32 %13, 16
  %15 = zext i8 %3 to i32
  %16 = shl nuw i32 %15, 24
  %17 = or disjoint i32 %14, %16
  %18 = or disjoint i32 %17, %10
  %19 = or disjoint i32 %18, %12
  ret i32 %19
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local i32 @incorrect_read_le32(ptr noundef readonly captures(none) %0, ptr noundef writeonly captures(none) initializes((0, 1)) %1) local_unnamed_addr #1 {
  %3 = load i32, ptr %0, align 1
  store i8 1, ptr %1, align 1, !tbaa !6
  ret i32 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local i32 @incorrect_read_be32(ptr noundef readonly captures(none) %0, ptr noundef writeonly captures(none) initializes((0, 1)) %1) local_unnamed_addr #1 {
  %3 = load i8, ptr %0, align 1, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 1
  %5 = load i8, ptr %4, align 1, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 2
  %7 = load i8, ptr %6, align 1, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 3
  %9 = load i8, ptr %8, align 1, !tbaa !6
  store i8 1, ptr %1, align 1, !tbaa !6
  %10 = zext i8 %9 to i32
  %11 = zext i8 %7 to i32
  %12 = shl nuw nsw i32 %11, 8
  %13 = zext i8 %5 to i32
  %14 = shl nuw nsw i32 %13, 16
  %15 = zext i8 %3 to i32
  %16 = shl nuw i32 %15, 24
  %17 = or disjoint i32 %14, %16
  %18 = or disjoint i32 %17, %10
  %19 = or disjoint i32 %18, %12
  ret i32 %19
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = alloca [4 x i8], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  store i32 -1987607165, ptr %1, align 4
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 2
  %3 = call i32 @fake_read_le32(ptr noundef nonnull %1, ptr noundef nonnull %2)
  %4 = icmp eq i32 %3, -1996388989
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  tail call void @abort() #7
  unreachable

6:                                                ; preds = %0
  store i8 -121, ptr %2, align 2, !tbaa !6
  %7 = call i32 @fake_read_be32(ptr noundef nonnull %1, ptr noundef nonnull %2)
  %8 = icmp eq i32 %7, -2088435319
  br i1 %8, label %10, label %9

9:                                                ; preds = %6
  tail call void @abort() #7
  unreachable

10:                                               ; preds = %6
  store i8 -121, ptr %2, align 2, !tbaa !6
  %11 = call i32 @incorrect_read_le32(ptr noundef nonnull %1, ptr noundef nonnull %2)
  %12 = icmp eq i32 %11, -1987607165
  br i1 %12, label %14, label %13

13:                                               ; preds = %10
  tail call void @abort() #7
  unreachable

14:                                               ; preds = %10
  store i8 -121, ptr %2, align 2, !tbaa !6
  %15 = call i32 @incorrect_read_be32(ptr noundef nonnull %1, ptr noundef nonnull %2)
  %16 = icmp eq i32 %15, -2088401015
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  tail call void @abort() #7
  unreachable

18:                                               ; preds = %14
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.or.v4i32(<4 x i32>) #5

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }

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
