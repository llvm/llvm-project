; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/memset-4.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/memset-4.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @f(ptr noundef writeonly captures(none) initializes((0, 15)) %0) local_unnamed_addr #0 {
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(15) %0, i8 0, i64 15, i1 false)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca [15 x i8], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 13
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 14
  call void @f(ptr noundef nonnull %1)
  %6 = load <8 x i8>, ptr %1, align 8
  %7 = freeze <8 x i8> %6
  %8 = icmp eq <8 x i8> %7, zeroinitializer
  %9 = load <4 x i8>, ptr %2, align 8
  %10 = freeze <4 x i8> %9
  %11 = icmp eq <4 x i8> %10, zeroinitializer
  %12 = load i8, ptr %3, align 4
  %13 = freeze i8 %12
  %14 = load i8, ptr %4, align 1
  %15 = freeze i8 %14
  %16 = load i8, ptr %5, align 2
  %17 = icmp eq i8 %16, 0
  %18 = shufflevector <8 x i1> %8, <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %19 = and <4 x i1> %18, %11
  %20 = shufflevector <4 x i1> %19, <4 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %21 = freeze <8 x i1> %20
  %22 = shufflevector <8 x i1> %21, <8 x i1> %8, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15>
  %23 = bitcast <8 x i1> %22 to i8
  %24 = icmp eq i8 %23, -1
  %25 = or i8 %15, %13
  %26 = icmp eq i8 %25, 0
  %27 = and i1 %24, %26
  %28 = select i1 %27, i1 %17, i1 false
  br i1 %28, label %29, label %30

29:                                               ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0

30:                                               ; preds = %0
  tail call void @abort() #6
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }
attributes #6 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
