; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/return-addr.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/return-addr.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [13 x i8] c"%p %p %p %p\0A\00", align 1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local nonnull ptr @get_max_2(ptr noundef readnone captures(ret: address, provenance) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 1
  ret ptr %2
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local nonnull ptr @get_max_3(ptr noundef readnone captures(address, ret: address, provenance) %0, ptr noundef readnone captures(address, ret: address, provenance) %1) local_unnamed_addr #0 {
  %3 = icmp ult ptr %0, %1
  %4 = select i1 %3, ptr %1, ptr %0
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 1
  ret ptr %5
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local nonnull ptr @get_min_2(ptr noundef readnone captures(ret: address, provenance) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds i8, ptr %0, i64 -1
  ret ptr %2
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local nonnull ptr @get_min_3(ptr noundef readnone captures(address, ret: address, provenance) %0, ptr noundef readnone captures(address, ret: address, provenance) %1) local_unnamed_addr #0 {
  %3 = icmp ult ptr %0, %1
  %4 = select i1 %3, ptr %0, ptr %1
  %5 = getelementptr inbounds i8, ptr %4, i64 -1
  ret ptr %5
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local nonnull ptr @test_max_2() local_unnamed_addr #0 {
  %1 = alloca i8, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  %2 = call ptr @get_max_2(ptr noundef nonnull %1)
  %3 = icmp ugt ptr %2, %1
  %4 = select i1 %3, ptr %2, ptr %1
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret ptr %4
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local nonnull ptr @test_max_3() local_unnamed_addr #0 {
  %1 = alloca i8, align 1
  %2 = alloca i8, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  %3 = call ptr @get_max_3(ptr noundef nonnull %1, ptr noundef nonnull %2)
  %4 = icmp ult ptr %3, %1
  %5 = icmp ult ptr %1, %2
  %6 = select i1 %5, ptr %2, ptr %1
  %7 = select i1 %4, ptr %6, ptr %3
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret ptr %7
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local nonnull ptr @test_min_2() local_unnamed_addr #0 {
  %1 = alloca i8, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  %2 = call ptr @get_min_2(ptr noundef nonnull %1)
  %3 = icmp ult ptr %2, %1
  %4 = select i1 %3, ptr %2, ptr %1
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret ptr %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local nonnull ptr @test_min_3() local_unnamed_addr #0 {
  %1 = alloca i8, align 1
  %2 = alloca i8, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  %3 = call ptr @get_min_3(ptr noundef nonnull %1, ptr noundef nonnull %2)
  %4 = icmp ugt ptr %3, %1
  %5 = icmp ugt ptr %1, %2
  %6 = select i1 %5, ptr %2, ptr %1
  %7 = select i1 %4, ptr %6, ptr %3
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret ptr %7
}

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef nonnull ptr @test_min_3_phi(i32 noundef %0) local_unnamed_addr #2 {
  %2 = alloca i8, align 1
  %3 = alloca i8, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  %4 = call ptr @get_min_3(ptr noundef nonnull %2, ptr noundef nonnull %3)
  %5 = icmp ult ptr %4, %2
  %6 = select i1 %5, ptr %4, ptr %2
  %7 = icmp ult ptr %4, %3
  %8 = select i1 %7, ptr %4, ptr %3
  %9 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %4, ptr noundef nonnull %4, ptr noundef nonnull %6, ptr noundef nonnull %8) #5
  %10 = icmp eq i32 %0, 1
  %11 = select i1 %10, ptr %6, ptr %8
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  ret ptr %11
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  %1 = tail call ptr @test_min_3_phi(i32 noundef 0)
  ret i32 0
}

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
