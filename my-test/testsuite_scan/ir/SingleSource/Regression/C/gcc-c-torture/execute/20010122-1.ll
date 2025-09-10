; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20010122-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20010122-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@func1 = dso_local local_unnamed_addr global [6 x ptr] [ptr @test1, ptr @test2, ptr @test3, ptr @test4, ptr @test5, ptr @test6], align 8
@save_ret1 = dso_local local_unnamed_addr global [6 x ptr] zeroinitializer, align 8
@ret_addr = internal unnamed_addr global ptr null, align 8
@func2 = dso_local local_unnamed_addr global [6 x ptr] [ptr @test7, ptr @test8, ptr @test9, ptr @test10, ptr @test11, ptr @test12], align 8
@save_ret2 = dso_local local_unnamed_addr global [6 x ptr] zeroinitializer, align 8

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local ptr @test1() #0 {
  %1 = tail call ptr @llvm.returnaddress(i32 0)
  ret ptr %1
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.returnaddress(i32 immarg) #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local ptr @test2() #0 {
  %1 = tail call ptr @llvm.returnaddress(i32 0)
  ret ptr %1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noalias noundef nonnull ptr @dummy() local_unnamed_addr #2 {
  %1 = alloca [4 x i8], align 16
  ret ptr %1
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local ptr @test3() #0 {
  %1 = tail call ptr @llvm.returnaddress(i32 0)
  ret ptr %1
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local ptr @test4() #0 {
  %1 = tail call ptr @test4a(ptr nonnull poison)
  ret ptr %1
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local ptr @test4a(ptr readnone captures(none) %0) local_unnamed_addr #0 {
  %2 = tail call ptr @llvm.returnaddress(i32 1)
  ret ptr %2
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local ptr @test5() #0 {
  %1 = tail call ptr @test5a(ptr nonnull poison)
  ret ptr %1
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local ptr @test5a(ptr readnone captures(none) %0) local_unnamed_addr #0 {
  %2 = tail call ptr @llvm.returnaddress(i32 1)
  ret ptr %2
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local ptr @test6() #0 {
  %1 = tail call ptr @test6a(ptr nonnull poison)
  ret ptr %1
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local ptr @test6a(ptr readnone captures(none) %0) local_unnamed_addr #0 {
  %2 = tail call ptr @llvm.returnaddress(i32 1)
  ret ptr %2
}

; Function Attrs: noinline nounwind uwtable
define dso_local noalias ptr @call_func1(i32 noundef %0) local_unnamed_addr #3 {
  %2 = sext i32 %0 to i64
  %3 = getelementptr inbounds ptr, ptr @func1, i64 %2
  %4 = load ptr, ptr %3, align 8, !tbaa !6
  %5 = tail call ptr %4() #9
  %6 = getelementptr inbounds ptr, ptr @save_ret1, i64 %2
  store ptr %5, ptr %6, align 8, !tbaa !6
  ret ptr undef
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @test7() #4 {
  %1 = tail call ptr @llvm.returnaddress(i32 0)
  store ptr %1, ptr @ret_addr, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @test8() #4 {
  %1 = tail call ptr @llvm.returnaddress(i32 0)
  store ptr %1, ptr @ret_addr, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @test9() #4 {
  %1 = tail call ptr @llvm.returnaddress(i32 0)
  store ptr %1, ptr @ret_addr, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, inaccessiblemem: none) uwtable
define dso_local void @test10() #5 {
  tail call void @test10a(ptr nonnull poison)
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @test10a(ptr readnone captures(none) %0) local_unnamed_addr #4 {
  %2 = tail call ptr @llvm.returnaddress(i32 1)
  store ptr %2, ptr @ret_addr, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, inaccessiblemem: none) uwtable
define dso_local void @test11() #5 {
  tail call void @test11a(ptr nonnull poison)
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @test11a(ptr readnone captures(none) %0) local_unnamed_addr #4 {
  %2 = tail call ptr @llvm.returnaddress(i32 1)
  store ptr %2, ptr @ret_addr, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, inaccessiblemem: none) uwtable
define dso_local void @test12() #5 {
  tail call void @test12a(ptr nonnull poison)
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @test12a(ptr readnone captures(none) %0) local_unnamed_addr #4 {
  %2 = tail call ptr @llvm.returnaddress(i32 1)
  store ptr %2, ptr @ret_addr, align 8, !tbaa !6
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @call_func2(i32 noundef %0) local_unnamed_addr #3 {
  %2 = sext i32 %0 to i64
  %3 = getelementptr inbounds ptr, ptr @func2, i64 %2
  %4 = load ptr, ptr %3, align 8, !tbaa !6
  tail call void %4() #9
  %5 = load ptr, ptr @ret_addr, align 8, !tbaa !6
  %6 = getelementptr inbounds ptr, ptr @save_ret2, i64 %2
  store ptr %5, ptr %6, align 8, !tbaa !6
  ret void
}

; Function Attrs: noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #6 {
  %1 = tail call ptr @call_func1(i32 noundef 0)
  %2 = tail call ptr @call_func1(i32 noundef 1)
  %3 = tail call ptr @call_func1(i32 noundef 2)
  %4 = tail call ptr @call_func1(i32 noundef 3)
  %5 = tail call ptr @call_func1(i32 noundef 4)
  %6 = tail call ptr @call_func1(i32 noundef 5)
  %7 = load ptr, ptr @save_ret1, align 8, !tbaa !6
  %8 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @save_ret1, i64 8), align 8, !tbaa !6
  %9 = icmp eq ptr %7, %8
  %10 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @save_ret1, i64 16), align 8
  %11 = icmp eq ptr %8, %10
  %12 = select i1 %9, i1 %11, i1 false
  br i1 %12, label %14, label %13

13:                                               ; preds = %0
  tail call void @abort() #10
  unreachable

14:                                               ; preds = %0
  %15 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @save_ret1, i64 24), align 8, !tbaa !6
  %16 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @save_ret1, i64 32), align 8, !tbaa !6
  %17 = icmp eq ptr %15, %16
  %18 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @save_ret1, i64 40), align 8
  %19 = icmp eq ptr %16, %18
  %20 = select i1 %17, i1 %19, i1 false
  br i1 %20, label %22, label %21

21:                                               ; preds = %14
  tail call void @abort() #10
  unreachable

22:                                               ; preds = %14
  %23 = icmp eq ptr %15, null
  %24 = icmp eq ptr %7, %15
  %25 = or i1 %23, %24
  br i1 %25, label %26, label %33

26:                                               ; preds = %22
  tail call void @call_func2(i32 noundef 0)
  tail call void @call_func2(i32 noundef 1)
  tail call void @call_func2(i32 noundef 2)
  tail call void @call_func2(i32 noundef 3)
  tail call void @call_func2(i32 noundef 4)
  tail call void @call_func2(i32 noundef 5)
  %27 = load ptr, ptr @save_ret2, align 8, !tbaa !6
  %28 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @save_ret2, i64 8), align 8, !tbaa !6
  %29 = icmp eq ptr %27, %28
  %30 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @save_ret2, i64 16), align 8
  %31 = icmp eq ptr %28, %30
  %32 = select i1 %29, i1 %31, i1 false
  br i1 %32, label %35, label %34

33:                                               ; preds = %22
  tail call void @abort() #10
  unreachable

34:                                               ; preds = %26
  tail call void @abort() #10
  unreachable

35:                                               ; preds = %26
  %36 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @save_ret2, i64 24), align 8, !tbaa !6
  %37 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @save_ret2, i64 32), align 8, !tbaa !6
  %38 = icmp eq ptr %36, %37
  %39 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @save_ret2, i64 40), align 8
  %40 = icmp eq ptr %37, %39
  %41 = select i1 %38, i1 %40, i1 false
  br i1 %41, label %43, label %42

42:                                               ; preds = %35
  tail call void @abort() #10
  unreachable

43:                                               ; preds = %35
  %44 = icmp eq ptr %36, null
  %45 = icmp eq ptr %27, %36
  %46 = or i1 %44, %45
  br i1 %46, label %48, label %47

47:                                               ; preds = %43
  tail call void @abort() #10
  unreachable

48:                                               ; preds = %43
  tail call void @exit(i32 noundef 0) #10
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #7

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #8

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nounwind }
attributes #10 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
