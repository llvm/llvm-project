; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/DefaultInitDynArrays.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/DefaultInitDynArrays.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [16 x i8] c"buckets[i] == 0\00", align 1
@.str.1 = private unnamed_addr constant [104 x i8] c"/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/DefaultInitDynArrays.cpp\00", align 1
@__PRETTY_FUNCTION__._Z4funcv = private unnamed_addr constant [12 x i8] c"void func()\00", align 1

; Function Attrs: mustprogress uwtable
define dso_local void @_Z4funcv() local_unnamed_addr #0 {
  %1 = tail call noalias noundef nonnull dereferenceable(444) ptr @_Znam(i64 noundef 444) #6
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(444) %1, i8 0, i64 444, i1 false)
  br label %12

2:                                                ; preds = %12
  %3 = add nuw nsw i64 %13, 1
  %4 = icmp eq i64 %3, 111
  br i1 %4, label %5, label %12, !llvm.loop !6

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 4
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %6, align 4, !tbaa !8
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 20
  store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr %7, align 4, !tbaa !8
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 36
  store <4 x i32> <i32 9, i32 10, i32 11, i32 12>, ptr %8, align 4, !tbaa !8
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 52
  store <4 x i32> <i32 13, i32 14, i32 15, i32 16>, ptr %9, align 4, !tbaa !8
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 68
  store <4 x i32> <i32 17, i32 18, i32 19, i32 20>, ptr %10, align 4, !tbaa !8
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 84
  store i32 21, ptr %11, align 4, !tbaa !8
  tail call void @_ZdaPv(ptr noundef nonnull %1) #7
  ret void

12:                                               ; preds = %0, %2
  %13 = phi i64 [ 0, %0 ], [ %3, %2 ]
  %14 = getelementptr inbounds nuw i32, ptr %1, i64 %13
  %15 = load i32, ptr %14, align 4, !tbaa !8
  %16 = icmp eq i32 %15, 0
  br i1 %16, label %2, label %17

17:                                               ; preds = %12
  tail call void @__assert_fail(ptr noundef nonnull @.str, ptr noundef nonnull @.str.1, i32 noundef 6, ptr noundef nonnull @__PRETTY_FUNCTION__._Z4funcv) #8
  unreachable
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

; Function Attrs: cold noreturn nounwind
declare void @__assert_fail(ptr noundef, ptr noundef, i32 noundef, ptr noundef) local_unnamed_addr #3

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(ptr noundef) local_unnamed_addr #4

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #5 {
  %3 = tail call noalias noundef nonnull dereferenceable(444) ptr @_Znam(i64 noundef 444) #6
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(444) %3, i8 0, i64 444, i1 false)
  br label %7

4:                                                ; preds = %7
  %5 = add nuw nsw i64 %8, 1
  %6 = icmp eq i64 %5, 111
  br i1 %6, label %13, label %7, !llvm.loop !6

7:                                                ; preds = %4, %2
  %8 = phi i64 [ 0, %2 ], [ %5, %4 ]
  %9 = getelementptr inbounds nuw i32, ptr %3, i64 %8
  %10 = load i32, ptr %9, align 4, !tbaa !8
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %4, label %12

12:                                               ; preds = %7
  tail call void @__assert_fail(ptr noundef nonnull @.str, ptr noundef nonnull @.str.1, i32 noundef 6, ptr noundef nonnull @__PRETTY_FUNCTION__._Z4funcv) #8
  unreachable

13:                                               ; preds = %4
  tail call void @_ZdaPv(ptr noundef nonnull %3) #7
  %14 = tail call noalias noundef nonnull dereferenceable(444) ptr @_Znam(i64 noundef 444) #6
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(444) %14, i8 0, i64 444, i1 false)
  br label %18

15:                                               ; preds = %18
  %16 = add nuw nsw i64 %19, 1
  %17 = icmp eq i64 %16, 111
  br i1 %17, label %24, label %18, !llvm.loop !6

18:                                               ; preds = %15, %13
  %19 = phi i64 [ 0, %13 ], [ %16, %15 ]
  %20 = getelementptr inbounds nuw i32, ptr %14, i64 %19
  %21 = load i32, ptr %20, align 4, !tbaa !8
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %15, label %23

23:                                               ; preds = %18
  tail call void @__assert_fail(ptr noundef nonnull @.str, ptr noundef nonnull @.str.1, i32 noundef 6, ptr noundef nonnull @__PRETTY_FUNCTION__._Z4funcv) #8
  unreachable

24:                                               ; preds = %15
  tail call void @_ZdaPv(ptr noundef nonnull %14) #7
  %25 = tail call noalias noundef nonnull dereferenceable(444) ptr @_Znam(i64 noundef 444) #6
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(444) %25, i8 0, i64 444, i1 false)
  br label %29

26:                                               ; preds = %29
  %27 = add nuw nsw i64 %30, 1
  %28 = icmp eq i64 %27, 111
  br i1 %28, label %35, label %29, !llvm.loop !6

29:                                               ; preds = %26, %24
  %30 = phi i64 [ 0, %24 ], [ %27, %26 ]
  %31 = getelementptr inbounds nuw i32, ptr %25, i64 %30
  %32 = load i32, ptr %31, align 4, !tbaa !8
  %33 = icmp eq i32 %32, 0
  br i1 %33, label %26, label %34

34:                                               ; preds = %29
  tail call void @__assert_fail(ptr noundef nonnull @.str, ptr noundef nonnull @.str.1, i32 noundef 6, ptr noundef nonnull @__PRETTY_FUNCTION__._Z4funcv) #8
  unreachable

35:                                               ; preds = %26
  tail call void @_ZdaPv(ptr noundef nonnull %25) #7
  %36 = tail call noalias noundef nonnull dereferenceable(444) ptr @_Znam(i64 noundef 444) #6
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(444) %36, i8 0, i64 444, i1 false)
  br label %40

37:                                               ; preds = %40
  %38 = add nuw nsw i64 %41, 1
  %39 = icmp eq i64 %38, 111
  br i1 %39, label %46, label %40, !llvm.loop !6

40:                                               ; preds = %37, %35
  %41 = phi i64 [ 0, %35 ], [ %38, %37 ]
  %42 = getelementptr inbounds nuw i32, ptr %36, i64 %41
  %43 = load i32, ptr %42, align 4, !tbaa !8
  %44 = icmp eq i32 %43, 0
  br i1 %44, label %37, label %45

45:                                               ; preds = %40
  tail call void @__assert_fail(ptr noundef nonnull @.str, ptr noundef nonnull @.str.1, i32 noundef 6, ptr noundef nonnull @__PRETTY_FUNCTION__._Z4funcv) #8
  unreachable

46:                                               ; preds = %37
  tail call void @_ZdaPv(ptr noundef nonnull %36) #7
  %47 = tail call noalias noundef nonnull dereferenceable(444) ptr @_Znam(i64 noundef 444) #6
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(444) %47, i8 0, i64 444, i1 false)
  br label %51

48:                                               ; preds = %51
  %49 = add nuw nsw i64 %52, 1
  %50 = icmp eq i64 %49, 111
  br i1 %50, label %57, label %51, !llvm.loop !6

51:                                               ; preds = %48, %46
  %52 = phi i64 [ 0, %46 ], [ %49, %48 ]
  %53 = getelementptr inbounds nuw i32, ptr %47, i64 %52
  %54 = load i32, ptr %53, align 4, !tbaa !8
  %55 = icmp eq i32 %54, 0
  br i1 %55, label %48, label %56

56:                                               ; preds = %51
  tail call void @__assert_fail(ptr noundef nonnull @.str, ptr noundef nonnull @.str.1, i32 noundef 6, ptr noundef nonnull @__PRETTY_FUNCTION__._Z4funcv) #8
  unreachable

57:                                               ; preds = %48
  tail call void @_ZdaPv(ptr noundef nonnull %47) #7
  ret i32 0
}

attributes #0 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { cold noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { builtin allocsize(0) }
attributes #7 = { builtin nounwind }
attributes #8 = { cold noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
