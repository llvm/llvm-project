; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/bcp-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/bcp-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@global = dso_local local_unnamed_addr global i32 0, align 4
@bad_t0 = dso_local global [6 x ptr] [ptr @bad0, ptr @bad1, ptr @bad5, ptr @bad7, ptr @bad8, ptr @bad10], align 8
@bad_t1 = dso_local global [3 x ptr] [ptr @bad2, ptr @bad3, ptr @bad6], align 8
@bad_t2 = dso_local global [2 x ptr] [ptr @bad4, ptr @bad9], align 8
@good_t0 = dso_local global [3 x ptr] [ptr @good0, ptr @good1, ptr @good2], align 8
@opt_t0 = dso_local global [3 x ptr] [ptr @opt0, ptr @opt1, ptr @opt2], align 8
@.str = private unnamed_addr constant [3 x i8] c"hi\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @bad0() #0 {
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @bad1() #1 {
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @bad5() #1 {
  ret i32 0
}

; Function Attrs: inlinehint nounwind uwtable
declare i32 @bad2(i32 noundef) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @bad7() #1 {
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @bad8() #1 {
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local range(i32 0, 2) i32 @bad9(ptr noundef readonly captures(none) %0) #3 {
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @bad10() #1 {
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @good0() #1 {
  ret i32 1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @good1() #1 {
  ret i32 1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @good2() #1 {
  ret i32 1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @opt0() #1 {
  ret i32 1
}

; Function Attrs: inlinehint nounwind uwtable
declare i32 @bad3(i32 noundef) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @opt1() #1 {
  ret i32 1
}

; Function Attrs: inlinehint nounwind uwtable
declare i32 @bad6(i32 noundef) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @opt2() #1 {
  ret i32 1
}

; Function Attrs: inlinehint nounwind uwtable
declare i32 @bad4(ptr noundef) #2

; Function Attrs: noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  %1 = load volatile ptr, ptr @bad_t0, align 8, !tbaa !6
  %2 = tail call i32 %1() #7
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %28

4:                                                ; preds = %0
  %5 = load volatile ptr, ptr getelementptr inbounds nuw (i8, ptr @bad_t0, i64 8), align 8, !tbaa !6
  %6 = tail call i32 %5() #7
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %28

8:                                                ; preds = %4
  %9 = load volatile ptr, ptr getelementptr inbounds nuw (i8, ptr @bad_t0, i64 16), align 8, !tbaa !6
  %10 = tail call i32 %9() #7
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %12, label %28

12:                                               ; preds = %8
  %13 = load volatile ptr, ptr getelementptr inbounds nuw (i8, ptr @bad_t0, i64 24), align 8, !tbaa !6
  %14 = tail call i32 %13() #7
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %16, label %28

16:                                               ; preds = %12
  %17 = load volatile ptr, ptr getelementptr inbounds nuw (i8, ptr @bad_t0, i64 32), align 8, !tbaa !6
  %18 = tail call i32 %17() #7
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %20, label %28

20:                                               ; preds = %16
  %21 = load volatile ptr, ptr getelementptr inbounds nuw (i8, ptr @bad_t0, i64 40), align 8, !tbaa !6
  %22 = tail call i32 %21() #7
  %23 = icmp eq i32 %22, 0
  br i1 %23, label %24, label %28

24:                                               ; preds = %20
  %25 = load volatile ptr, ptr @bad_t1, align 8, !tbaa !6
  %26 = tail call i32 %25(i32 noundef 1) #7
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %29, label %41

28:                                               ; preds = %20, %16, %12, %8, %4, %0
  tail call void @abort() #8
  unreachable

29:                                               ; preds = %24
  %30 = load volatile ptr, ptr getelementptr inbounds nuw (i8, ptr @bad_t1, i64 8), align 8, !tbaa !6
  %31 = tail call i32 %30(i32 noundef 1) #7
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %33, label %41

33:                                               ; preds = %29
  %34 = load volatile ptr, ptr getelementptr inbounds nuw (i8, ptr @bad_t1, i64 16), align 8, !tbaa !6
  %35 = tail call i32 %34(i32 noundef 1) #7
  %36 = icmp eq i32 %35, 0
  br i1 %36, label %37, label %41

37:                                               ; preds = %33
  %38 = load volatile ptr, ptr @bad_t2, align 8, !tbaa !6
  %39 = tail call i32 %38(ptr noundef nonnull @.str) #7
  %40 = icmp eq i32 %39, 0
  br i1 %40, label %42, label %50

41:                                               ; preds = %33, %29, %24
  tail call void @abort() #8
  unreachable

42:                                               ; preds = %37
  %43 = load volatile ptr, ptr getelementptr inbounds nuw (i8, ptr @bad_t2, i64 8), align 8, !tbaa !6
  %44 = tail call i32 %43(ptr noundef nonnull @.str) #7
  %45 = icmp eq i32 %44, 0
  br i1 %45, label %46, label %50

46:                                               ; preds = %42
  %47 = load volatile ptr, ptr @good_t0, align 8, !tbaa !6
  %48 = tail call i32 %47() #7
  %49 = icmp eq i32 %48, 0
  br i1 %49, label %63, label %51

50:                                               ; preds = %42, %37
  tail call void @abort() #8
  unreachable

51:                                               ; preds = %46
  %52 = load volatile ptr, ptr getelementptr inbounds nuw (i8, ptr @good_t0, i64 8), align 8, !tbaa !6
  %53 = tail call i32 %52() #7
  %54 = icmp eq i32 %53, 0
  br i1 %54, label %63, label %55

55:                                               ; preds = %51
  %56 = load volatile ptr, ptr getelementptr inbounds nuw (i8, ptr @good_t0, i64 16), align 8, !tbaa !6
  %57 = tail call i32 %56() #7
  %58 = icmp eq i32 %57, 0
  br i1 %58, label %63, label %59

59:                                               ; preds = %55
  %60 = load volatile ptr, ptr @opt_t0, align 8, !tbaa !6
  %61 = tail call i32 %60() #7
  %62 = icmp eq i32 %61, 0
  br i1 %62, label %73, label %64

63:                                               ; preds = %55, %51, %46
  tail call void @abort() #8
  unreachable

64:                                               ; preds = %59
  %65 = load volatile ptr, ptr getelementptr inbounds nuw (i8, ptr @opt_t0, i64 8), align 8, !tbaa !6
  %66 = tail call i32 %65() #7
  %67 = icmp eq i32 %66, 0
  br i1 %67, label %73, label %68

68:                                               ; preds = %64
  %69 = load volatile ptr, ptr getelementptr inbounds nuw (i8, ptr @opt_t0, i64 16), align 8, !tbaa !6
  %70 = tail call i32 %69() #7
  %71 = icmp eq i32 %70, 0
  br i1 %71, label %73, label %72

72:                                               ; preds = %68
  tail call void @exit(i32 noundef 0) #8
  unreachable

73:                                               ; preds = %68, %64, %59
  tail call void @abort() #8
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #5

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { inlinehint nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nounwind }
attributes #8 = { noreturn nounwind }

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
