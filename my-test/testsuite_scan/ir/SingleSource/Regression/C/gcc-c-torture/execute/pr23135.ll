; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr23135.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr23135.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%union.anon = type { <2 x i32> }

@i = dso_local local_unnamed_addr global <2 x i32> <i32 150, i32 100>, align 8
@j = dso_local local_unnamed_addr global <2 x i32> <i32 10, i32 13>, align 8
@res = dso_local local_unnamed_addr global %union.anon zeroinitializer, align 8
@k = dso_local local_unnamed_addr global <2 x i32> zeroinitializer, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @verify(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, ptr dead_on_return noundef readnone captures(none) %4) local_unnamed_addr #0 {
  %6 = icmp eq i32 %0, %2
  %7 = icmp eq i32 %1, %3
  %8 = and i1 %6, %7
  br i1 %8, label %10, label %9

9:                                                ; preds = %5
  tail call void @abort() #4
  unreachable

10:                                               ; preds = %5
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = load <2 x i32>, ptr @i, align 8, !tbaa !6
  %2 = sub <2 x i32> zeroinitializer, %1
  %3 = load <2 x i32>, ptr @j, align 8, !tbaa !6
  %4 = add <2 x i32> %3, %1
  store <2 x i32> %4, ptr @res, align 8, !tbaa !6
  %5 = icmp eq <2 x i32> %4, <i32 160, i32 113>
  %6 = shufflevector <2 x i1> %5, <2 x i1> poison, <2 x i32> <i32 1, i32 poison>
  %7 = and <2 x i1> %5, %6
  %8 = extractelement <2 x i1> %7, i64 0
  br i1 %8, label %10, label %9

9:                                                ; preds = %0
  tail call void @abort() #4
  unreachable

10:                                               ; preds = %0
  %11 = mul <2 x i32> %3, %1
  store <2 x i32> %11, ptr @res, align 8, !tbaa !6
  %12 = icmp eq <2 x i32> %11, <i32 1500, i32 1300>
  %13 = shufflevector <2 x i1> %12, <2 x i1> poison, <2 x i32> <i32 1, i32 poison>
  %14 = and <2 x i1> %12, %13
  %15 = extractelement <2 x i1> %14, i64 0
  br i1 %15, label %17, label %16

16:                                               ; preds = %10
  tail call void @abort() #4
  unreachable

17:                                               ; preds = %10
  %18 = sdiv <2 x i32> %1, %3
  store <2 x i32> %18, ptr @res, align 8, !tbaa !6
  %19 = icmp eq <2 x i32> %18, <i32 15, i32 7>
  %20 = shufflevector <2 x i1> %19, <2 x i1> poison, <2 x i32> <i32 1, i32 poison>
  %21 = and <2 x i1> %19, %20
  %22 = extractelement <2 x i1> %21, i64 0
  br i1 %22, label %24, label %23

23:                                               ; preds = %17
  tail call void @abort() #4
  unreachable

24:                                               ; preds = %17
  %25 = and <2 x i32> %3, %1
  store <2 x i32> %25, ptr @res, align 8, !tbaa !6
  %26 = icmp eq <2 x i32> %25, <i32 2, i32 4>
  %27 = shufflevector <2 x i1> %26, <2 x i1> poison, <2 x i32> <i32 1, i32 poison>
  %28 = and <2 x i1> %26, %27
  %29 = extractelement <2 x i1> %28, i64 0
  br i1 %29, label %31, label %30

30:                                               ; preds = %24
  tail call void @abort() #4
  unreachable

31:                                               ; preds = %24
  %32 = or <2 x i32> %3, %1
  store <2 x i32> %32, ptr @res, align 8, !tbaa !6
  %33 = icmp eq <2 x i32> %32, <i32 158, i32 109>
  %34 = shufflevector <2 x i1> %33, <2 x i1> poison, <2 x i32> <i32 1, i32 poison>
  %35 = and <2 x i1> %33, %34
  %36 = extractelement <2 x i1> %35, i64 0
  br i1 %36, label %38, label %37

37:                                               ; preds = %31
  tail call void @abort() #4
  unreachable

38:                                               ; preds = %31
  %39 = xor <2 x i32> %3, %1
  store <2 x i32> %39, ptr @res, align 8, !tbaa !6
  %40 = icmp eq <2 x i32> %39, <i32 156, i32 105>
  %41 = shufflevector <2 x i1> %40, <2 x i1> poison, <2 x i32> <i32 1, i32 poison>
  %42 = and <2 x i1> %40, %41
  %43 = extractelement <2 x i1> %42, i64 0
  br i1 %43, label %45, label %44

44:                                               ; preds = %38
  tail call void @abort() #4
  unreachable

45:                                               ; preds = %38
  store <2 x i32> %2, ptr @res, align 8, !tbaa !6
  %46 = icmp eq <2 x i32> %2, <i32 -150, i32 -100>
  %47 = shufflevector <2 x i1> %46, <2 x i1> poison, <2 x i32> <i32 1, i32 poison>
  %48 = and <2 x i1> %46, %47
  %49 = extractelement <2 x i1> %48, i64 0
  br i1 %49, label %51, label %50

50:                                               ; preds = %45
  tail call void @abort() #4
  unreachable

51:                                               ; preds = %45
  %52 = xor <2 x i32> %1, splat (i32 -1)
  store <2 x i32> %52, ptr @res, align 8, !tbaa !6
  %53 = icmp eq <2 x i32> %1, <i32 150, i32 100>
  %54 = shufflevector <2 x i1> %53, <2 x i1> poison, <2 x i32> <i32 1, i32 poison>
  %55 = and <2 x i1> %53, %54
  %56 = extractelement <2 x i1> %55, i64 0
  br i1 %56, label %58, label %57

57:                                               ; preds = %51
  tail call void @abort() #4
  unreachable

58:                                               ; preds = %51
  %59 = sub <2 x i32> %52, %1
  %60 = add <2 x i32> %59, %11
  %61 = add <2 x i32> %60, %4
  %62 = add <2 x i32> %61, %25
  %63 = add <2 x i32> %62, %32
  %64 = add <2 x i32> %63, %39
  store <2 x i32> %64, ptr @k, align 8, !tbaa !6
  store <2 x i32> %64, ptr @res, align 8, !tbaa !6
  %65 = icmp eq <2 x i32> %64, <i32 1675, i32 1430>
  %66 = shufflevector <2 x i1> %65, <2 x i1> poison, <2 x i32> <i32 1, i32 poison>
  %67 = and <2 x i1> %65, %66
  %68 = extractelement <2 x i1> %67, i64 0
  br i1 %68, label %70, label %69

69:                                               ; preds = %58
  tail call void @abort() #4
  unreachable

70:                                               ; preds = %58
  %71 = mul <2 x i32> %52, %2
  %72 = mul <2 x i32> %71, %11
  %73 = mul <2 x i32> %72, %4
  %74 = mul <2 x i32> %73, %25
  %75 = mul <2 x i32> %74, %32
  %76 = mul <2 x i32> %75, %39
  store <2 x i32> %76, ptr @k, align 8, !tbaa !6
  store <2 x i32> %76, ptr @res, align 8, !tbaa !6
  %77 = icmp eq <2 x i32> %76, <i32 1456467968, i32 -1579586240>
  %78 = shufflevector <2 x i1> %77, <2 x i1> poison, <2 x i32> <i32 1, i32 poison>
  %79 = and <2 x i1> %77, %78
  %80 = extractelement <2 x i1> %79, i64 0
  br i1 %80, label %82, label %81

81:                                               ; preds = %70
  tail call void @abort() #4
  unreachable

82:                                               ; preds = %70
  %83 = sdiv <2 x i32> %4, %11
  %84 = sdiv <2 x i32> %83, %18
  %85 = sdiv <2 x i32> %84, %25
  %86 = sdiv <2 x i32> %85, %32
  %87 = sdiv <2 x i32> %86, %39
  %88 = sdiv <2 x i32> %87, %2
  %89 = sdiv <2 x i32> %88, %52
  store <2 x i32> %89, ptr @k, align 8, !tbaa !6
  store <2 x i32> %89, ptr @res, align 8, !tbaa !6
  %90 = shufflevector <2 x i32> %89, <2 x i32> poison, <2 x i32> <i32 1, i32 poison>
  %91 = or <2 x i32> %90, %89
  %92 = extractelement <2 x i32> %91, i64 0
  %93 = icmp eq i32 %92, 0
  br i1 %93, label %95, label %94

94:                                               ; preds = %82
  tail call void @abort() #4
  unreachable

95:                                               ; preds = %82
  tail call void @exit(i32 noundef 0) #4
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #3

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }

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
