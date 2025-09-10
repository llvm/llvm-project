; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/complex-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/complex-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  %1 = alloca float, align 4
  %2 = alloca float, align 4
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  %5 = alloca fp128, align 16
  %6 = alloca fp128, align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  call void @llvm.lifetime.start.p0(ptr nonnull %2)
  store volatile float 1.000000e+00, ptr %1, align 4
  store volatile float 2.000000e+00, ptr %2, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %3)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  store volatile double 1.000000e+00, ptr %3, align 8
  store volatile double 2.000000e+00, ptr %4, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store volatile fp128 0xL00000000000000003FFF000000000000, ptr %5, align 16
  store volatile fp128 0xL00000000000000004000000000000000, ptr %6, align 16
  %7 = load volatile float, ptr %1, align 4
  %8 = load volatile float, ptr %2, align 4
  %9 = fcmp une float %7, 1.000000e+00
  %10 = fcmp une float %8, 2.000000e+00
  %11 = or i1 %9, %10
  br i1 %11, label %12, label %13

12:                                               ; preds = %0
  tail call void @abort() #3
  unreachable

13:                                               ; preds = %0
  %14 = load volatile float, ptr %1, align 4
  %15 = load volatile float, ptr %2, align 4
  %16 = fcmp une float %14, 1.000000e+00
  %17 = fcmp une float %15, 2.000000e+00
  %18 = or i1 %16, %17
  br i1 %18, label %19, label %20

19:                                               ; preds = %13
  tail call void @abort() #3
  unreachable

20:                                               ; preds = %13
  %21 = load volatile float, ptr %1, align 4
  %22 = load volatile float, ptr %2, align 4
  %23 = fcmp une float %21, 1.000000e+00
  br i1 %23, label %24, label %25

24:                                               ; preds = %20
  tail call void @abort() #3
  unreachable

25:                                               ; preds = %20
  %26 = load volatile float, ptr %1, align 4
  %27 = load volatile float, ptr %2, align 4
  %28 = fcmp une float %26, 1.000000e+00
  br i1 %28, label %29, label %30

29:                                               ; preds = %25
  tail call void @abort() #3
  unreachable

30:                                               ; preds = %25
  %31 = load volatile float, ptr %1, align 4
  %32 = load volatile float, ptr %2, align 4
  %33 = fcmp une float %32, 2.000000e+00
  br i1 %33, label %34, label %35

34:                                               ; preds = %30
  tail call void @abort() #3
  unreachable

35:                                               ; preds = %30
  %36 = load volatile float, ptr %1, align 4
  %37 = load volatile float, ptr %2, align 4
  %38 = fcmp une float %37, 2.000000e+00
  br i1 %38, label %39, label %40

39:                                               ; preds = %35
  tail call void @abort() #3
  unreachable

40:                                               ; preds = %35
  %41 = load volatile double, ptr %3, align 8
  %42 = load volatile double, ptr %4, align 8
  %43 = fcmp une double %41, 1.000000e+00
  %44 = fcmp une double %42, 2.000000e+00
  %45 = or i1 %43, %44
  br i1 %45, label %46, label %47

46:                                               ; preds = %40
  tail call void @abort() #3
  unreachable

47:                                               ; preds = %40
  %48 = load volatile double, ptr %3, align 8
  %49 = load volatile double, ptr %4, align 8
  %50 = fcmp une double %48, 1.000000e+00
  %51 = fcmp une double %49, 2.000000e+00
  %52 = or i1 %50, %51
  br i1 %52, label %53, label %54

53:                                               ; preds = %47
  tail call void @abort() #3
  unreachable

54:                                               ; preds = %47
  %55 = load volatile double, ptr %3, align 8
  %56 = load volatile double, ptr %4, align 8
  %57 = fcmp une double %55, 1.000000e+00
  br i1 %57, label %58, label %59

58:                                               ; preds = %54
  tail call void @abort() #3
  unreachable

59:                                               ; preds = %54
  %60 = load volatile double, ptr %3, align 8
  %61 = load volatile double, ptr %4, align 8
  %62 = fcmp une double %60, 1.000000e+00
  br i1 %62, label %63, label %64

63:                                               ; preds = %59
  tail call void @abort() #3
  unreachable

64:                                               ; preds = %59
  %65 = load volatile double, ptr %3, align 8
  %66 = load volatile double, ptr %4, align 8
  %67 = fcmp une double %66, 2.000000e+00
  br i1 %67, label %68, label %69

68:                                               ; preds = %64
  tail call void @abort() #3
  unreachable

69:                                               ; preds = %64
  %70 = load volatile double, ptr %3, align 8
  %71 = load volatile double, ptr %4, align 8
  %72 = fcmp une double %71, 2.000000e+00
  br i1 %72, label %73, label %74

73:                                               ; preds = %69
  tail call void @abort() #3
  unreachable

74:                                               ; preds = %69
  %75 = load volatile fp128, ptr %5, align 16
  %76 = load volatile fp128, ptr %6, align 16
  %77 = fcmp une fp128 %75, 0xL00000000000000003FFF000000000000
  %78 = fcmp une fp128 %76, 0xL00000000000000004000000000000000
  %79 = or i1 %77, %78
  br i1 %79, label %80, label %81

80:                                               ; preds = %74
  tail call void @abort() #3
  unreachable

81:                                               ; preds = %74
  %82 = load volatile fp128, ptr %5, align 16
  %83 = load volatile fp128, ptr %6, align 16
  %84 = fcmp une fp128 %82, 0xL00000000000000003FFF000000000000
  %85 = fcmp une fp128 %83, 0xL00000000000000004000000000000000
  %86 = or i1 %84, %85
  br i1 %86, label %87, label %88

87:                                               ; preds = %81
  tail call void @abort() #3
  unreachable

88:                                               ; preds = %81
  %89 = load volatile fp128, ptr %5, align 16
  %90 = load volatile fp128, ptr %6, align 16
  %91 = fcmp une fp128 %89, 0xL00000000000000003FFF000000000000
  br i1 %91, label %92, label %93

92:                                               ; preds = %88
  tail call void @abort() #3
  unreachable

93:                                               ; preds = %88
  %94 = load volatile fp128, ptr %5, align 16
  %95 = load volatile fp128, ptr %6, align 16
  %96 = fcmp une fp128 %94, 0xL00000000000000003FFF000000000000
  br i1 %96, label %97, label %98

97:                                               ; preds = %93
  tail call void @abort() #3
  unreachable

98:                                               ; preds = %93
  %99 = load volatile fp128, ptr %5, align 16
  %100 = load volatile fp128, ptr %6, align 16
  %101 = fcmp une fp128 %100, 0xL00000000000000004000000000000000
  br i1 %101, label %102, label %103

102:                                              ; preds = %98
  tail call void @abort() #3
  unreachable

103:                                              ; preds = %98
  %104 = load volatile fp128, ptr %5, align 16
  %105 = load volatile fp128, ptr %6, align 16
  %106 = fcmp une fp128 %105, 0xL00000000000000004000000000000000
  br i1 %106, label %107, label %108

107:                                              ; preds = %103
  tail call void @abort() #3
  unreachable

108:                                              ; preds = %103
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  call void @llvm.lifetime.end.p0(ptr nonnull %2)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
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
