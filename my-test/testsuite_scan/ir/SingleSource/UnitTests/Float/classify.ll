; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Float/classify.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Float/classify.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@FloatQNaNValues = dso_local local_unnamed_addr global [8 x i32] [i32 2147483647, i32 -1, i32 2143289344, i32 -4194304, i32 2145386496, i32 -3145728, i32 2143289345, i32 -4194302], align 4
@FloatSNaNValues = dso_local local_unnamed_addr global [6 x i32] [i32 2143289343, i32 -4194305, i32 2141192192, i32 -7340032, i32 2139095041, i32 -8388606], align 4
@FloatInfValues = dso_local local_unnamed_addr global [2 x i32] [i32 2139095040, i32 -8388608], align 4
@FloatZeroValues = dso_local local_unnamed_addr global [2 x i32] [i32 0, i32 -2147483648], align 4
@FloatDenormValues = dso_local local_unnamed_addr global [4 x i32] [i32 1, i32 -2147483647, i32 8388607, i32 -2139095041], align 4
@FloatNormalValues = dso_local local_unnamed_addr global [26 x i32] [i32 8388608, i32 -2139095040, i32 2139095039, i32 -8388609, i32 1065353216, i32 1065353215, i32 1065353217, i32 1069547520, i32 1067450368, i32 1066401792, i32 1056964608, i32 1048576000, i32 1040187392, i32 -1082130432, i32 -1082130433, i32 -1082130431, i32 -1077936128, i32 -1080033280, i32 -1081081856, i32 -1090519040, i32 -1098907648, i32 -1107296256, i32 1073741824, i32 1077936128, i32 -1073741824, i32 -1069547520], align 4
@.str = private unnamed_addr constant [58 x i8] c"Check '%s' in file '%s' at line %d failed for the value '\00", align 1
@.str.1 = private unnamed_addr constant [19 x i8] c"__builtin_isnan(X)\00", align 1
@.str.2 = private unnamed_addr constant [100 x i8] c"/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Float/classify-f32.h\00", align 1
@.str.3 = private unnamed_addr constant [3 x i8] c"%x\00", align 1
@.str.5 = private unnamed_addr constant [26 x i8] c"!__builtin_issignaling(X)\00", align 1
@.str.6 = private unnamed_addr constant [20 x i8] c"!__builtin_isinf(X)\00", align 1
@.str.8 = private unnamed_addr constant [23 x i8] c"!__builtin_isnormal(X)\00", align 1
@.str.9 = private unnamed_addr constant [26 x i8] c"!__builtin_issubnormal(X)\00", align 1
@.str.12 = private unnamed_addr constant [25 x i8] c"__builtin_issignaling(X)\00", align 1
@.str.13 = private unnamed_addr constant [20 x i8] c"!__builtin_isnan(X)\00", align 1
@.str.14 = private unnamed_addr constant [19 x i8] c"__builtin_isinf(X)\00", align 1
@.str.19 = private unnamed_addr constant [25 x i8] c"__builtin_issubnormal(X)\00", align 1
@.str.21 = private unnamed_addr constant [22 x i8] c"__builtin_isnormal(X)\00", align 1
@.str.23 = private unnamed_addr constant [53 x i8] c"Check '%s' failed for the value '%x' , FPCLASS=0x%x\0A\00", align 1
@.str.24 = private unnamed_addr constant [61 x i8] c"!!((fcSNan)&FPCLASS) == !!__builtin_isfpclass((x), (fcSNan))\00", align 1
@.str.25 = private unnamed_addr constant [61 x i8] c"!!((fcQNan)&FPCLASS) == !!__builtin_isfpclass((x), (fcQNan))\00", align 1
@.str.27 = private unnamed_addr constant [65 x i8] c"!!((fcPosInf)&FPCLASS) == !!__builtin_isfpclass((x), (fcPosInf))\00", align 1
@.str.28 = private unnamed_addr constant [65 x i8] c"!!((fcNegInf)&FPCLASS) == !!__builtin_isfpclass((x), (fcNegInf))\00", align 1
@.str.30 = private unnamed_addr constant [71 x i8] c"!!((fcPosNormal)&FPCLASS) == !!__builtin_isfpclass((x), (fcPosNormal))\00", align 1
@.str.31 = private unnamed_addr constant [71 x i8] c"!!((fcNegNormal)&FPCLASS) == !!__builtin_isfpclass((x), (fcNegNormal))\00", align 1
@.str.33 = private unnamed_addr constant [77 x i8] c"!!((fcPosSubnormal)&FPCLASS) == !!__builtin_isfpclass((x), (fcPosSubnormal))\00", align 1
@.str.34 = private unnamed_addr constant [77 x i8] c"!!((fcNegSubnormal)&FPCLASS) == !!__builtin_isfpclass((x), (fcNegSubnormal))\00", align 1
@.str.36 = private unnamed_addr constant [67 x i8] c"!!((fcPosZero)&FPCLASS) == !!__builtin_isfpclass((x), (fcPosZero))\00", align 1
@DoubleQNaNValues = dso_local local_unnamed_addr global [8 x i64] [i64 9223372036854775807, i64 -1, i64 9221120237041090560, i64 -2251799813685248, i64 9222246136947933184, i64 -1688849860263936, i64 9221120237041090561, i64 -2251799813685246], align 8
@DoubleSNaNValues = dso_local local_unnamed_addr global [6 x i64] [i64 9221120237041090559, i64 -2251799813685249, i64 9219994337134247936, i64 -3940649673949184, i64 9218868437227405313, i64 -4503599627370494], align 8
@DoubleInfValues = dso_local local_unnamed_addr global [2 x i64] [i64 9218868437227405312, i64 -4503599627370496], align 8
@DoubleZeroValues = dso_local local_unnamed_addr global [2 x i64] [i64 0, i64 -9223372036854775808], align 8
@DoubleDenormValues = dso_local local_unnamed_addr global [4 x i64] [i64 1, i64 -9223372036854775807, i64 4503599627370495, i64 -9218868437227405313], align 8
@DoubleNormalValues = dso_local local_unnamed_addr global [26 x i64] [i64 4503599627370496, i64 -9218868437227405312, i64 9218868437227405311, i64 -4503599627370497, i64 4607182418800017408, i64 4607182418800017407, i64 4607182418800017409, i64 4609434218613702656, i64 4608308318706860032, i64 4607745368753438720, i64 4602678819172646912, i64 4598175219545276416, i64 4593671619917905920, i64 -4616189618054758400, i64 -4616189618054758401, i64 -4616189618054758399, i64 -4613937818241073152, i64 -4615063718147915776, i64 -4615626668101337088, i64 -4620693217682128896, i64 -4625196817309499392, i64 -4629700416936869888, i64 4611686018427387904, i64 4613937818241073152, i64 -4611686018427387904, i64 -4609434218613702656], align 8
@.str.169 = private unnamed_addr constant [100 x i8] c"/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Float/classify-f64.h\00", align 1
@.str.170 = private unnamed_addr constant [4 x i8] c"%lx\00", align 1
@.str.171 = private unnamed_addr constant [54 x i8] c"Check '%s' failed for the value '%lx' , FPCLASS=0x%x\0A\00", align 1
@LongDoubleZeroValues = dso_local local_unnamed_addr global [2 x fp128] [fp128 0xL00000000000000000000000000000000, fp128 0xL00000000000000008000000000000000], align 16
@LongDoubleQNaNValues = dso_local local_unnamed_addr global [4 x fp128] zeroinitializer, align 16
@LongDoubleSNaNValues = dso_local local_unnamed_addr global [4 x fp128] zeroinitializer, align 16
@LongDoubleInfValues = dso_local local_unnamed_addr global [2 x fp128] zeroinitializer, align 16
@LongDoubleDenormValues = dso_local local_unnamed_addr global [2 x fp128] zeroinitializer, align 16
@LongDoubleNormalValues = dso_local local_unnamed_addr global [6 x fp128] zeroinitializer, align 16
@.str.172 = private unnamed_addr constant [104 x i8] c"/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Float/classify-ldouble.h\00", align 1
@.str.173 = private unnamed_addr constant [4 x i8] c"%Lg\00", align 1
@.str.174 = private unnamed_addr constant [54 x i8] c"Check '%s' failed for the value '%Lg' , FPCLASS=0x%x\0A\00", align 1
@str.224 = private unnamed_addr constant [2 x i8] c"'\00", align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @test_float() local_unnamed_addr #0 {
  %1 = load float, ptr @FloatQNaNValues, align 4, !tbaa !6
  %2 = fcmp uno float %1, 0.000000e+00
  br i1 %2, label %47, label %41

3:                                                ; preds = %47
  %4 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 4), align 4, !tbaa !6
  %5 = fcmp uno float %4, 0.000000e+00
  br i1 %5, label %6, label %41

6:                                                ; preds = %3
  %7 = tail call i1 @llvm.is.fpclass.f32(float %4, i32 1)
  br i1 %7, label %49, label %8

8:                                                ; preds = %6
  %9 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 8), align 4, !tbaa !6
  %10 = fcmp uno float %9, 0.000000e+00
  br i1 %10, label %11, label %41

11:                                               ; preds = %8
  %12 = tail call i1 @llvm.is.fpclass.f32(float %9, i32 1)
  br i1 %12, label %49, label %13

13:                                               ; preds = %11
  %14 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 12), align 4, !tbaa !6
  %15 = fcmp uno float %14, 0.000000e+00
  br i1 %15, label %16, label %41

16:                                               ; preds = %13
  %17 = tail call i1 @llvm.is.fpclass.f32(float %14, i32 1)
  br i1 %17, label %49, label %18

18:                                               ; preds = %16
  %19 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 16), align 4, !tbaa !6
  %20 = fcmp uno float %19, 0.000000e+00
  br i1 %20, label %21, label %41

21:                                               ; preds = %18
  %22 = tail call i1 @llvm.is.fpclass.f32(float %19, i32 1)
  br i1 %22, label %49, label %23

23:                                               ; preds = %21
  %24 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 20), align 4, !tbaa !6
  %25 = fcmp uno float %24, 0.000000e+00
  br i1 %25, label %26, label %41

26:                                               ; preds = %23
  %27 = tail call i1 @llvm.is.fpclass.f32(float %24, i32 1)
  br i1 %27, label %49, label %28

28:                                               ; preds = %26
  %29 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 24), align 4, !tbaa !6
  %30 = fcmp uno float %29, 0.000000e+00
  br i1 %30, label %31, label %41

31:                                               ; preds = %28
  %32 = tail call i1 @llvm.is.fpclass.f32(float %29, i32 1)
  br i1 %32, label %49, label %33

33:                                               ; preds = %31
  %34 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 28), align 4, !tbaa !6
  %35 = fcmp uno float %34, 0.000000e+00
  br i1 %35, label %36, label %41

36:                                               ; preds = %33
  %37 = tail call i1 @llvm.is.fpclass.f32(float %34, i32 1)
  br i1 %37, label %49, label %38

38:                                               ; preds = %36
  %39 = load float, ptr @FloatSNaNValues, align 4, !tbaa !6
  %40 = fcmp uno float %39, 0.000000e+00
  br i1 %40, label %89, label %83

41:                                               ; preds = %33, %28, %23, %18, %13, %8, %3, %0
  %42 = phi ptr [ @FloatQNaNValues, %0 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 4), %3 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 8), %8 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 12), %13 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 16), %18 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 20), %23 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 24), %28 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 28), %33 ]
  %43 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.2, i32 noundef 100)
  %44 = load i32, ptr %42, align 4, !tbaa !10
  %45 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %44)
  %46 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

47:                                               ; preds = %0
  %48 = tail call i1 @llvm.is.fpclass.f32(float %1, i32 1)
  br i1 %48, label %49, label %3

49:                                               ; preds = %36, %31, %26, %21, %16, %11, %6, %47
  %50 = phi ptr [ @FloatQNaNValues, %47 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 4), %6 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 8), %11 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 12), %16 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 16), %21 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 20), %26 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 24), %31 ], [ getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 28), %36 ]
  %51 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.2, i32 noundef 101)
  %52 = load i32, ptr %50, align 4, !tbaa !10
  %53 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %52)
  %54 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

55:                                               ; preds = %89
  %56 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 4), align 4, !tbaa !6
  %57 = fcmp uno float %56, 0.000000e+00
  br i1 %57, label %58, label %83

58:                                               ; preds = %55
  %59 = tail call i1 @llvm.is.fpclass.f32(float %56, i32 1)
  br i1 %59, label %60, label %91

60:                                               ; preds = %58
  %61 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 8), align 4, !tbaa !6
  %62 = fcmp uno float %61, 0.000000e+00
  br i1 %62, label %63, label %83

63:                                               ; preds = %60
  %64 = tail call i1 @llvm.is.fpclass.f32(float %61, i32 1)
  br i1 %64, label %65, label %91

65:                                               ; preds = %63
  %66 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 12), align 4, !tbaa !6
  %67 = fcmp uno float %66, 0.000000e+00
  br i1 %67, label %68, label %83

68:                                               ; preds = %65
  %69 = tail call i1 @llvm.is.fpclass.f32(float %66, i32 1)
  br i1 %69, label %70, label %91

70:                                               ; preds = %68
  %71 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 16), align 4, !tbaa !6
  %72 = fcmp uno float %71, 0.000000e+00
  br i1 %72, label %73, label %83

73:                                               ; preds = %70
  %74 = tail call i1 @llvm.is.fpclass.f32(float %71, i32 1)
  br i1 %74, label %75, label %91

75:                                               ; preds = %73
  %76 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 20), align 4, !tbaa !6
  %77 = fcmp uno float %76, 0.000000e+00
  br i1 %77, label %78, label %83

78:                                               ; preds = %75
  %79 = tail call i1 @llvm.is.fpclass.f32(float %76, i32 1)
  br i1 %79, label %80, label %91

80:                                               ; preds = %78
  %81 = load float, ptr @FloatInfValues, align 4, !tbaa !6
  %82 = fcmp uno float %81, 0.000000e+00
  br i1 %82, label %106, label %112

83:                                               ; preds = %75, %70, %65, %60, %55, %38
  %84 = phi ptr [ @FloatSNaNValues, %38 ], [ getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 4), %55 ], [ getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 8), %60 ], [ getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 12), %65 ], [ getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 16), %70 ], [ getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 20), %75 ]
  %85 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.2, i32 noundef 112)
  %86 = load i32, ptr %84, align 4, !tbaa !10
  %87 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %86)
  %88 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

89:                                               ; preds = %38
  %90 = tail call i1 @llvm.is.fpclass.f32(float %39, i32 1)
  br i1 %90, label %55, label %91

91:                                               ; preds = %78, %73, %68, %63, %58, %89
  %92 = phi ptr [ @FloatSNaNValues, %89 ], [ getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 4), %58 ], [ getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 8), %63 ], [ getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 12), %68 ], [ getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 16), %73 ], [ getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 20), %78 ]
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.12, ptr noundef nonnull @.str.2, i32 noundef 113)
  %94 = load i32, ptr %92, align 4, !tbaa !10
  %95 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %94)
  %96 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

97:                                               ; preds = %112
  %98 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatInfValues, i64 4), align 4, !tbaa !6
  %99 = fcmp uno float %98, 0.000000e+00
  br i1 %99, label %106, label %100

100:                                              ; preds = %97
  %101 = tail call float @llvm.fabs.f32(float %98)
  %102 = fcmp oeq float %101, 0x7FF0000000000000
  br i1 %102, label %103, label %115

103:                                              ; preds = %100
  %104 = load float, ptr @FloatZeroValues, align 4, !tbaa !6
  %105 = fcmp uno float %104, 0.000000e+00
  br i1 %105, label %134, label %140

106:                                              ; preds = %97, %80
  %107 = phi ptr [ @FloatInfValues, %80 ], [ getelementptr inbounds nuw (i8, ptr @FloatInfValues, i64 4), %97 ]
  %108 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.2, i32 noundef 124)
  %109 = load i32, ptr %107, align 4, !tbaa !10
  %110 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %109)
  %111 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

112:                                              ; preds = %80
  %113 = tail call float @llvm.fabs.f32(float %81)
  %114 = fcmp oeq float %113, 0x7FF0000000000000
  br i1 %114, label %97, label %115

115:                                              ; preds = %100, %112
  %116 = phi ptr [ @FloatInfValues, %112 ], [ getelementptr inbounds nuw (i8, ptr @FloatInfValues, i64 4), %100 ]
  %117 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.14, ptr noundef nonnull @.str.2, i32 noundef 126)
  %118 = load i32, ptr %116, align 4, !tbaa !10
  %119 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %118)
  %120 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

121:                                              ; preds = %157
  %122 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatZeroValues, i64 4), align 4, !tbaa !6
  %123 = fcmp uno float %122, 0.000000e+00
  br i1 %123, label %134, label %124

124:                                              ; preds = %121
  %125 = tail call float @llvm.fabs.f32(float %122)
  %126 = fcmp oeq float %125, 0x7FF0000000000000
  br i1 %126, label %143, label %127

127:                                              ; preds = %124
  %128 = tail call i1 @llvm.is.fpclass.f32(float %122, i32 264)
  br i1 %128, label %151, label %129

129:                                              ; preds = %127
  %130 = tail call i1 @llvm.is.fpclass.f32(float %122, i32 144)
  br i1 %130, label %159, label %131

131:                                              ; preds = %129
  %132 = load float, ptr @FloatDenormValues, align 4, !tbaa !6
  %133 = fcmp uno float %132, 0.000000e+00
  br i1 %133, label %198, label %204

134:                                              ; preds = %121, %103
  %135 = phi ptr [ @FloatZeroValues, %103 ], [ getelementptr inbounds nuw (i8, ptr @FloatZeroValues, i64 4), %121 ]
  %136 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.2, i32 noundef 136)
  %137 = load i32, ptr %135, align 4, !tbaa !10
  %138 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %137)
  %139 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

140:                                              ; preds = %103
  %141 = tail call float @llvm.fabs.f32(float %104)
  %142 = fcmp oeq float %141, 0x7FF0000000000000
  br i1 %142, label %143, label %149

143:                                              ; preds = %124, %140
  %144 = phi ptr [ @FloatZeroValues, %140 ], [ getelementptr inbounds nuw (i8, ptr @FloatZeroValues, i64 4), %124 ]
  %145 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.2, i32 noundef 138)
  %146 = load i32, ptr %144, align 4, !tbaa !10
  %147 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %146)
  %148 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

149:                                              ; preds = %140
  %150 = tail call i1 @llvm.is.fpclass.f32(float %104, i32 264)
  br i1 %150, label %151, label %157

151:                                              ; preds = %127, %149
  %152 = phi ptr [ @FloatZeroValues, %149 ], [ getelementptr inbounds nuw (i8, ptr @FloatZeroValues, i64 4), %127 ]
  %153 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.2, i32 noundef 140)
  %154 = load i32, ptr %152, align 4, !tbaa !10
  %155 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %154)
  %156 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

157:                                              ; preds = %149
  %158 = tail call i1 @llvm.is.fpclass.f32(float %104, i32 144)
  br i1 %158, label %159, label %121

159:                                              ; preds = %129, %157
  %160 = phi ptr [ @FloatZeroValues, %157 ], [ getelementptr inbounds nuw (i8, ptr @FloatZeroValues, i64 4), %129 ]
  %161 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.2, i32 noundef 141)
  %162 = load i32, ptr %160, align 4, !tbaa !10
  %163 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %162)
  %164 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

165:                                              ; preds = %221
  %166 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 4), align 4, !tbaa !6
  %167 = fcmp uno float %166, 0.000000e+00
  br i1 %167, label %198, label %168

168:                                              ; preds = %165
  %169 = tail call float @llvm.fabs.f32(float %166)
  %170 = fcmp oeq float %169, 0x7FF0000000000000
  br i1 %170, label %207, label %171

171:                                              ; preds = %168
  %172 = tail call i1 @llvm.is.fpclass.f32(float %166, i32 264)
  br i1 %172, label %215, label %173

173:                                              ; preds = %171
  %174 = tail call i1 @llvm.is.fpclass.f32(float %166, i32 144)
  br i1 %174, label %175, label %223

175:                                              ; preds = %173
  %176 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 8), align 4, !tbaa !6
  %177 = fcmp uno float %176, 0.000000e+00
  br i1 %177, label %198, label %178

178:                                              ; preds = %175
  %179 = tail call float @llvm.fabs.f32(float %176)
  %180 = fcmp oeq float %179, 0x7FF0000000000000
  br i1 %180, label %207, label %181

181:                                              ; preds = %178
  %182 = tail call i1 @llvm.is.fpclass.f32(float %176, i32 264)
  br i1 %182, label %215, label %183

183:                                              ; preds = %181
  %184 = tail call i1 @llvm.is.fpclass.f32(float %176, i32 144)
  br i1 %184, label %185, label %223

185:                                              ; preds = %183
  %186 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 12), align 4, !tbaa !6
  %187 = fcmp uno float %186, 0.000000e+00
  br i1 %187, label %198, label %188

188:                                              ; preds = %185
  %189 = tail call float @llvm.fabs.f32(float %186)
  %190 = fcmp oeq float %189, 0x7FF0000000000000
  br i1 %190, label %207, label %191

191:                                              ; preds = %188
  %192 = tail call i1 @llvm.is.fpclass.f32(float %186, i32 264)
  br i1 %192, label %215, label %193

193:                                              ; preds = %191
  %194 = tail call i1 @llvm.is.fpclass.f32(float %186, i32 144)
  br i1 %194, label %195, label %223

195:                                              ; preds = %193
  %196 = load float, ptr @FloatNormalValues, align 4, !tbaa !6
  %197 = fcmp uno float %196, 0.000000e+00
  br i1 %197, label %430, label %436

198:                                              ; preds = %185, %175, %165, %131
  %199 = phi ptr [ @FloatDenormValues, %131 ], [ getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 4), %165 ], [ getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 8), %175 ], [ getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 12), %185 ]
  %200 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.2, i32 noundef 148)
  %201 = load i32, ptr %199, align 4, !tbaa !10
  %202 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %201)
  %203 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

204:                                              ; preds = %131
  %205 = tail call float @llvm.fabs.f32(float %132)
  %206 = fcmp oeq float %205, 0x7FF0000000000000
  br i1 %206, label %207, label %213

207:                                              ; preds = %188, %178, %168, %204
  %208 = phi ptr [ @FloatDenormValues, %204 ], [ getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 4), %168 ], [ getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 8), %178 ], [ getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 12), %188 ]
  %209 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.2, i32 noundef 150)
  %210 = load i32, ptr %208, align 4, !tbaa !10
  %211 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %210)
  %212 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

213:                                              ; preds = %204
  %214 = tail call i1 @llvm.is.fpclass.f32(float %132, i32 264)
  br i1 %214, label %215, label %221

215:                                              ; preds = %191, %181, %171, %213
  %216 = phi ptr [ @FloatDenormValues, %213 ], [ getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 4), %171 ], [ getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 8), %181 ], [ getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 12), %191 ]
  %217 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.2, i32 noundef 152)
  %218 = load i32, ptr %216, align 4, !tbaa !10
  %219 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %218)
  %220 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

221:                                              ; preds = %213
  %222 = tail call i1 @llvm.is.fpclass.f32(float %132, i32 144)
  br i1 %222, label %165, label %223

223:                                              ; preds = %193, %183, %173, %221
  %224 = phi ptr [ @FloatDenormValues, %221 ], [ getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 4), %173 ], [ getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 8), %183 ], [ getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 12), %193 ]
  %225 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.19, ptr noundef nonnull @.str.2, i32 noundef 153)
  %226 = load i32, ptr %224, align 4, !tbaa !10
  %227 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %226)
  %228 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

229:                                              ; preds = %445
  %230 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 4), align 4, !tbaa !6
  %231 = fcmp uno float %230, 0.000000e+00
  br i1 %231, label %430, label %232

232:                                              ; preds = %229
  %233 = tail call float @llvm.fabs.f32(float %230)
  %234 = fcmp oeq float %233, 0x7FF0000000000000
  br i1 %234, label %439, label %235

235:                                              ; preds = %232
  %236 = tail call i1 @llvm.is.fpclass.f32(float %230, i32 264)
  br i1 %236, label %237, label %447

237:                                              ; preds = %235
  %238 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 8), align 4, !tbaa !6
  %239 = fcmp uno float %238, 0.000000e+00
  br i1 %239, label %430, label %240

240:                                              ; preds = %237
  %241 = tail call float @llvm.fabs.f32(float %238)
  %242 = fcmp oeq float %241, 0x7FF0000000000000
  br i1 %242, label %439, label %243

243:                                              ; preds = %240
  %244 = tail call i1 @llvm.is.fpclass.f32(float %238, i32 264)
  br i1 %244, label %245, label %447

245:                                              ; preds = %243
  %246 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 12), align 4, !tbaa !6
  %247 = fcmp uno float %246, 0.000000e+00
  br i1 %247, label %430, label %248

248:                                              ; preds = %245
  %249 = tail call float @llvm.fabs.f32(float %246)
  %250 = fcmp oeq float %249, 0x7FF0000000000000
  br i1 %250, label %439, label %251

251:                                              ; preds = %248
  %252 = tail call i1 @llvm.is.fpclass.f32(float %246, i32 264)
  br i1 %252, label %253, label %447

253:                                              ; preds = %251
  %254 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 16), align 4, !tbaa !6
  %255 = fcmp uno float %254, 0.000000e+00
  br i1 %255, label %430, label %256

256:                                              ; preds = %253
  %257 = tail call float @llvm.fabs.f32(float %254)
  %258 = fcmp oeq float %257, 0x7FF0000000000000
  br i1 %258, label %439, label %259

259:                                              ; preds = %256
  %260 = tail call i1 @llvm.is.fpclass.f32(float %254, i32 264)
  br i1 %260, label %261, label %447

261:                                              ; preds = %259
  %262 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 20), align 4, !tbaa !6
  %263 = fcmp uno float %262, 0.000000e+00
  br i1 %263, label %430, label %264

264:                                              ; preds = %261
  %265 = tail call float @llvm.fabs.f32(float %262)
  %266 = fcmp oeq float %265, 0x7FF0000000000000
  br i1 %266, label %439, label %267

267:                                              ; preds = %264
  %268 = tail call i1 @llvm.is.fpclass.f32(float %262, i32 264)
  br i1 %268, label %269, label %447

269:                                              ; preds = %267
  %270 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 24), align 4, !tbaa !6
  %271 = fcmp uno float %270, 0.000000e+00
  br i1 %271, label %430, label %272

272:                                              ; preds = %269
  %273 = tail call float @llvm.fabs.f32(float %270)
  %274 = fcmp oeq float %273, 0x7FF0000000000000
  br i1 %274, label %439, label %275

275:                                              ; preds = %272
  %276 = tail call i1 @llvm.is.fpclass.f32(float %270, i32 264)
  br i1 %276, label %277, label %447

277:                                              ; preds = %275
  %278 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 28), align 4, !tbaa !6
  %279 = fcmp uno float %278, 0.000000e+00
  br i1 %279, label %430, label %280

280:                                              ; preds = %277
  %281 = tail call float @llvm.fabs.f32(float %278)
  %282 = fcmp oeq float %281, 0x7FF0000000000000
  br i1 %282, label %439, label %283

283:                                              ; preds = %280
  %284 = tail call i1 @llvm.is.fpclass.f32(float %278, i32 264)
  br i1 %284, label %285, label %447

285:                                              ; preds = %283
  %286 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 32), align 4, !tbaa !6
  %287 = fcmp uno float %286, 0.000000e+00
  br i1 %287, label %430, label %288

288:                                              ; preds = %285
  %289 = tail call float @llvm.fabs.f32(float %286)
  %290 = fcmp oeq float %289, 0x7FF0000000000000
  br i1 %290, label %439, label %291

291:                                              ; preds = %288
  %292 = tail call i1 @llvm.is.fpclass.f32(float %286, i32 264)
  br i1 %292, label %293, label %447

293:                                              ; preds = %291
  %294 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 36), align 4, !tbaa !6
  %295 = fcmp uno float %294, 0.000000e+00
  br i1 %295, label %430, label %296

296:                                              ; preds = %293
  %297 = tail call float @llvm.fabs.f32(float %294)
  %298 = fcmp oeq float %297, 0x7FF0000000000000
  br i1 %298, label %439, label %299

299:                                              ; preds = %296
  %300 = tail call i1 @llvm.is.fpclass.f32(float %294, i32 264)
  br i1 %300, label %301, label %447

301:                                              ; preds = %299
  %302 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 40), align 4, !tbaa !6
  %303 = fcmp uno float %302, 0.000000e+00
  br i1 %303, label %430, label %304

304:                                              ; preds = %301
  %305 = tail call float @llvm.fabs.f32(float %302)
  %306 = fcmp oeq float %305, 0x7FF0000000000000
  br i1 %306, label %439, label %307

307:                                              ; preds = %304
  %308 = tail call i1 @llvm.is.fpclass.f32(float %302, i32 264)
  br i1 %308, label %309, label %447

309:                                              ; preds = %307
  %310 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 44), align 4, !tbaa !6
  %311 = fcmp uno float %310, 0.000000e+00
  br i1 %311, label %430, label %312

312:                                              ; preds = %309
  %313 = tail call float @llvm.fabs.f32(float %310)
  %314 = fcmp oeq float %313, 0x7FF0000000000000
  br i1 %314, label %439, label %315

315:                                              ; preds = %312
  %316 = tail call i1 @llvm.is.fpclass.f32(float %310, i32 264)
  br i1 %316, label %317, label %447

317:                                              ; preds = %315
  %318 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 48), align 4, !tbaa !6
  %319 = fcmp uno float %318, 0.000000e+00
  br i1 %319, label %430, label %320

320:                                              ; preds = %317
  %321 = tail call float @llvm.fabs.f32(float %318)
  %322 = fcmp oeq float %321, 0x7FF0000000000000
  br i1 %322, label %439, label %323

323:                                              ; preds = %320
  %324 = tail call i1 @llvm.is.fpclass.f32(float %318, i32 264)
  br i1 %324, label %325, label %447

325:                                              ; preds = %323
  %326 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 52), align 4, !tbaa !6
  %327 = fcmp uno float %326, 0.000000e+00
  br i1 %327, label %430, label %328

328:                                              ; preds = %325
  %329 = tail call float @llvm.fabs.f32(float %326)
  %330 = fcmp oeq float %329, 0x7FF0000000000000
  br i1 %330, label %439, label %331

331:                                              ; preds = %328
  %332 = tail call i1 @llvm.is.fpclass.f32(float %326, i32 264)
  br i1 %332, label %333, label %447

333:                                              ; preds = %331
  %334 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 56), align 4, !tbaa !6
  %335 = fcmp uno float %334, 0.000000e+00
  br i1 %335, label %430, label %336

336:                                              ; preds = %333
  %337 = tail call float @llvm.fabs.f32(float %334)
  %338 = fcmp oeq float %337, 0x7FF0000000000000
  br i1 %338, label %439, label %339

339:                                              ; preds = %336
  %340 = tail call i1 @llvm.is.fpclass.f32(float %334, i32 264)
  br i1 %340, label %341, label %447

341:                                              ; preds = %339
  %342 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 60), align 4, !tbaa !6
  %343 = fcmp uno float %342, 0.000000e+00
  br i1 %343, label %430, label %344

344:                                              ; preds = %341
  %345 = tail call float @llvm.fabs.f32(float %342)
  %346 = fcmp oeq float %345, 0x7FF0000000000000
  br i1 %346, label %439, label %347

347:                                              ; preds = %344
  %348 = tail call i1 @llvm.is.fpclass.f32(float %342, i32 264)
  br i1 %348, label %349, label %447

349:                                              ; preds = %347
  %350 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 64), align 4, !tbaa !6
  %351 = fcmp uno float %350, 0.000000e+00
  br i1 %351, label %430, label %352

352:                                              ; preds = %349
  %353 = tail call float @llvm.fabs.f32(float %350)
  %354 = fcmp oeq float %353, 0x7FF0000000000000
  br i1 %354, label %439, label %355

355:                                              ; preds = %352
  %356 = tail call i1 @llvm.is.fpclass.f32(float %350, i32 264)
  br i1 %356, label %357, label %447

357:                                              ; preds = %355
  %358 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 68), align 4, !tbaa !6
  %359 = fcmp uno float %358, 0.000000e+00
  br i1 %359, label %430, label %360

360:                                              ; preds = %357
  %361 = tail call float @llvm.fabs.f32(float %358)
  %362 = fcmp oeq float %361, 0x7FF0000000000000
  br i1 %362, label %439, label %363

363:                                              ; preds = %360
  %364 = tail call i1 @llvm.is.fpclass.f32(float %358, i32 264)
  br i1 %364, label %365, label %447

365:                                              ; preds = %363
  %366 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 72), align 4, !tbaa !6
  %367 = fcmp uno float %366, 0.000000e+00
  br i1 %367, label %430, label %368

368:                                              ; preds = %365
  %369 = tail call float @llvm.fabs.f32(float %366)
  %370 = fcmp oeq float %369, 0x7FF0000000000000
  br i1 %370, label %439, label %371

371:                                              ; preds = %368
  %372 = tail call i1 @llvm.is.fpclass.f32(float %366, i32 264)
  br i1 %372, label %373, label %447

373:                                              ; preds = %371
  %374 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 76), align 4, !tbaa !6
  %375 = fcmp uno float %374, 0.000000e+00
  br i1 %375, label %430, label %376

376:                                              ; preds = %373
  %377 = tail call float @llvm.fabs.f32(float %374)
  %378 = fcmp oeq float %377, 0x7FF0000000000000
  br i1 %378, label %439, label %379

379:                                              ; preds = %376
  %380 = tail call i1 @llvm.is.fpclass.f32(float %374, i32 264)
  br i1 %380, label %381, label %447

381:                                              ; preds = %379
  %382 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 80), align 4, !tbaa !6
  %383 = fcmp uno float %382, 0.000000e+00
  br i1 %383, label %430, label %384

384:                                              ; preds = %381
  %385 = tail call float @llvm.fabs.f32(float %382)
  %386 = fcmp oeq float %385, 0x7FF0000000000000
  br i1 %386, label %439, label %387

387:                                              ; preds = %384
  %388 = tail call i1 @llvm.is.fpclass.f32(float %382, i32 264)
  br i1 %388, label %389, label %447

389:                                              ; preds = %387
  %390 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 84), align 4, !tbaa !6
  %391 = fcmp uno float %390, 0.000000e+00
  br i1 %391, label %430, label %392

392:                                              ; preds = %389
  %393 = tail call float @llvm.fabs.f32(float %390)
  %394 = fcmp oeq float %393, 0x7FF0000000000000
  br i1 %394, label %439, label %395

395:                                              ; preds = %392
  %396 = tail call i1 @llvm.is.fpclass.f32(float %390, i32 264)
  br i1 %396, label %397, label %447

397:                                              ; preds = %395
  %398 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 88), align 4, !tbaa !6
  %399 = fcmp uno float %398, 0.000000e+00
  br i1 %399, label %430, label %400

400:                                              ; preds = %397
  %401 = tail call float @llvm.fabs.f32(float %398)
  %402 = fcmp oeq float %401, 0x7FF0000000000000
  br i1 %402, label %439, label %403

403:                                              ; preds = %400
  %404 = tail call i1 @llvm.is.fpclass.f32(float %398, i32 264)
  br i1 %404, label %405, label %447

405:                                              ; preds = %403
  %406 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 92), align 4, !tbaa !6
  %407 = fcmp uno float %406, 0.000000e+00
  br i1 %407, label %430, label %408

408:                                              ; preds = %405
  %409 = tail call float @llvm.fabs.f32(float %406)
  %410 = fcmp oeq float %409, 0x7FF0000000000000
  br i1 %410, label %439, label %411

411:                                              ; preds = %408
  %412 = tail call i1 @llvm.is.fpclass.f32(float %406, i32 264)
  br i1 %412, label %413, label %447

413:                                              ; preds = %411
  %414 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 96), align 4, !tbaa !6
  %415 = fcmp uno float %414, 0.000000e+00
  br i1 %415, label %430, label %416

416:                                              ; preds = %413
  %417 = tail call float @llvm.fabs.f32(float %414)
  %418 = fcmp oeq float %417, 0x7FF0000000000000
  br i1 %418, label %439, label %419

419:                                              ; preds = %416
  %420 = tail call i1 @llvm.is.fpclass.f32(float %414, i32 264)
  br i1 %420, label %421, label %447

421:                                              ; preds = %419
  %422 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 100), align 4, !tbaa !6
  %423 = fcmp uno float %422, 0.000000e+00
  br i1 %423, label %430, label %424

424:                                              ; preds = %421
  %425 = tail call float @llvm.fabs.f32(float %422)
  %426 = fcmp oeq float %425, 0x7FF0000000000000
  br i1 %426, label %439, label %427

427:                                              ; preds = %424
  %428 = tail call i1 @llvm.is.fpclass.f32(float %422, i32 264)
  br i1 %428, label %429, label %447

429:                                              ; preds = %427
  ret i32 0

430:                                              ; preds = %421, %413, %405, %397, %389, %381, %373, %365, %357, %349, %341, %333, %325, %317, %309, %301, %293, %285, %277, %269, %261, %253, %245, %237, %229, %195
  %431 = phi ptr [ @FloatNormalValues, %195 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 4), %229 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 8), %237 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 12), %245 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 16), %253 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 20), %261 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 24), %269 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 28), %277 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 32), %285 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 36), %293 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 40), %301 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 44), %309 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 48), %317 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 52), %325 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 56), %333 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 60), %341 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 64), %349 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 68), %357 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 72), %365 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 76), %373 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 80), %381 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 84), %389 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 88), %397 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 92), %405 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 96), %413 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 100), %421 ]
  %432 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.2, i32 noundef 160)
  %433 = load i32, ptr %431, align 4, !tbaa !10
  %434 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %433)
  %435 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

436:                                              ; preds = %195
  %437 = tail call float @llvm.fabs.f32(float %196)
  %438 = fcmp oeq float %437, 0x7FF0000000000000
  br i1 %438, label %439, label %445

439:                                              ; preds = %424, %416, %408, %400, %392, %384, %376, %368, %360, %352, %344, %336, %328, %320, %312, %304, %296, %288, %280, %272, %264, %256, %248, %240, %232, %436
  %440 = phi ptr [ @FloatNormalValues, %436 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 4), %232 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 8), %240 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 12), %248 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 16), %256 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 20), %264 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 24), %272 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 28), %280 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 32), %288 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 36), %296 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 40), %304 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 44), %312 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 48), %320 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 52), %328 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 56), %336 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 60), %344 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 64), %352 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 68), %360 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 72), %368 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 76), %376 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 80), %384 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 84), %392 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 88), %400 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 92), %408 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 96), %416 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 100), %424 ]
  %441 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.2, i32 noundef 162)
  %442 = load i32, ptr %440, align 4, !tbaa !10
  %443 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %442)
  %444 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

445:                                              ; preds = %436
  %446 = tail call i1 @llvm.is.fpclass.f32(float %196, i32 264)
  br i1 %446, label %229, label %447

447:                                              ; preds = %427, %419, %411, %403, %395, %387, %379, %371, %363, %355, %347, %339, %331, %323, %315, %307, %299, %291, %283, %275, %267, %259, %251, %243, %235, %445
  %448 = phi ptr [ @FloatNormalValues, %445 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 4), %235 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 8), %243 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 12), %251 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 16), %259 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 20), %267 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 24), %275 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 28), %283 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 32), %291 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 36), %299 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 40), %307 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 44), %315 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 48), %323 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 52), %331 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 56), %339 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 60), %347 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 64), %355 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 68), %363 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 72), %371 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 76), %379 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 80), %387 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 84), %395 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 88), %403 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 92), %411 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 96), %419 ], [ getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 100), %427 ]
  %449 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.21, ptr noundef nonnull @.str.2, i32 noundef 164)
  %450 = load i32, ptr %448, align 4, !tbaa !10
  %451 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %450)
  %452 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f32(float, i32 immarg) #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcSNan_float(float noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 1)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast float %0 to i32
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.24, i32 noundef %4, i32 noundef 1)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcQNan_float(float noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast float %0 to i32
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.24, i32 noundef %4, i32 noundef 2)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 2)
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast float %0 to i32
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.25, i32 noundef %9, i32 noundef 2)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcPosInf_float(float noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast float %0 to i32
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.24, i32 noundef %4, i32 noundef 512)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord float %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast float %0 to i32
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.25, i32 noundef %9, i32 noundef 512)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp oeq float %0, 0x7FF0000000000000
  br i1 %12, label %16, label %13

13:                                               ; preds = %11
  %14 = bitcast float %0 to i32
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.27, i32 noundef %14, i32 noundef 512)
  tail call void @exit(i32 noundef -1) #6
  unreachable

16:                                               ; preds = %11
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcNegInf_float(float noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast float %0 to i32
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.24, i32 noundef %4, i32 noundef 4)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord float %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast float %0 to i32
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.25, i32 noundef %9, i32 noundef 4)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une float %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.27, i32 noundef 2139095040, i32 noundef 4)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp oeq float %0, 0xFFF0000000000000
  br i1 %16, label %20, label %17

17:                                               ; preds = %15
  %18 = bitcast float %0 to i32
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.28, i32 noundef %18, i32 noundef 4)
  tail call void @exit(i32 noundef -1) #6
  unreachable

20:                                               ; preds = %15
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcPosNormal_float(float noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast float %0 to i32
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.24, i32 noundef %4, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord float %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast float %0 to i32
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.25, i32 noundef %9, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une float %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.27, i32 noundef 2139095040, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp une float %0, 0xFFF0000000000000
  br i1 %16, label %19, label %17

17:                                               ; preds = %15
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.28, i32 noundef -8388608, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

19:                                               ; preds = %15
  %20 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 256)
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  %22 = bitcast float %0 to i32
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.30, i32 noundef %22, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

24:                                               ; preds = %19
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcNegNormal_float(float noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast float %0 to i32
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.24, i32 noundef %4, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord float %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast float %0 to i32
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.25, i32 noundef %9, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une float %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.27, i32 noundef 2139095040, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp une float %0, 0xFFF0000000000000
  br i1 %16, label %19, label %17

17:                                               ; preds = %15
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.28, i32 noundef -8388608, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

19:                                               ; preds = %15
  %20 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 248)
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  %22 = bitcast float %0 to i32
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.30, i32 noundef %22, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

24:                                               ; preds = %19
  %25 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 8)
  br i1 %25, label %29, label %26

26:                                               ; preds = %24
  %27 = bitcast float %0 to i32
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.31, i32 noundef %27, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %24
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcPosSubnormal_float(float noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast float %0 to i32
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.24, i32 noundef %4, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord float %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast float %0 to i32
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.25, i32 noundef %9, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une float %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.27, i32 noundef 2139095040, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp une float %0, 0xFFF0000000000000
  br i1 %16, label %19, label %17

17:                                               ; preds = %15
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.28, i32 noundef -8388608, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

19:                                               ; preds = %15
  %20 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 248)
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  %22 = bitcast float %0 to i32
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.30, i32 noundef %22, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

24:                                               ; preds = %19
  %25 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 240)
  br i1 %25, label %29, label %26

26:                                               ; preds = %24
  %27 = bitcast float %0 to i32
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.31, i32 noundef %27, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %24
  %30 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 128)
  br i1 %30, label %34, label %31

31:                                               ; preds = %29
  %32 = bitcast float %0 to i32
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.33, i32 noundef %32, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

34:                                               ; preds = %29
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcNegSubnormal_float(float noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast float %0 to i32
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.24, i32 noundef %4, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord float %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast float %0 to i32
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.25, i32 noundef %9, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une float %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.27, i32 noundef 2139095040, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp une float %0, 0xFFF0000000000000
  br i1 %16, label %19, label %17

17:                                               ; preds = %15
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.28, i32 noundef -8388608, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

19:                                               ; preds = %15
  %20 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 248)
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  %22 = bitcast float %0 to i32
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.30, i32 noundef %22, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

24:                                               ; preds = %19
  %25 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 240)
  br i1 %25, label %29, label %26

26:                                               ; preds = %24
  %27 = bitcast float %0 to i32
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.31, i32 noundef %27, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %24
  %30 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 112)
  br i1 %30, label %34, label %31

31:                                               ; preds = %29
  %32 = bitcast float %0 to i32
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.33, i32 noundef %32, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

34:                                               ; preds = %29
  %35 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 16)
  br i1 %35, label %39, label %36

36:                                               ; preds = %34
  %37 = bitcast float %0 to i32
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.34, i32 noundef %37, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

39:                                               ; preds = %34
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcPosZero_float(float noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast float %0 to i32
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.24, i32 noundef %4, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord float %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast float %0 to i32
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.25, i32 noundef %9, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une float %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.27, i32 noundef 2139095040, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp une float %0, 0xFFF0000000000000
  br i1 %16, label %19, label %17

17:                                               ; preds = %15
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.28, i32 noundef -8388608, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

19:                                               ; preds = %15
  %20 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 248)
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  %22 = bitcast float %0 to i32
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.30, i32 noundef %22, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

24:                                               ; preds = %19
  %25 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 240)
  br i1 %25, label %29, label %26

26:                                               ; preds = %24
  %27 = bitcast float %0 to i32
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.31, i32 noundef %27, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %24
  %30 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 112)
  br i1 %30, label %34, label %31

31:                                               ; preds = %29
  %32 = bitcast float %0 to i32
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.33, i32 noundef %32, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

34:                                               ; preds = %29
  %35 = fcmp oeq float %0, 0.000000e+00
  br i1 %35, label %39, label %36

36:                                               ; preds = %34
  %37 = bitcast float %0 to i32
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.34, i32 noundef %37, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

39:                                               ; preds = %34
  %40 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 64)
  br i1 %40, label %44, label %41

41:                                               ; preds = %39
  %42 = bitcast float %0 to i32
  %43 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.36, i32 noundef %42, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

44:                                               ; preds = %39
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcNegZero_float(float noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast float %0 to i32
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.24, i32 noundef %4, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord float %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast float %0 to i32
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.25, i32 noundef %9, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une float %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.27, i32 noundef 2139095040, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp une float %0, 0xFFF0000000000000
  br i1 %16, label %19, label %17

17:                                               ; preds = %15
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.28, i32 noundef -8388608, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

19:                                               ; preds = %15
  %20 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 248)
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  %22 = bitcast float %0 to i32
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.30, i32 noundef %22, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

24:                                               ; preds = %19
  %25 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 240)
  br i1 %25, label %29, label %26

26:                                               ; preds = %24
  %27 = bitcast float %0 to i32
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.31, i32 noundef %27, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %24
  %30 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 112)
  br i1 %30, label %34, label %31

31:                                               ; preds = %29
  %32 = bitcast float %0 to i32
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.33, i32 noundef %32, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

34:                                               ; preds = %29
  %35 = fcmp oeq float %0, 0.000000e+00
  br i1 %35, label %39, label %36

36:                                               ; preds = %34
  %37 = bitcast float %0 to i32
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.34, i32 noundef %37, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

39:                                               ; preds = %34
  %40 = tail call i1 @llvm.is.fpclass.f32(float %0, i32 32)
  br i1 %40, label %44, label %41

41:                                               ; preds = %39
  %42 = bitcast float %0 to i32
  %43 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.36, i32 noundef %42, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

44:                                               ; preds = %39
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_isfpclass_float() local_unnamed_addr #0 {
  %1 = load float, ptr @FloatZeroValues, align 4, !tbaa !6
  %2 = bitcast float %1 to i32
  %3 = icmp sgt i32 %2, -1
  br i1 %3, label %4, label %5

4:                                                ; preds = %0
  tail call void @test_fcPosZero_float(float noundef %1)
  br label %6

5:                                                ; preds = %0
  tail call void @test_fcNegZero_float(float noundef %1)
  br label %6

6:                                                ; preds = %5, %4
  %7 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatZeroValues, i64 4), align 4, !tbaa !6
  %8 = bitcast float %7 to i32
  %9 = icmp sgt i32 %8, -1
  br i1 %9, label %11, label %10

10:                                               ; preds = %6
  tail call void @test_fcNegZero_float(float noundef %7)
  br label %12

11:                                               ; preds = %6
  tail call void @test_fcPosZero_float(float noundef %7)
  br label %12

12:                                               ; preds = %11, %10
  %13 = load float, ptr @FloatDenormValues, align 4, !tbaa !6
  %14 = fcmp olt float %13, 0.000000e+00
  br i1 %14, label %15, label %16

15:                                               ; preds = %12
  tail call void @test_fcNegSubnormal_float(float noundef %13)
  br label %17

16:                                               ; preds = %12
  tail call void @test_fcPosSubnormal_float(float noundef %13)
  br label %17

17:                                               ; preds = %16, %15
  %18 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 4), align 4, !tbaa !6
  %19 = fcmp olt float %18, 0.000000e+00
  br i1 %19, label %21, label %20

20:                                               ; preds = %17
  tail call void @test_fcPosSubnormal_float(float noundef %18)
  br label %22

21:                                               ; preds = %17
  tail call void @test_fcNegSubnormal_float(float noundef %18)
  br label %22

22:                                               ; preds = %21, %20
  %23 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 8), align 4, !tbaa !6
  %24 = fcmp olt float %23, 0.000000e+00
  br i1 %24, label %26, label %25

25:                                               ; preds = %22
  tail call void @test_fcPosSubnormal_float(float noundef %23)
  br label %27

26:                                               ; preds = %22
  tail call void @test_fcNegSubnormal_float(float noundef %23)
  br label %27

27:                                               ; preds = %26, %25
  %28 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatDenormValues, i64 12), align 4, !tbaa !6
  %29 = fcmp olt float %28, 0.000000e+00
  br i1 %29, label %31, label %30

30:                                               ; preds = %27
  tail call void @test_fcPosSubnormal_float(float noundef %28)
  br label %32

31:                                               ; preds = %27
  tail call void @test_fcNegSubnormal_float(float noundef %28)
  br label %32

32:                                               ; preds = %31, %30
  %33 = load float, ptr @FloatNormalValues, align 4, !tbaa !6
  %34 = fcmp olt float %33, 0.000000e+00
  br i1 %34, label %35, label %36

35:                                               ; preds = %32
  tail call void @test_fcNegNormal_float(float noundef %33)
  br label %37

36:                                               ; preds = %32
  tail call void @test_fcPosNormal_float(float noundef %33)
  br label %37

37:                                               ; preds = %36, %35
  %38 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 4), align 4, !tbaa !6
  %39 = fcmp olt float %38, 0.000000e+00
  br i1 %39, label %41, label %40

40:                                               ; preds = %37
  tail call void @test_fcPosNormal_float(float noundef %38)
  br label %42

41:                                               ; preds = %37
  tail call void @test_fcNegNormal_float(float noundef %38)
  br label %42

42:                                               ; preds = %41, %40
  %43 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 8), align 4, !tbaa !6
  %44 = fcmp olt float %43, 0.000000e+00
  br i1 %44, label %46, label %45

45:                                               ; preds = %42
  tail call void @test_fcPosNormal_float(float noundef %43)
  br label %47

46:                                               ; preds = %42
  tail call void @test_fcNegNormal_float(float noundef %43)
  br label %47

47:                                               ; preds = %46, %45
  %48 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 12), align 4, !tbaa !6
  %49 = fcmp olt float %48, 0.000000e+00
  br i1 %49, label %51, label %50

50:                                               ; preds = %47
  tail call void @test_fcPosNormal_float(float noundef %48)
  br label %52

51:                                               ; preds = %47
  tail call void @test_fcNegNormal_float(float noundef %48)
  br label %52

52:                                               ; preds = %51, %50
  %53 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 16), align 4, !tbaa !6
  %54 = fcmp olt float %53, 0.000000e+00
  br i1 %54, label %56, label %55

55:                                               ; preds = %52
  tail call void @test_fcPosNormal_float(float noundef %53)
  br label %57

56:                                               ; preds = %52
  tail call void @test_fcNegNormal_float(float noundef %53)
  br label %57

57:                                               ; preds = %56, %55
  %58 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 20), align 4, !tbaa !6
  %59 = fcmp olt float %58, 0.000000e+00
  br i1 %59, label %61, label %60

60:                                               ; preds = %57
  tail call void @test_fcPosNormal_float(float noundef %58)
  br label %62

61:                                               ; preds = %57
  tail call void @test_fcNegNormal_float(float noundef %58)
  br label %62

62:                                               ; preds = %61, %60
  %63 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 24), align 4, !tbaa !6
  %64 = fcmp olt float %63, 0.000000e+00
  br i1 %64, label %66, label %65

65:                                               ; preds = %62
  tail call void @test_fcPosNormal_float(float noundef %63)
  br label %67

66:                                               ; preds = %62
  tail call void @test_fcNegNormal_float(float noundef %63)
  br label %67

67:                                               ; preds = %66, %65
  %68 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 28), align 4, !tbaa !6
  %69 = fcmp olt float %68, 0.000000e+00
  br i1 %69, label %71, label %70

70:                                               ; preds = %67
  tail call void @test_fcPosNormal_float(float noundef %68)
  br label %72

71:                                               ; preds = %67
  tail call void @test_fcNegNormal_float(float noundef %68)
  br label %72

72:                                               ; preds = %71, %70
  %73 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 32), align 4, !tbaa !6
  %74 = fcmp olt float %73, 0.000000e+00
  br i1 %74, label %76, label %75

75:                                               ; preds = %72
  tail call void @test_fcPosNormal_float(float noundef %73)
  br label %77

76:                                               ; preds = %72
  tail call void @test_fcNegNormal_float(float noundef %73)
  br label %77

77:                                               ; preds = %76, %75
  %78 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 36), align 4, !tbaa !6
  %79 = fcmp olt float %78, 0.000000e+00
  br i1 %79, label %81, label %80

80:                                               ; preds = %77
  tail call void @test_fcPosNormal_float(float noundef %78)
  br label %82

81:                                               ; preds = %77
  tail call void @test_fcNegNormal_float(float noundef %78)
  br label %82

82:                                               ; preds = %81, %80
  %83 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 40), align 4, !tbaa !6
  %84 = fcmp olt float %83, 0.000000e+00
  br i1 %84, label %86, label %85

85:                                               ; preds = %82
  tail call void @test_fcPosNormal_float(float noundef %83)
  br label %87

86:                                               ; preds = %82
  tail call void @test_fcNegNormal_float(float noundef %83)
  br label %87

87:                                               ; preds = %86, %85
  %88 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 44), align 4, !tbaa !6
  %89 = fcmp olt float %88, 0.000000e+00
  br i1 %89, label %91, label %90

90:                                               ; preds = %87
  tail call void @test_fcPosNormal_float(float noundef %88)
  br label %92

91:                                               ; preds = %87
  tail call void @test_fcNegNormal_float(float noundef %88)
  br label %92

92:                                               ; preds = %91, %90
  %93 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 48), align 4, !tbaa !6
  %94 = fcmp olt float %93, 0.000000e+00
  br i1 %94, label %96, label %95

95:                                               ; preds = %92
  tail call void @test_fcPosNormal_float(float noundef %93)
  br label %97

96:                                               ; preds = %92
  tail call void @test_fcNegNormal_float(float noundef %93)
  br label %97

97:                                               ; preds = %96, %95
  %98 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 52), align 4, !tbaa !6
  %99 = fcmp olt float %98, 0.000000e+00
  br i1 %99, label %101, label %100

100:                                              ; preds = %97
  tail call void @test_fcPosNormal_float(float noundef %98)
  br label %102

101:                                              ; preds = %97
  tail call void @test_fcNegNormal_float(float noundef %98)
  br label %102

102:                                              ; preds = %101, %100
  %103 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 56), align 4, !tbaa !6
  %104 = fcmp olt float %103, 0.000000e+00
  br i1 %104, label %106, label %105

105:                                              ; preds = %102
  tail call void @test_fcPosNormal_float(float noundef %103)
  br label %107

106:                                              ; preds = %102
  tail call void @test_fcNegNormal_float(float noundef %103)
  br label %107

107:                                              ; preds = %106, %105
  %108 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 60), align 4, !tbaa !6
  %109 = fcmp olt float %108, 0.000000e+00
  br i1 %109, label %111, label %110

110:                                              ; preds = %107
  tail call void @test_fcPosNormal_float(float noundef %108)
  br label %112

111:                                              ; preds = %107
  tail call void @test_fcNegNormal_float(float noundef %108)
  br label %112

112:                                              ; preds = %111, %110
  %113 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 64), align 4, !tbaa !6
  %114 = fcmp olt float %113, 0.000000e+00
  br i1 %114, label %116, label %115

115:                                              ; preds = %112
  tail call void @test_fcPosNormal_float(float noundef %113)
  br label %117

116:                                              ; preds = %112
  tail call void @test_fcNegNormal_float(float noundef %113)
  br label %117

117:                                              ; preds = %116, %115
  %118 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 68), align 4, !tbaa !6
  %119 = fcmp olt float %118, 0.000000e+00
  br i1 %119, label %121, label %120

120:                                              ; preds = %117
  tail call void @test_fcPosNormal_float(float noundef %118)
  br label %122

121:                                              ; preds = %117
  tail call void @test_fcNegNormal_float(float noundef %118)
  br label %122

122:                                              ; preds = %121, %120
  %123 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 72), align 4, !tbaa !6
  %124 = fcmp olt float %123, 0.000000e+00
  br i1 %124, label %126, label %125

125:                                              ; preds = %122
  tail call void @test_fcPosNormal_float(float noundef %123)
  br label %127

126:                                              ; preds = %122
  tail call void @test_fcNegNormal_float(float noundef %123)
  br label %127

127:                                              ; preds = %126, %125
  %128 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 76), align 4, !tbaa !6
  %129 = fcmp olt float %128, 0.000000e+00
  br i1 %129, label %131, label %130

130:                                              ; preds = %127
  tail call void @test_fcPosNormal_float(float noundef %128)
  br label %132

131:                                              ; preds = %127
  tail call void @test_fcNegNormal_float(float noundef %128)
  br label %132

132:                                              ; preds = %131, %130
  %133 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 80), align 4, !tbaa !6
  %134 = fcmp olt float %133, 0.000000e+00
  br i1 %134, label %136, label %135

135:                                              ; preds = %132
  tail call void @test_fcPosNormal_float(float noundef %133)
  br label %137

136:                                              ; preds = %132
  tail call void @test_fcNegNormal_float(float noundef %133)
  br label %137

137:                                              ; preds = %136, %135
  %138 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 84), align 4, !tbaa !6
  %139 = fcmp olt float %138, 0.000000e+00
  br i1 %139, label %141, label %140

140:                                              ; preds = %137
  tail call void @test_fcPosNormal_float(float noundef %138)
  br label %142

141:                                              ; preds = %137
  tail call void @test_fcNegNormal_float(float noundef %138)
  br label %142

142:                                              ; preds = %141, %140
  %143 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 88), align 4, !tbaa !6
  %144 = fcmp olt float %143, 0.000000e+00
  br i1 %144, label %146, label %145

145:                                              ; preds = %142
  tail call void @test_fcPosNormal_float(float noundef %143)
  br label %147

146:                                              ; preds = %142
  tail call void @test_fcNegNormal_float(float noundef %143)
  br label %147

147:                                              ; preds = %146, %145
  %148 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 92), align 4, !tbaa !6
  %149 = fcmp olt float %148, 0.000000e+00
  br i1 %149, label %151, label %150

150:                                              ; preds = %147
  tail call void @test_fcPosNormal_float(float noundef %148)
  br label %152

151:                                              ; preds = %147
  tail call void @test_fcNegNormal_float(float noundef %148)
  br label %152

152:                                              ; preds = %151, %150
  %153 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 96), align 4, !tbaa !6
  %154 = fcmp olt float %153, 0.000000e+00
  br i1 %154, label %156, label %155

155:                                              ; preds = %152
  tail call void @test_fcPosNormal_float(float noundef %153)
  br label %157

156:                                              ; preds = %152
  tail call void @test_fcNegNormal_float(float noundef %153)
  br label %157

157:                                              ; preds = %156, %155
  %158 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatNormalValues, i64 100), align 4, !tbaa !6
  %159 = fcmp olt float %158, 0.000000e+00
  br i1 %159, label %161, label %160

160:                                              ; preds = %157
  tail call void @test_fcPosNormal_float(float noundef %158)
  br label %162

161:                                              ; preds = %157
  tail call void @test_fcNegNormal_float(float noundef %158)
  br label %162

162:                                              ; preds = %161, %160
  %163 = load float, ptr @FloatInfValues, align 4, !tbaa !6
  %164 = fcmp ogt float %163, 0.000000e+00
  br i1 %164, label %165, label %166

165:                                              ; preds = %162
  tail call void @test_fcPosInf_float(float noundef %163)
  br label %167

166:                                              ; preds = %162
  tail call void @test_fcNegInf_float(float noundef %163)
  br label %167

167:                                              ; preds = %166, %165
  %168 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatInfValues, i64 4), align 4, !tbaa !6
  %169 = fcmp ogt float %168, 0.000000e+00
  br i1 %169, label %171, label %170

170:                                              ; preds = %167
  tail call void @test_fcNegInf_float(float noundef %168)
  br label %172

171:                                              ; preds = %167
  tail call void @test_fcPosInf_float(float noundef %168)
  br label %172

172:                                              ; preds = %171, %170
  %173 = load float, ptr @FloatQNaNValues, align 4, !tbaa !6
  %174 = tail call i1 @llvm.is.fpclass.f32(float %173, i32 1022)
  br i1 %174, label %217, label %213

175:                                              ; preds = %217
  %176 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 4), align 4, !tbaa !6
  %177 = tail call i1 @llvm.is.fpclass.f32(float %176, i32 1022)
  br i1 %177, label %178, label %213

178:                                              ; preds = %175
  %179 = tail call i1 @llvm.is.fpclass.f32(float %176, i32 2)
  br i1 %179, label %180, label %219

180:                                              ; preds = %178
  %181 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 8), align 4, !tbaa !6
  %182 = tail call i1 @llvm.is.fpclass.f32(float %181, i32 1022)
  br i1 %182, label %183, label %213

183:                                              ; preds = %180
  %184 = tail call i1 @llvm.is.fpclass.f32(float %181, i32 2)
  br i1 %184, label %185, label %219

185:                                              ; preds = %183
  %186 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 12), align 4, !tbaa !6
  %187 = tail call i1 @llvm.is.fpclass.f32(float %186, i32 1022)
  br i1 %187, label %188, label %213

188:                                              ; preds = %185
  %189 = tail call i1 @llvm.is.fpclass.f32(float %186, i32 2)
  br i1 %189, label %190, label %219

190:                                              ; preds = %188
  %191 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 16), align 4, !tbaa !6
  %192 = tail call i1 @llvm.is.fpclass.f32(float %191, i32 1022)
  br i1 %192, label %193, label %213

193:                                              ; preds = %190
  %194 = tail call i1 @llvm.is.fpclass.f32(float %191, i32 2)
  br i1 %194, label %195, label %219

195:                                              ; preds = %193
  %196 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 20), align 4, !tbaa !6
  %197 = tail call i1 @llvm.is.fpclass.f32(float %196, i32 1022)
  br i1 %197, label %198, label %213

198:                                              ; preds = %195
  %199 = tail call i1 @llvm.is.fpclass.f32(float %196, i32 2)
  br i1 %199, label %200, label %219

200:                                              ; preds = %198
  %201 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 24), align 4, !tbaa !6
  %202 = tail call i1 @llvm.is.fpclass.f32(float %201, i32 1022)
  br i1 %202, label %203, label %213

203:                                              ; preds = %200
  %204 = tail call i1 @llvm.is.fpclass.f32(float %201, i32 2)
  br i1 %204, label %205, label %219

205:                                              ; preds = %203
  %206 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatQNaNValues, i64 28), align 4, !tbaa !6
  %207 = tail call i1 @llvm.is.fpclass.f32(float %206, i32 1022)
  br i1 %207, label %208, label %213

208:                                              ; preds = %205
  %209 = tail call i1 @llvm.is.fpclass.f32(float %206, i32 2)
  br i1 %209, label %210, label %219

210:                                              ; preds = %208
  %211 = load float, ptr @FloatSNaNValues, align 4, !tbaa !6
  %212 = tail call i1 @llvm.is.fpclass.f32(float %211, i32 1)
  br i1 %212, label %223, label %239

213:                                              ; preds = %205, %200, %195, %190, %185, %180, %175, %172
  %214 = phi float [ %173, %172 ], [ %176, %175 ], [ %181, %180 ], [ %186, %185 ], [ %191, %190 ], [ %196, %195 ], [ %201, %200 ], [ %206, %205 ]
  %215 = bitcast float %214 to i32
  %216 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.24, i32 noundef %215, i32 noundef 2)
  tail call void @exit(i32 noundef -1) #6
  unreachable

217:                                              ; preds = %172
  %218 = tail call i1 @llvm.is.fpclass.f32(float %173, i32 2)
  br i1 %218, label %175, label %219

219:                                              ; preds = %208, %203, %198, %193, %188, %183, %178, %217
  %220 = phi float [ %173, %217 ], [ %176, %178 ], [ %181, %183 ], [ %186, %188 ], [ %191, %193 ], [ %196, %198 ], [ %201, %203 ], [ %206, %208 ]
  %221 = bitcast float %220 to i32
  %222 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.25, i32 noundef %221, i32 noundef 2)
  tail call void @exit(i32 noundef -1) #6
  unreachable

223:                                              ; preds = %210
  %224 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 4), align 4, !tbaa !6
  %225 = tail call i1 @llvm.is.fpclass.f32(float %224, i32 1)
  br i1 %225, label %226, label %239

226:                                              ; preds = %223
  %227 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 8), align 4, !tbaa !6
  %228 = tail call i1 @llvm.is.fpclass.f32(float %227, i32 1)
  br i1 %228, label %229, label %239

229:                                              ; preds = %226
  %230 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 12), align 4, !tbaa !6
  %231 = tail call i1 @llvm.is.fpclass.f32(float %230, i32 1)
  br i1 %231, label %232, label %239

232:                                              ; preds = %229
  %233 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 16), align 4, !tbaa !6
  %234 = tail call i1 @llvm.is.fpclass.f32(float %233, i32 1)
  br i1 %234, label %235, label %239

235:                                              ; preds = %232
  %236 = load float, ptr getelementptr inbounds nuw (i8, ptr @FloatSNaNValues, i64 20), align 4, !tbaa !6
  %237 = tail call i1 @llvm.is.fpclass.f32(float %236, i32 1)
  br i1 %237, label %238, label %239

238:                                              ; preds = %235
  ret void

239:                                              ; preds = %235, %232, %229, %226, %223, %210
  %240 = phi float [ %211, %210 ], [ %224, %223 ], [ %227, %226 ], [ %230, %229 ], [ %233, %232 ], [ %236, %235 ]
  %241 = bitcast float %240 to i32
  %242 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, ptr noundef nonnull @.str.24, i32 noundef %241, i32 noundef 1)
  tail call void @exit(i32 noundef -1) #6
  unreachable
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @test_double() local_unnamed_addr #0 {
  %1 = load double, ptr @DoubleQNaNValues, align 8, !tbaa !12
  %2 = fcmp uno double %1, 0.000000e+00
  br i1 %2, label %47, label %41

3:                                                ; preds = %47
  %4 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 8), align 8, !tbaa !12
  %5 = fcmp uno double %4, 0.000000e+00
  br i1 %5, label %6, label %41

6:                                                ; preds = %3
  %7 = tail call i1 @llvm.is.fpclass.f64(double %4, i32 1)
  br i1 %7, label %49, label %8

8:                                                ; preds = %6
  %9 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 16), align 8, !tbaa !12
  %10 = fcmp uno double %9, 0.000000e+00
  br i1 %10, label %11, label %41

11:                                               ; preds = %8
  %12 = tail call i1 @llvm.is.fpclass.f64(double %9, i32 1)
  br i1 %12, label %49, label %13

13:                                               ; preds = %11
  %14 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 24), align 8, !tbaa !12
  %15 = fcmp uno double %14, 0.000000e+00
  br i1 %15, label %16, label %41

16:                                               ; preds = %13
  %17 = tail call i1 @llvm.is.fpclass.f64(double %14, i32 1)
  br i1 %17, label %49, label %18

18:                                               ; preds = %16
  %19 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 32), align 8, !tbaa !12
  %20 = fcmp uno double %19, 0.000000e+00
  br i1 %20, label %21, label %41

21:                                               ; preds = %18
  %22 = tail call i1 @llvm.is.fpclass.f64(double %19, i32 1)
  br i1 %22, label %49, label %23

23:                                               ; preds = %21
  %24 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 40), align 8, !tbaa !12
  %25 = fcmp uno double %24, 0.000000e+00
  br i1 %25, label %26, label %41

26:                                               ; preds = %23
  %27 = tail call i1 @llvm.is.fpclass.f64(double %24, i32 1)
  br i1 %27, label %49, label %28

28:                                               ; preds = %26
  %29 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 48), align 8, !tbaa !12
  %30 = fcmp uno double %29, 0.000000e+00
  br i1 %30, label %31, label %41

31:                                               ; preds = %28
  %32 = tail call i1 @llvm.is.fpclass.f64(double %29, i32 1)
  br i1 %32, label %49, label %33

33:                                               ; preds = %31
  %34 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 56), align 8, !tbaa !12
  %35 = fcmp uno double %34, 0.000000e+00
  br i1 %35, label %36, label %41

36:                                               ; preds = %33
  %37 = tail call i1 @llvm.is.fpclass.f64(double %34, i32 1)
  br i1 %37, label %49, label %38

38:                                               ; preds = %36
  %39 = load double, ptr @DoubleSNaNValues, align 8, !tbaa !12
  %40 = fcmp uno double %39, 0.000000e+00
  br i1 %40, label %89, label %83

41:                                               ; preds = %33, %28, %23, %18, %13, %8, %3, %0
  %42 = phi ptr [ @DoubleQNaNValues, %0 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 8), %3 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 16), %8 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 24), %13 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 32), %18 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 40), %23 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 48), %28 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 56), %33 ]
  %43 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.169, i32 noundef 101)
  %44 = load i64, ptr %42, align 8, !tbaa !14
  %45 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %44)
  %46 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

47:                                               ; preds = %0
  %48 = tail call i1 @llvm.is.fpclass.f64(double %1, i32 1)
  br i1 %48, label %49, label %3

49:                                               ; preds = %36, %31, %26, %21, %16, %11, %6, %47
  %50 = phi ptr [ @DoubleQNaNValues, %47 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 8), %6 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 16), %11 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 24), %16 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 32), %21 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 40), %26 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 48), %31 ], [ getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 56), %36 ]
  %51 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.169, i32 noundef 102)
  %52 = load i64, ptr %50, align 8, !tbaa !14
  %53 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %52)
  %54 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

55:                                               ; preds = %89
  %56 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 8), align 8, !tbaa !12
  %57 = fcmp uno double %56, 0.000000e+00
  br i1 %57, label %58, label %83

58:                                               ; preds = %55
  %59 = tail call i1 @llvm.is.fpclass.f64(double %56, i32 1)
  br i1 %59, label %60, label %91

60:                                               ; preds = %58
  %61 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 16), align 8, !tbaa !12
  %62 = fcmp uno double %61, 0.000000e+00
  br i1 %62, label %63, label %83

63:                                               ; preds = %60
  %64 = tail call i1 @llvm.is.fpclass.f64(double %61, i32 1)
  br i1 %64, label %65, label %91

65:                                               ; preds = %63
  %66 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 24), align 8, !tbaa !12
  %67 = fcmp uno double %66, 0.000000e+00
  br i1 %67, label %68, label %83

68:                                               ; preds = %65
  %69 = tail call i1 @llvm.is.fpclass.f64(double %66, i32 1)
  br i1 %69, label %70, label %91

70:                                               ; preds = %68
  %71 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 32), align 8, !tbaa !12
  %72 = fcmp uno double %71, 0.000000e+00
  br i1 %72, label %73, label %83

73:                                               ; preds = %70
  %74 = tail call i1 @llvm.is.fpclass.f64(double %71, i32 1)
  br i1 %74, label %75, label %91

75:                                               ; preds = %73
  %76 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 40), align 8, !tbaa !12
  %77 = fcmp uno double %76, 0.000000e+00
  br i1 %77, label %78, label %83

78:                                               ; preds = %75
  %79 = tail call i1 @llvm.is.fpclass.f64(double %76, i32 1)
  br i1 %79, label %80, label %91

80:                                               ; preds = %78
  %81 = load double, ptr @DoubleInfValues, align 8, !tbaa !12
  %82 = fcmp uno double %81, 0.000000e+00
  br i1 %82, label %106, label %112

83:                                               ; preds = %75, %70, %65, %60, %55, %38
  %84 = phi ptr [ @DoubleSNaNValues, %38 ], [ getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 8), %55 ], [ getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 16), %60 ], [ getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 24), %65 ], [ getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 32), %70 ], [ getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 40), %75 ]
  %85 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.169, i32 noundef 113)
  %86 = load i64, ptr %84, align 8, !tbaa !14
  %87 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %86)
  %88 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

89:                                               ; preds = %38
  %90 = tail call i1 @llvm.is.fpclass.f64(double %39, i32 1)
  br i1 %90, label %55, label %91

91:                                               ; preds = %78, %73, %68, %63, %58, %89
  %92 = phi ptr [ @DoubleSNaNValues, %89 ], [ getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 8), %58 ], [ getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 16), %63 ], [ getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 24), %68 ], [ getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 32), %73 ], [ getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 40), %78 ]
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.12, ptr noundef nonnull @.str.169, i32 noundef 114)
  %94 = load i64, ptr %92, align 8, !tbaa !14
  %95 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %94)
  %96 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

97:                                               ; preds = %112
  %98 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleInfValues, i64 8), align 8, !tbaa !12
  %99 = fcmp uno double %98, 0.000000e+00
  br i1 %99, label %106, label %100

100:                                              ; preds = %97
  %101 = tail call double @llvm.fabs.f64(double %98)
  %102 = fcmp oeq double %101, 0x7FF0000000000000
  br i1 %102, label %103, label %115

103:                                              ; preds = %100
  %104 = load double, ptr @DoubleZeroValues, align 8, !tbaa !12
  %105 = fcmp uno double %104, 0.000000e+00
  br i1 %105, label %134, label %140

106:                                              ; preds = %97, %80
  %107 = phi ptr [ @DoubleInfValues, %80 ], [ getelementptr inbounds nuw (i8, ptr @DoubleInfValues, i64 8), %97 ]
  %108 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.169, i32 noundef 125)
  %109 = load i64, ptr %107, align 8, !tbaa !14
  %110 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %109)
  %111 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

112:                                              ; preds = %80
  %113 = tail call double @llvm.fabs.f64(double %81)
  %114 = fcmp oeq double %113, 0x7FF0000000000000
  br i1 %114, label %97, label %115

115:                                              ; preds = %100, %112
  %116 = phi ptr [ @DoubleInfValues, %112 ], [ getelementptr inbounds nuw (i8, ptr @DoubleInfValues, i64 8), %100 ]
  %117 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.14, ptr noundef nonnull @.str.169, i32 noundef 127)
  %118 = load i64, ptr %116, align 8, !tbaa !14
  %119 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %118)
  %120 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

121:                                              ; preds = %157
  %122 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleZeroValues, i64 8), align 8, !tbaa !12
  %123 = fcmp uno double %122, 0.000000e+00
  br i1 %123, label %134, label %124

124:                                              ; preds = %121
  %125 = tail call double @llvm.fabs.f64(double %122)
  %126 = fcmp oeq double %125, 0x7FF0000000000000
  br i1 %126, label %143, label %127

127:                                              ; preds = %124
  %128 = tail call i1 @llvm.is.fpclass.f64(double %122, i32 264)
  br i1 %128, label %151, label %129

129:                                              ; preds = %127
  %130 = tail call i1 @llvm.is.fpclass.f64(double %122, i32 144)
  br i1 %130, label %159, label %131

131:                                              ; preds = %129
  %132 = load double, ptr @DoubleDenormValues, align 8, !tbaa !12
  %133 = fcmp uno double %132, 0.000000e+00
  br i1 %133, label %198, label %204

134:                                              ; preds = %121, %103
  %135 = phi ptr [ @DoubleZeroValues, %103 ], [ getelementptr inbounds nuw (i8, ptr @DoubleZeroValues, i64 8), %121 ]
  %136 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.169, i32 noundef 137)
  %137 = load i64, ptr %135, align 8, !tbaa !14
  %138 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %137)
  %139 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

140:                                              ; preds = %103
  %141 = tail call double @llvm.fabs.f64(double %104)
  %142 = fcmp oeq double %141, 0x7FF0000000000000
  br i1 %142, label %143, label %149

143:                                              ; preds = %124, %140
  %144 = phi ptr [ @DoubleZeroValues, %140 ], [ getelementptr inbounds nuw (i8, ptr @DoubleZeroValues, i64 8), %124 ]
  %145 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.169, i32 noundef 139)
  %146 = load i64, ptr %144, align 8, !tbaa !14
  %147 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %146)
  %148 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

149:                                              ; preds = %140
  %150 = tail call i1 @llvm.is.fpclass.f64(double %104, i32 264)
  br i1 %150, label %151, label %157

151:                                              ; preds = %127, %149
  %152 = phi ptr [ @DoubleZeroValues, %149 ], [ getelementptr inbounds nuw (i8, ptr @DoubleZeroValues, i64 8), %127 ]
  %153 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.169, i32 noundef 141)
  %154 = load i64, ptr %152, align 8, !tbaa !14
  %155 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %154)
  %156 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

157:                                              ; preds = %149
  %158 = tail call i1 @llvm.is.fpclass.f64(double %104, i32 144)
  br i1 %158, label %159, label %121

159:                                              ; preds = %129, %157
  %160 = phi ptr [ @DoubleZeroValues, %157 ], [ getelementptr inbounds nuw (i8, ptr @DoubleZeroValues, i64 8), %129 ]
  %161 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.169, i32 noundef 142)
  %162 = load i64, ptr %160, align 8, !tbaa !14
  %163 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %162)
  %164 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

165:                                              ; preds = %221
  %166 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 8), align 8, !tbaa !12
  %167 = fcmp uno double %166, 0.000000e+00
  br i1 %167, label %198, label %168

168:                                              ; preds = %165
  %169 = tail call double @llvm.fabs.f64(double %166)
  %170 = fcmp oeq double %169, 0x7FF0000000000000
  br i1 %170, label %207, label %171

171:                                              ; preds = %168
  %172 = tail call i1 @llvm.is.fpclass.f64(double %166, i32 264)
  br i1 %172, label %215, label %173

173:                                              ; preds = %171
  %174 = tail call i1 @llvm.is.fpclass.f64(double %166, i32 144)
  br i1 %174, label %175, label %223

175:                                              ; preds = %173
  %176 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 16), align 8, !tbaa !12
  %177 = fcmp uno double %176, 0.000000e+00
  br i1 %177, label %198, label %178

178:                                              ; preds = %175
  %179 = tail call double @llvm.fabs.f64(double %176)
  %180 = fcmp oeq double %179, 0x7FF0000000000000
  br i1 %180, label %207, label %181

181:                                              ; preds = %178
  %182 = tail call i1 @llvm.is.fpclass.f64(double %176, i32 264)
  br i1 %182, label %215, label %183

183:                                              ; preds = %181
  %184 = tail call i1 @llvm.is.fpclass.f64(double %176, i32 144)
  br i1 %184, label %185, label %223

185:                                              ; preds = %183
  %186 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 24), align 8, !tbaa !12
  %187 = fcmp uno double %186, 0.000000e+00
  br i1 %187, label %198, label %188

188:                                              ; preds = %185
  %189 = tail call double @llvm.fabs.f64(double %186)
  %190 = fcmp oeq double %189, 0x7FF0000000000000
  br i1 %190, label %207, label %191

191:                                              ; preds = %188
  %192 = tail call i1 @llvm.is.fpclass.f64(double %186, i32 264)
  br i1 %192, label %215, label %193

193:                                              ; preds = %191
  %194 = tail call i1 @llvm.is.fpclass.f64(double %186, i32 144)
  br i1 %194, label %195, label %223

195:                                              ; preds = %193
  %196 = load double, ptr @DoubleNormalValues, align 8, !tbaa !12
  %197 = fcmp uno double %196, 0.000000e+00
  br i1 %197, label %430, label %436

198:                                              ; preds = %185, %175, %165, %131
  %199 = phi ptr [ @DoubleDenormValues, %131 ], [ getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 8), %165 ], [ getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 16), %175 ], [ getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 24), %185 ]
  %200 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.169, i32 noundef 149)
  %201 = load i64, ptr %199, align 8, !tbaa !14
  %202 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %201)
  %203 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

204:                                              ; preds = %131
  %205 = tail call double @llvm.fabs.f64(double %132)
  %206 = fcmp oeq double %205, 0x7FF0000000000000
  br i1 %206, label %207, label %213

207:                                              ; preds = %188, %178, %168, %204
  %208 = phi ptr [ @DoubleDenormValues, %204 ], [ getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 8), %168 ], [ getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 16), %178 ], [ getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 24), %188 ]
  %209 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.169, i32 noundef 151)
  %210 = load i64, ptr %208, align 8, !tbaa !14
  %211 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %210)
  %212 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

213:                                              ; preds = %204
  %214 = tail call i1 @llvm.is.fpclass.f64(double %132, i32 264)
  br i1 %214, label %215, label %221

215:                                              ; preds = %191, %181, %171, %213
  %216 = phi ptr [ @DoubleDenormValues, %213 ], [ getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 8), %171 ], [ getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 16), %181 ], [ getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 24), %191 ]
  %217 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.169, i32 noundef 153)
  %218 = load i64, ptr %216, align 8, !tbaa !14
  %219 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %218)
  %220 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

221:                                              ; preds = %213
  %222 = tail call i1 @llvm.is.fpclass.f64(double %132, i32 144)
  br i1 %222, label %165, label %223

223:                                              ; preds = %193, %183, %173, %221
  %224 = phi ptr [ @DoubleDenormValues, %221 ], [ getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 8), %173 ], [ getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 16), %183 ], [ getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 24), %193 ]
  %225 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.19, ptr noundef nonnull @.str.169, i32 noundef 154)
  %226 = load i64, ptr %224, align 8, !tbaa !14
  %227 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %226)
  %228 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

229:                                              ; preds = %445
  %230 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 8), align 8, !tbaa !12
  %231 = fcmp uno double %230, 0.000000e+00
  br i1 %231, label %430, label %232

232:                                              ; preds = %229
  %233 = tail call double @llvm.fabs.f64(double %230)
  %234 = fcmp oeq double %233, 0x7FF0000000000000
  br i1 %234, label %439, label %235

235:                                              ; preds = %232
  %236 = tail call i1 @llvm.is.fpclass.f64(double %230, i32 264)
  br i1 %236, label %237, label %447

237:                                              ; preds = %235
  %238 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 16), align 8, !tbaa !12
  %239 = fcmp uno double %238, 0.000000e+00
  br i1 %239, label %430, label %240

240:                                              ; preds = %237
  %241 = tail call double @llvm.fabs.f64(double %238)
  %242 = fcmp oeq double %241, 0x7FF0000000000000
  br i1 %242, label %439, label %243

243:                                              ; preds = %240
  %244 = tail call i1 @llvm.is.fpclass.f64(double %238, i32 264)
  br i1 %244, label %245, label %447

245:                                              ; preds = %243
  %246 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 24), align 8, !tbaa !12
  %247 = fcmp uno double %246, 0.000000e+00
  br i1 %247, label %430, label %248

248:                                              ; preds = %245
  %249 = tail call double @llvm.fabs.f64(double %246)
  %250 = fcmp oeq double %249, 0x7FF0000000000000
  br i1 %250, label %439, label %251

251:                                              ; preds = %248
  %252 = tail call i1 @llvm.is.fpclass.f64(double %246, i32 264)
  br i1 %252, label %253, label %447

253:                                              ; preds = %251
  %254 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 32), align 8, !tbaa !12
  %255 = fcmp uno double %254, 0.000000e+00
  br i1 %255, label %430, label %256

256:                                              ; preds = %253
  %257 = tail call double @llvm.fabs.f64(double %254)
  %258 = fcmp oeq double %257, 0x7FF0000000000000
  br i1 %258, label %439, label %259

259:                                              ; preds = %256
  %260 = tail call i1 @llvm.is.fpclass.f64(double %254, i32 264)
  br i1 %260, label %261, label %447

261:                                              ; preds = %259
  %262 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 40), align 8, !tbaa !12
  %263 = fcmp uno double %262, 0.000000e+00
  br i1 %263, label %430, label %264

264:                                              ; preds = %261
  %265 = tail call double @llvm.fabs.f64(double %262)
  %266 = fcmp oeq double %265, 0x7FF0000000000000
  br i1 %266, label %439, label %267

267:                                              ; preds = %264
  %268 = tail call i1 @llvm.is.fpclass.f64(double %262, i32 264)
  br i1 %268, label %269, label %447

269:                                              ; preds = %267
  %270 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 48), align 8, !tbaa !12
  %271 = fcmp uno double %270, 0.000000e+00
  br i1 %271, label %430, label %272

272:                                              ; preds = %269
  %273 = tail call double @llvm.fabs.f64(double %270)
  %274 = fcmp oeq double %273, 0x7FF0000000000000
  br i1 %274, label %439, label %275

275:                                              ; preds = %272
  %276 = tail call i1 @llvm.is.fpclass.f64(double %270, i32 264)
  br i1 %276, label %277, label %447

277:                                              ; preds = %275
  %278 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 56), align 8, !tbaa !12
  %279 = fcmp uno double %278, 0.000000e+00
  br i1 %279, label %430, label %280

280:                                              ; preds = %277
  %281 = tail call double @llvm.fabs.f64(double %278)
  %282 = fcmp oeq double %281, 0x7FF0000000000000
  br i1 %282, label %439, label %283

283:                                              ; preds = %280
  %284 = tail call i1 @llvm.is.fpclass.f64(double %278, i32 264)
  br i1 %284, label %285, label %447

285:                                              ; preds = %283
  %286 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 64), align 8, !tbaa !12
  %287 = fcmp uno double %286, 0.000000e+00
  br i1 %287, label %430, label %288

288:                                              ; preds = %285
  %289 = tail call double @llvm.fabs.f64(double %286)
  %290 = fcmp oeq double %289, 0x7FF0000000000000
  br i1 %290, label %439, label %291

291:                                              ; preds = %288
  %292 = tail call i1 @llvm.is.fpclass.f64(double %286, i32 264)
  br i1 %292, label %293, label %447

293:                                              ; preds = %291
  %294 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 72), align 8, !tbaa !12
  %295 = fcmp uno double %294, 0.000000e+00
  br i1 %295, label %430, label %296

296:                                              ; preds = %293
  %297 = tail call double @llvm.fabs.f64(double %294)
  %298 = fcmp oeq double %297, 0x7FF0000000000000
  br i1 %298, label %439, label %299

299:                                              ; preds = %296
  %300 = tail call i1 @llvm.is.fpclass.f64(double %294, i32 264)
  br i1 %300, label %301, label %447

301:                                              ; preds = %299
  %302 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 80), align 8, !tbaa !12
  %303 = fcmp uno double %302, 0.000000e+00
  br i1 %303, label %430, label %304

304:                                              ; preds = %301
  %305 = tail call double @llvm.fabs.f64(double %302)
  %306 = fcmp oeq double %305, 0x7FF0000000000000
  br i1 %306, label %439, label %307

307:                                              ; preds = %304
  %308 = tail call i1 @llvm.is.fpclass.f64(double %302, i32 264)
  br i1 %308, label %309, label %447

309:                                              ; preds = %307
  %310 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 88), align 8, !tbaa !12
  %311 = fcmp uno double %310, 0.000000e+00
  br i1 %311, label %430, label %312

312:                                              ; preds = %309
  %313 = tail call double @llvm.fabs.f64(double %310)
  %314 = fcmp oeq double %313, 0x7FF0000000000000
  br i1 %314, label %439, label %315

315:                                              ; preds = %312
  %316 = tail call i1 @llvm.is.fpclass.f64(double %310, i32 264)
  br i1 %316, label %317, label %447

317:                                              ; preds = %315
  %318 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 96), align 8, !tbaa !12
  %319 = fcmp uno double %318, 0.000000e+00
  br i1 %319, label %430, label %320

320:                                              ; preds = %317
  %321 = tail call double @llvm.fabs.f64(double %318)
  %322 = fcmp oeq double %321, 0x7FF0000000000000
  br i1 %322, label %439, label %323

323:                                              ; preds = %320
  %324 = tail call i1 @llvm.is.fpclass.f64(double %318, i32 264)
  br i1 %324, label %325, label %447

325:                                              ; preds = %323
  %326 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 104), align 8, !tbaa !12
  %327 = fcmp uno double %326, 0.000000e+00
  br i1 %327, label %430, label %328

328:                                              ; preds = %325
  %329 = tail call double @llvm.fabs.f64(double %326)
  %330 = fcmp oeq double %329, 0x7FF0000000000000
  br i1 %330, label %439, label %331

331:                                              ; preds = %328
  %332 = tail call i1 @llvm.is.fpclass.f64(double %326, i32 264)
  br i1 %332, label %333, label %447

333:                                              ; preds = %331
  %334 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 112), align 8, !tbaa !12
  %335 = fcmp uno double %334, 0.000000e+00
  br i1 %335, label %430, label %336

336:                                              ; preds = %333
  %337 = tail call double @llvm.fabs.f64(double %334)
  %338 = fcmp oeq double %337, 0x7FF0000000000000
  br i1 %338, label %439, label %339

339:                                              ; preds = %336
  %340 = tail call i1 @llvm.is.fpclass.f64(double %334, i32 264)
  br i1 %340, label %341, label %447

341:                                              ; preds = %339
  %342 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 120), align 8, !tbaa !12
  %343 = fcmp uno double %342, 0.000000e+00
  br i1 %343, label %430, label %344

344:                                              ; preds = %341
  %345 = tail call double @llvm.fabs.f64(double %342)
  %346 = fcmp oeq double %345, 0x7FF0000000000000
  br i1 %346, label %439, label %347

347:                                              ; preds = %344
  %348 = tail call i1 @llvm.is.fpclass.f64(double %342, i32 264)
  br i1 %348, label %349, label %447

349:                                              ; preds = %347
  %350 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 128), align 8, !tbaa !12
  %351 = fcmp uno double %350, 0.000000e+00
  br i1 %351, label %430, label %352

352:                                              ; preds = %349
  %353 = tail call double @llvm.fabs.f64(double %350)
  %354 = fcmp oeq double %353, 0x7FF0000000000000
  br i1 %354, label %439, label %355

355:                                              ; preds = %352
  %356 = tail call i1 @llvm.is.fpclass.f64(double %350, i32 264)
  br i1 %356, label %357, label %447

357:                                              ; preds = %355
  %358 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 136), align 8, !tbaa !12
  %359 = fcmp uno double %358, 0.000000e+00
  br i1 %359, label %430, label %360

360:                                              ; preds = %357
  %361 = tail call double @llvm.fabs.f64(double %358)
  %362 = fcmp oeq double %361, 0x7FF0000000000000
  br i1 %362, label %439, label %363

363:                                              ; preds = %360
  %364 = tail call i1 @llvm.is.fpclass.f64(double %358, i32 264)
  br i1 %364, label %365, label %447

365:                                              ; preds = %363
  %366 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 144), align 8, !tbaa !12
  %367 = fcmp uno double %366, 0.000000e+00
  br i1 %367, label %430, label %368

368:                                              ; preds = %365
  %369 = tail call double @llvm.fabs.f64(double %366)
  %370 = fcmp oeq double %369, 0x7FF0000000000000
  br i1 %370, label %439, label %371

371:                                              ; preds = %368
  %372 = tail call i1 @llvm.is.fpclass.f64(double %366, i32 264)
  br i1 %372, label %373, label %447

373:                                              ; preds = %371
  %374 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 152), align 8, !tbaa !12
  %375 = fcmp uno double %374, 0.000000e+00
  br i1 %375, label %430, label %376

376:                                              ; preds = %373
  %377 = tail call double @llvm.fabs.f64(double %374)
  %378 = fcmp oeq double %377, 0x7FF0000000000000
  br i1 %378, label %439, label %379

379:                                              ; preds = %376
  %380 = tail call i1 @llvm.is.fpclass.f64(double %374, i32 264)
  br i1 %380, label %381, label %447

381:                                              ; preds = %379
  %382 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 160), align 8, !tbaa !12
  %383 = fcmp uno double %382, 0.000000e+00
  br i1 %383, label %430, label %384

384:                                              ; preds = %381
  %385 = tail call double @llvm.fabs.f64(double %382)
  %386 = fcmp oeq double %385, 0x7FF0000000000000
  br i1 %386, label %439, label %387

387:                                              ; preds = %384
  %388 = tail call i1 @llvm.is.fpclass.f64(double %382, i32 264)
  br i1 %388, label %389, label %447

389:                                              ; preds = %387
  %390 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 168), align 8, !tbaa !12
  %391 = fcmp uno double %390, 0.000000e+00
  br i1 %391, label %430, label %392

392:                                              ; preds = %389
  %393 = tail call double @llvm.fabs.f64(double %390)
  %394 = fcmp oeq double %393, 0x7FF0000000000000
  br i1 %394, label %439, label %395

395:                                              ; preds = %392
  %396 = tail call i1 @llvm.is.fpclass.f64(double %390, i32 264)
  br i1 %396, label %397, label %447

397:                                              ; preds = %395
  %398 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 176), align 8, !tbaa !12
  %399 = fcmp uno double %398, 0.000000e+00
  br i1 %399, label %430, label %400

400:                                              ; preds = %397
  %401 = tail call double @llvm.fabs.f64(double %398)
  %402 = fcmp oeq double %401, 0x7FF0000000000000
  br i1 %402, label %439, label %403

403:                                              ; preds = %400
  %404 = tail call i1 @llvm.is.fpclass.f64(double %398, i32 264)
  br i1 %404, label %405, label %447

405:                                              ; preds = %403
  %406 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 184), align 8, !tbaa !12
  %407 = fcmp uno double %406, 0.000000e+00
  br i1 %407, label %430, label %408

408:                                              ; preds = %405
  %409 = tail call double @llvm.fabs.f64(double %406)
  %410 = fcmp oeq double %409, 0x7FF0000000000000
  br i1 %410, label %439, label %411

411:                                              ; preds = %408
  %412 = tail call i1 @llvm.is.fpclass.f64(double %406, i32 264)
  br i1 %412, label %413, label %447

413:                                              ; preds = %411
  %414 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 192), align 8, !tbaa !12
  %415 = fcmp uno double %414, 0.000000e+00
  br i1 %415, label %430, label %416

416:                                              ; preds = %413
  %417 = tail call double @llvm.fabs.f64(double %414)
  %418 = fcmp oeq double %417, 0x7FF0000000000000
  br i1 %418, label %439, label %419

419:                                              ; preds = %416
  %420 = tail call i1 @llvm.is.fpclass.f64(double %414, i32 264)
  br i1 %420, label %421, label %447

421:                                              ; preds = %419
  %422 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 200), align 8, !tbaa !12
  %423 = fcmp uno double %422, 0.000000e+00
  br i1 %423, label %430, label %424

424:                                              ; preds = %421
  %425 = tail call double @llvm.fabs.f64(double %422)
  %426 = fcmp oeq double %425, 0x7FF0000000000000
  br i1 %426, label %439, label %427

427:                                              ; preds = %424
  %428 = tail call i1 @llvm.is.fpclass.f64(double %422, i32 264)
  br i1 %428, label %429, label %447

429:                                              ; preds = %427
  ret i32 0

430:                                              ; preds = %421, %413, %405, %397, %389, %381, %373, %365, %357, %349, %341, %333, %325, %317, %309, %301, %293, %285, %277, %269, %261, %253, %245, %237, %229, %195
  %431 = phi ptr [ @DoubleNormalValues, %195 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 8), %229 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 16), %237 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 24), %245 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 32), %253 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 40), %261 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 48), %269 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 56), %277 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 64), %285 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 72), %293 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 80), %301 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 88), %309 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 96), %317 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 104), %325 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 112), %333 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 120), %341 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 128), %349 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 136), %357 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 144), %365 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 152), %373 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 160), %381 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 168), %389 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 176), %397 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 184), %405 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 192), %413 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 200), %421 ]
  %432 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.169, i32 noundef 161)
  %433 = load i64, ptr %431, align 8, !tbaa !14
  %434 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %433)
  %435 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

436:                                              ; preds = %195
  %437 = tail call double @llvm.fabs.f64(double %196)
  %438 = fcmp oeq double %437, 0x7FF0000000000000
  br i1 %438, label %439, label %445

439:                                              ; preds = %424, %416, %408, %400, %392, %384, %376, %368, %360, %352, %344, %336, %328, %320, %312, %304, %296, %288, %280, %272, %264, %256, %248, %240, %232, %436
  %440 = phi ptr [ @DoubleNormalValues, %436 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 8), %232 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 16), %240 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 24), %248 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 32), %256 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 40), %264 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 48), %272 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 56), %280 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 64), %288 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 72), %296 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 80), %304 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 88), %312 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 96), %320 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 104), %328 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 112), %336 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 120), %344 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 128), %352 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 136), %360 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 144), %368 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 152), %376 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 160), %384 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 168), %392 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 176), %400 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 184), %408 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 192), %416 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 200), %424 ]
  %441 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.169, i32 noundef 163)
  %442 = load i64, ptr %440, align 8, !tbaa !14
  %443 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %442)
  %444 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

445:                                              ; preds = %436
  %446 = tail call i1 @llvm.is.fpclass.f64(double %196, i32 264)
  br i1 %446, label %229, label %447

447:                                              ; preds = %427, %419, %411, %403, %395, %387, %379, %371, %363, %355, %347, %339, %331, %323, %315, %307, %299, %291, %283, %275, %267, %259, %251, %243, %235, %445
  %448 = phi ptr [ @DoubleNormalValues, %445 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 8), %235 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 16), %243 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 24), %251 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 32), %259 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 40), %267 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 48), %275 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 56), %283 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 64), %291 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 72), %299 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 80), %307 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 88), %315 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 96), %323 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 104), %331 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 112), %339 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 120), %347 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 128), %355 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 136), %363 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 144), %371 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 152), %379 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 160), %387 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 168), %395 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 176), %403 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 184), %411 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 192), %419 ], [ getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 200), %427 ]
  %449 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.21, ptr noundef nonnull @.str.169, i32 noundef 165)
  %450 = load i64, ptr %448, align 8, !tbaa !14
  %451 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.170, i64 noundef %450)
  %452 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f64(double, i32 immarg) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcSNan_double(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 1)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast double %0 to i64
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.24, i64 noundef %4, i32 noundef 1)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcQNan_double(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast double %0 to i64
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.24, i64 noundef %4, i32 noundef 2)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 2)
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast double %0 to i64
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.25, i64 noundef %9, i32 noundef 2)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcPosInf_double(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast double %0 to i64
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.24, i64 noundef %4, i32 noundef 512)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord double %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast double %0 to i64
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.25, i64 noundef %9, i32 noundef 512)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp oeq double %0, 0x7FF0000000000000
  br i1 %12, label %16, label %13

13:                                               ; preds = %11
  %14 = bitcast double %0 to i64
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.27, i64 noundef %14, i32 noundef 512)
  tail call void @exit(i32 noundef -1) #6
  unreachable

16:                                               ; preds = %11
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcNegInf_double(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast double %0 to i64
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.24, i64 noundef %4, i32 noundef 4)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord double %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast double %0 to i64
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.25, i64 noundef %9, i32 noundef 4)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une double %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.27, i64 noundef 9218868437227405312, i32 noundef 4)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp oeq double %0, 0xFFF0000000000000
  br i1 %16, label %20, label %17

17:                                               ; preds = %15
  %18 = bitcast double %0 to i64
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.28, i64 noundef %18, i32 noundef 4)
  tail call void @exit(i32 noundef -1) #6
  unreachable

20:                                               ; preds = %15
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcPosNormal_double(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast double %0 to i64
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.24, i64 noundef %4, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord double %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast double %0 to i64
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.25, i64 noundef %9, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une double %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.27, i64 noundef 9218868437227405312, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp une double %0, 0xFFF0000000000000
  br i1 %16, label %19, label %17

17:                                               ; preds = %15
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.28, i64 noundef -4503599627370496, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

19:                                               ; preds = %15
  %20 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 256)
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  %22 = bitcast double %0 to i64
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.30, i64 noundef %22, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

24:                                               ; preds = %19
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcNegNormal_double(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast double %0 to i64
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.24, i64 noundef %4, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord double %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast double %0 to i64
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.25, i64 noundef %9, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une double %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.27, i64 noundef 9218868437227405312, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp une double %0, 0xFFF0000000000000
  br i1 %16, label %19, label %17

17:                                               ; preds = %15
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.28, i64 noundef -4503599627370496, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

19:                                               ; preds = %15
  %20 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 248)
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  %22 = bitcast double %0 to i64
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.30, i64 noundef %22, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

24:                                               ; preds = %19
  %25 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 8)
  br i1 %25, label %29, label %26

26:                                               ; preds = %24
  %27 = bitcast double %0 to i64
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.31, i64 noundef %27, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %24
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcPosSubnormal_double(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast double %0 to i64
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.24, i64 noundef %4, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord double %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast double %0 to i64
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.25, i64 noundef %9, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une double %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.27, i64 noundef 9218868437227405312, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp une double %0, 0xFFF0000000000000
  br i1 %16, label %19, label %17

17:                                               ; preds = %15
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.28, i64 noundef -4503599627370496, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

19:                                               ; preds = %15
  %20 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 248)
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  %22 = bitcast double %0 to i64
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.30, i64 noundef %22, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

24:                                               ; preds = %19
  %25 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 240)
  br i1 %25, label %29, label %26

26:                                               ; preds = %24
  %27 = bitcast double %0 to i64
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.31, i64 noundef %27, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %24
  %30 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 128)
  br i1 %30, label %34, label %31

31:                                               ; preds = %29
  %32 = bitcast double %0 to i64
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.33, i64 noundef %32, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

34:                                               ; preds = %29
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcNegSubnormal_double(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast double %0 to i64
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.24, i64 noundef %4, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord double %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast double %0 to i64
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.25, i64 noundef %9, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une double %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.27, i64 noundef 9218868437227405312, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp une double %0, 0xFFF0000000000000
  br i1 %16, label %19, label %17

17:                                               ; preds = %15
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.28, i64 noundef -4503599627370496, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

19:                                               ; preds = %15
  %20 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 248)
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  %22 = bitcast double %0 to i64
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.30, i64 noundef %22, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

24:                                               ; preds = %19
  %25 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 240)
  br i1 %25, label %29, label %26

26:                                               ; preds = %24
  %27 = bitcast double %0 to i64
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.31, i64 noundef %27, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %24
  %30 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 112)
  br i1 %30, label %34, label %31

31:                                               ; preds = %29
  %32 = bitcast double %0 to i64
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.33, i64 noundef %32, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

34:                                               ; preds = %29
  %35 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 16)
  br i1 %35, label %39, label %36

36:                                               ; preds = %34
  %37 = bitcast double %0 to i64
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.34, i64 noundef %37, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

39:                                               ; preds = %34
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcPosZero_double(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast double %0 to i64
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.24, i64 noundef %4, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord double %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast double %0 to i64
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.25, i64 noundef %9, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une double %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.27, i64 noundef 9218868437227405312, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp une double %0, 0xFFF0000000000000
  br i1 %16, label %19, label %17

17:                                               ; preds = %15
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.28, i64 noundef -4503599627370496, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

19:                                               ; preds = %15
  %20 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 248)
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  %22 = bitcast double %0 to i64
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.30, i64 noundef %22, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

24:                                               ; preds = %19
  %25 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 240)
  br i1 %25, label %29, label %26

26:                                               ; preds = %24
  %27 = bitcast double %0 to i64
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.31, i64 noundef %27, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %24
  %30 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 112)
  br i1 %30, label %34, label %31

31:                                               ; preds = %29
  %32 = bitcast double %0 to i64
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.33, i64 noundef %32, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

34:                                               ; preds = %29
  %35 = fcmp oeq double %0, 0.000000e+00
  br i1 %35, label %39, label %36

36:                                               ; preds = %34
  %37 = bitcast double %0 to i64
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.34, i64 noundef %37, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

39:                                               ; preds = %34
  %40 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 64)
  br i1 %40, label %44, label %41

41:                                               ; preds = %39
  %42 = bitcast double %0 to i64
  %43 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.36, i64 noundef %42, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

44:                                               ; preds = %39
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcNegZero_double(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 1022)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = bitcast double %0 to i64
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.24, i64 noundef %4, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

6:                                                ; preds = %1
  %7 = fcmp ord double %0, 0.000000e+00
  br i1 %7, label %11, label %8

8:                                                ; preds = %6
  %9 = bitcast double %0 to i64
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.25, i64 noundef %9, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

11:                                               ; preds = %6
  %12 = fcmp une double %0, 0x7FF0000000000000
  br i1 %12, label %15, label %13

13:                                               ; preds = %11
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.27, i64 noundef 9218868437227405312, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

15:                                               ; preds = %11
  %16 = fcmp une double %0, 0xFFF0000000000000
  br i1 %16, label %19, label %17

17:                                               ; preds = %15
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.28, i64 noundef -4503599627370496, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

19:                                               ; preds = %15
  %20 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 248)
  br i1 %20, label %24, label %21

21:                                               ; preds = %19
  %22 = bitcast double %0 to i64
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.30, i64 noundef %22, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

24:                                               ; preds = %19
  %25 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 240)
  br i1 %25, label %29, label %26

26:                                               ; preds = %24
  %27 = bitcast double %0 to i64
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.31, i64 noundef %27, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %24
  %30 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 112)
  br i1 %30, label %34, label %31

31:                                               ; preds = %29
  %32 = bitcast double %0 to i64
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.33, i64 noundef %32, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

34:                                               ; preds = %29
  %35 = fcmp oeq double %0, 0.000000e+00
  br i1 %35, label %39, label %36

36:                                               ; preds = %34
  %37 = bitcast double %0 to i64
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.34, i64 noundef %37, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

39:                                               ; preds = %34
  %40 = tail call i1 @llvm.is.fpclass.f64(double %0, i32 32)
  br i1 %40, label %44, label %41

41:                                               ; preds = %39
  %42 = bitcast double %0 to i64
  %43 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.36, i64 noundef %42, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

44:                                               ; preds = %39
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_isfpclass_double() local_unnamed_addr #0 {
  %1 = load double, ptr @DoubleZeroValues, align 8, !tbaa !12
  %2 = bitcast double %1 to i64
  %3 = icmp sgt i64 %2, -1
  br i1 %3, label %4, label %5

4:                                                ; preds = %0
  tail call void @test_fcPosZero_double(double noundef %1)
  br label %6

5:                                                ; preds = %0
  tail call void @test_fcNegZero_double(double noundef %1)
  br label %6

6:                                                ; preds = %5, %4
  %7 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleZeroValues, i64 8), align 8, !tbaa !12
  %8 = bitcast double %7 to i64
  %9 = icmp sgt i64 %8, -1
  br i1 %9, label %11, label %10

10:                                               ; preds = %6
  tail call void @test_fcNegZero_double(double noundef %7)
  br label %12

11:                                               ; preds = %6
  tail call void @test_fcPosZero_double(double noundef %7)
  br label %12

12:                                               ; preds = %11, %10
  %13 = load double, ptr @DoubleDenormValues, align 8, !tbaa !12
  %14 = fcmp olt double %13, 0.000000e+00
  br i1 %14, label %15, label %16

15:                                               ; preds = %12
  tail call void @test_fcNegSubnormal_double(double noundef %13)
  br label %17

16:                                               ; preds = %12
  tail call void @test_fcPosSubnormal_double(double noundef %13)
  br label %17

17:                                               ; preds = %16, %15
  %18 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 8), align 8, !tbaa !12
  %19 = fcmp olt double %18, 0.000000e+00
  br i1 %19, label %21, label %20

20:                                               ; preds = %17
  tail call void @test_fcPosSubnormal_double(double noundef %18)
  br label %22

21:                                               ; preds = %17
  tail call void @test_fcNegSubnormal_double(double noundef %18)
  br label %22

22:                                               ; preds = %21, %20
  %23 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 16), align 8, !tbaa !12
  %24 = fcmp olt double %23, 0.000000e+00
  br i1 %24, label %26, label %25

25:                                               ; preds = %22
  tail call void @test_fcPosSubnormal_double(double noundef %23)
  br label %27

26:                                               ; preds = %22
  tail call void @test_fcNegSubnormal_double(double noundef %23)
  br label %27

27:                                               ; preds = %26, %25
  %28 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleDenormValues, i64 24), align 8, !tbaa !12
  %29 = fcmp olt double %28, 0.000000e+00
  br i1 %29, label %31, label %30

30:                                               ; preds = %27
  tail call void @test_fcPosSubnormal_double(double noundef %28)
  br label %32

31:                                               ; preds = %27
  tail call void @test_fcNegSubnormal_double(double noundef %28)
  br label %32

32:                                               ; preds = %31, %30
  %33 = load double, ptr @DoubleNormalValues, align 8, !tbaa !12
  %34 = fcmp olt double %33, 0.000000e+00
  br i1 %34, label %35, label %36

35:                                               ; preds = %32
  tail call void @test_fcNegNormal_double(double noundef %33)
  br label %37

36:                                               ; preds = %32
  tail call void @test_fcPosNormal_double(double noundef %33)
  br label %37

37:                                               ; preds = %36, %35
  %38 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 8), align 8, !tbaa !12
  %39 = fcmp olt double %38, 0.000000e+00
  br i1 %39, label %41, label %40

40:                                               ; preds = %37
  tail call void @test_fcPosNormal_double(double noundef %38)
  br label %42

41:                                               ; preds = %37
  tail call void @test_fcNegNormal_double(double noundef %38)
  br label %42

42:                                               ; preds = %41, %40
  %43 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 16), align 8, !tbaa !12
  %44 = fcmp olt double %43, 0.000000e+00
  br i1 %44, label %46, label %45

45:                                               ; preds = %42
  tail call void @test_fcPosNormal_double(double noundef %43)
  br label %47

46:                                               ; preds = %42
  tail call void @test_fcNegNormal_double(double noundef %43)
  br label %47

47:                                               ; preds = %46, %45
  %48 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 24), align 8, !tbaa !12
  %49 = fcmp olt double %48, 0.000000e+00
  br i1 %49, label %51, label %50

50:                                               ; preds = %47
  tail call void @test_fcPosNormal_double(double noundef %48)
  br label %52

51:                                               ; preds = %47
  tail call void @test_fcNegNormal_double(double noundef %48)
  br label %52

52:                                               ; preds = %51, %50
  %53 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 32), align 8, !tbaa !12
  %54 = fcmp olt double %53, 0.000000e+00
  br i1 %54, label %56, label %55

55:                                               ; preds = %52
  tail call void @test_fcPosNormal_double(double noundef %53)
  br label %57

56:                                               ; preds = %52
  tail call void @test_fcNegNormal_double(double noundef %53)
  br label %57

57:                                               ; preds = %56, %55
  %58 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 40), align 8, !tbaa !12
  %59 = fcmp olt double %58, 0.000000e+00
  br i1 %59, label %61, label %60

60:                                               ; preds = %57
  tail call void @test_fcPosNormal_double(double noundef %58)
  br label %62

61:                                               ; preds = %57
  tail call void @test_fcNegNormal_double(double noundef %58)
  br label %62

62:                                               ; preds = %61, %60
  %63 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 48), align 8, !tbaa !12
  %64 = fcmp olt double %63, 0.000000e+00
  br i1 %64, label %66, label %65

65:                                               ; preds = %62
  tail call void @test_fcPosNormal_double(double noundef %63)
  br label %67

66:                                               ; preds = %62
  tail call void @test_fcNegNormal_double(double noundef %63)
  br label %67

67:                                               ; preds = %66, %65
  %68 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 56), align 8, !tbaa !12
  %69 = fcmp olt double %68, 0.000000e+00
  br i1 %69, label %71, label %70

70:                                               ; preds = %67
  tail call void @test_fcPosNormal_double(double noundef %68)
  br label %72

71:                                               ; preds = %67
  tail call void @test_fcNegNormal_double(double noundef %68)
  br label %72

72:                                               ; preds = %71, %70
  %73 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 64), align 8, !tbaa !12
  %74 = fcmp olt double %73, 0.000000e+00
  br i1 %74, label %76, label %75

75:                                               ; preds = %72
  tail call void @test_fcPosNormal_double(double noundef %73)
  br label %77

76:                                               ; preds = %72
  tail call void @test_fcNegNormal_double(double noundef %73)
  br label %77

77:                                               ; preds = %76, %75
  %78 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 72), align 8, !tbaa !12
  %79 = fcmp olt double %78, 0.000000e+00
  br i1 %79, label %81, label %80

80:                                               ; preds = %77
  tail call void @test_fcPosNormal_double(double noundef %78)
  br label %82

81:                                               ; preds = %77
  tail call void @test_fcNegNormal_double(double noundef %78)
  br label %82

82:                                               ; preds = %81, %80
  %83 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 80), align 8, !tbaa !12
  %84 = fcmp olt double %83, 0.000000e+00
  br i1 %84, label %86, label %85

85:                                               ; preds = %82
  tail call void @test_fcPosNormal_double(double noundef %83)
  br label %87

86:                                               ; preds = %82
  tail call void @test_fcNegNormal_double(double noundef %83)
  br label %87

87:                                               ; preds = %86, %85
  %88 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 88), align 8, !tbaa !12
  %89 = fcmp olt double %88, 0.000000e+00
  br i1 %89, label %91, label %90

90:                                               ; preds = %87
  tail call void @test_fcPosNormal_double(double noundef %88)
  br label %92

91:                                               ; preds = %87
  tail call void @test_fcNegNormal_double(double noundef %88)
  br label %92

92:                                               ; preds = %91, %90
  %93 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 96), align 8, !tbaa !12
  %94 = fcmp olt double %93, 0.000000e+00
  br i1 %94, label %96, label %95

95:                                               ; preds = %92
  tail call void @test_fcPosNormal_double(double noundef %93)
  br label %97

96:                                               ; preds = %92
  tail call void @test_fcNegNormal_double(double noundef %93)
  br label %97

97:                                               ; preds = %96, %95
  %98 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 104), align 8, !tbaa !12
  %99 = fcmp olt double %98, 0.000000e+00
  br i1 %99, label %101, label %100

100:                                              ; preds = %97
  tail call void @test_fcPosNormal_double(double noundef %98)
  br label %102

101:                                              ; preds = %97
  tail call void @test_fcNegNormal_double(double noundef %98)
  br label %102

102:                                              ; preds = %101, %100
  %103 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 112), align 8, !tbaa !12
  %104 = fcmp olt double %103, 0.000000e+00
  br i1 %104, label %106, label %105

105:                                              ; preds = %102
  tail call void @test_fcPosNormal_double(double noundef %103)
  br label %107

106:                                              ; preds = %102
  tail call void @test_fcNegNormal_double(double noundef %103)
  br label %107

107:                                              ; preds = %106, %105
  %108 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 120), align 8, !tbaa !12
  %109 = fcmp olt double %108, 0.000000e+00
  br i1 %109, label %111, label %110

110:                                              ; preds = %107
  tail call void @test_fcPosNormal_double(double noundef %108)
  br label %112

111:                                              ; preds = %107
  tail call void @test_fcNegNormal_double(double noundef %108)
  br label %112

112:                                              ; preds = %111, %110
  %113 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 128), align 8, !tbaa !12
  %114 = fcmp olt double %113, 0.000000e+00
  br i1 %114, label %116, label %115

115:                                              ; preds = %112
  tail call void @test_fcPosNormal_double(double noundef %113)
  br label %117

116:                                              ; preds = %112
  tail call void @test_fcNegNormal_double(double noundef %113)
  br label %117

117:                                              ; preds = %116, %115
  %118 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 136), align 8, !tbaa !12
  %119 = fcmp olt double %118, 0.000000e+00
  br i1 %119, label %121, label %120

120:                                              ; preds = %117
  tail call void @test_fcPosNormal_double(double noundef %118)
  br label %122

121:                                              ; preds = %117
  tail call void @test_fcNegNormal_double(double noundef %118)
  br label %122

122:                                              ; preds = %121, %120
  %123 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 144), align 8, !tbaa !12
  %124 = fcmp olt double %123, 0.000000e+00
  br i1 %124, label %126, label %125

125:                                              ; preds = %122
  tail call void @test_fcPosNormal_double(double noundef %123)
  br label %127

126:                                              ; preds = %122
  tail call void @test_fcNegNormal_double(double noundef %123)
  br label %127

127:                                              ; preds = %126, %125
  %128 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 152), align 8, !tbaa !12
  %129 = fcmp olt double %128, 0.000000e+00
  br i1 %129, label %131, label %130

130:                                              ; preds = %127
  tail call void @test_fcPosNormal_double(double noundef %128)
  br label %132

131:                                              ; preds = %127
  tail call void @test_fcNegNormal_double(double noundef %128)
  br label %132

132:                                              ; preds = %131, %130
  %133 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 160), align 8, !tbaa !12
  %134 = fcmp olt double %133, 0.000000e+00
  br i1 %134, label %136, label %135

135:                                              ; preds = %132
  tail call void @test_fcPosNormal_double(double noundef %133)
  br label %137

136:                                              ; preds = %132
  tail call void @test_fcNegNormal_double(double noundef %133)
  br label %137

137:                                              ; preds = %136, %135
  %138 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 168), align 8, !tbaa !12
  %139 = fcmp olt double %138, 0.000000e+00
  br i1 %139, label %141, label %140

140:                                              ; preds = %137
  tail call void @test_fcPosNormal_double(double noundef %138)
  br label %142

141:                                              ; preds = %137
  tail call void @test_fcNegNormal_double(double noundef %138)
  br label %142

142:                                              ; preds = %141, %140
  %143 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 176), align 8, !tbaa !12
  %144 = fcmp olt double %143, 0.000000e+00
  br i1 %144, label %146, label %145

145:                                              ; preds = %142
  tail call void @test_fcPosNormal_double(double noundef %143)
  br label %147

146:                                              ; preds = %142
  tail call void @test_fcNegNormal_double(double noundef %143)
  br label %147

147:                                              ; preds = %146, %145
  %148 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 184), align 8, !tbaa !12
  %149 = fcmp olt double %148, 0.000000e+00
  br i1 %149, label %151, label %150

150:                                              ; preds = %147
  tail call void @test_fcPosNormal_double(double noundef %148)
  br label %152

151:                                              ; preds = %147
  tail call void @test_fcNegNormal_double(double noundef %148)
  br label %152

152:                                              ; preds = %151, %150
  %153 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 192), align 8, !tbaa !12
  %154 = fcmp olt double %153, 0.000000e+00
  br i1 %154, label %156, label %155

155:                                              ; preds = %152
  tail call void @test_fcPosNormal_double(double noundef %153)
  br label %157

156:                                              ; preds = %152
  tail call void @test_fcNegNormal_double(double noundef %153)
  br label %157

157:                                              ; preds = %156, %155
  %158 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleNormalValues, i64 200), align 8, !tbaa !12
  %159 = fcmp olt double %158, 0.000000e+00
  br i1 %159, label %161, label %160

160:                                              ; preds = %157
  tail call void @test_fcPosNormal_double(double noundef %158)
  br label %162

161:                                              ; preds = %157
  tail call void @test_fcNegNormal_double(double noundef %158)
  br label %162

162:                                              ; preds = %161, %160
  %163 = load double, ptr @DoubleInfValues, align 8, !tbaa !12
  %164 = fcmp ogt double %163, 0.000000e+00
  br i1 %164, label %165, label %166

165:                                              ; preds = %162
  tail call void @test_fcPosInf_double(double noundef %163)
  br label %167

166:                                              ; preds = %162
  tail call void @test_fcNegInf_double(double noundef %163)
  br label %167

167:                                              ; preds = %166, %165
  %168 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleInfValues, i64 8), align 8, !tbaa !12
  %169 = fcmp ogt double %168, 0.000000e+00
  br i1 %169, label %171, label %170

170:                                              ; preds = %167
  tail call void @test_fcNegInf_double(double noundef %168)
  br label %172

171:                                              ; preds = %167
  tail call void @test_fcPosInf_double(double noundef %168)
  br label %172

172:                                              ; preds = %171, %170
  %173 = load double, ptr @DoubleQNaNValues, align 8, !tbaa !12
  %174 = tail call i1 @llvm.is.fpclass.f64(double %173, i32 1022)
  br i1 %174, label %217, label %213

175:                                              ; preds = %217
  %176 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 8), align 8, !tbaa !12
  %177 = tail call i1 @llvm.is.fpclass.f64(double %176, i32 1022)
  br i1 %177, label %178, label %213

178:                                              ; preds = %175
  %179 = tail call i1 @llvm.is.fpclass.f64(double %176, i32 2)
  br i1 %179, label %180, label %219

180:                                              ; preds = %178
  %181 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 16), align 8, !tbaa !12
  %182 = tail call i1 @llvm.is.fpclass.f64(double %181, i32 1022)
  br i1 %182, label %183, label %213

183:                                              ; preds = %180
  %184 = tail call i1 @llvm.is.fpclass.f64(double %181, i32 2)
  br i1 %184, label %185, label %219

185:                                              ; preds = %183
  %186 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 24), align 8, !tbaa !12
  %187 = tail call i1 @llvm.is.fpclass.f64(double %186, i32 1022)
  br i1 %187, label %188, label %213

188:                                              ; preds = %185
  %189 = tail call i1 @llvm.is.fpclass.f64(double %186, i32 2)
  br i1 %189, label %190, label %219

190:                                              ; preds = %188
  %191 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 32), align 8, !tbaa !12
  %192 = tail call i1 @llvm.is.fpclass.f64(double %191, i32 1022)
  br i1 %192, label %193, label %213

193:                                              ; preds = %190
  %194 = tail call i1 @llvm.is.fpclass.f64(double %191, i32 2)
  br i1 %194, label %195, label %219

195:                                              ; preds = %193
  %196 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 40), align 8, !tbaa !12
  %197 = tail call i1 @llvm.is.fpclass.f64(double %196, i32 1022)
  br i1 %197, label %198, label %213

198:                                              ; preds = %195
  %199 = tail call i1 @llvm.is.fpclass.f64(double %196, i32 2)
  br i1 %199, label %200, label %219

200:                                              ; preds = %198
  %201 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 48), align 8, !tbaa !12
  %202 = tail call i1 @llvm.is.fpclass.f64(double %201, i32 1022)
  br i1 %202, label %203, label %213

203:                                              ; preds = %200
  %204 = tail call i1 @llvm.is.fpclass.f64(double %201, i32 2)
  br i1 %204, label %205, label %219

205:                                              ; preds = %203
  %206 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleQNaNValues, i64 56), align 8, !tbaa !12
  %207 = tail call i1 @llvm.is.fpclass.f64(double %206, i32 1022)
  br i1 %207, label %208, label %213

208:                                              ; preds = %205
  %209 = tail call i1 @llvm.is.fpclass.f64(double %206, i32 2)
  br i1 %209, label %210, label %219

210:                                              ; preds = %208
  %211 = load double, ptr @DoubleSNaNValues, align 8, !tbaa !12
  %212 = tail call i1 @llvm.is.fpclass.f64(double %211, i32 1)
  br i1 %212, label %223, label %239

213:                                              ; preds = %205, %200, %195, %190, %185, %180, %175, %172
  %214 = phi double [ %173, %172 ], [ %176, %175 ], [ %181, %180 ], [ %186, %185 ], [ %191, %190 ], [ %196, %195 ], [ %201, %200 ], [ %206, %205 ]
  %215 = bitcast double %214 to i64
  %216 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.24, i64 noundef %215, i32 noundef 2)
  tail call void @exit(i32 noundef -1) #6
  unreachable

217:                                              ; preds = %172
  %218 = tail call i1 @llvm.is.fpclass.f64(double %173, i32 2)
  br i1 %218, label %175, label %219

219:                                              ; preds = %208, %203, %198, %193, %188, %183, %178, %217
  %220 = phi double [ %173, %217 ], [ %176, %178 ], [ %181, %183 ], [ %186, %188 ], [ %191, %193 ], [ %196, %198 ], [ %201, %203 ], [ %206, %208 ]
  %221 = bitcast double %220 to i64
  %222 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.25, i64 noundef %221, i32 noundef 2)
  tail call void @exit(i32 noundef -1) #6
  unreachable

223:                                              ; preds = %210
  %224 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 8), align 8, !tbaa !12
  %225 = tail call i1 @llvm.is.fpclass.f64(double %224, i32 1)
  br i1 %225, label %226, label %239

226:                                              ; preds = %223
  %227 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 16), align 8, !tbaa !12
  %228 = tail call i1 @llvm.is.fpclass.f64(double %227, i32 1)
  br i1 %228, label %229, label %239

229:                                              ; preds = %226
  %230 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 24), align 8, !tbaa !12
  %231 = tail call i1 @llvm.is.fpclass.f64(double %230, i32 1)
  br i1 %231, label %232, label %239

232:                                              ; preds = %229
  %233 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 32), align 8, !tbaa !12
  %234 = tail call i1 @llvm.is.fpclass.f64(double %233, i32 1)
  br i1 %234, label %235, label %239

235:                                              ; preds = %232
  %236 = load double, ptr getelementptr inbounds nuw (i8, ptr @DoubleSNaNValues, i64 40), align 8, !tbaa !12
  %237 = tail call i1 @llvm.is.fpclass.f64(double %236, i32 1)
  br i1 %237, label %238, label %239

238:                                              ; preds = %235
  ret void

239:                                              ; preds = %235, %232, %229, %226, %223, %210
  %240 = phi double [ %211, %210 ], [ %224, %223 ], [ %227, %226 ], [ %230, %229 ], [ %233, %232 ], [ %236, %235 ]
  %241 = bitcast double %240 to i64
  %242 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.171, ptr noundef nonnull @.str.24, i64 noundef %241, i32 noundef 1)
  tail call void @exit(i32 noundef -1) #6
  unreachable
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @prepare_ldouble_tables() local_unnamed_addr #4 {
  store fp128 0xL00000000000000007FFF800000000000, ptr @LongDoubleQNaNValues, align 16, !tbaa !16
  store fp128 0xL0000000000000000FFFF800000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 16), align 16, !tbaa !16
  store fp128 0xL00000000000000017FFF800000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 32), align 16, !tbaa !16
  store fp128 0xL0000000000000001FFFF800000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 48), align 16, !tbaa !16
  store fp128 0xL00000000000000007FFF400000000000, ptr @LongDoubleSNaNValues, align 16, !tbaa !16
  store fp128 0xL0000000000000000FFFF400000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 16), align 16, !tbaa !16
  store fp128 0xL00000000000000017FFF000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 32), align 16, !tbaa !16
  store fp128 0xL0000000000000001FFFF000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 48), align 16, !tbaa !16
  store fp128 0xL00000000000000007FFF000000000000, ptr @LongDoubleInfValues, align 16, !tbaa !16
  store fp128 0xL0000000000000000FFFF000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleInfValues, i64 16), align 16, !tbaa !16
  store fp128 0xL00000000000000010000000000000000, ptr @LongDoubleDenormValues, align 16, !tbaa !16
  store fp128 0xL00000000000000018000000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleDenormValues, i64 16), align 16, !tbaa !16
  store fp128 0xL00000000000000003FFF000000000000, ptr @LongDoubleNormalValues, align 16, !tbaa !16
  store fp128 0xL0000000000000000BFFF000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 16), align 16, !tbaa !16
  store fp128 0xLFFFFFFFFFFFFFFFF7FFEFFFFFFFFFFFF, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 32), align 16, !tbaa !16
  store fp128 0xLFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFF, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 48), align 16, !tbaa !16
  store fp128 0xL00000000000000000001000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 64), align 16, !tbaa !16
  store fp128 0xL00000000000000008001000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 80), align 16, !tbaa !16
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @test_ldouble() local_unnamed_addr #0 {
  %1 = load fp128, ptr @LongDoubleQNaNValues, align 16, !tbaa !16
  %2 = fcmp uno fp128 %1, 0xL00000000000000000000000000000000
  br i1 %2, label %27, label %21

3:                                                ; preds = %27
  %4 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 16), align 16, !tbaa !16
  %5 = fcmp uno fp128 %4, 0xL00000000000000000000000000000000
  br i1 %5, label %6, label %21

6:                                                ; preds = %3
  %7 = tail call i1 @llvm.is.fpclass.f128(fp128 %4, i32 1)
  br i1 %7, label %29, label %8

8:                                                ; preds = %6
  %9 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 32), align 16, !tbaa !16
  %10 = fcmp uno fp128 %9, 0xL00000000000000000000000000000000
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = tail call i1 @llvm.is.fpclass.f128(fp128 %9, i32 1)
  br i1 %12, label %29, label %13

13:                                               ; preds = %11
  %14 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 48), align 16, !tbaa !16
  %15 = fcmp uno fp128 %14, 0xL00000000000000000000000000000000
  br i1 %15, label %16, label %21

16:                                               ; preds = %13
  %17 = tail call i1 @llvm.is.fpclass.f128(fp128 %14, i32 1)
  br i1 %17, label %29, label %18

18:                                               ; preds = %16
  %19 = load fp128, ptr @LongDoubleSNaNValues, align 16, !tbaa !16
  %20 = fcmp uno fp128 %19, 0xL00000000000000000000000000000000
  br i1 %20, label %59, label %53

21:                                               ; preds = %13, %8, %3, %0
  %22 = phi ptr [ @LongDoubleQNaNValues, %0 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 16), %3 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 32), %8 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 48), %13 ]
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.172, i32 noundef 77)
  %24 = load fp128, ptr %22, align 16, !tbaa !16
  %25 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %24)
  %26 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

27:                                               ; preds = %0
  %28 = tail call i1 @llvm.is.fpclass.f128(fp128 %1, i32 1)
  br i1 %28, label %29, label %3

29:                                               ; preds = %16, %11, %6, %27
  %30 = phi ptr [ @LongDoubleQNaNValues, %27 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 16), %6 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 32), %11 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 48), %16 ]
  %31 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.172, i32 noundef 78)
  %32 = load fp128, ptr %30, align 16, !tbaa !16
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %32)
  %34 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

35:                                               ; preds = %59
  %36 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 16), align 16, !tbaa !16
  %37 = fcmp uno fp128 %36, 0xL00000000000000000000000000000000
  br i1 %37, label %38, label %53

38:                                               ; preds = %35
  %39 = tail call i1 @llvm.is.fpclass.f128(fp128 %36, i32 1)
  br i1 %39, label %40, label %61

40:                                               ; preds = %38
  %41 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 32), align 16, !tbaa !16
  %42 = fcmp uno fp128 %41, 0xL00000000000000000000000000000000
  br i1 %42, label %43, label %53

43:                                               ; preds = %40
  %44 = tail call i1 @llvm.is.fpclass.f128(fp128 %41, i32 1)
  br i1 %44, label %45, label %61

45:                                               ; preds = %43
  %46 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 48), align 16, !tbaa !16
  %47 = fcmp uno fp128 %46, 0xL00000000000000000000000000000000
  br i1 %47, label %48, label %53

48:                                               ; preds = %45
  %49 = tail call i1 @llvm.is.fpclass.f128(fp128 %46, i32 1)
  br i1 %49, label %50, label %61

50:                                               ; preds = %48
  %51 = load fp128, ptr @LongDoubleInfValues, align 16, !tbaa !16
  %52 = fcmp uno fp128 %51, 0xL00000000000000000000000000000000
  br i1 %52, label %76, label %82

53:                                               ; preds = %45, %40, %35, %18
  %54 = phi ptr [ @LongDoubleSNaNValues, %18 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 16), %35 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 32), %40 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 48), %45 ]
  %55 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.172, i32 noundef 89)
  %56 = load fp128, ptr %54, align 16, !tbaa !16
  %57 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %56)
  %58 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

59:                                               ; preds = %18
  %60 = tail call i1 @llvm.is.fpclass.f128(fp128 %19, i32 1)
  br i1 %60, label %35, label %61

61:                                               ; preds = %48, %43, %38, %59
  %62 = phi ptr [ @LongDoubleSNaNValues, %59 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 16), %38 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 32), %43 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 48), %48 ]
  %63 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.12, ptr noundef nonnull @.str.172, i32 noundef 90)
  %64 = load fp128, ptr %62, align 16, !tbaa !16
  %65 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %64)
  %66 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

67:                                               ; preds = %82
  %68 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleInfValues, i64 16), align 16, !tbaa !16
  %69 = fcmp uno fp128 %68, 0xL00000000000000000000000000000000
  br i1 %69, label %76, label %70

70:                                               ; preds = %67
  %71 = tail call fp128 @llvm.fabs.f128(fp128 %68)
  %72 = fcmp oeq fp128 %71, 0xL00000000000000007FFF000000000000
  br i1 %72, label %73, label %85

73:                                               ; preds = %70
  %74 = load fp128, ptr @LongDoubleZeroValues, align 16, !tbaa !16
  %75 = fcmp uno fp128 %74, 0xL00000000000000000000000000000000
  br i1 %75, label %104, label %110

76:                                               ; preds = %67, %50
  %77 = phi ptr [ @LongDoubleInfValues, %50 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleInfValues, i64 16), %67 ]
  %78 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.172, i32 noundef 103)
  %79 = load fp128, ptr %77, align 16, !tbaa !16
  %80 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %79)
  %81 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

82:                                               ; preds = %50
  %83 = tail call fp128 @llvm.fabs.f128(fp128 %51)
  %84 = fcmp oeq fp128 %83, 0xL00000000000000007FFF000000000000
  br i1 %84, label %67, label %85

85:                                               ; preds = %70, %82
  %86 = phi ptr [ @LongDoubleInfValues, %82 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleInfValues, i64 16), %70 ]
  %87 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.14, ptr noundef nonnull @.str.172, i32 noundef 105)
  %88 = load fp128, ptr %86, align 16, !tbaa !16
  %89 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %88)
  %90 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

91:                                               ; preds = %127
  %92 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleZeroValues, i64 16), align 16, !tbaa !16
  %93 = fcmp uno fp128 %92, 0xL00000000000000000000000000000000
  br i1 %93, label %104, label %94

94:                                               ; preds = %91
  %95 = tail call fp128 @llvm.fabs.f128(fp128 %92)
  %96 = fcmp oeq fp128 %95, 0xL00000000000000007FFF000000000000
  br i1 %96, label %113, label %97

97:                                               ; preds = %94
  %98 = tail call i1 @llvm.is.fpclass.f128(fp128 %92, i32 264)
  br i1 %98, label %121, label %99

99:                                               ; preds = %97
  %100 = tail call i1 @llvm.is.fpclass.f128(fp128 %92, i32 144)
  br i1 %100, label %129, label %101

101:                                              ; preds = %99
  %102 = load fp128, ptr @LongDoubleDenormValues, align 16, !tbaa !16
  %103 = fcmp uno fp128 %102, 0xL00000000000000000000000000000000
  br i1 %103, label %148, label %154

104:                                              ; preds = %91, %73
  %105 = phi ptr [ @LongDoubleZeroValues, %73 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleZeroValues, i64 16), %91 ]
  %106 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.172, i32 noundef 116)
  %107 = load fp128, ptr %105, align 16, !tbaa !16
  %108 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %107)
  %109 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

110:                                              ; preds = %73
  %111 = tail call fp128 @llvm.fabs.f128(fp128 %74)
  %112 = fcmp oeq fp128 %111, 0xL00000000000000007FFF000000000000
  br i1 %112, label %113, label %119

113:                                              ; preds = %94, %110
  %114 = phi ptr [ @LongDoubleZeroValues, %110 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleZeroValues, i64 16), %94 ]
  %115 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.172, i32 noundef 118)
  %116 = load fp128, ptr %114, align 16, !tbaa !16
  %117 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %116)
  %118 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

119:                                              ; preds = %110
  %120 = tail call i1 @llvm.is.fpclass.f128(fp128 %74, i32 264)
  br i1 %120, label %121, label %127

121:                                              ; preds = %97, %119
  %122 = phi ptr [ @LongDoubleZeroValues, %119 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleZeroValues, i64 16), %97 ]
  %123 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.172, i32 noundef 120)
  %124 = load fp128, ptr %122, align 16, !tbaa !16
  %125 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %124)
  %126 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

127:                                              ; preds = %119
  %128 = tail call i1 @llvm.is.fpclass.f128(fp128 %74, i32 144)
  br i1 %128, label %129, label %91

129:                                              ; preds = %99, %127
  %130 = phi ptr [ @LongDoubleZeroValues, %127 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleZeroValues, i64 16), %99 ]
  %131 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.172, i32 noundef 121)
  %132 = load fp128, ptr %130, align 16, !tbaa !16
  %133 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %132)
  %134 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

135:                                              ; preds = %171
  %136 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleDenormValues, i64 16), align 16, !tbaa !16
  %137 = fcmp uno fp128 %136, 0xL00000000000000000000000000000000
  br i1 %137, label %148, label %138

138:                                              ; preds = %135
  %139 = tail call fp128 @llvm.fabs.f128(fp128 %136)
  %140 = fcmp oeq fp128 %139, 0xL00000000000000007FFF000000000000
  br i1 %140, label %157, label %141

141:                                              ; preds = %138
  %142 = tail call i1 @llvm.is.fpclass.f128(fp128 %136, i32 264)
  br i1 %142, label %165, label %143

143:                                              ; preds = %141
  %144 = tail call i1 @llvm.is.fpclass.f128(fp128 %136, i32 144)
  br i1 %144, label %145, label %173

145:                                              ; preds = %143
  %146 = load fp128, ptr @LongDoubleNormalValues, align 16, !tbaa !16
  %147 = fcmp uno fp128 %146, 0xL00000000000000000000000000000000
  br i1 %147, label %220, label %226

148:                                              ; preds = %135, %101
  %149 = phi ptr [ @LongDoubleDenormValues, %101 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleDenormValues, i64 16), %135 ]
  %150 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.172, i32 noundef 129)
  %151 = load fp128, ptr %149, align 16, !tbaa !16
  %152 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %151)
  %153 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

154:                                              ; preds = %101
  %155 = tail call fp128 @llvm.fabs.f128(fp128 %102)
  %156 = fcmp oeq fp128 %155, 0xL00000000000000007FFF000000000000
  br i1 %156, label %157, label %163

157:                                              ; preds = %138, %154
  %158 = phi ptr [ @LongDoubleDenormValues, %154 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleDenormValues, i64 16), %138 ]
  %159 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.172, i32 noundef 131)
  %160 = load fp128, ptr %158, align 16, !tbaa !16
  %161 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %160)
  %162 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

163:                                              ; preds = %154
  %164 = tail call i1 @llvm.is.fpclass.f128(fp128 %102, i32 264)
  br i1 %164, label %165, label %171

165:                                              ; preds = %141, %163
  %166 = phi ptr [ @LongDoubleDenormValues, %163 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleDenormValues, i64 16), %141 ]
  %167 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.172, i32 noundef 133)
  %168 = load fp128, ptr %166, align 16, !tbaa !16
  %169 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %168)
  %170 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

171:                                              ; preds = %163
  %172 = tail call i1 @llvm.is.fpclass.f128(fp128 %102, i32 144)
  br i1 %172, label %135, label %173

173:                                              ; preds = %143, %171
  %174 = phi ptr [ @LongDoubleDenormValues, %171 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleDenormValues, i64 16), %143 ]
  %175 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.19, ptr noundef nonnull @.str.172, i32 noundef 134)
  %176 = load fp128, ptr %174, align 16, !tbaa !16
  %177 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %176)
  %178 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

179:                                              ; preds = %235
  %180 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 16), align 16, !tbaa !16
  %181 = fcmp uno fp128 %180, 0xL00000000000000000000000000000000
  br i1 %181, label %220, label %182

182:                                              ; preds = %179
  %183 = tail call fp128 @llvm.fabs.f128(fp128 %180)
  %184 = fcmp oeq fp128 %183, 0xL00000000000000007FFF000000000000
  br i1 %184, label %229, label %185

185:                                              ; preds = %182
  %186 = tail call i1 @llvm.is.fpclass.f128(fp128 %180, i32 264)
  br i1 %186, label %187, label %237

187:                                              ; preds = %185
  %188 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 32), align 16, !tbaa !16
  %189 = fcmp uno fp128 %188, 0xL00000000000000000000000000000000
  br i1 %189, label %220, label %190

190:                                              ; preds = %187
  %191 = tail call fp128 @llvm.fabs.f128(fp128 %188)
  %192 = fcmp oeq fp128 %191, 0xL00000000000000007FFF000000000000
  br i1 %192, label %229, label %193

193:                                              ; preds = %190
  %194 = tail call i1 @llvm.is.fpclass.f128(fp128 %188, i32 264)
  br i1 %194, label %195, label %237

195:                                              ; preds = %193
  %196 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 48), align 16, !tbaa !16
  %197 = fcmp uno fp128 %196, 0xL00000000000000000000000000000000
  br i1 %197, label %220, label %198

198:                                              ; preds = %195
  %199 = tail call fp128 @llvm.fabs.f128(fp128 %196)
  %200 = fcmp oeq fp128 %199, 0xL00000000000000007FFF000000000000
  br i1 %200, label %229, label %201

201:                                              ; preds = %198
  %202 = tail call i1 @llvm.is.fpclass.f128(fp128 %196, i32 264)
  br i1 %202, label %203, label %237

203:                                              ; preds = %201
  %204 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 64), align 16, !tbaa !16
  %205 = fcmp uno fp128 %204, 0xL00000000000000000000000000000000
  br i1 %205, label %220, label %206

206:                                              ; preds = %203
  %207 = tail call fp128 @llvm.fabs.f128(fp128 %204)
  %208 = fcmp oeq fp128 %207, 0xL00000000000000007FFF000000000000
  br i1 %208, label %229, label %209

209:                                              ; preds = %206
  %210 = tail call i1 @llvm.is.fpclass.f128(fp128 %204, i32 264)
  br i1 %210, label %211, label %237

211:                                              ; preds = %209
  %212 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 80), align 16, !tbaa !16
  %213 = fcmp uno fp128 %212, 0xL00000000000000000000000000000000
  br i1 %213, label %220, label %214

214:                                              ; preds = %211
  %215 = tail call fp128 @llvm.fabs.f128(fp128 %212)
  %216 = fcmp oeq fp128 %215, 0xL00000000000000007FFF000000000000
  br i1 %216, label %229, label %217

217:                                              ; preds = %214
  %218 = tail call i1 @llvm.is.fpclass.f128(fp128 %212, i32 264)
  br i1 %218, label %219, label %237

219:                                              ; preds = %217
  ret i32 0

220:                                              ; preds = %211, %203, %195, %187, %179, %145
  %221 = phi ptr [ @LongDoubleNormalValues, %145 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 16), %179 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 32), %187 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 48), %195 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 64), %203 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 80), %211 ]
  %222 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.172, i32 noundef 142)
  %223 = load fp128, ptr %221, align 16, !tbaa !16
  %224 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %223)
  %225 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

226:                                              ; preds = %145
  %227 = tail call fp128 @llvm.fabs.f128(fp128 %146)
  %228 = fcmp oeq fp128 %227, 0xL00000000000000007FFF000000000000
  br i1 %228, label %229, label %235

229:                                              ; preds = %214, %206, %198, %190, %182, %226
  %230 = phi ptr [ @LongDoubleNormalValues, %226 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 16), %182 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 32), %190 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 48), %198 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 64), %206 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 80), %214 ]
  %231 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.172, i32 noundef 144)
  %232 = load fp128, ptr %230, align 16, !tbaa !16
  %233 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %232)
  %234 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable

235:                                              ; preds = %226
  %236 = tail call i1 @llvm.is.fpclass.f128(fp128 %146, i32 264)
  br i1 %236, label %179, label %237

237:                                              ; preds = %217, %209, %201, %193, %185, %235
  %238 = phi ptr [ @LongDoubleNormalValues, %235 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 16), %185 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 32), %193 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 48), %201 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 64), %209 ], [ getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 80), %217 ]
  %239 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.21, ptr noundef nonnull @.str.172, i32 noundef 146)
  %240 = load fp128, ptr %238, align 16, !tbaa !16
  %241 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.173, fp128 noundef %240)
  %242 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.224)
  tail call void @exit(i32 noundef -1) #6
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f128(fp128, i32 immarg) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare fp128 @llvm.fabs.f128(fp128) #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcSNan_ldouble(fp128 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 1)
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.24, fp128 noundef %0, i32 noundef 1)
  tail call void @exit(i32 noundef -1) #6
  unreachable

5:                                                ; preds = %1
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcQNan_ldouble(fp128 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 1022)
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.24, fp128 noundef %0, i32 noundef 2)
  tail call void @exit(i32 noundef -1) #6
  unreachable

5:                                                ; preds = %1
  %6 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 2)
  br i1 %6, label %9, label %7

7:                                                ; preds = %5
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.25, fp128 noundef %0, i32 noundef 2)
  tail call void @exit(i32 noundef -1) #6
  unreachable

9:                                                ; preds = %5
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcPosInf_ldouble(fp128 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 1022)
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.24, fp128 noundef %0, i32 noundef 512)
  tail call void @exit(i32 noundef -1) #6
  unreachable

5:                                                ; preds = %1
  %6 = fcmp ord fp128 %0, 0xL00000000000000000000000000000000
  br i1 %6, label %9, label %7

7:                                                ; preds = %5
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.25, fp128 noundef %0, i32 noundef 512)
  tail call void @exit(i32 noundef -1) #6
  unreachable

9:                                                ; preds = %5
  %10 = fcmp oeq fp128 %0, 0xL00000000000000007FFF000000000000
  br i1 %10, label %13, label %11

11:                                               ; preds = %9
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.27, fp128 noundef %0, i32 noundef 512)
  tail call void @exit(i32 noundef -1) #6
  unreachable

13:                                               ; preds = %9
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcNegInf_ldouble(fp128 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 1022)
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.24, fp128 noundef %0, i32 noundef 4)
  tail call void @exit(i32 noundef -1) #6
  unreachable

5:                                                ; preds = %1
  %6 = fcmp ord fp128 %0, 0xL00000000000000000000000000000000
  br i1 %6, label %9, label %7

7:                                                ; preds = %5
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.25, fp128 noundef %0, i32 noundef 4)
  tail call void @exit(i32 noundef -1) #6
  unreachable

9:                                                ; preds = %5
  %10 = fcmp une fp128 %0, 0xL00000000000000007FFF000000000000
  br i1 %10, label %13, label %11

11:                                               ; preds = %9
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.27, fp128 noundef 0xL00000000000000007FFF000000000000, i32 noundef 4)
  tail call void @exit(i32 noundef -1) #6
  unreachable

13:                                               ; preds = %9
  %14 = fcmp oeq fp128 %0, 0xL0000000000000000FFFF000000000000
  br i1 %14, label %17, label %15

15:                                               ; preds = %13
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.28, fp128 noundef %0, i32 noundef 4)
  tail call void @exit(i32 noundef -1) #6
  unreachable

17:                                               ; preds = %13
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcPosNormal_ldouble(fp128 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 1022)
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.24, fp128 noundef %0, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

5:                                                ; preds = %1
  %6 = fcmp ord fp128 %0, 0xL00000000000000000000000000000000
  br i1 %6, label %9, label %7

7:                                                ; preds = %5
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.25, fp128 noundef %0, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

9:                                                ; preds = %5
  %10 = fcmp une fp128 %0, 0xL00000000000000007FFF000000000000
  br i1 %10, label %13, label %11

11:                                               ; preds = %9
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.27, fp128 noundef 0xL00000000000000007FFF000000000000, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

13:                                               ; preds = %9
  %14 = fcmp une fp128 %0, 0xL0000000000000000FFFF000000000000
  br i1 %14, label %17, label %15

15:                                               ; preds = %13
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.28, fp128 noundef 0xL0000000000000000FFFF000000000000, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

17:                                               ; preds = %13
  %18 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 256)
  br i1 %18, label %21, label %19

19:                                               ; preds = %17
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.30, fp128 noundef %0, i32 noundef 256)
  tail call void @exit(i32 noundef -1) #6
  unreachable

21:                                               ; preds = %17
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcNegNormal_ldouble(fp128 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 1022)
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.24, fp128 noundef %0, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

5:                                                ; preds = %1
  %6 = fcmp ord fp128 %0, 0xL00000000000000000000000000000000
  br i1 %6, label %9, label %7

7:                                                ; preds = %5
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.25, fp128 noundef %0, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

9:                                                ; preds = %5
  %10 = fcmp une fp128 %0, 0xL00000000000000007FFF000000000000
  br i1 %10, label %13, label %11

11:                                               ; preds = %9
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.27, fp128 noundef 0xL00000000000000007FFF000000000000, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

13:                                               ; preds = %9
  %14 = fcmp une fp128 %0, 0xL0000000000000000FFFF000000000000
  br i1 %14, label %17, label %15

15:                                               ; preds = %13
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.28, fp128 noundef 0xL0000000000000000FFFF000000000000, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

17:                                               ; preds = %13
  %18 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 248)
  br i1 %18, label %21, label %19

19:                                               ; preds = %17
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.30, fp128 noundef %0, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

21:                                               ; preds = %17
  %22 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 8)
  br i1 %22, label %25, label %23

23:                                               ; preds = %21
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.31, fp128 noundef %0, i32 noundef 8)
  tail call void @exit(i32 noundef -1) #6
  unreachable

25:                                               ; preds = %21
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcPosSubnormal_ldouble(fp128 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 1022)
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.24, fp128 noundef %0, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

5:                                                ; preds = %1
  %6 = fcmp ord fp128 %0, 0xL00000000000000000000000000000000
  br i1 %6, label %9, label %7

7:                                                ; preds = %5
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.25, fp128 noundef %0, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

9:                                                ; preds = %5
  %10 = fcmp une fp128 %0, 0xL00000000000000007FFF000000000000
  br i1 %10, label %13, label %11

11:                                               ; preds = %9
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.27, fp128 noundef 0xL00000000000000007FFF000000000000, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

13:                                               ; preds = %9
  %14 = fcmp une fp128 %0, 0xL0000000000000000FFFF000000000000
  br i1 %14, label %17, label %15

15:                                               ; preds = %13
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.28, fp128 noundef 0xL0000000000000000FFFF000000000000, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

17:                                               ; preds = %13
  %18 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 248)
  br i1 %18, label %21, label %19

19:                                               ; preds = %17
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.30, fp128 noundef %0, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

21:                                               ; preds = %17
  %22 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 240)
  br i1 %22, label %25, label %23

23:                                               ; preds = %21
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.31, fp128 noundef %0, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

25:                                               ; preds = %21
  %26 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 128)
  br i1 %26, label %29, label %27

27:                                               ; preds = %25
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.33, fp128 noundef %0, i32 noundef 128)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %25
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcNegSubnormal_ldouble(fp128 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 1022)
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.24, fp128 noundef %0, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

5:                                                ; preds = %1
  %6 = fcmp ord fp128 %0, 0xL00000000000000000000000000000000
  br i1 %6, label %9, label %7

7:                                                ; preds = %5
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.25, fp128 noundef %0, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

9:                                                ; preds = %5
  %10 = fcmp une fp128 %0, 0xL00000000000000007FFF000000000000
  br i1 %10, label %13, label %11

11:                                               ; preds = %9
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.27, fp128 noundef 0xL00000000000000007FFF000000000000, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

13:                                               ; preds = %9
  %14 = fcmp une fp128 %0, 0xL0000000000000000FFFF000000000000
  br i1 %14, label %17, label %15

15:                                               ; preds = %13
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.28, fp128 noundef 0xL0000000000000000FFFF000000000000, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

17:                                               ; preds = %13
  %18 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 248)
  br i1 %18, label %21, label %19

19:                                               ; preds = %17
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.30, fp128 noundef %0, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

21:                                               ; preds = %17
  %22 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 240)
  br i1 %22, label %25, label %23

23:                                               ; preds = %21
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.31, fp128 noundef %0, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

25:                                               ; preds = %21
  %26 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 112)
  br i1 %26, label %29, label %27

27:                                               ; preds = %25
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.33, fp128 noundef %0, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %25
  %30 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 16)
  br i1 %30, label %33, label %31

31:                                               ; preds = %29
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.34, fp128 noundef %0, i32 noundef 16)
  tail call void @exit(i32 noundef -1) #6
  unreachable

33:                                               ; preds = %29
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcPosZero_ldouble(fp128 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 1022)
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.24, fp128 noundef %0, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

5:                                                ; preds = %1
  %6 = fcmp ord fp128 %0, 0xL00000000000000000000000000000000
  br i1 %6, label %9, label %7

7:                                                ; preds = %5
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.25, fp128 noundef %0, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

9:                                                ; preds = %5
  %10 = fcmp une fp128 %0, 0xL00000000000000007FFF000000000000
  br i1 %10, label %13, label %11

11:                                               ; preds = %9
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.27, fp128 noundef 0xL00000000000000007FFF000000000000, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

13:                                               ; preds = %9
  %14 = fcmp une fp128 %0, 0xL0000000000000000FFFF000000000000
  br i1 %14, label %17, label %15

15:                                               ; preds = %13
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.28, fp128 noundef 0xL0000000000000000FFFF000000000000, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

17:                                               ; preds = %13
  %18 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 248)
  br i1 %18, label %21, label %19

19:                                               ; preds = %17
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.30, fp128 noundef %0, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

21:                                               ; preds = %17
  %22 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 240)
  br i1 %22, label %25, label %23

23:                                               ; preds = %21
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.31, fp128 noundef %0, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

25:                                               ; preds = %21
  %26 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 112)
  br i1 %26, label %29, label %27

27:                                               ; preds = %25
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.33, fp128 noundef %0, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %25
  %30 = fcmp oeq fp128 %0, 0xL00000000000000000000000000000000
  br i1 %30, label %33, label %31

31:                                               ; preds = %29
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.34, fp128 noundef %0, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

33:                                               ; preds = %29
  %34 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 64)
  br i1 %34, label %37, label %35

35:                                               ; preds = %33
  %36 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.36, fp128 noundef %0, i32 noundef 64)
  tail call void @exit(i32 noundef -1) #6
  unreachable

37:                                               ; preds = %33
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_fcNegZero_ldouble(fp128 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 1022)
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.24, fp128 noundef %0, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

5:                                                ; preds = %1
  %6 = fcmp ord fp128 %0, 0xL00000000000000000000000000000000
  br i1 %6, label %9, label %7

7:                                                ; preds = %5
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.25, fp128 noundef %0, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

9:                                                ; preds = %5
  %10 = fcmp une fp128 %0, 0xL00000000000000007FFF000000000000
  br i1 %10, label %13, label %11

11:                                               ; preds = %9
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.27, fp128 noundef 0xL00000000000000007FFF000000000000, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

13:                                               ; preds = %9
  %14 = fcmp une fp128 %0, 0xL0000000000000000FFFF000000000000
  br i1 %14, label %17, label %15

15:                                               ; preds = %13
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.28, fp128 noundef 0xL0000000000000000FFFF000000000000, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

17:                                               ; preds = %13
  %18 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 248)
  br i1 %18, label %21, label %19

19:                                               ; preds = %17
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.30, fp128 noundef %0, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

21:                                               ; preds = %17
  %22 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 240)
  br i1 %22, label %25, label %23

23:                                               ; preds = %21
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.31, fp128 noundef %0, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

25:                                               ; preds = %21
  %26 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 112)
  br i1 %26, label %29, label %27

27:                                               ; preds = %25
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.33, fp128 noundef %0, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

29:                                               ; preds = %25
  %30 = fcmp oeq fp128 %0, 0xL00000000000000000000000000000000
  br i1 %30, label %33, label %31

31:                                               ; preds = %29
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.34, fp128 noundef %0, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

33:                                               ; preds = %29
  %34 = tail call i1 @llvm.is.fpclass.f128(fp128 %0, i32 32)
  br i1 %34, label %37, label %35

35:                                               ; preds = %33
  %36 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.36, fp128 noundef %0, i32 noundef 32)
  tail call void @exit(i32 noundef -1) #6
  unreachable

37:                                               ; preds = %33
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_isfpclass_ldouble() local_unnamed_addr #0 {
  %1 = load fp128, ptr @LongDoubleZeroValues, align 16, !tbaa !16
  %2 = bitcast fp128 %1 to i128
  %3 = icmp slt i128 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %0
  tail call void @test_fcNegZero_ldouble(fp128 noundef %1)
  br label %6

5:                                                ; preds = %0
  tail call void @test_fcPosZero_ldouble(fp128 noundef %1)
  br label %6

6:                                                ; preds = %5, %4
  %7 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleZeroValues, i64 16), align 16, !tbaa !16
  %8 = bitcast fp128 %7 to i128
  %9 = icmp slt i128 %8, 0
  br i1 %9, label %11, label %10

10:                                               ; preds = %6
  tail call void @test_fcPosZero_ldouble(fp128 noundef %7)
  br label %12

11:                                               ; preds = %6
  tail call void @test_fcNegZero_ldouble(fp128 noundef %7)
  br label %12

12:                                               ; preds = %11, %10
  %13 = load fp128, ptr @LongDoubleDenormValues, align 16, !tbaa !16
  %14 = fcmp olt fp128 %13, 0xL00000000000000000000000000000000
  br i1 %14, label %15, label %16

15:                                               ; preds = %12
  tail call void @test_fcNegSubnormal_ldouble(fp128 noundef %13)
  br label %17

16:                                               ; preds = %12
  tail call void @test_fcPosSubnormal_ldouble(fp128 noundef %13)
  br label %17

17:                                               ; preds = %16, %15
  %18 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleDenormValues, i64 16), align 16, !tbaa !16
  %19 = fcmp olt fp128 %18, 0xL00000000000000000000000000000000
  br i1 %19, label %21, label %20

20:                                               ; preds = %17
  tail call void @test_fcPosSubnormal_ldouble(fp128 noundef %18)
  br label %22

21:                                               ; preds = %17
  tail call void @test_fcNegSubnormal_ldouble(fp128 noundef %18)
  br label %22

22:                                               ; preds = %21, %20
  %23 = load fp128, ptr @LongDoubleNormalValues, align 16, !tbaa !16
  %24 = fcmp olt fp128 %23, 0xL00000000000000000000000000000000
  br i1 %24, label %25, label %26

25:                                               ; preds = %22
  tail call void @test_fcNegNormal_ldouble(fp128 noundef %23)
  br label %27

26:                                               ; preds = %22
  tail call void @test_fcPosNormal_ldouble(fp128 noundef %23)
  br label %27

27:                                               ; preds = %26, %25
  %28 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 16), align 16, !tbaa !16
  %29 = fcmp olt fp128 %28, 0xL00000000000000000000000000000000
  br i1 %29, label %31, label %30

30:                                               ; preds = %27
  tail call void @test_fcPosNormal_ldouble(fp128 noundef %28)
  br label %32

31:                                               ; preds = %27
  tail call void @test_fcNegNormal_ldouble(fp128 noundef %28)
  br label %32

32:                                               ; preds = %31, %30
  %33 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 32), align 16, !tbaa !16
  %34 = fcmp olt fp128 %33, 0xL00000000000000000000000000000000
  br i1 %34, label %36, label %35

35:                                               ; preds = %32
  tail call void @test_fcPosNormal_ldouble(fp128 noundef %33)
  br label %37

36:                                               ; preds = %32
  tail call void @test_fcNegNormal_ldouble(fp128 noundef %33)
  br label %37

37:                                               ; preds = %36, %35
  %38 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 48), align 16, !tbaa !16
  %39 = fcmp olt fp128 %38, 0xL00000000000000000000000000000000
  br i1 %39, label %41, label %40

40:                                               ; preds = %37
  tail call void @test_fcPosNormal_ldouble(fp128 noundef %38)
  br label %42

41:                                               ; preds = %37
  tail call void @test_fcNegNormal_ldouble(fp128 noundef %38)
  br label %42

42:                                               ; preds = %41, %40
  %43 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 64), align 16, !tbaa !16
  %44 = fcmp olt fp128 %43, 0xL00000000000000000000000000000000
  br i1 %44, label %46, label %45

45:                                               ; preds = %42
  tail call void @test_fcPosNormal_ldouble(fp128 noundef %43)
  br label %47

46:                                               ; preds = %42
  tail call void @test_fcNegNormal_ldouble(fp128 noundef %43)
  br label %47

47:                                               ; preds = %46, %45
  %48 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 80), align 16, !tbaa !16
  %49 = fcmp olt fp128 %48, 0xL00000000000000000000000000000000
  br i1 %49, label %51, label %50

50:                                               ; preds = %47
  tail call void @test_fcPosNormal_ldouble(fp128 noundef %48)
  br label %52

51:                                               ; preds = %47
  tail call void @test_fcNegNormal_ldouble(fp128 noundef %48)
  br label %52

52:                                               ; preds = %51, %50
  %53 = load fp128, ptr @LongDoubleInfValues, align 16, !tbaa !16
  %54 = fcmp ogt fp128 %53, 0xL00000000000000000000000000000000
  br i1 %54, label %55, label %56

55:                                               ; preds = %52
  tail call void @test_fcPosInf_ldouble(fp128 noundef %53)
  br label %57

56:                                               ; preds = %52
  tail call void @test_fcNegInf_ldouble(fp128 noundef %53)
  br label %57

57:                                               ; preds = %56, %55
  %58 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleInfValues, i64 16), align 16, !tbaa !16
  %59 = fcmp ogt fp128 %58, 0xL00000000000000000000000000000000
  br i1 %59, label %61, label %60

60:                                               ; preds = %57
  tail call void @test_fcNegInf_ldouble(fp128 noundef %58)
  br label %62

61:                                               ; preds = %57
  tail call void @test_fcPosInf_ldouble(fp128 noundef %58)
  br label %62

62:                                               ; preds = %61, %60
  %63 = load fp128, ptr @LongDoubleQNaNValues, align 16, !tbaa !16
  %64 = tail call i1 @llvm.is.fpclass.f128(fp128 %63, i32 1022)
  br i1 %64, label %86, label %83

65:                                               ; preds = %86
  %66 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 16), align 16, !tbaa !16
  %67 = tail call i1 @llvm.is.fpclass.f128(fp128 %66, i32 1022)
  br i1 %67, label %68, label %83

68:                                               ; preds = %65
  %69 = tail call i1 @llvm.is.fpclass.f128(fp128 %66, i32 2)
  br i1 %69, label %70, label %88

70:                                               ; preds = %68
  %71 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 32), align 16, !tbaa !16
  %72 = tail call i1 @llvm.is.fpclass.f128(fp128 %71, i32 1022)
  br i1 %72, label %73, label %83

73:                                               ; preds = %70
  %74 = tail call i1 @llvm.is.fpclass.f128(fp128 %71, i32 2)
  br i1 %74, label %75, label %88

75:                                               ; preds = %73
  %76 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 48), align 16, !tbaa !16
  %77 = tail call i1 @llvm.is.fpclass.f128(fp128 %76, i32 1022)
  br i1 %77, label %78, label %83

78:                                               ; preds = %75
  %79 = tail call i1 @llvm.is.fpclass.f128(fp128 %76, i32 2)
  br i1 %79, label %80, label %88

80:                                               ; preds = %78
  %81 = load fp128, ptr @LongDoubleSNaNValues, align 16, !tbaa !16
  %82 = tail call i1 @llvm.is.fpclass.f128(fp128 %81, i32 1)
  br i1 %82, label %91, label %101

83:                                               ; preds = %75, %70, %65, %62
  %84 = phi fp128 [ %63, %62 ], [ %66, %65 ], [ %71, %70 ], [ %76, %75 ]
  %85 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.24, fp128 noundef %84, i32 noundef 2)
  tail call void @exit(i32 noundef -1) #6
  unreachable

86:                                               ; preds = %62
  %87 = tail call i1 @llvm.is.fpclass.f128(fp128 %63, i32 2)
  br i1 %87, label %65, label %88

88:                                               ; preds = %78, %73, %68, %86
  %89 = phi fp128 [ %63, %86 ], [ %66, %68 ], [ %71, %73 ], [ %76, %78 ]
  %90 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.25, fp128 noundef %89, i32 noundef 2)
  tail call void @exit(i32 noundef -1) #6
  unreachable

91:                                               ; preds = %80
  %92 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 16), align 16, !tbaa !16
  %93 = tail call i1 @llvm.is.fpclass.f128(fp128 %92, i32 1)
  br i1 %93, label %94, label %101

94:                                               ; preds = %91
  %95 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 32), align 16, !tbaa !16
  %96 = tail call i1 @llvm.is.fpclass.f128(fp128 %95, i32 1)
  br i1 %96, label %97, label %101

97:                                               ; preds = %94
  %98 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 48), align 16, !tbaa !16
  %99 = tail call i1 @llvm.is.fpclass.f128(fp128 %98, i32 1)
  br i1 %99, label %100, label %101

100:                                              ; preds = %97
  ret void

101:                                              ; preds = %97, %94, %91, %80
  %102 = phi fp128 [ %81, %80 ], [ %92, %91 ], [ %95, %94 ], [ %98, %97 ]
  %103 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.174, ptr noundef nonnull @.str.24, fp128 noundef %102, i32 noundef 1)
  tail call void @exit(i32 noundef -1) #6
  unreachable
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = tail call i32 @test_float()
  tail call void @test_isfpclass_float()
  %2 = tail call i32 @test_double()
  tail call void @test_isfpclass_double()
  store fp128 0xL00000000000000007FFF800000000000, ptr @LongDoubleQNaNValues, align 16, !tbaa !16
  store fp128 0xL0000000000000000FFFF800000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 16), align 16, !tbaa !16
  store fp128 0xL00000000000000017FFF800000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 32), align 16, !tbaa !16
  store fp128 0xL0000000000000001FFFF800000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleQNaNValues, i64 48), align 16, !tbaa !16
  store fp128 0xL00000000000000007FFF400000000000, ptr @LongDoubleSNaNValues, align 16, !tbaa !16
  store fp128 0xL0000000000000000FFFF400000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 16), align 16, !tbaa !16
  store fp128 0xL00000000000000017FFF000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 32), align 16, !tbaa !16
  store fp128 0xL0000000000000001FFFF000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleSNaNValues, i64 48), align 16, !tbaa !16
  store fp128 0xL00000000000000007FFF000000000000, ptr @LongDoubleInfValues, align 16, !tbaa !16
  store fp128 0xL0000000000000000FFFF000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleInfValues, i64 16), align 16, !tbaa !16
  store fp128 0xL00000000000000010000000000000000, ptr @LongDoubleDenormValues, align 16, !tbaa !16
  store fp128 0xL00000000000000018000000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleDenormValues, i64 16), align 16, !tbaa !16
  store fp128 0xL00000000000000003FFF000000000000, ptr @LongDoubleNormalValues, align 16, !tbaa !16
  store fp128 0xL0000000000000000BFFF000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 16), align 16, !tbaa !16
  store fp128 0xLFFFFFFFFFFFFFFFF7FFEFFFFFFFFFFFF, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 32), align 16, !tbaa !16
  store fp128 0xLFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFF, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 48), align 16, !tbaa !16
  store fp128 0xL00000000000000000001000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 64), align 16, !tbaa !16
  store fp128 0xL00000000000000008001000000000000, ptr getelementptr inbounds nuw (i8, ptr @LongDoubleNormalValues, i64 80), align 16, !tbaa !16
  %3 = tail call i32 @test_ldouble()
  tail call void @test_isfpclass_ldouble()
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #5

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind }
attributes #6 = { cold noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"float", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"double", !8, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"long", !8, i64 0}
!16 = !{!17, !17, i64 0}
!17 = !{!"long double", !8, i64 0}
