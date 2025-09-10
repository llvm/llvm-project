; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr59643.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr59643.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@expected = dso_local local_unnamed_addr global [32 x double] [double 0.000000e+00, double 1.000000e+01, double 4.400000e+01, double 1.100000e+02, double 2.320000e+02, double 4.900000e+02, double 1.020000e+03, double 2.078000e+03, double 4.152000e+03, double 8.314000e+03, double 1.665200e+04, double 3.332600e+04, double 6.666400e+04, double 1.333540e+05, double 2.667480e+05, double 5.335340e+05, double 0x4130483800000000, double 0x4140483D00000000, double 4.268300e+06, double 0x41604845C0000000, double 0x4170484680000000, double 0x4180484750000000, double 0x41904847F0000000, double 0x41A048483C000000, double 0x41B0484838000000, double 0x41C048483D000000, double 0x41D0484843000000, double 0x41E0484845C00000, double 0x41F0484846800000, double 0x4200484847500000, double 0x4210484847F00000, double 6.000000e+00], align 8

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @foo(ptr noundef captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef readonly captures(none) %2, double noundef %3, double noundef %4, i32 noundef %5) local_unnamed_addr #0 {
  %7 = icmp sgt i32 %5, 2
  br i1 %7, label %8, label %32

8:                                                ; preds = %6
  %9 = add nsw i32 %5, -1
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr i8, ptr %0, i64 8
  %12 = load double, ptr %11, align 8, !tbaa !6
  %13 = load double, ptr %0, align 8
  br label %14

14:                                               ; preds = %8, %14
  %15 = phi double [ %13, %8 ], [ %30, %14 ]
  %16 = phi double [ %12, %8 ], [ %27, %14 ]
  %17 = phi i64 [ 1, %8 ], [ %25, %14 ]
  %18 = getelementptr inbounds nuw double, ptr %1, i64 %17
  %19 = load double, ptr %18, align 8, !tbaa !6
  %20 = getelementptr inbounds nuw double, ptr %2, i64 %17
  %21 = load double, ptr %20, align 8, !tbaa !6
  %22 = fadd double %19, %21
  %23 = getelementptr double, ptr %0, i64 %17
  %24 = fadd double %22, %15
  %25 = add nuw nsw i64 %17, 1
  %26 = getelementptr inbounds nuw double, ptr %0, i64 %25
  %27 = load double, ptr %26, align 8, !tbaa !6
  %28 = fadd double %24, %27
  %29 = fmul double %4, %16
  %30 = tail call double @llvm.fmuladd.f64(double %3, double %28, double %29)
  store double %30, ptr %23, align 8, !tbaa !6
  %31 = icmp eq i64 %25, %10
  br i1 %31, label %32, label %14, !llvm.loop !10

32:                                               ; preds = %14, %6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = alloca [32 x double], align 16
  %2 = alloca [32 x double], align 16
  %3 = alloca [32 x double], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store <2 x double> <double 0.000000e+00, double 2.000000e+00>, ptr %1, align 16, !tbaa !6
  store <2 x double> <double 4.000000e+00, double 6.000000e+00>, ptr %4, align 16, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store <2 x double> <double -4.000000e+00, double -3.000000e+00>, ptr %2, align 16, !tbaa !6
  store <2 x double> <double -2.000000e+00, double -1.000000e+00>, ptr %5, align 16, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store <2 x double> <double 0.000000e+00, double 1.000000e+00>, ptr %3, align 16, !tbaa !6
  store <2 x double> <double 2.000000e+00, double 3.000000e+00>, ptr %6, align 16, !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 48
  store <2 x double> <double 0.000000e+00, double 2.000000e+00>, ptr %7, align 16, !tbaa !6
  store <2 x double> <double 4.000000e+00, double 6.000000e+00>, ptr %8, align 16, !tbaa !6
  %9 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 48
  store <2 x double> <double 0.000000e+00, double 1.000000e+00>, ptr %9, align 16, !tbaa !6
  store <2 x double> <double 2.000000e+00, double 3.000000e+00>, ptr %10, align 16, !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %12 = getelementptr inbounds nuw i8, ptr %3, i64 48
  store <2 x double> <double 4.000000e+00, double 5.000000e+00>, ptr %11, align 16, !tbaa !6
  store <2 x double> <double 6.000000e+00, double 7.000000e+00>, ptr %12, align 16, !tbaa !6
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 80
  store <2 x double> <double 0.000000e+00, double 2.000000e+00>, ptr %13, align 16, !tbaa !6
  store <2 x double> <double 4.000000e+00, double 6.000000e+00>, ptr %14, align 16, !tbaa !6
  %15 = getelementptr inbounds nuw i8, ptr %2, i64 64
  %16 = getelementptr inbounds nuw i8, ptr %2, i64 80
  store <2 x double> <double -4.000000e+00, double -3.000000e+00>, ptr %15, align 16, !tbaa !6
  store <2 x double> <double -2.000000e+00, double -1.000000e+00>, ptr %16, align 16, !tbaa !6
  %17 = getelementptr inbounds nuw i8, ptr %3, i64 64
  %18 = getelementptr inbounds nuw i8, ptr %3, i64 80
  store <2 x double> <double 0.000000e+00, double 1.000000e+00>, ptr %17, align 16, !tbaa !6
  store <2 x double> <double 2.000000e+00, double 3.000000e+00>, ptr %18, align 16, !tbaa !6
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 112
  store <2 x double> <double 0.000000e+00, double 2.000000e+00>, ptr %19, align 16, !tbaa !6
  store <2 x double> <double 4.000000e+00, double 6.000000e+00>, ptr %20, align 16, !tbaa !6
  %21 = getelementptr inbounds nuw i8, ptr %2, i64 96
  %22 = getelementptr inbounds nuw i8, ptr %2, i64 112
  store <2 x double> <double 0.000000e+00, double 1.000000e+00>, ptr %21, align 16, !tbaa !6
  store <2 x double> <double 2.000000e+00, double 3.000000e+00>, ptr %22, align 16, !tbaa !6
  %23 = getelementptr inbounds nuw i8, ptr %3, i64 96
  %24 = getelementptr inbounds nuw i8, ptr %3, i64 112
  store <2 x double> <double 4.000000e+00, double 5.000000e+00>, ptr %23, align 16, !tbaa !6
  store <2 x double> <double 6.000000e+00, double 7.000000e+00>, ptr %24, align 16, !tbaa !6
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 144
  store <2 x double> <double 0.000000e+00, double 2.000000e+00>, ptr %25, align 16, !tbaa !6
  store <2 x double> <double 4.000000e+00, double 6.000000e+00>, ptr %26, align 16, !tbaa !6
  %27 = getelementptr inbounds nuw i8, ptr %2, i64 128
  %28 = getelementptr inbounds nuw i8, ptr %2, i64 144
  store <2 x double> <double -4.000000e+00, double -3.000000e+00>, ptr %27, align 16, !tbaa !6
  store <2 x double> <double -2.000000e+00, double -1.000000e+00>, ptr %28, align 16, !tbaa !6
  %29 = getelementptr inbounds nuw i8, ptr %3, i64 128
  %30 = getelementptr inbounds nuw i8, ptr %3, i64 144
  store <2 x double> <double 0.000000e+00, double 1.000000e+00>, ptr %29, align 16, !tbaa !6
  store <2 x double> <double 2.000000e+00, double 3.000000e+00>, ptr %30, align 16, !tbaa !6
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 176
  store <2 x double> <double 0.000000e+00, double 2.000000e+00>, ptr %31, align 16, !tbaa !6
  store <2 x double> <double 4.000000e+00, double 6.000000e+00>, ptr %32, align 16, !tbaa !6
  %33 = getelementptr inbounds nuw i8, ptr %2, i64 160
  %34 = getelementptr inbounds nuw i8, ptr %2, i64 176
  store <2 x double> <double 0.000000e+00, double 1.000000e+00>, ptr %33, align 16, !tbaa !6
  store <2 x double> <double 2.000000e+00, double 3.000000e+00>, ptr %34, align 16, !tbaa !6
  %35 = getelementptr inbounds nuw i8, ptr %3, i64 160
  %36 = getelementptr inbounds nuw i8, ptr %3, i64 176
  store <2 x double> <double 4.000000e+00, double 5.000000e+00>, ptr %35, align 16, !tbaa !6
  store <2 x double> <double 6.000000e+00, double 7.000000e+00>, ptr %36, align 16, !tbaa !6
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 208
  store <2 x double> <double 0.000000e+00, double 2.000000e+00>, ptr %37, align 16, !tbaa !6
  store <2 x double> <double 4.000000e+00, double 6.000000e+00>, ptr %38, align 16, !tbaa !6
  %39 = getelementptr inbounds nuw i8, ptr %2, i64 192
  %40 = getelementptr inbounds nuw i8, ptr %2, i64 208
  store <2 x double> <double -4.000000e+00, double -3.000000e+00>, ptr %39, align 16, !tbaa !6
  store <2 x double> <double -2.000000e+00, double -1.000000e+00>, ptr %40, align 16, !tbaa !6
  %41 = getelementptr inbounds nuw i8, ptr %3, i64 192
  %42 = getelementptr inbounds nuw i8, ptr %3, i64 208
  store <2 x double> <double 0.000000e+00, double 1.000000e+00>, ptr %41, align 16, !tbaa !6
  store <2 x double> <double 2.000000e+00, double 3.000000e+00>, ptr %42, align 16, !tbaa !6
  %43 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %44 = getelementptr inbounds nuw i8, ptr %1, i64 240
  store <2 x double> <double 0.000000e+00, double 2.000000e+00>, ptr %43, align 16, !tbaa !6
  store <2 x double> <double 4.000000e+00, double 6.000000e+00>, ptr %44, align 16, !tbaa !6
  %45 = getelementptr inbounds nuw i8, ptr %2, i64 224
  %46 = getelementptr inbounds nuw i8, ptr %2, i64 240
  store <2 x double> <double 0.000000e+00, double 1.000000e+00>, ptr %45, align 16, !tbaa !6
  store <2 x double> <double 2.000000e+00, double 3.000000e+00>, ptr %46, align 16, !tbaa !6
  %47 = getelementptr inbounds nuw i8, ptr %3, i64 224
  %48 = getelementptr inbounds nuw i8, ptr %3, i64 240
  store <2 x double> <double 4.000000e+00, double 5.000000e+00>, ptr %47, align 16, !tbaa !6
  store <2 x double> <double 6.000000e+00, double 7.000000e+00>, ptr %48, align 16, !tbaa !6
  call void @foo(ptr noundef nonnull %1, ptr noundef nonnull %2, ptr noundef nonnull %3, double noundef 2.000000e+00, double noundef 3.000000e+00, i32 noundef 32)
  %49 = load double, ptr %1, align 16, !tbaa !6
  %50 = load double, ptr @expected, align 8, !tbaa !6
  %51 = fcmp une double %49, %50
  br i1 %51, label %208, label %52

52:                                               ; preds = %0
  %53 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %54 = load double, ptr %53, align 8, !tbaa !6
  %55 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 8), align 8, !tbaa !6
  %56 = fcmp une double %54, %55
  br i1 %56, label %208, label %57

57:                                               ; preds = %52
  %58 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %59 = load double, ptr %58, align 16, !tbaa !6
  %60 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 16), align 8, !tbaa !6
  %61 = fcmp une double %59, %60
  br i1 %61, label %208, label %62

62:                                               ; preds = %57
  %63 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %64 = load double, ptr %63, align 8, !tbaa !6
  %65 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 24), align 8, !tbaa !6
  %66 = fcmp une double %64, %65
  br i1 %66, label %208, label %67

67:                                               ; preds = %62
  %68 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %69 = load double, ptr %68, align 16, !tbaa !6
  %70 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 32), align 8, !tbaa !6
  %71 = fcmp une double %69, %70
  br i1 %71, label %208, label %72

72:                                               ; preds = %67
  %73 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %74 = load double, ptr %73, align 8, !tbaa !6
  %75 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 40), align 8, !tbaa !6
  %76 = fcmp une double %74, %75
  br i1 %76, label %208, label %77

77:                                               ; preds = %72
  %78 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %79 = load double, ptr %78, align 16, !tbaa !6
  %80 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 48), align 8, !tbaa !6
  %81 = fcmp une double %79, %80
  br i1 %81, label %208, label %82

82:                                               ; preds = %77
  %83 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %84 = load double, ptr %83, align 8, !tbaa !6
  %85 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 56), align 8, !tbaa !6
  %86 = fcmp une double %84, %85
  br i1 %86, label %208, label %87

87:                                               ; preds = %82
  %88 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %89 = load double, ptr %88, align 16, !tbaa !6
  %90 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 64), align 8, !tbaa !6
  %91 = fcmp une double %89, %90
  br i1 %91, label %208, label %92

92:                                               ; preds = %87
  %93 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %94 = load double, ptr %93, align 8, !tbaa !6
  %95 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 72), align 8, !tbaa !6
  %96 = fcmp une double %94, %95
  br i1 %96, label %208, label %97

97:                                               ; preds = %92
  %98 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %99 = load double, ptr %98, align 16, !tbaa !6
  %100 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 80), align 8, !tbaa !6
  %101 = fcmp une double %99, %100
  br i1 %101, label %208, label %102

102:                                              ; preds = %97
  %103 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %104 = load double, ptr %103, align 8, !tbaa !6
  %105 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 88), align 8, !tbaa !6
  %106 = fcmp une double %104, %105
  br i1 %106, label %208, label %107

107:                                              ; preds = %102
  %108 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %109 = load double, ptr %108, align 16, !tbaa !6
  %110 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 96), align 8, !tbaa !6
  %111 = fcmp une double %109, %110
  br i1 %111, label %208, label %112

112:                                              ; preds = %107
  %113 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %114 = load double, ptr %113, align 8, !tbaa !6
  %115 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 104), align 8, !tbaa !6
  %116 = fcmp une double %114, %115
  br i1 %116, label %208, label %117

117:                                              ; preds = %112
  %118 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %119 = load double, ptr %118, align 16, !tbaa !6
  %120 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 112), align 8, !tbaa !6
  %121 = fcmp une double %119, %120
  br i1 %121, label %208, label %122

122:                                              ; preds = %117
  %123 = getelementptr inbounds nuw i8, ptr %1, i64 120
  %124 = load double, ptr %123, align 8, !tbaa !6
  %125 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 120), align 8, !tbaa !6
  %126 = fcmp une double %124, %125
  br i1 %126, label %208, label %127

127:                                              ; preds = %122
  %128 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %129 = load double, ptr %128, align 16, !tbaa !6
  %130 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 128), align 8, !tbaa !6
  %131 = fcmp une double %129, %130
  br i1 %131, label %208, label %132

132:                                              ; preds = %127
  %133 = getelementptr inbounds nuw i8, ptr %1, i64 136
  %134 = load double, ptr %133, align 8, !tbaa !6
  %135 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 136), align 8, !tbaa !6
  %136 = fcmp une double %134, %135
  br i1 %136, label %208, label %137

137:                                              ; preds = %132
  %138 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %139 = load double, ptr %138, align 16, !tbaa !6
  %140 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 144), align 8, !tbaa !6
  %141 = fcmp une double %139, %140
  br i1 %141, label %208, label %142

142:                                              ; preds = %137
  %143 = getelementptr inbounds nuw i8, ptr %1, i64 152
  %144 = load double, ptr %143, align 8, !tbaa !6
  %145 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 152), align 8, !tbaa !6
  %146 = fcmp une double %144, %145
  br i1 %146, label %208, label %147

147:                                              ; preds = %142
  %148 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %149 = load double, ptr %148, align 16, !tbaa !6
  %150 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 160), align 8, !tbaa !6
  %151 = fcmp une double %149, %150
  br i1 %151, label %208, label %152

152:                                              ; preds = %147
  %153 = getelementptr inbounds nuw i8, ptr %1, i64 168
  %154 = load double, ptr %153, align 8, !tbaa !6
  %155 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 168), align 8, !tbaa !6
  %156 = fcmp une double %154, %155
  br i1 %156, label %208, label %157

157:                                              ; preds = %152
  %158 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %159 = load double, ptr %158, align 16, !tbaa !6
  %160 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 176), align 8, !tbaa !6
  %161 = fcmp une double %159, %160
  br i1 %161, label %208, label %162

162:                                              ; preds = %157
  %163 = getelementptr inbounds nuw i8, ptr %1, i64 184
  %164 = load double, ptr %163, align 8, !tbaa !6
  %165 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 184), align 8, !tbaa !6
  %166 = fcmp une double %164, %165
  br i1 %166, label %208, label %167

167:                                              ; preds = %162
  %168 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %169 = load double, ptr %168, align 16, !tbaa !6
  %170 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 192), align 8, !tbaa !6
  %171 = fcmp une double %169, %170
  br i1 %171, label %208, label %172

172:                                              ; preds = %167
  %173 = getelementptr inbounds nuw i8, ptr %1, i64 200
  %174 = load double, ptr %173, align 8, !tbaa !6
  %175 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 200), align 8, !tbaa !6
  %176 = fcmp une double %174, %175
  br i1 %176, label %208, label %177

177:                                              ; preds = %172
  %178 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %179 = load double, ptr %178, align 16, !tbaa !6
  %180 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 208), align 8, !tbaa !6
  %181 = fcmp une double %179, %180
  br i1 %181, label %208, label %182

182:                                              ; preds = %177
  %183 = getelementptr inbounds nuw i8, ptr %1, i64 216
  %184 = load double, ptr %183, align 8, !tbaa !6
  %185 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 216), align 8, !tbaa !6
  %186 = fcmp une double %184, %185
  br i1 %186, label %208, label %187

187:                                              ; preds = %182
  %188 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %189 = load double, ptr %188, align 16, !tbaa !6
  %190 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 224), align 8, !tbaa !6
  %191 = fcmp une double %189, %190
  br i1 %191, label %208, label %192

192:                                              ; preds = %187
  %193 = getelementptr inbounds nuw i8, ptr %1, i64 232
  %194 = load double, ptr %193, align 8, !tbaa !6
  %195 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 232), align 8, !tbaa !6
  %196 = fcmp une double %194, %195
  br i1 %196, label %208, label %197

197:                                              ; preds = %192
  %198 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %199 = load double, ptr %198, align 16, !tbaa !6
  %200 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 240), align 8, !tbaa !6
  %201 = fcmp une double %199, %200
  br i1 %201, label %208, label %202

202:                                              ; preds = %197
  %203 = getelementptr inbounds nuw i8, ptr %1, i64 248
  %204 = load double, ptr %203, align 8, !tbaa !6
  %205 = load double, ptr getelementptr inbounds nuw (i8, ptr @expected, i64 248), align 8, !tbaa !6
  %206 = fcmp une double %204, %205
  br i1 %206, label %208, label %207

207:                                              ; preds = %202
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0

208:                                              ; preds = %202, %197, %192, %187, %182, %177, %172, %167, %162, %157, %152, %147, %142, %137, %132, %127, %122, %117, %112, %107, %102, %97, %92, %87, %82, %77, %72, %67, %62, %57, %52, %0
  tail call void @abort() #6
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

attributes #0 = { nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
