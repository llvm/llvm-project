; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/ffbench.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/ffbench.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@main.nsize.0 = internal unnamed_addr global i1 false, align 4
@main.nsize.1 = internal unnamed_addr global i1 false, align 4
@stderr = external local_unnamed_addr global ptr, align 8
@.str = private unnamed_addr constant [28 x i8] c"Can't allocate data array.\0A\00", align 1
@.str.1 = private unnamed_addr constant [48 x i8] c"Wrong answer at (%d,%d)!  Expected %d, got %d.\0A\00", align 1
@.str.2 = private unnamed_addr constant [35 x i8] c"%d passes.  No errors in results.\0A\00", align 1
@.str.3 = private unnamed_addr constant [35 x i8] c"%d passes.  %d errors in results.\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  store i1 true, ptr @main.nsize.1, align 4
  store i1 true, ptr @main.nsize.0, align 4
  %1 = tail call dereferenceable_or_null(1048592) ptr @calloc(i64 1, i64 1048592)
  %2 = icmp eq ptr %1, null
  br i1 %2, label %3, label %6

3:                                                ; preds = %0
  %4 = load ptr, ptr @stderr, align 8, !tbaa !6
  %5 = tail call i64 @fwrite(ptr nonnull @.str, i64 27, i64 1, ptr %4) #9
  tail call void @exit(i32 noundef 1) #10
  unreachable

6:                                                ; preds = %0, %33
  %7 = phi i64 [ %34, %33 ], [ 0, %0 ]
  %8 = and i64 %7, 15
  %9 = icmp eq i64 %8, 8
  %10 = shl nsw i64 %7, 12
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 %10
  br i1 %9, label %12, label %22

12:                                               ; preds = %6, %12
  %13 = phi i64 [ %20, %12 ], [ 0, %6 ]
  %14 = shl nuw nsw i64 %13, 4
  %15 = shl i64 %13, 4
  %16 = getelementptr inbounds nuw i8, ptr %11, i64 %14
  %17 = getelementptr inbounds nuw i8, ptr %11, i64 %15
  %18 = getelementptr inbounds nuw i8, ptr %16, i64 8
  %19 = getelementptr inbounds nuw i8, ptr %17, i64 24
  store double 1.280000e+02, ptr %18, align 8, !tbaa !11
  store double 1.280000e+02, ptr %19, align 8, !tbaa !11
  %20 = add nuw i64 %13, 2
  %21 = icmp eq i64 %20, 256
  br i1 %21, label %33, label %12, !llvm.loop !13

22:                                               ; preds = %6, %30
  %23 = phi i64 [ %31, %30 ], [ 0, %6 ]
  %24 = and i64 %23, 15
  %25 = icmp eq i64 %24, 8
  br i1 %25, label %26, label %30

26:                                               ; preds = %22
  %27 = shl nuw nsw i64 %23, 4
  %28 = getelementptr inbounds nuw i8, ptr %11, i64 %27
  %29 = getelementptr inbounds nuw i8, ptr %28, i64 8
  store double 1.280000e+02, ptr %29, align 8, !tbaa !11
  br label %30

30:                                               ; preds = %22, %26
  %31 = add nuw nsw i64 %23, 1
  %32 = icmp eq i64 %31, 256
  br i1 %32, label %33, label %22, !llvm.loop !17

33:                                               ; preds = %30, %12
  %34 = add nuw nsw i64 %7, 1
  %35 = icmp eq i64 %34, 256
  br i1 %35, label %36, label %6, !llvm.loop !18

36:                                               ; preds = %33, %36
  %37 = phi i32 [ %38, %36 ], [ 0, %33 ]
  tail call fastcc void @fourn(ptr noundef %1, i32 noundef 1)
  tail call fastcc void @fourn(ptr noundef %1, i32 noundef -1)
  %38 = add nuw nsw i32 %37, 1
  %39 = icmp eq i32 %38, 63
  br i1 %39, label %40, label %36, !llvm.loop !19

40:                                               ; preds = %36, %40
  %41 = phi i64 [ %50, %40 ], [ 1, %36 ]
  %42 = phi double [ %49, %40 ], [ -1.000000e+10, %36 ]
  %43 = phi double [ %47, %40 ], [ 1.000000e+10, %36 ]
  %44 = getelementptr inbounds nuw double, ptr %1, i64 %41
  %45 = load double, ptr %44, align 8, !tbaa !11
  %46 = fcmp ole double %45, %43
  %47 = select i1 %46, double %45, double %43
  %48 = fcmp ogt double %45, %42
  %49 = select i1 %48, double %45, double %42
  %50 = add nuw nsw i64 %41, 2
  %51 = icmp samesign ult i64 %41, 65535
  br i1 %51, label %40, label %52, !llvm.loop !20

52:                                               ; preds = %40
  %53 = fsub double %49, %47
  %54 = fdiv double 2.550000e+02, %53
  br label %55

55:                                               ; preds = %52, %106
  %56 = phi i64 [ 0, %52 ], [ %108, %106 ]
  %57 = phi i32 [ 0, %52 ], [ %107, %106 ]
  %58 = trunc nuw nsw i64 %56 to i32
  %59 = and i32 %58, 15
  %60 = icmp eq i32 %59, 8
  %61 = shl nsw i64 %56, 12
  %62 = getelementptr inbounds nuw i8, ptr %1, i64 %61
  br i1 %60, label %63, label %83

63:                                               ; preds = %55, %79
  %64 = phi i64 [ %81, %79 ], [ 0, %55 ]
  %65 = phi i32 [ %80, %79 ], [ %57, %55 ]
  %66 = shl nuw nsw i64 %64, 4
  %67 = getelementptr inbounds nuw i8, ptr %62, i64 %66
  %68 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %69 = load double, ptr %68, align 8, !tbaa !11
  %70 = fsub double %69, %47
  %71 = fmul double %54, %70
  %72 = fptosi double %71 to i32
  %73 = icmp eq i32 %72, 255
  br i1 %73, label %79, label %74

74:                                               ; preds = %63
  %75 = add nsw i32 %65, 1
  %76 = load ptr, ptr @stderr, align 8, !tbaa !6
  %77 = trunc nuw nsw i64 %64 to i32
  %78 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %76, ptr noundef nonnull @.str.1, i32 noundef %58, i32 noundef %77, i32 noundef 255, i32 noundef %72) #11
  br label %79

79:                                               ; preds = %63, %74
  %80 = phi i32 [ %75, %74 ], [ %65, %63 ]
  %81 = add nuw nsw i64 %64, 1
  %82 = icmp eq i64 %81, 256
  br i1 %82, label %106, label %63, !llvm.loop !21

83:                                               ; preds = %55, %102
  %84 = phi i64 [ %104, %102 ], [ 0, %55 ]
  %85 = phi i32 [ %103, %102 ], [ %57, %55 ]
  %86 = shl nuw nsw i64 %84, 4
  %87 = getelementptr inbounds nuw i8, ptr %62, i64 %86
  %88 = getelementptr inbounds nuw i8, ptr %87, i64 8
  %89 = load double, ptr %88, align 8, !tbaa !11
  %90 = fsub double %89, %47
  %91 = fmul double %54, %90
  %92 = fptosi double %91 to i32
  %93 = trunc nuw nsw i64 %84 to i32
  %94 = and i32 %93, 15
  %95 = icmp eq i32 %94, 8
  %96 = select i1 %95, i32 255, i32 0
  %97 = icmp eq i32 %96, %92
  br i1 %97, label %102, label %98

98:                                               ; preds = %83
  %99 = add nsw i32 %85, 1
  %100 = load ptr, ptr @stderr, align 8, !tbaa !6
  %101 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %100, ptr noundef nonnull @.str.1, i32 noundef %58, i32 noundef %93, i32 noundef %96, i32 noundef %92) #11
  br label %102

102:                                              ; preds = %83, %98
  %103 = phi i32 [ %99, %98 ], [ %85, %83 ]
  %104 = add nuw nsw i64 %84, 1
  %105 = icmp eq i64 %104, 256
  br i1 %105, label %106, label %83, !llvm.loop !21

106:                                              ; preds = %102, %79
  %107 = phi i32 [ %80, %79 ], [ %103, %102 ]
  %108 = add nuw nsw i64 %56, 1
  %109 = icmp eq i64 %108, 256
  br i1 %109, label %110, label %55, !llvm.loop !22

110:                                              ; preds = %106
  %111 = icmp eq i32 %107, 0
  %112 = load ptr, ptr @stderr, align 8, !tbaa !6
  br i1 %111, label %113, label %115

113:                                              ; preds = %110
  %114 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %112, ptr noundef nonnull @.str.2, i32 noundef 63) #11
  br label %117

115:                                              ; preds = %110
  %116 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %112, ptr noundef nonnull @.str.3, i32 noundef 63, i32 noundef %107) #11
  br label %117

117:                                              ; preds = %115, %113
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #2

; Function Attrs: nofree norecurse nounwind memory(read, argmem: readwrite, inaccessiblemem: none, errnomem: readwrite) uwtable
define internal fastcc void @fourn(ptr noundef nonnull captures(none) %0, i32 noundef range(i32 -1, 2) %1) unnamed_addr #3 {
  %3 = load i1, ptr @main.nsize.0, align 4
  %4 = select i1 %3, i32 256, i32 0
  %5 = load i1, ptr @main.nsize.1, align 4
  %6 = select i1 %5, i32 256, i32 0
  %7 = mul nuw nsw i32 %6, %4
  %8 = sitofp i32 %1 to double
  %9 = fmul double %8, 0x401921FB54442D1C
  %10 = shl nuw nsw i32 %6, 1
  %11 = mul nuw nsw i32 %10, %4
  br i1 %5, label %12, label %123

12:                                               ; preds = %2
  %13 = zext nneg i32 %10 to i64
  %14 = zext nneg i32 %11 to i64
  br label %17

15:                                               ; preds = %49
  %16 = zext nneg i32 %11 to i64
  br label %55

17:                                               ; preds = %12, %49
  %18 = phi i64 [ 1, %12 ], [ %51, %49 ]
  %19 = phi i32 [ 1, %12 ], [ %50, %49 ]
  %20 = trunc i64 %18 to i32
  %21 = icmp sgt i32 %19, %20
  br i1 %21, label %22, label %40

22:                                               ; preds = %17
  %23 = sub i32 %19, %20
  %24 = icmp samesign ugt i64 %18, %14
  br i1 %24, label %40, label %25

25:                                               ; preds = %22, %25
  %26 = phi i64 [ %38, %25 ], [ %18, %22 ]
  %27 = trunc nsw i64 %26 to i32
  %28 = add i32 %23, %27
  %29 = getelementptr inbounds nuw double, ptr %0, i64 %26
  %30 = load double, ptr %29, align 8, !tbaa !11
  %31 = sext i32 %28 to i64
  %32 = getelementptr inbounds double, ptr %0, i64 %31
  %33 = load double, ptr %32, align 8, !tbaa !11
  store double %33, ptr %29, align 8, !tbaa !11
  store double %30, ptr %32, align 8, !tbaa !11
  %34 = getelementptr i8, ptr %29, i64 8
  %35 = load double, ptr %34, align 8, !tbaa !11
  %36 = getelementptr i8, ptr %32, i64 8
  %37 = load double, ptr %36, align 8, !tbaa !11
  store double %37, ptr %34, align 8, !tbaa !11
  store double %35, ptr %36, align 8, !tbaa !11
  %38 = add nuw nsw i64 %26, %13
  %39 = icmp samesign ugt i64 %38, %14
  br i1 %39, label %40, label %25, !llvm.loop !23

40:                                               ; preds = %25, %22, %17
  br label %41

41:                                               ; preds = %40, %41
  %42 = phi i32 [ %44, %41 ], [ %10, %40 ]
  %43 = phi i32 [ %48, %41 ], [ %19, %40 ]
  %44 = lshr i32 %42, 1
  %45 = icmp samesign ugt i32 %42, 3
  %46 = icmp sgt i32 %43, %44
  %47 = select i1 %45, i1 %46, i1 false
  %48 = sub nsw i32 %43, %44
  br i1 %47, label %41, label %49, !llvm.loop !24

49:                                               ; preds = %41
  %50 = add nsw i32 %43, %44
  %51 = add nuw nsw i64 %18, 2
  %52 = icmp samesign ugt i64 %51, %13
  br i1 %52, label %15, label %17, !llvm.loop !25

53:                                               ; preds = %111, %55
  %54 = icmp slt i32 %57, %10
  br i1 %54, label %55, label %119, !llvm.loop !26

55:                                               ; preds = %15, %53
  %56 = phi i32 [ %57, %53 ], [ 2, %15 ]
  %57 = shl i32 %56, 1
  %58 = ashr exact i32 %57, 1
  %59 = sitofp i32 %58 to double
  %60 = fdiv double %9, %59
  %61 = fmul double %60, 5.000000e-01
  %62 = tail call double @sin(double noundef %61) #12, !tbaa !27
  %63 = fmul double %62, -2.000000e+00
  %64 = fmul double %62, %63
  %65 = tail call double @sin(double noundef %60) #12, !tbaa !27
  %66 = icmp slt i32 %56, 1
  br i1 %66, label %53, label %67

67:                                               ; preds = %55
  %68 = fneg double %65
  %69 = sext i32 %57 to i64
  %70 = zext nneg i32 %56 to i64
  %71 = getelementptr double, ptr %0, i64 %70
  br label %72

72:                                               ; preds = %67, %111
  %73 = phi i64 [ 1, %67 ], [ %77, %111 ]
  %74 = phi double [ 1.000000e+00, %67 ], [ %114, %111 ]
  %75 = phi double [ 0.000000e+00, %67 ], [ %117, %111 ]
  %76 = trunc i64 %73 to i32
  %77 = add nuw nsw i64 %73, 2
  %78 = trunc i64 %77 to i32
  %79 = add nsw i32 %78, -2
  %80 = icmp slt i32 %79, %76
  %81 = icmp samesign ugt i64 %73, %16
  %82 = select i1 %80, i1 true, i1 %81
  br i1 %82, label %111, label %83

83:                                               ; preds = %72
  %84 = insertelement <2 x double> poison, double %75, i64 0
  %85 = shufflevector <2 x double> %84, <2 x double> poison, <2 x i32> zeroinitializer
  %86 = insertelement <2 x double> poison, double %74, i64 0
  %87 = shufflevector <2 x double> %86, <2 x double> poison, <2 x i32> zeroinitializer
  br label %88

88:                                               ; preds = %83, %88
  %89 = phi i64 [ %109, %88 ], [ %73, %83 ]
  %90 = getelementptr double, ptr %71, i64 %89
  %91 = getelementptr i8, ptr %90, i64 8
  %92 = getelementptr inbounds double, ptr %0, i64 %89
  %93 = load double, ptr %92, align 8, !tbaa !11
  %94 = getelementptr i8, ptr %92, i64 8
  %95 = load double, ptr %91, align 8, !tbaa !11
  %96 = load <2 x double>, ptr %90, align 8, !tbaa !11
  %97 = fneg double %95
  %98 = shufflevector <2 x double> %96, <2 x double> poison, <2 x i32> <i32 poison, i32 0>
  %99 = insertelement <2 x double> %98, double %97, i64 0
  %100 = fmul <2 x double> %85, %99
  %101 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %87, <2 x double> %96, <2 x double> %100)
  %102 = extractelement <2 x double> %101, i64 0
  %103 = fsub double %93, %102
  store double %103, ptr %90, align 8, !tbaa !11
  %104 = load double, ptr %94, align 8, !tbaa !11
  %105 = load <2 x double>, ptr %92, align 8, !tbaa !11
  %106 = extractelement <2 x double> %101, i64 1
  %107 = fsub double %104, %106
  store double %107, ptr %91, align 8, !tbaa !11
  %108 = fadd <2 x double> %101, %105
  store <2 x double> %108, ptr %92, align 8, !tbaa !11
  %109 = add nsw i64 %89, %69
  %110 = icmp sgt i64 %109, %16
  br i1 %110, label %111, label %88, !llvm.loop !29

111:                                              ; preds = %88, %72
  %112 = fmul double %75, %68
  %113 = tail call double @llvm.fmuladd.f64(double %74, double %64, double %112)
  %114 = fadd double %74, %113
  %115 = fmul double %65, %74
  %116 = tail call double @llvm.fmuladd.f64(double %75, double %64, double %115)
  %117 = fadd double %75, %116
  %118 = icmp slt i32 %56, %78
  br i1 %118, label %53, label %72, !llvm.loop !30

119:                                              ; preds = %53
  %120 = load i1, ptr @main.nsize.0, align 4
  %121 = select i1 %120, i32 256, i32 0
  %122 = mul nuw nsw i32 %121, %6
  br label %123

123:                                              ; preds = %2, %119
  %124 = phi i32 [ %122, %119 ], [ %7, %2 ]
  %125 = phi i32 [ %121, %119 ], [ %4, %2 ]
  %126 = sdiv i32 %7, %124
  %127 = shl nuw nsw i32 %6, 1
  %128 = mul nuw nsw i32 %125, %127
  %129 = mul nsw i32 %128, %126
  %130 = icmp eq i32 %128, 0
  br i1 %130, label %183, label %131

131:                                              ; preds = %123
  %132 = add nsw i32 %127, -2
  %133 = zext nneg i32 %127 to i64
  %134 = zext nneg i32 %128 to i64
  %135 = sext i32 %129 to i64
  br label %136

136:                                              ; preds = %178, %131
  %137 = phi i64 [ 1, %131 ], [ %180, %178 ]
  %138 = phi i32 [ 1, %131 ], [ %179, %178 ]
  %139 = trunc i64 %137 to i32
  %140 = icmp sgt i32 %138, %139
  br i1 %140, label %141, label %169

141:                                              ; preds = %136
  %142 = add i32 %132, %139
  %143 = icmp slt i32 %142, %139
  br i1 %143, label %169, label %144

144:                                              ; preds = %141
  %145 = sub i32 %138, %139
  br label %146

146:                                              ; preds = %165, %144
  %147 = phi i64 [ %137, %144 ], [ %166, %165 ]
  %148 = trunc i64 %147 to i32
  %149 = icmp slt i32 %129, %148
  br i1 %149, label %165, label %150

150:                                              ; preds = %146, %150
  %151 = phi i64 [ %163, %150 ], [ %147, %146 ]
  %152 = trunc nsw i64 %151 to i32
  %153 = add i32 %145, %152
  %154 = getelementptr inbounds double, ptr %0, i64 %151
  %155 = load double, ptr %154, align 8, !tbaa !11
  %156 = sext i32 %153 to i64
  %157 = getelementptr inbounds double, ptr %0, i64 %156
  %158 = load double, ptr %157, align 8, !tbaa !11
  store double %158, ptr %154, align 8, !tbaa !11
  store double %155, ptr %157, align 8, !tbaa !11
  %159 = getelementptr i8, ptr %154, i64 8
  %160 = load double, ptr %159, align 8, !tbaa !11
  %161 = getelementptr i8, ptr %157, i64 8
  %162 = load double, ptr %161, align 8, !tbaa !11
  store double %162, ptr %159, align 8, !tbaa !11
  store double %160, ptr %161, align 8, !tbaa !11
  %163 = add nsw i64 %151, %134
  %164 = icmp sgt i64 %163, %135
  br i1 %164, label %165, label %150, !llvm.loop !23

165:                                              ; preds = %150, %146
  %166 = add nsw i64 %147, 2
  %167 = trunc i64 %166 to i32
  %168 = icmp slt i32 %142, %167
  br i1 %168, label %169, label %146, !llvm.loop !31

169:                                              ; preds = %165, %141, %136
  br label %170

170:                                              ; preds = %169, %170
  %171 = phi i32 [ %173, %170 ], [ %128, %169 ]
  %172 = phi i32 [ %177, %170 ], [ %138, %169 ]
  %173 = lshr i32 %171, 1
  %174 = icmp samesign uge i32 %173, %127
  %175 = icmp sgt i32 %172, %173
  %176 = select i1 %174, i1 %175, i1 false
  %177 = sub nsw i32 %172, %173
  br i1 %176, label %170, label %178, !llvm.loop !24

178:                                              ; preds = %170
  %179 = add nsw i32 %172, %173
  %180 = add i64 %137, %133
  %181 = trunc i64 %180 to i32
  %182 = icmp slt i32 %128, %181
  br i1 %182, label %183, label %136, !llvm.loop !25

183:                                              ; preds = %178, %123
  %184 = icmp samesign ult i32 %127, %128
  br i1 %184, label %185, label %260

185:                                              ; preds = %183
  %186 = zext nneg i32 %127 to i64
  %187 = sext i32 %129 to i64
  br label %188

188:                                              ; preds = %258, %185
  %189 = phi i32 [ %190, %258 ], [ %127, %185 ]
  %190 = shl i32 %189, 1
  %191 = sdiv i32 %190, %127
  %192 = sitofp i32 %191 to double
  %193 = fdiv double %9, %192
  %194 = fmul double %193, 5.000000e-01
  %195 = tail call double @sin(double noundef %194) #12, !tbaa !27
  %196 = fmul double %195, -2.000000e+00
  %197 = fmul double %195, %196
  %198 = tail call double @sin(double noundef %193) #12, !tbaa !27
  %199 = icmp slt i32 %189, 1
  br i1 %199, label %258, label %200

200:                                              ; preds = %188
  %201 = fneg double %198
  %202 = sext i32 %190 to i64
  %203 = zext nneg i32 %189 to i64
  %204 = getelementptr double, ptr %0, i64 %203
  br label %205

205:                                              ; preds = %250, %200
  %206 = phi i64 [ 1, %200 ], [ %210, %250 ]
  %207 = phi double [ 1.000000e+00, %200 ], [ %253, %250 ]
  %208 = phi double [ 0.000000e+00, %200 ], [ %256, %250 ]
  %209 = trunc i64 %206 to i32
  %210 = add i64 %206, %186
  %211 = trunc i64 %210 to i32
  %212 = add nsw i32 %211, -2
  %213 = icmp slt i32 %212, %209
  br i1 %213, label %250, label %214

214:                                              ; preds = %205
  %215 = insertelement <2 x double> poison, double %208, i64 0
  %216 = shufflevector <2 x double> %215, <2 x double> poison, <2 x i32> zeroinitializer
  %217 = insertelement <2 x double> poison, double %207, i64 0
  %218 = shufflevector <2 x double> %217, <2 x double> poison, <2 x i32> zeroinitializer
  br label %219

219:                                              ; preds = %214, %246
  %220 = phi i64 [ %247, %246 ], [ %206, %214 ]
  %221 = trunc i64 %220 to i32
  %222 = icmp slt i32 %129, %221
  br i1 %222, label %246, label %223

223:                                              ; preds = %219, %223
  %224 = phi i64 [ %244, %223 ], [ %220, %219 ]
  %225 = getelementptr double, ptr %204, i64 %224
  %226 = getelementptr i8, ptr %225, i64 8
  %227 = getelementptr inbounds double, ptr %0, i64 %224
  %228 = load double, ptr %227, align 8, !tbaa !11
  %229 = getelementptr i8, ptr %227, i64 8
  %230 = load double, ptr %226, align 8, !tbaa !11
  %231 = load <2 x double>, ptr %225, align 8, !tbaa !11
  %232 = fneg double %230
  %233 = shufflevector <2 x double> %231, <2 x double> poison, <2 x i32> <i32 poison, i32 0>
  %234 = insertelement <2 x double> %233, double %232, i64 0
  %235 = fmul <2 x double> %216, %234
  %236 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %218, <2 x double> %231, <2 x double> %235)
  %237 = extractelement <2 x double> %236, i64 0
  %238 = fsub double %228, %237
  store double %238, ptr %225, align 8, !tbaa !11
  %239 = load double, ptr %229, align 8, !tbaa !11
  %240 = load <2 x double>, ptr %227, align 8, !tbaa !11
  %241 = extractelement <2 x double> %236, i64 1
  %242 = fsub double %239, %241
  store double %242, ptr %226, align 8, !tbaa !11
  %243 = fadd <2 x double> %236, %240
  store <2 x double> %243, ptr %227, align 8, !tbaa !11
  %244 = add nsw i64 %224, %202
  %245 = icmp sgt i64 %244, %187
  br i1 %245, label %246, label %223, !llvm.loop !29

246:                                              ; preds = %223, %219
  %247 = add nsw i64 %220, 2
  %248 = trunc i64 %247 to i32
  %249 = icmp slt i32 %212, %248
  br i1 %249, label %250, label %219, !llvm.loop !32

250:                                              ; preds = %246, %205
  %251 = fmul double %208, %201
  %252 = tail call double @llvm.fmuladd.f64(double %207, double %197, double %251)
  %253 = fadd double %207, %252
  %254 = fmul double %198, %207
  %255 = tail call double @llvm.fmuladd.f64(double %208, double %197, double %254)
  %256 = fadd double %208, %255
  %257 = icmp slt i32 %189, %211
  br i1 %257, label %258, label %205, !llvm.loop !30

258:                                              ; preds = %250, %188
  %259 = icmp slt i32 %190, %128
  br i1 %259, label %188, label %260, !llvm.loop !26

260:                                              ; preds = %258, %183
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @sin(double noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #5

; Function Attrs: nofree nounwind
declare noundef i64 @fwrite(ptr noundef readonly captures(none), i64 noundef, i64 noundef, ptr noundef captures(none)) local_unnamed_addr #6

; Function Attrs: nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) local_unnamed_addr #7

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #8

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree norecurse nounwind memory(read, argmem: readwrite, inaccessiblemem: none, errnomem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nofree nounwind }
attributes #7 = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
attributes #8 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { cold }
attributes #10 = { cold noreturn nounwind }
attributes #11 = { cold nounwind }
attributes #12 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 _ZTS8_IO_FILE", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !9, i64 0}
!13 = distinct !{!13, !14, !15, !16}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !14}
!18 = distinct !{!18, !14}
!19 = distinct !{!19, !14}
!20 = distinct !{!20, !14}
!21 = distinct !{!21, !14}
!22 = distinct !{!22, !14}
!23 = distinct !{!23, !14}
!24 = distinct !{!24, !14}
!25 = distinct !{!25, !14}
!26 = distinct !{!26, !14}
!27 = !{!28, !28, i64 0}
!28 = !{!"int", !9, i64 0}
!29 = distinct !{!29, !14}
!30 = distinct !{!30, !14}
!31 = distinct !{!31, !14}
!32 = distinct !{!32, !14}
