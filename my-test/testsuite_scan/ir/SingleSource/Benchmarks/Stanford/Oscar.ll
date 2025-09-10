; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/Oscar.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/Oscar.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.complex = type { float, float }
%struct.element = type { i32, i32 }

@seed = dso_local local_unnamed_addr global i64 0, align 8
@.str.1 = private unnamed_addr constant [15 x i8] c"  %15.3f%15.3f\00", align 1
@e = dso_local local_unnamed_addr global [130 x %struct.complex] zeroinitializer, align 4
@zr = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@zi = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@z = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 4
@w = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 4
@value = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@fixed = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@floated = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@permarray = dso_local local_unnamed_addr global [11 x i32] zeroinitializer, align 4
@pctr = dso_local local_unnamed_addr global i32 0, align 4
@tree = dso_local local_unnamed_addr global ptr null, align 8
@stack = dso_local local_unnamed_addr global [4 x i32] zeroinitializer, align 4
@cellspace = dso_local local_unnamed_addr global [19 x %struct.element] zeroinitializer, align 4
@freelist = dso_local local_unnamed_addr global i32 0, align 4
@movesdone = dso_local local_unnamed_addr global i32 0, align 4
@ima = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@imb = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@imr = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@rma = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@rmb = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@rmr = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@piececount = dso_local local_unnamed_addr global [4 x i32] zeroinitializer, align 4
@class = dso_local local_unnamed_addr global [13 x i32] zeroinitializer, align 4
@piecemax = dso_local local_unnamed_addr global [13 x i32] zeroinitializer, align 4
@puzzl = dso_local local_unnamed_addr global [512 x i32] zeroinitializer, align 4
@p = dso_local local_unnamed_addr global [13 x [512 x i32]] zeroinitializer, align 4
@n = dso_local local_unnamed_addr global i32 0, align 4
@kount = dso_local local_unnamed_addr global i32 0, align 4
@sortlist = dso_local local_unnamed_addr global [5001 x i32] zeroinitializer, align 4
@biggest = dso_local local_unnamed_addr global i32 0, align 4
@littlest = dso_local local_unnamed_addr global i32 0, align 4
@top = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @Initrand() local_unnamed_addr #0 {
  store i64 74755, ptr @seed, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 65536) i32 @Rand() local_unnamed_addr #1 {
  %1 = load i64, ptr @seed, align 8, !tbaa !6
  %2 = mul nsw i64 %1, 1309
  %3 = add nsw i64 %2, 13849
  %4 = and i64 %3, 65535
  store i64 %4, ptr @seed, align 8, !tbaa !6
  %5 = trunc nuw nsw i64 %4 to i32
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local float @Cos(float noundef %0) local_unnamed_addr #2 {
  %2 = fmul float %0, %0
  %3 = fmul float %2, 5.000000e-01
  %4 = fsub float 1.000000e+00, %3
  %5 = fmul float %0, %2
  %6 = fmul float %0, %5
  %7 = fdiv float %6, 2.400000e+01
  %8 = fadd float %4, %7
  %9 = fmul float %0, %6
  %10 = fmul float %0, %9
  %11 = fdiv float %10, 7.200000e+02
  %12 = fsub float %8, %11
  %13 = fmul float %0, %10
  %14 = fmul float %0, %13
  %15 = fdiv float %14, 4.032000e+04
  %16 = fadd float %12, %15
  %17 = fmul float %0, %14
  %18 = fmul float %0, %17
  %19 = fdiv float %18, 3.628800e+06
  %20 = fsub float %16, %19
  ret float %20
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @Min0(i32 noundef %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = tail call i32 @llvm.smin.i32(i32 %0, i32 %1)
  ret i32 %3
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @Printcomplex(ptr noundef readonly captures(none) %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #4 {
  %5 = tail call i32 @putchar(i32 10)
  br label %6

6:                                                ; preds = %6, %4
  %7 = phi i32 [ %1, %4 ], [ %26, %6 ]
  %8 = sext i32 %7 to i64
  %9 = getelementptr inbounds %struct.complex, ptr %0, i64 %8
  %10 = load float, ptr %9, align 4, !tbaa !10
  %11 = fpext float %10 to double
  %12 = getelementptr inbounds nuw i8, ptr %9, i64 4
  %13 = load float, ptr %12, align 4, !tbaa !13
  %14 = fpext float %13 to double
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %11, double noundef %14)
  %16 = add nsw i32 %7, %3
  %17 = sext i32 %16 to i64
  %18 = getelementptr inbounds %struct.complex, ptr %0, i64 %17
  %19 = load float, ptr %18, align 4, !tbaa !10
  %20 = fpext float %19 to double
  %21 = getelementptr inbounds nuw i8, ptr %18, i64 4
  %22 = load float, ptr %21, align 4, !tbaa !13
  %23 = fpext float %22 to double
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %20, double noundef %23)
  %25 = tail call i32 @putchar(i32 10)
  %26 = add nsw i32 %16, %3
  %27 = icmp sgt i32 %26, %2
  br i1 %27, label %28, label %6, !llvm.loop !14

28:                                               ; preds = %6
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #5

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @Uniform11(ptr noundef captures(none) %0, ptr noundef writeonly captures(none) initializes((0, 4)) %1) local_unnamed_addr #6 {
  %3 = load i32, ptr %0, align 4, !tbaa !16
  %4 = mul nsw i32 %3, 4855
  %5 = add nsw i32 %4, 1731
  %6 = and i32 %5, 8191
  store i32 %6, ptr %0, align 4, !tbaa !16
  %7 = uitofp nneg i32 %6 to float
  %8 = fmul float %7, 0x3F20000000000000
  store float %8, ptr %1, align 4, !tbaa !18
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @Exptab(i32 noundef %0, ptr noundef captures(none) %1) local_unnamed_addr #7 {
  %3 = alloca [26 x float], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #12
  br label %4

4:                                                ; preds = %2, %4
  %5 = phi i64 [ 1, %2 ], [ %31, %4 ]
  %6 = phi float [ 4.000000e+00, %2 ], [ %30, %4 ]
  %7 = fdiv float 0x400921FB60000000, %6
  %8 = fmul float %7, %7
  %9 = fmul float %8, 5.000000e-01
  %10 = fsub float 1.000000e+00, %9
  %11 = fmul float %7, %8
  %12 = fmul float %7, %11
  %13 = fdiv float %12, 2.400000e+01
  %14 = fadd float %10, %13
  %15 = fmul float %7, %12
  %16 = fmul float %7, %15
  %17 = fdiv float %16, 7.200000e+02
  %18 = fsub float %14, %17
  %19 = fmul float %7, %16
  %20 = fmul float %7, %19
  %21 = fdiv float %20, 4.032000e+04
  %22 = fadd float %18, %21
  %23 = fmul float %7, %20
  %24 = fmul float %7, %23
  %25 = fdiv float %24, 3.628800e+06
  %26 = fsub float %22, %25
  %27 = fmul float %26, 2.000000e+00
  %28 = fdiv float 1.000000e+00, %27
  %29 = getelementptr inbounds nuw float, ptr %3, i64 %5
  store float %28, ptr %29, align 4, !tbaa !18
  %30 = fadd float %6, %6
  %31 = add nuw nsw i64 %5, 1
  %32 = icmp eq i64 %31, 26
  br i1 %32, label %33, label %4, !llvm.loop !19

33:                                               ; preds = %4
  %34 = sdiv i32 %0, 2
  %35 = sdiv i32 %0, 4
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store <2 x float> <float 1.000000e+00, float 0.000000e+00>, ptr %36, align 4, !tbaa !18
  %37 = sext i32 %35 to i64
  %38 = getelementptr %struct.complex, ptr %1, i64 %37
  %39 = getelementptr i8, ptr %38, i64 8
  store <2 x float> <float 0.000000e+00, float 1.000000e+00>, ptr %39, align 4, !tbaa !18
  %40 = sext i32 %34 to i64
  %41 = getelementptr %struct.complex, ptr %1, i64 %40
  %42 = getelementptr i8, ptr %41, i64 8
  store <2 x float> <float -1.000000e+00, float 0.000000e+00>, ptr %42, align 4, !tbaa !18
  br label %43

43:                                               ; preds = %70, %33
  %44 = phi i32 [ 1, %33 ], [ %72, %70 ]
  %45 = phi i32 [ %35, %33 ], [ %46, %70 ]
  %46 = sdiv i32 %45, 2
  %47 = sext i32 %44 to i64
  %48 = getelementptr inbounds float, ptr %3, i64 %47
  %49 = load float, ptr %48, align 4, !tbaa !18
  %50 = sext i32 %46 to i64
  %51 = sext i32 %45 to i64
  %52 = getelementptr %struct.complex, ptr %1, i64 %50
  %53 = insertelement <2 x float> poison, float %49, i64 0
  %54 = shufflevector <2 x float> %53, <2 x float> poison, <2 x i32> zeroinitializer
  br label %55

55:                                               ; preds = %55, %43
  %56 = phi i64 [ %68, %55 ], [ %50, %43 ]
  %57 = getelementptr %struct.complex, ptr %52, i64 %56
  %58 = getelementptr i8, ptr %57, i64 8
  %59 = sub nsw i64 %56, %50
  %60 = getelementptr %struct.complex, ptr %1, i64 %59
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = getelementptr %struct.complex, ptr %1, i64 %56
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = load <2 x float>, ptr %58, align 4, !tbaa !18
  %65 = load <2 x float>, ptr %61, align 4, !tbaa !18
  %66 = fadd <2 x float> %64, %65
  %67 = fmul <2 x float> %54, %66
  store <2 x float> %67, ptr %63, align 4, !tbaa !18
  %68 = add nsw i64 %56, %51
  %69 = icmp sgt i64 %68, %40
  br i1 %69, label %70, label %55, !llvm.loop !20

70:                                               ; preds = %55
  %71 = tail call i32 @llvm.smin.i32(i32 %44, i32 24)
  %72 = add nsw i32 %71, 1
  %73 = icmp sgt i32 %45, 3
  br i1 %73, label %43, label %74, !llvm.loop !21

74:                                               ; preds = %70
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #12
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @Fft(i32 noundef %0, ptr noundef captures(none) %1, ptr noundef captures(none) %2, ptr noundef readonly captures(none) %3, float noundef %4) local_unnamed_addr #7 {
  %6 = ptrtoint ptr %2 to i64
  %7 = ptrtoint ptr %1 to i64
  %8 = sdiv i32 %0, 2
  %9 = sext i32 %8 to i64
  %10 = tail call i32 @llvm.smax.i32(i32 %0, i32 1)
  %11 = add nuw i32 %10, 1
  %12 = getelementptr %struct.complex, ptr %1, i64 %9
  %13 = zext i32 %11 to i64
  %14 = sub i64 %7, %6
  %15 = zext nneg i32 %10 to i64
  %16 = getelementptr i8, ptr %2, i64 4
  %17 = getelementptr i8, ptr %1, i64 4
  %18 = shl nsw i64 %9, 3
  %19 = getelementptr i8, ptr %1, i64 %18
  %20 = getelementptr i8, ptr %1, i64 4
  %21 = getelementptr i8, ptr %3, i64 8
  %22 = getelementptr i8, ptr %3, i64 12
  %23 = getelementptr i8, ptr %2, i64 4
  %24 = getelementptr i8, ptr %2, i64 8
  %25 = getelementptr i8, ptr %1, i64 4
  %26 = getelementptr i8, ptr %1, i64 8
  %27 = getelementptr i8, ptr %1, i64 %18
  %28 = getelementptr i8, ptr %27, i64 4
  %29 = getelementptr i8, ptr %1, i64 8
  %30 = getelementptr i8, ptr %3, i64 16
  %31 = icmp slt i32 %0, 4
  %32 = icmp ult i64 %14, 32
  %33 = or i1 %31, %32
  %34 = and i64 %15, 2147483644
  %35 = or disjoint i64 %34, 1
  %36 = icmp eq i64 %34, %15
  br label %37

37:                                               ; preds = %257, %5
  %38 = phi i32 [ 1, %5 ], [ %258, %257 ]
  %39 = sext i32 %38 to i64
  %40 = shl nsw i64 %39, 3
  %41 = shl nsw i64 %39, 3
  %42 = getelementptr i8, ptr %2, i64 %40
  %43 = getelementptr i8, ptr %42, i64 8
  %44 = getelementptr i8, ptr %2, i64 %40
  %45 = getelementptr i8, ptr %44, i64 4
  %46 = getelementptr i8, ptr %2, i64 %40
  br label %47

47:                                               ; preds = %230, %37
  %48 = phi i64 [ %234, %230 ], [ 0, %37 ]
  %49 = phi i32 [ %233, %230 ], [ 0, %37 ]
  %50 = phi i64 [ %231, %230 ], [ %39, %37 ]
  %51 = phi i64 [ %50, %230 ], [ 0, %37 ]
  %52 = phi i64 [ %228, %230 ], [ 1, %37 ]
  %53 = shl i64 %51, 32
  %54 = ashr exact i64 %53, 32
  %55 = getelementptr %struct.complex, ptr %3, i64 %54
  %56 = getelementptr i8, ptr %55, i64 8
  %57 = getelementptr i8, ptr %55, i64 12
  %58 = shl i64 %52, 32
  %59 = ashr exact i64 %58, 32
  %60 = getelementptr %struct.complex, ptr %2, i64 %54
  %61 = getelementptr %struct.complex, ptr %2, i64 %50
  %62 = tail call i64 @llvm.smax.i64(i64 %50, i64 %59)
  %63 = add i64 %62, 1
  %64 = sub i64 %63, %59
  %65 = icmp ult i64 %64, 9
  br i1 %65, label %197, label %66

66:                                               ; preds = %47
  %67 = mul i32 %38, %49
  %68 = sext i32 %67 to i64
  %69 = shl nsw i64 %68, 3
  %70 = getelementptr i8, ptr %30, i64 %69
  %71 = mul i64 %41, %48
  %72 = getelementptr i8, ptr %43, i64 %71
  %73 = getelementptr i8, ptr %22, i64 %69
  %74 = getelementptr i8, ptr %21, i64 %69
  %75 = getelementptr i8, ptr %45, i64 %71
  %76 = getelementptr i8, ptr %46, i64 %71
  %77 = ashr exact i64 %58, 29
  %78 = add nsw i64 %69, %77
  %79 = getelementptr i8, ptr %2, i64 %78
  %80 = tail call i64 @llvm.smax.i64(i64 %50, i64 %59)
  %81 = add i64 %80, %68
  %82 = shl i64 %81, 3
  %83 = getelementptr i8, ptr %16, i64 %82
  %84 = getelementptr i8, ptr %76, i64 %77
  %85 = shl i64 %80, 3
  %86 = getelementptr i8, ptr %75, i64 %85
  %87 = getelementptr i8, ptr %1, i64 %77
  %88 = getelementptr i8, ptr %17, i64 %85
  %89 = getelementptr i8, ptr %19, i64 %77
  %90 = add i64 %80, %9
  %91 = shl i64 %90, 3
  %92 = getelementptr i8, ptr %20, i64 %91
  %93 = getelementptr i8, ptr %23, i64 %78
  %94 = getelementptr i8, ptr %24, i64 %82
  %95 = getelementptr i8, ptr %75, i64 %77
  %96 = getelementptr i8, ptr %72, i64 %85
  %97 = getelementptr i8, ptr %25, i64 %77
  %98 = getelementptr i8, ptr %26, i64 %85
  %99 = getelementptr i8, ptr %28, i64 %77
  %100 = getelementptr i8, ptr %29, i64 %91
  %101 = icmp ult ptr %79, %86
  %102 = icmp ult ptr %84, %83
  %103 = and i1 %101, %102
  %104 = icmp ult ptr %79, %88
  %105 = icmp ult ptr %87, %83
  %106 = and i1 %104, %105
  %107 = or i1 %103, %106
  %108 = icmp ult ptr %79, %92
  %109 = icmp ult ptr %89, %83
  %110 = and i1 %108, %109
  %111 = or i1 %107, %110
  %112 = icmp ult ptr %79, %73
  %113 = icmp ult ptr %74, %83
  %114 = and i1 %112, %113
  %115 = or i1 %111, %114
  %116 = icmp ult ptr %84, %88
  %117 = icmp ult ptr %87, %86
  %118 = and i1 %116, %117
  %119 = or i1 %115, %118
  %120 = icmp ult ptr %84, %92
  %121 = icmp ult ptr %89, %86
  %122 = and i1 %120, %121
  %123 = or i1 %119, %122
  %124 = icmp ult ptr %84, %73
  %125 = icmp ult ptr %74, %86
  %126 = and i1 %124, %125
  %127 = or i1 %123, %126
  %128 = icmp ult ptr %93, %96
  %129 = icmp ult ptr %95, %94
  %130 = and i1 %128, %129
  %131 = or i1 %127, %130
  %132 = icmp ult ptr %93, %98
  %133 = icmp ult ptr %97, %94
  %134 = and i1 %132, %133
  %135 = or i1 %131, %134
  %136 = icmp ult ptr %93, %100
  %137 = icmp ult ptr %99, %94
  %138 = and i1 %136, %137
  %139 = or i1 %135, %138
  %140 = icmp ult ptr %93, %70
  %141 = icmp ult ptr %73, %94
  %142 = and i1 %140, %141
  %143 = or i1 %139, %142
  %144 = icmp ult ptr %95, %98
  %145 = icmp ult ptr %97, %96
  %146 = and i1 %144, %145
  %147 = or i1 %143, %146
  %148 = icmp ult ptr %95, %100
  %149 = icmp ult ptr %99, %96
  %150 = and i1 %148, %149
  %151 = or i1 %147, %150
  %152 = icmp ult ptr %95, %70
  %153 = icmp ult ptr %73, %96
  %154 = and i1 %152, %153
  %155 = or i1 %151, %154
  br i1 %155, label %197, label %156

156:                                              ; preds = %66
  %157 = and i64 %64, 3
  %158 = icmp eq i64 %157, 0
  %159 = select i1 %158, i64 4, i64 %157
  %160 = sub i64 %64, %159
  %161 = add i64 %59, %160
  br label %162

162:                                              ; preds = %162, %156
  %163 = phi i64 [ 0, %156 ], [ %195, %162 ]
  %164 = add i64 %59, %163
  %165 = getelementptr inbounds %struct.complex, ptr %1, i64 %164
  %166 = load <8 x float>, ptr %165, align 4, !tbaa !10, !alias.scope !22
  %167 = getelementptr %struct.complex, ptr %12, i64 %164
  %168 = load <8 x float>, ptr %167, align 4, !tbaa !10, !alias.scope !25
  %169 = getelementptr %struct.complex, ptr %60, i64 %164
  %170 = fadd <8 x float> %166, %168
  store <8 x float> %170, ptr %169, align 4, !tbaa !18
  %171 = load float, ptr %56, align 4, !tbaa !10, !alias.scope !27
  %172 = insertelement <4 x float> poison, float %171, i64 0
  %173 = shufflevector <4 x float> %172, <4 x float> poison, <4 x i32> zeroinitializer
  %174 = fsub <8 x float> %166, %168
  %175 = shufflevector <8 x float> %174, <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %176 = load float, ptr %57, align 4, !tbaa !13, !alias.scope !29
  %177 = insertelement <4 x float> poison, float %176, i64 0
  %178 = shufflevector <4 x float> %177, <4 x float> poison, <4 x i32> zeroinitializer
  %179 = load <8 x float>, ptr %165, align 4, !tbaa !18
  %180 = load <8 x float>, ptr %167, align 4, !tbaa !18
  %181 = fsub <8 x float> %179, %180
  %182 = shufflevector <8 x float> %181, <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %183 = fneg <4 x float> %182
  %184 = fmul <4 x float> %178, %183
  %185 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %173, <4 x float> %175, <4 x float> %184)
  %186 = getelementptr %struct.complex, ptr %61, i64 %164
  %187 = load float, ptr %56, align 4, !tbaa !10, !alias.scope !27
  %188 = insertelement <4 x float> poison, float %187, i64 0
  %189 = shufflevector <4 x float> %188, <4 x float> poison, <4 x i32> zeroinitializer
  %190 = fsub <8 x float> %179, %180
  %191 = shufflevector <8 x float> %190, <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %192 = fmul <4 x float> %178, %191
  %193 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %189, <4 x float> %182, <4 x float> %192)
  %194 = shufflevector <4 x float> %185, <4 x float> %193, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x float> %194, ptr %186, align 4, !tbaa !18
  %195 = add nuw i64 %163, 4
  %196 = icmp eq i64 %195, %160
  br i1 %196, label %197, label %162, !llvm.loop !31

197:                                              ; preds = %162, %66, %47
  %198 = phi i64 [ %59, %66 ], [ %59, %47 ], [ %161, %162 ]
  br label %199

199:                                              ; preds = %197, %199
  %200 = phi i64 [ %228, %199 ], [ %198, %197 ]
  %201 = getelementptr inbounds %struct.complex, ptr %1, i64 %200
  %202 = getelementptr %struct.complex, ptr %12, i64 %200
  %203 = getelementptr %struct.complex, ptr %60, i64 %200
  %204 = getelementptr inbounds nuw i8, ptr %201, i64 4
  %205 = getelementptr inbounds nuw i8, ptr %202, i64 4
  %206 = load <2 x float>, ptr %201, align 4, !tbaa !18
  %207 = load <2 x float>, ptr %202, align 4, !tbaa !18
  %208 = fadd <2 x float> %206, %207
  store <2 x float> %208, ptr %203, align 4, !tbaa !18
  %209 = load float, ptr %56, align 4, !tbaa !10
  %210 = load float, ptr %201, align 4, !tbaa !10
  %211 = load float, ptr %202, align 4, !tbaa !10
  %212 = fsub float %210, %211
  %213 = load float, ptr %57, align 4, !tbaa !13
  %214 = load float, ptr %204, align 4, !tbaa !13
  %215 = load float, ptr %205, align 4, !tbaa !13
  %216 = fsub float %214, %215
  %217 = fneg float %216
  %218 = fmul float %213, %217
  %219 = tail call float @llvm.fmuladd.f32(float %209, float %212, float %218)
  %220 = getelementptr %struct.complex, ptr %61, i64 %200
  store float %219, ptr %220, align 4, !tbaa !10
  %221 = load float, ptr %56, align 4, !tbaa !10
  %222 = load float, ptr %201, align 4, !tbaa !10
  %223 = load float, ptr %202, align 4, !tbaa !10
  %224 = fsub float %222, %223
  %225 = fmul float %213, %224
  %226 = tail call float @llvm.fmuladd.f32(float %221, float %216, float %225)
  %227 = getelementptr inbounds nuw i8, ptr %220, i64 4
  store float %226, ptr %227, align 4, !tbaa !13
  %228 = add nsw i64 %200, 1
  %229 = icmp slt i64 %200, %50
  br i1 %229, label %199, label %230, !llvm.loop !34

230:                                              ; preds = %199
  %231 = add nsw i64 %50, %39
  %232 = icmp sgt i64 %231, %9
  %233 = add i32 %49, 1
  %234 = add i64 %48, 1
  br i1 %232, label %235, label %47, !llvm.loop !35

235:                                              ; preds = %230
  br i1 %33, label %248, label %236

236:                                              ; preds = %235, %236
  %237 = phi i64 [ %245, %236 ], [ 0, %235 ]
  %238 = or disjoint i64 %237, 1
  %239 = getelementptr inbounds nuw %struct.complex, ptr %1, i64 %238
  %240 = getelementptr inbounds nuw %struct.complex, ptr %2, i64 %238
  %241 = getelementptr inbounds nuw i8, ptr %240, i64 16
  %242 = load <2 x i64>, ptr %240, align 4
  %243 = load <2 x i64>, ptr %241, align 4
  %244 = getelementptr inbounds nuw i8, ptr %239, i64 16
  store <2 x i64> %242, ptr %239, align 4
  store <2 x i64> %243, ptr %244, align 4
  %245 = add nuw i64 %237, 4
  %246 = icmp eq i64 %245, %34
  br i1 %246, label %247, label %236, !llvm.loop !36

247:                                              ; preds = %236
  br i1 %36, label %257, label %248

248:                                              ; preds = %235, %247
  %249 = phi i64 [ 1, %235 ], [ %35, %247 ]
  br label %250

250:                                              ; preds = %248, %250
  %251 = phi i64 [ %255, %250 ], [ %249, %248 ]
  %252 = getelementptr inbounds nuw %struct.complex, ptr %1, i64 %251
  %253 = getelementptr inbounds nuw %struct.complex, ptr %2, i64 %251
  %254 = load i64, ptr %253, align 4
  store i64 %254, ptr %252, align 4
  %255 = add nuw nsw i64 %251, 1
  %256 = icmp eq i64 %255, %13
  br i1 %256, label %257, label %250, !llvm.loop !37

257:                                              ; preds = %250, %247
  %258 = shl nsw i32 %38, 1
  %259 = icmp sgt i32 %258, %8
  br i1 %259, label %260, label %37, !llvm.loop !38

260:                                              ; preds = %257
  %261 = icmp slt i32 %0, 1
  br i1 %261, label %308, label %262

262:                                              ; preds = %260
  %263 = fneg float %4
  %264 = add nuw i32 %0, 1
  %265 = zext i32 %264 to i64
  %266 = zext nneg i32 %0 to i64
  %267 = icmp ult i32 %0, 8
  br i1 %267, label %297, label %268

268:                                              ; preds = %262
  %269 = and i64 %266, 2147483640
  %270 = or disjoint i64 %269, 1
  %271 = insertelement <4 x float> poison, float %263, i64 0
  %272 = shufflevector <4 x float> %271, <4 x float> poison, <4 x i32> zeroinitializer
  %273 = insertelement <4 x float> poison, float %4, i64 0
  %274 = shufflevector <4 x float> %273, <4 x float> poison, <4 x i32> zeroinitializer
  br label %275

275:                                              ; preds = %275, %268
  %276 = phi i64 [ 0, %268 ], [ %293, %275 ]
  %277 = getelementptr inbounds nuw %struct.complex, ptr %1, i64 %276
  %278 = getelementptr inbounds nuw i8, ptr %277, i64 8
  %279 = getelementptr inbounds nuw %struct.complex, ptr %1, i64 %276
  %280 = getelementptr inbounds nuw i8, ptr %279, i64 40
  %281 = load <8 x float>, ptr %278, align 4, !tbaa !18
  %282 = shufflevector <8 x float> %281, <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %283 = shufflevector <8 x float> %281, <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %284 = load <8 x float>, ptr %280, align 4, !tbaa !18
  %285 = shufflevector <8 x float> %284, <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %286 = shufflevector <8 x float> %284, <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %287 = fmul <4 x float> %274, %282
  %288 = fmul <4 x float> %274, %285
  %289 = fmul <4 x float> %283, %272
  %290 = fmul <4 x float> %286, %272
  %291 = shufflevector <4 x float> %287, <4 x float> %289, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x float> %291, ptr %278, align 4, !tbaa !18
  %292 = shufflevector <4 x float> %288, <4 x float> %290, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x float> %292, ptr %280, align 4, !tbaa !18
  %293 = add nuw i64 %276, 8
  %294 = icmp eq i64 %293, %269
  br i1 %294, label %295, label %275, !llvm.loop !39

295:                                              ; preds = %275
  %296 = icmp eq i64 %269, %266
  br i1 %296, label %308, label %297

297:                                              ; preds = %262, %295
  %298 = phi i64 [ 1, %262 ], [ %270, %295 ]
  %299 = insertelement <2 x float> poison, float %4, i64 0
  %300 = insertelement <2 x float> %299, float %263, i64 1
  br label %301

301:                                              ; preds = %297, %301
  %302 = phi i64 [ %306, %301 ], [ %298, %297 ]
  %303 = getelementptr inbounds nuw %struct.complex, ptr %1, i64 %302
  %304 = load <2 x float>, ptr %303, align 4, !tbaa !18
  %305 = fmul <2 x float> %304, %300
  store <2 x float> %305, ptr %303, align 4, !tbaa !18
  %306 = add nuw nsw i64 %302, 1
  %307 = icmp eq i64 %306, %265
  br i1 %307, label %308, label %301, !llvm.loop !40

308:                                              ; preds = %301, %295, %260
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #8

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #9

; Function Attrs: nofree nounwind uwtable
define dso_local void @Oscar() local_unnamed_addr #4 {
  %1 = alloca [26 x float], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #12
  br label %2

2:                                                ; preds = %2, %0
  %3 = phi i64 [ 1, %0 ], [ %29, %2 ]
  %4 = phi float [ 4.000000e+00, %0 ], [ %28, %2 ]
  %5 = fdiv float 0x400921FB60000000, %4
  %6 = fmul float %5, %5
  %7 = fmul float %6, 5.000000e-01
  %8 = fsub float 1.000000e+00, %7
  %9 = fmul float %5, %6
  %10 = fmul float %5, %9
  %11 = fdiv float %10, 2.400000e+01
  %12 = fadd float %8, %11
  %13 = fmul float %5, %10
  %14 = fmul float %5, %13
  %15 = fdiv float %14, 7.200000e+02
  %16 = fsub float %12, %15
  %17 = fmul float %5, %14
  %18 = fmul float %5, %17
  %19 = fdiv float %18, 4.032000e+04
  %20 = fadd float %16, %19
  %21 = fmul float %5, %18
  %22 = fmul float %5, %21
  %23 = fdiv float %22, 3.628800e+06
  %24 = fsub float %20, %23
  %25 = fmul float %24, 2.000000e+00
  %26 = fdiv float 1.000000e+00, %25
  %27 = getelementptr inbounds nuw float, ptr %1, i64 %3
  store float %26, ptr %27, align 4, !tbaa !18
  %28 = fadd float %4, %4
  %29 = add nuw nsw i64 %3, 1
  %30 = icmp eq i64 %29, 26
  br i1 %30, label %31, label %2, !llvm.loop !19

31:                                               ; preds = %2
  store <2 x float> <float 1.000000e+00, float 0.000000e+00>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 8), align 4, !tbaa !18
  store <2 x float> <float 0.000000e+00, float 1.000000e+00>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 520), align 4, !tbaa !18
  store <2 x float> <float -1.000000e+00, float 0.000000e+00>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1032), align 4, !tbaa !18
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %33 = load float, ptr %32, align 4, !tbaa !18
  store float %33, ptr getelementptr inbounds nuw (i8, ptr @e, i64 264), align 4, !tbaa !10
  store float %33, ptr getelementptr inbounds nuw (i8, ptr @e, i64 268), align 4, !tbaa !13
  %34 = fneg float %33
  store float %34, ptr getelementptr inbounds nuw (i8, ptr @e, i64 776), align 4, !tbaa !10
  store float %33, ptr getelementptr inbounds nuw (i8, ptr @e, i64 780), align 4, !tbaa !13
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %36 = load float, ptr %35, align 4, !tbaa !18
  %37 = fadd float %33, 1.000000e+00
  %38 = fmul float %36, %37
  store float %38, ptr getelementptr inbounds nuw (i8, ptr @e, i64 136), align 4, !tbaa !10
  store float %38, ptr getelementptr inbounds nuw (i8, ptr @e, i64 396), align 4, !tbaa !13
  %39 = fsub float 0.000000e+00, %33
  %40 = fmul float %36, %39
  store float %40, ptr getelementptr inbounds nuw (i8, ptr @e, i64 648), align 4, !tbaa !10
  store float %38, ptr getelementptr inbounds nuw (i8, ptr @e, i64 652), align 4, !tbaa !13
  %41 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %42 = load float, ptr %41, align 4, !tbaa !18
  %43 = fadd float %38, 1.000000e+00
  %44 = fmul float %42, %43
  store float %44, ptr getelementptr inbounds nuw (i8, ptr @e, i64 72), align 4, !tbaa !10
  %45 = fadd float %33, %38
  %46 = fmul float %42, %45
  store float %46, ptr getelementptr inbounds nuw (i8, ptr @e, i64 200), align 4, !tbaa !10
  store float %46, ptr getelementptr inbounds nuw (i8, ptr @e, i64 332), align 4, !tbaa !13
  store float %44, ptr getelementptr inbounds nuw (i8, ptr @e, i64 460), align 4, !tbaa !13
  %47 = fadd float %40, 0.000000e+00
  %48 = fmul float %42, %47
  store float %48, ptr getelementptr inbounds nuw (i8, ptr @e, i64 584), align 4, !tbaa !10
  store float %44, ptr getelementptr inbounds nuw (i8, ptr @e, i64 588), align 4, !tbaa !13
  %49 = fsub float %40, %33
  %50 = fmul float %42, %49
  store float %50, ptr getelementptr inbounds nuw (i8, ptr @e, i64 712), align 4, !tbaa !10
  store float %46, ptr getelementptr inbounds nuw (i8, ptr @e, i64 716), align 4, !tbaa !13
  %51 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %52 = load float, ptr %51, align 4, !tbaa !18
  %53 = fadd float %44, 1.000000e+00
  %54 = fmul float %52, %53
  store float %54, ptr getelementptr inbounds nuw (i8, ptr @e, i64 40), align 4, !tbaa !10
  %55 = fadd float %38, %44
  %56 = fmul float %52, %55
  store float %56, ptr getelementptr inbounds nuw (i8, ptr @e, i64 104), align 4, !tbaa !10
  %57 = fadd float %46, %38
  %58 = fmul float %52, %57
  store float %58, ptr getelementptr inbounds nuw (i8, ptr @e, i64 168), align 4, !tbaa !10
  %59 = fadd float %33, %46
  %60 = fmul float %52, %59
  store float %60, ptr getelementptr inbounds nuw (i8, ptr @e, i64 232), align 4, !tbaa !10
  store float %60, ptr getelementptr inbounds nuw (i8, ptr @e, i64 300), align 4, !tbaa !13
  store float %58, ptr getelementptr inbounds nuw (i8, ptr @e, i64 364), align 4, !tbaa !13
  store float %56, ptr getelementptr inbounds nuw (i8, ptr @e, i64 428), align 4, !tbaa !13
  store float %54, ptr getelementptr inbounds nuw (i8, ptr @e, i64 492), align 4, !tbaa !13
  %61 = fadd float %48, 0.000000e+00
  %62 = fmul float %52, %61
  store float %62, ptr getelementptr inbounds nuw (i8, ptr @e, i64 552), align 4, !tbaa !10
  store float %54, ptr getelementptr inbounds nuw (i8, ptr @e, i64 556), align 4, !tbaa !13
  %63 = fadd float %40, %48
  %64 = fmul float %52, %63
  store float %64, ptr getelementptr inbounds nuw (i8, ptr @e, i64 616), align 4, !tbaa !10
  store float %56, ptr getelementptr inbounds nuw (i8, ptr @e, i64 620), align 4, !tbaa !13
  %65 = fadd float %50, %40
  %66 = fmul float %52, %65
  store float %66, ptr getelementptr inbounds nuw (i8, ptr @e, i64 680), align 4, !tbaa !10
  store float %58, ptr getelementptr inbounds nuw (i8, ptr @e, i64 684), align 4, !tbaa !13
  %67 = load float, ptr getelementptr inbounds nuw (i8, ptr @e, i64 776), align 4, !tbaa !10
  %68 = fadd float %67, %50
  %69 = fmul float %52, %68
  store float %69, ptr getelementptr inbounds nuw (i8, ptr @e, i64 744), align 4, !tbaa !10
  %70 = load float, ptr getelementptr inbounds nuw (i8, ptr @e, i64 780), align 4, !tbaa !13
  %71 = fadd float %70, %46
  %72 = fmul float %52, %71
  store float %72, ptr getelementptr inbounds nuw (i8, ptr @e, i64 748), align 4, !tbaa !13
  %73 = insertelement <2 x float> poison, float %33, i64 0
  %74 = shufflevector <2 x float> %73, <2 x float> poison, <2 x i32> zeroinitializer
  %75 = fsub <2 x float> <float -1.000000e+00, float poison>, %74
  %76 = fadd <2 x float> %74, <float poison, float 0.000000e+00>
  %77 = shufflevector <2 x float> %75, <2 x float> %76, <2 x i32> <i32 0, i32 3>
  %78 = insertelement <2 x float> poison, float %36, i64 0
  %79 = shufflevector <2 x float> %78, <2 x float> poison, <2 x i32> zeroinitializer
  %80 = fmul <2 x float> %79, %77
  %81 = extractelement <2 x float> %80, i64 1
  store float %81, ptr getelementptr inbounds nuw (i8, ptr @e, i64 140), align 4, !tbaa !13
  store float %81, ptr getelementptr inbounds nuw (i8, ptr @e, i64 392), align 4, !tbaa !10
  %82 = extractelement <2 x float> %80, i64 0
  store float %82, ptr getelementptr inbounds nuw (i8, ptr @e, i64 904), align 4, !tbaa !10
  store float %81, ptr getelementptr inbounds nuw (i8, ptr @e, i64 908), align 4, !tbaa !13
  %83 = fadd <2 x float> %80, <float -1.000000e+00, float 0.000000e+00>
  %84 = insertelement <2 x float> poison, float %42, i64 0
  %85 = shufflevector <2 x float> %84, <2 x float> poison, <2 x i32> zeroinitializer
  %86 = fmul <2 x float> %85, %83
  %87 = extractelement <2 x float> %86, i64 1
  store float %87, ptr getelementptr inbounds nuw (i8, ptr @e, i64 76), align 4, !tbaa !13
  %88 = fadd float %33, %81
  %89 = fmul float %42, %88
  store float %89, ptr getelementptr inbounds nuw (i8, ptr @e, i64 204), align 4, !tbaa !13
  store float %89, ptr getelementptr inbounds nuw (i8, ptr @e, i64 328), align 4, !tbaa !10
  store float %87, ptr getelementptr inbounds nuw (i8, ptr @e, i64 456), align 4, !tbaa !10
  %90 = fsub float %82, %33
  %91 = fmul float %42, %90
  store float %91, ptr getelementptr inbounds nuw (i8, ptr @e, i64 840), align 4, !tbaa !10
  store float %89, ptr getelementptr inbounds nuw (i8, ptr @e, i64 844), align 4, !tbaa !13
  store <2 x float> %86, ptr getelementptr inbounds nuw (i8, ptr @e, i64 968), align 4, !tbaa !18
  %92 = fadd float %87, 0.000000e+00
  %93 = fmul float %52, %92
  store float %93, ptr getelementptr inbounds nuw (i8, ptr @e, i64 44), align 4, !tbaa !13
  %94 = fadd float %81, %87
  %95 = fmul float %52, %94
  store float %95, ptr getelementptr inbounds nuw (i8, ptr @e, i64 108), align 4, !tbaa !13
  %96 = fadd float %89, %81
  %97 = fmul float %52, %96
  store float %97, ptr getelementptr inbounds nuw (i8, ptr @e, i64 172), align 4, !tbaa !13
  %98 = fadd float %33, %89
  %99 = fmul float %52, %98
  store float %99, ptr getelementptr inbounds nuw (i8, ptr @e, i64 236), align 4, !tbaa !13
  store float %99, ptr getelementptr inbounds nuw (i8, ptr @e, i64 296), align 4, !tbaa !10
  store float %97, ptr getelementptr inbounds nuw (i8, ptr @e, i64 360), align 4, !tbaa !10
  store float %95, ptr getelementptr inbounds nuw (i8, ptr @e, i64 424), align 4, !tbaa !10
  store float %93, ptr getelementptr inbounds nuw (i8, ptr @e, i64 488), align 4, !tbaa !10
  %100 = fadd float %91, %67
  %101 = fmul float %52, %100
  store float %101, ptr getelementptr inbounds nuw (i8, ptr @e, i64 808), align 4, !tbaa !10
  %102 = fadd float %89, %70
  %103 = fmul float %52, %102
  store float %103, ptr getelementptr inbounds nuw (i8, ptr @e, i64 812), align 4, !tbaa !13
  %104 = fadd float %82, %91
  %105 = fmul float %52, %104
  store float %105, ptr getelementptr inbounds nuw (i8, ptr @e, i64 872), align 4, !tbaa !10
  store float %97, ptr getelementptr inbounds nuw (i8, ptr @e, i64 876), align 4, !tbaa !13
  %106 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 904), align 4, !tbaa !18
  %107 = fadd <2 x float> %86, %106
  %108 = insertelement <2 x float> poison, float %52, i64 0
  %109 = shufflevector <2 x float> %108, <2 x float> poison, <2 x i32> zeroinitializer
  %110 = fmul <2 x float> %109, %107
  store <2 x float> %110, ptr getelementptr inbounds nuw (i8, ptr @e, i64 936), align 4, !tbaa !18
  %111 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1032), align 4, !tbaa !18
  %112 = fadd <2 x float> %111, %86
  %113 = fmul <2 x float> %109, %112
  store <2 x float> %113, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1000), align 4, !tbaa !18
  %114 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %115 = load <2 x float>, ptr %114, align 4
  %116 = shufflevector <2 x float> %115, <2 x float> poison, <2 x i32> zeroinitializer
  %117 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 40), align 4, !tbaa !18
  %118 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 8), align 4, !tbaa !18
  %119 = fadd <2 x float> %117, %118
  %120 = fmul <2 x float> %116, %119
  store <2 x float> %120, ptr getelementptr inbounds nuw (i8, ptr @e, i64 24), align 4, !tbaa !18
  %121 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 72), align 4, !tbaa !18
  %122 = fadd <2 x float> %121, %117
  %123 = fmul <2 x float> %116, %122
  store <2 x float> %123, ptr getelementptr inbounds nuw (i8, ptr @e, i64 56), align 4, !tbaa !18
  %124 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 104), align 4, !tbaa !18
  %125 = fadd <2 x float> %124, %121
  %126 = fmul <2 x float> %116, %125
  store <2 x float> %126, ptr getelementptr inbounds nuw (i8, ptr @e, i64 88), align 4, !tbaa !18
  %127 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 136), align 4, !tbaa !18
  %128 = fadd <2 x float> %127, %124
  %129 = fmul <2 x float> %116, %128
  store <2 x float> %129, ptr getelementptr inbounds nuw (i8, ptr @e, i64 120), align 4, !tbaa !18
  %130 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 168), align 4, !tbaa !18
  %131 = fadd <2 x float> %130, %127
  %132 = fmul <2 x float> %116, %131
  store <2 x float> %132, ptr getelementptr inbounds nuw (i8, ptr @e, i64 152), align 4, !tbaa !18
  %133 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 200), align 4, !tbaa !18
  %134 = fadd <2 x float> %133, %130
  %135 = fmul <2 x float> %116, %134
  store <2 x float> %135, ptr getelementptr inbounds nuw (i8, ptr @e, i64 184), align 4, !tbaa !18
  %136 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 232), align 4, !tbaa !18
  %137 = fadd <2 x float> %136, %133
  %138 = fmul <2 x float> %116, %137
  store <2 x float> %138, ptr getelementptr inbounds nuw (i8, ptr @e, i64 216), align 4, !tbaa !18
  %139 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 264), align 4, !tbaa !18
  %140 = fadd <2 x float> %139, %136
  %141 = fmul <2 x float> %116, %140
  store <2 x float> %141, ptr getelementptr inbounds nuw (i8, ptr @e, i64 248), align 4, !tbaa !18
  %142 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 296), align 4, !tbaa !18
  %143 = fadd <2 x float> %142, %139
  %144 = fmul <2 x float> %116, %143
  store <2 x float> %144, ptr getelementptr inbounds nuw (i8, ptr @e, i64 280), align 4, !tbaa !18
  %145 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 328), align 4, !tbaa !18
  %146 = fadd <2 x float> %145, %142
  %147 = fmul <2 x float> %116, %146
  store <2 x float> %147, ptr getelementptr inbounds nuw (i8, ptr @e, i64 312), align 4, !tbaa !18
  %148 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 360), align 4, !tbaa !18
  %149 = fadd <2 x float> %148, %145
  %150 = fmul <2 x float> %116, %149
  store <2 x float> %150, ptr getelementptr inbounds nuw (i8, ptr @e, i64 344), align 4, !tbaa !18
  %151 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 392), align 4, !tbaa !18
  %152 = fadd <2 x float> %151, %148
  %153 = fmul <2 x float> %116, %152
  store <2 x float> %153, ptr getelementptr inbounds nuw (i8, ptr @e, i64 376), align 4, !tbaa !18
  %154 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 424), align 4, !tbaa !18
  %155 = fadd <2 x float> %154, %151
  %156 = fmul <2 x float> %116, %155
  store <2 x float> %156, ptr getelementptr inbounds nuw (i8, ptr @e, i64 408), align 4, !tbaa !18
  %157 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 456), align 4, !tbaa !18
  %158 = fadd <2 x float> %157, %154
  %159 = fmul <2 x float> %116, %158
  store <2 x float> %159, ptr getelementptr inbounds nuw (i8, ptr @e, i64 440), align 4, !tbaa !18
  %160 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 488), align 4, !tbaa !18
  %161 = fadd <2 x float> %160, %157
  %162 = fmul <2 x float> %116, %161
  store <2 x float> %162, ptr getelementptr inbounds nuw (i8, ptr @e, i64 472), align 4, !tbaa !18
  %163 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 520), align 4, !tbaa !18
  %164 = fadd <2 x float> %163, %160
  %165 = fmul <2 x float> %116, %164
  store <2 x float> %165, ptr getelementptr inbounds nuw (i8, ptr @e, i64 504), align 4, !tbaa !18
  %166 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 552), align 4, !tbaa !18
  %167 = fadd <2 x float> %166, %163
  %168 = fmul <2 x float> %116, %167
  store <2 x float> %168, ptr getelementptr inbounds nuw (i8, ptr @e, i64 536), align 4, !tbaa !18
  %169 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 584), align 4, !tbaa !18
  %170 = fadd <2 x float> %169, %166
  %171 = fmul <2 x float> %116, %170
  store <2 x float> %171, ptr getelementptr inbounds nuw (i8, ptr @e, i64 568), align 4, !tbaa !18
  %172 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 616), align 4, !tbaa !18
  %173 = fadd <2 x float> %172, %169
  %174 = fmul <2 x float> %116, %173
  store <2 x float> %174, ptr getelementptr inbounds nuw (i8, ptr @e, i64 600), align 4, !tbaa !18
  %175 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 648), align 4, !tbaa !18
  %176 = fadd <2 x float> %175, %172
  %177 = fmul <2 x float> %116, %176
  store <2 x float> %177, ptr getelementptr inbounds nuw (i8, ptr @e, i64 632), align 4, !tbaa !18
  %178 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 680), align 4, !tbaa !18
  %179 = fadd <2 x float> %178, %175
  %180 = fmul <2 x float> %116, %179
  store <2 x float> %180, ptr getelementptr inbounds nuw (i8, ptr @e, i64 664), align 4, !tbaa !18
  %181 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 712), align 4, !tbaa !18
  %182 = fadd <2 x float> %181, %178
  %183 = fmul <2 x float> %116, %182
  store <2 x float> %183, ptr getelementptr inbounds nuw (i8, ptr @e, i64 696), align 4, !tbaa !18
  %184 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 744), align 4, !tbaa !18
  %185 = fadd <2 x float> %184, %181
  %186 = fmul <2 x float> %116, %185
  store <2 x float> %186, ptr getelementptr inbounds nuw (i8, ptr @e, i64 728), align 4, !tbaa !18
  %187 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 776), align 4, !tbaa !18
  %188 = fadd <2 x float> %187, %184
  %189 = fmul <2 x float> %116, %188
  store <2 x float> %189, ptr getelementptr inbounds nuw (i8, ptr @e, i64 760), align 4, !tbaa !18
  %190 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 808), align 4, !tbaa !18
  %191 = fadd <2 x float> %190, %187
  %192 = fmul <2 x float> %116, %191
  store <2 x float> %192, ptr getelementptr inbounds nuw (i8, ptr @e, i64 792), align 4, !tbaa !18
  %193 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 840), align 4, !tbaa !18
  %194 = fadd <2 x float> %193, %190
  %195 = fmul <2 x float> %116, %194
  store <2 x float> %195, ptr getelementptr inbounds nuw (i8, ptr @e, i64 824), align 4, !tbaa !18
  %196 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 872), align 4, !tbaa !18
  %197 = fadd <2 x float> %196, %193
  %198 = fmul <2 x float> %116, %197
  store <2 x float> %198, ptr getelementptr inbounds nuw (i8, ptr @e, i64 856), align 4, !tbaa !18
  %199 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 904), align 4, !tbaa !18
  %200 = fadd <2 x float> %199, %196
  %201 = fmul <2 x float> %116, %200
  store <2 x float> %201, ptr getelementptr inbounds nuw (i8, ptr @e, i64 888), align 4, !tbaa !18
  %202 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 936), align 4, !tbaa !18
  %203 = fadd <2 x float> %202, %199
  %204 = fmul <2 x float> %116, %203
  store <2 x float> %204, ptr getelementptr inbounds nuw (i8, ptr @e, i64 920), align 4, !tbaa !18
  %205 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 968), align 4, !tbaa !18
  %206 = fadd <2 x float> %205, %202
  %207 = fmul <2 x float> %116, %206
  store <2 x float> %207, ptr getelementptr inbounds nuw (i8, ptr @e, i64 952), align 4, !tbaa !18
  %208 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1000), align 4, !tbaa !18
  %209 = fadd <2 x float> %208, %205
  %210 = fmul <2 x float> %116, %209
  store <2 x float> %210, ptr getelementptr inbounds nuw (i8, ptr @e, i64 984), align 4, !tbaa !18
  %211 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1032), align 4, !tbaa !18
  %212 = fadd <2 x float> %211, %208
  %213 = fmul <2 x float> %116, %212
  store <2 x float> %213, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1016), align 4, !tbaa !18
  %214 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %215 = load float, ptr %214, align 4, !tbaa !18
  %216 = insertelement <4 x float> poison, float %215, i64 0
  %217 = shufflevector <4 x float> %216, <4 x float> poison, <4 x i32> zeroinitializer
  br label %218

218:                                              ; preds = %218, %31
  %219 = phi i64 [ 0, %31 ], [ %246, %218 ]
  %220 = shl i64 %219, 1
  %221 = or disjoint i64 %220, 1
  %222 = getelementptr %struct.complex, ptr getelementptr inbounds nuw (i8, ptr @e, i64 8), i64 %221
  %223 = getelementptr i8, ptr %222, i64 8
  %224 = load <16 x float>, ptr %223, align 4, !tbaa !18
  %225 = getelementptr %struct.complex, ptr @e, i64 %221
  %226 = load <16 x float>, ptr %225, align 4, !tbaa !18
  %227 = fadd <16 x float> %224, %226
  %228 = shufflevector <16 x float> %227, <16 x float> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %229 = fmul <4 x float> %217, %228
  %230 = getelementptr %struct.complex, ptr @e, i64 %220
  %231 = getelementptr %struct.complex, ptr @e, i64 %220
  %232 = getelementptr %struct.complex, ptr @e, i64 %220
  %233 = getelementptr i8, ptr %225, i64 8
  %234 = getelementptr i8, ptr %230, i64 32
  %235 = getelementptr i8, ptr %231, i64 48
  %236 = getelementptr i8, ptr %232, i64 64
  %237 = extractelement <4 x float> %229, i64 0
  store float %237, ptr %233, align 4, !tbaa !10
  %238 = fadd <16 x float> %224, %226
  %239 = shufflevector <16 x float> %238, <16 x float> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %240 = fmul <4 x float> %217, %239
  %241 = getelementptr i8, ptr %225, i64 12
  %242 = extractelement <4 x float> %240, i64 0
  store float %242, ptr %241, align 4, !tbaa !13
  %243 = shufflevector <4 x float> %229, <4 x float> %240, <2 x i32> <i32 1, i32 5>
  store <2 x float> %243, ptr %234, align 4, !tbaa !18
  %244 = shufflevector <4 x float> %229, <4 x float> %240, <2 x i32> <i32 2, i32 6>
  store <2 x float> %244, ptr %235, align 4, !tbaa !18
  %245 = shufflevector <4 x float> %229, <4 x float> %240, <2 x i32> <i32 3, i32 7>
  store <2 x float> %245, ptr %236, align 4, !tbaa !18
  %246 = add nuw i64 %219, 4
  %247 = icmp eq i64 %246, 60
  br i1 %247, label %248, label %218, !llvm.loop !41

248:                                              ; preds = %218
  %249 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 984), align 4, !tbaa !18
  %250 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 968), align 4, !tbaa !18
  %251 = fadd <2 x float> %249, %250
  %252 = insertelement <2 x float> poison, float %215, i64 0
  %253 = shufflevector <2 x float> %252, <2 x float> poison, <2 x i32> zeroinitializer
  %254 = fmul <2 x float> %253, %251
  store <2 x float> %254, ptr getelementptr inbounds nuw (i8, ptr @e, i64 976), align 4, !tbaa !18
  %255 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1000), align 4, !tbaa !18
  %256 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 984), align 4, !tbaa !18
  %257 = fadd <2 x float> %255, %256
  %258 = insertelement <2 x float> poison, float %215, i64 0
  %259 = shufflevector <2 x float> %258, <2 x float> poison, <2 x i32> zeroinitializer
  %260 = fmul <2 x float> %259, %257
  store <2 x float> %260, ptr getelementptr inbounds nuw (i8, ptr @e, i64 992), align 4, !tbaa !18
  %261 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1016), align 4, !tbaa !18
  %262 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1000), align 4, !tbaa !18
  %263 = fadd <2 x float> %261, %262
  %264 = insertelement <2 x float> poison, float %215, i64 0
  %265 = shufflevector <2 x float> %264, <2 x float> poison, <2 x i32> zeroinitializer
  %266 = fmul <2 x float> %265, %263
  store <2 x float> %266, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1008), align 4, !tbaa !18
  %267 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1032), align 4, !tbaa !18
  %268 = load <2 x float>, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1016), align 4, !tbaa !18
  %269 = fadd <2 x float> %267, %268
  %270 = insertelement <2 x float> poison, float %215, i64 0
  %271 = shufflevector <2 x float> %270, <2 x float> poison, <2 x i32> zeroinitializer
  %272 = fmul <2 x float> %271, %269
  store <2 x float> %272, ptr getelementptr inbounds nuw (i8, ptr @e, i64 1024), align 4, !tbaa !18
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #12
  br label %275

273:                                              ; preds = %275
  %274 = zext nneg i32 %285 to i64
  store i64 %274, ptr @seed, align 8, !tbaa !6
  store float %282, ptr @zr, align 4, !tbaa !18
  store float %287, ptr @zi, align 4, !tbaa !18
  br label %294

275:                                              ; preds = %248, %275
  %276 = phi i64 [ 1, %248 ], [ %292, %275 ]
  %277 = phi i32 [ 5767, %248 ], [ %285, %275 ]
  %278 = mul nuw nsw i32 %277, 4855
  %279 = add nuw nsw i32 %278, 1731
  %280 = and i32 %279, 8191
  %281 = uitofp nneg i32 %280 to float
  %282 = fmul float %281, 0x3F20000000000000
  %283 = mul i32 %279, 4855
  %284 = add i32 %283, 1731
  %285 = and i32 %284, 8191
  %286 = uitofp nneg i32 %285 to float
  %287 = fmul float %286, 0x3F20000000000000
  %288 = tail call float @llvm.fmuladd.f32(float %282, float 2.000000e+01, float -1.000000e+01)
  %289 = getelementptr inbounds nuw %struct.complex, ptr @z, i64 %276
  store float %288, ptr %289, align 4, !tbaa !10
  %290 = tail call float @llvm.fmuladd.f32(float %287, float 2.000000e+01, float -1.000000e+01)
  %291 = getelementptr inbounds nuw i8, ptr %289, i64 4
  store float %290, ptr %291, align 4, !tbaa !13
  %292 = add nuw nsw i64 %276, 1
  %293 = icmp eq i64 %292, 257
  br i1 %293, label %273, label %275, !llvm.loop !42

294:                                              ; preds = %273, %433
  %295 = phi i32 [ 1, %273 ], [ %434, %433 ]
  br label %296

296:                                              ; preds = %294, %410
  %297 = phi i32 [ %411, %410 ], [ 1, %294 ]
  %298 = zext nneg i32 %297 to i64
  %299 = shl nuw nsw i64 %298, 3
  %300 = shl nuw nsw i64 %298, 3
  %301 = getelementptr i8, ptr @w, i64 %299
  %302 = getelementptr i8, ptr %301, i64 8
  %303 = getelementptr i8, ptr @w, i64 %299
  %304 = getelementptr i8, ptr %303, i64 4
  %305 = getelementptr i8, ptr @w, i64 %299
  br label %306

306:                                              ; preds = %406, %296
  %307 = phi i64 [ %409, %406 ], [ 0, %296 ]
  %308 = phi i64 [ %407, %406 ], [ %298, %296 ]
  %309 = phi i64 [ %308, %406 ], [ 0, %296 ]
  %310 = phi i64 [ %321, %406 ], [ 1, %296 ]
  %311 = getelementptr %struct.complex, ptr @e, i64 %309
  %312 = getelementptr i8, ptr %311, i64 8
  %313 = getelementptr i8, ptr %311, i64 12
  %314 = shl i64 %310, 32
  %315 = ashr exact i64 %314, 32
  %316 = getelementptr %struct.complex, ptr @w, i64 %309
  %317 = getelementptr %struct.complex, ptr @w, i64 %308
  %318 = load float, ptr %312, align 4, !tbaa !10
  %319 = load float, ptr %313, align 4, !tbaa !13
  %320 = tail call i64 @llvm.smax.i64(i64 %308, i64 %315)
  %321 = add nuw nsw i64 %320, 1
  %322 = add nuw i64 %320, 1
  %323 = sub i64 %322, %315
  %324 = icmp ult i64 %323, 4
  br i1 %324, label %382, label %325

325:                                              ; preds = %306
  %326 = mul i64 %300, %307
  %327 = getelementptr i8, ptr %302, i64 %326
  %328 = getelementptr i8, ptr %304, i64 %326
  %329 = getelementptr i8, ptr %305, i64 %326
  %330 = shl nuw nsw i64 %309, 3
  %331 = ashr exact i64 %314, 29
  %332 = add i64 %330, %331
  %333 = getelementptr i8, ptr @w, i64 %332
  %334 = add i64 %309, %320
  %335 = shl nuw nsw i64 %334, 3
  %336 = getelementptr i8, ptr getelementptr inbounds nuw (i8, ptr @w, i64 4), i64 %335
  %337 = getelementptr i8, ptr %329, i64 %331
  %338 = shl nuw nsw i64 %320, 3
  %339 = getelementptr i8, ptr %328, i64 %338
  %340 = getelementptr i8, ptr getelementptr inbounds nuw (i8, ptr @w, i64 4), i64 %332
  %341 = getelementptr i8, ptr getelementptr inbounds nuw (i8, ptr @w, i64 8), i64 %335
  %342 = getelementptr i8, ptr %328, i64 %331
  %343 = getelementptr i8, ptr %327, i64 %338
  %344 = icmp ult ptr %333, %339
  %345 = icmp ult ptr %337, %336
  %346 = and i1 %344, %345
  %347 = icmp ult ptr %340, %343
  %348 = icmp ult ptr %342, %341
  %349 = and i1 %347, %348
  %350 = or i1 %346, %349
  br i1 %350, label %382, label %351

351:                                              ; preds = %325
  %352 = and i64 %323, -4
  %353 = add i64 %315, %352
  %354 = insertelement <4 x float> poison, float %318, i64 0
  %355 = shufflevector <4 x float> %354, <4 x float> poison, <4 x i32> zeroinitializer
  %356 = insertelement <4 x float> poison, float %319, i64 0
  %357 = shufflevector <4 x float> %356, <4 x float> poison, <4 x i32> zeroinitializer
  br label %358

358:                                              ; preds = %358, %351
  %359 = phi i64 [ 0, %351 ], [ %378, %358 ]
  %360 = add i64 %315, %359
  %361 = getelementptr inbounds %struct.complex, ptr @z, i64 %360
  %362 = load <8 x float>, ptr %361, align 4, !tbaa !18
  %363 = getelementptr %struct.complex, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1024), i64 %360
  %364 = load <8 x float>, ptr %363, align 4, !tbaa !18
  %365 = getelementptr %struct.complex, ptr %316, i64 %360
  %366 = fadd <8 x float> %362, %364
  store <8 x float> %366, ptr %365, align 4, !tbaa !18
  %367 = fsub <8 x float> %362, %364
  %368 = shufflevector <8 x float> %367, <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %369 = fsub <8 x float> %362, %364
  %370 = shufflevector <8 x float> %369, <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %371 = fneg <4 x float> %370
  %372 = fmul <4 x float> %357, %371
  %373 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %355, <4 x float> %368, <4 x float> %372)
  %374 = getelementptr %struct.complex, ptr %317, i64 %360
  %375 = fmul <4 x float> %368, %357
  %376 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %355, <4 x float> %370, <4 x float> %375)
  %377 = shufflevector <4 x float> %373, <4 x float> %376, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x float> %377, ptr %374, align 4, !tbaa !18
  %378 = add nuw i64 %359, 4
  %379 = icmp eq i64 %378, %352
  br i1 %379, label %380, label %358, !llvm.loop !43

380:                                              ; preds = %358
  %381 = icmp eq i64 %323, %352
  br i1 %381, label %406, label %382

382:                                              ; preds = %325, %306, %380
  %383 = phi i64 [ %315, %325 ], [ %315, %306 ], [ %353, %380 ]
  %384 = insertelement <2 x float> poison, float %319, i64 0
  %385 = shufflevector <2 x float> %384, <2 x float> poison, <2 x i32> zeroinitializer
  %386 = insertelement <2 x float> poison, float %318, i64 0
  %387 = shufflevector <2 x float> %386, <2 x float> poison, <2 x i32> zeroinitializer
  br label %388

388:                                              ; preds = %382, %388
  %389 = phi i64 [ %404, %388 ], [ %383, %382 ]
  %390 = getelementptr inbounds %struct.complex, ptr @z, i64 %389
  %391 = getelementptr %struct.complex, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1024), i64 %389
  %392 = getelementptr %struct.complex, ptr %316, i64 %389
  %393 = load <2 x float>, ptr %390, align 4, !tbaa !18
  %394 = load <2 x float>, ptr %391, align 4, !tbaa !18
  %395 = fadd <2 x float> %393, %394
  store <2 x float> %395, ptr %392, align 4, !tbaa !18
  %396 = getelementptr %struct.complex, ptr %317, i64 %389
  %397 = fsub <2 x float> %393, %394
  %398 = extractelement <2 x float> %397, i64 1
  %399 = fneg float %398
  %400 = shufflevector <2 x float> %397, <2 x float> poison, <2 x i32> <i32 poison, i32 0>
  %401 = insertelement <2 x float> %400, float %399, i64 0
  %402 = fmul <2 x float> %385, %401
  %403 = tail call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %387, <2 x float> %397, <2 x float> %402)
  store <2 x float> %403, ptr %396, align 4, !tbaa !18
  %404 = add nsw i64 %389, 1
  %405 = icmp eq i64 %389, %320
  br i1 %405, label %406, label %388, !llvm.loop !44

406:                                              ; preds = %388, %380
  %407 = add nuw nsw i64 %308, %298
  %408 = icmp samesign ugt i64 %407, 128
  %409 = add i64 %307, 1
  br i1 %408, label %410, label %306, !llvm.loop !35

410:                                              ; preds = %406
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(2048) getelementptr inbounds nuw (i8, ptr @z, i64 8), ptr noundef nonnull align 4 dereferenceable(2048) getelementptr inbounds nuw (i8, ptr @w, i64 8), i64 2048, i1 false)
  %411 = shl nuw nsw i32 %297, 1
  %412 = icmp samesign ugt i32 %297, 64
  br i1 %412, label %413, label %296, !llvm.loop !38

413:                                              ; preds = %410, %413
  %414 = phi i64 [ %431, %413 ], [ 0, %410 ]
  %415 = getelementptr inbounds nuw %struct.complex, ptr @z, i64 %414
  %416 = getelementptr inbounds nuw i8, ptr %415, i64 8
  %417 = getelementptr inbounds nuw %struct.complex, ptr @z, i64 %414
  %418 = getelementptr inbounds nuw i8, ptr %417, i64 40
  %419 = load <8 x float>, ptr %416, align 4, !tbaa !18
  %420 = shufflevector <8 x float> %419, <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %421 = shufflevector <8 x float> %419, <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %422 = load <8 x float>, ptr %418, align 4, !tbaa !18
  %423 = shufflevector <8 x float> %422, <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %424 = shufflevector <8 x float> %422, <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %425 = fmul <4 x float> %420, splat (float 6.250000e-02)
  %426 = fmul <4 x float> %423, splat (float 6.250000e-02)
  %427 = fmul <4 x float> %421, splat (float -6.250000e-02)
  %428 = fmul <4 x float> %424, splat (float -6.250000e-02)
  %429 = shufflevector <4 x float> %425, <4 x float> %427, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x float> %429, ptr %416, align 4, !tbaa !18
  %430 = shufflevector <4 x float> %426, <4 x float> %428, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x float> %430, ptr %418, align 4, !tbaa !18
  %431 = add nuw i64 %414, 8
  %432 = icmp eq i64 %431, 256
  br i1 %432, label %433, label %413, !llvm.loop !45

433:                                              ; preds = %413
  %434 = add nuw nsw i32 %295, 1
  %435 = icmp eq i32 %434, 21
  br i1 %435, label %436, label %294, !llvm.loop !46

436:                                              ; preds = %433
  %437 = tail call i32 @putchar(i32 10)
  %438 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 8), align 4, !tbaa !10
  %439 = fpext float %438 to double
  %440 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 12), align 4, !tbaa !13
  %441 = fpext float %440 to double
  %442 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %439, double noundef %441)
  %443 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 144), align 4, !tbaa !10
  %444 = fpext float %443 to double
  %445 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 148), align 4, !tbaa !13
  %446 = fpext float %445 to double
  %447 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %444, double noundef %446)
  %448 = tail call i32 @putchar(i32 10)
  %449 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 280), align 4, !tbaa !10
  %450 = fpext float %449 to double
  %451 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 284), align 4, !tbaa !13
  %452 = fpext float %451 to double
  %453 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %450, double noundef %452)
  %454 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 416), align 4, !tbaa !10
  %455 = fpext float %454 to double
  %456 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 420), align 4, !tbaa !13
  %457 = fpext float %456 to double
  %458 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %455, double noundef %457)
  %459 = tail call i32 @putchar(i32 10)
  %460 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 552), align 4, !tbaa !10
  %461 = fpext float %460 to double
  %462 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 556), align 4, !tbaa !13
  %463 = fpext float %462 to double
  %464 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %461, double noundef %463)
  %465 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 688), align 4, !tbaa !10
  %466 = fpext float %465 to double
  %467 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 692), align 4, !tbaa !13
  %468 = fpext float %467 to double
  %469 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %466, double noundef %468)
  %470 = tail call i32 @putchar(i32 10)
  %471 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 824), align 4, !tbaa !10
  %472 = fpext float %471 to double
  %473 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 828), align 4, !tbaa !13
  %474 = fpext float %473 to double
  %475 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %472, double noundef %474)
  %476 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 960), align 4, !tbaa !10
  %477 = fpext float %476 to double
  %478 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 964), align 4, !tbaa !13
  %479 = fpext float %478 to double
  %480 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %477, double noundef %479)
  %481 = tail call i32 @putchar(i32 10)
  %482 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1096), align 4, !tbaa !10
  %483 = fpext float %482 to double
  %484 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1100), align 4, !tbaa !13
  %485 = fpext float %484 to double
  %486 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %483, double noundef %485)
  %487 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1232), align 4, !tbaa !10
  %488 = fpext float %487 to double
  %489 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1236), align 4, !tbaa !13
  %490 = fpext float %489 to double
  %491 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %488, double noundef %490)
  %492 = tail call i32 @putchar(i32 10)
  %493 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1368), align 4, !tbaa !10
  %494 = fpext float %493 to double
  %495 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1372), align 4, !tbaa !13
  %496 = fpext float %495 to double
  %497 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %494, double noundef %496)
  %498 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1504), align 4, !tbaa !10
  %499 = fpext float %498 to double
  %500 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1508), align 4, !tbaa !13
  %501 = fpext float %500 to double
  %502 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %499, double noundef %501)
  %503 = tail call i32 @putchar(i32 10)
  %504 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1640), align 4, !tbaa !10
  %505 = fpext float %504 to double
  %506 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1644), align 4, !tbaa !13
  %507 = fpext float %506 to double
  %508 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %505, double noundef %507)
  %509 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1776), align 4, !tbaa !10
  %510 = fpext float %509 to double
  %511 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1780), align 4, !tbaa !13
  %512 = fpext float %511 to double
  %513 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %510, double noundef %512)
  %514 = tail call i32 @putchar(i32 10)
  %515 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1912), align 4, !tbaa !10
  %516 = fpext float %515 to double
  %517 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 1916), align 4, !tbaa !13
  %518 = fpext float %517 to double
  %519 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %516, double noundef %518)
  %520 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 2048), align 4, !tbaa !10
  %521 = fpext float %520 to double
  %522 = load float, ptr getelementptr inbounds nuw (i8, ptr @z, i64 2052), align 4, !tbaa !13
  %523 = fpext float %522 to double
  %524 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %521, double noundef %523)
  %525 = tail call i32 @putchar(i32 10)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  tail call void @Oscar()
  tail call void @Oscar()
  tail call void @Oscar()
  tail call void @Oscar()
  tail call void @Oscar()
  tail call void @Oscar()
  tail call void @Oscar()
  tail call void @Oscar()
  tail call void @Oscar()
  tail call void @Oscar()
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #11

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #11

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #11

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #11

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x float> @llvm.fmuladd.v2f32(<2 x float>, <2 x float>, <2 x float>) #11

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #10 = { nofree nounwind }
attributes #11 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
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
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !12, i64 0}
!11 = !{!"complex", !12, i64 0, !12, i64 4}
!12 = !{!"float", !8, i64 0}
!13 = !{!11, !12, i64 4}
!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.mustprogress"}
!16 = !{!17, !17, i64 0}
!17 = !{!"int", !8, i64 0}
!18 = !{!12, !12, i64 0}
!19 = distinct !{!19, !15}
!20 = distinct !{!20, !15}
!21 = distinct !{!21, !15}
!22 = !{!23}
!23 = distinct !{!23, !24}
!24 = distinct !{!24, !"LVerDomain"}
!25 = !{!26}
!26 = distinct !{!26, !24}
!27 = !{!28}
!28 = distinct !{!28, !24}
!29 = !{!30}
!30 = distinct !{!30, !24}
!31 = distinct !{!31, !15, !32, !33}
!32 = !{!"llvm.loop.isvectorized", i32 1}
!33 = !{!"llvm.loop.unroll.runtime.disable"}
!34 = distinct !{!34, !15, !32}
!35 = distinct !{!35, !15}
!36 = distinct !{!36, !15, !32, !33}
!37 = distinct !{!37, !15, !32}
!38 = distinct !{!38, !15}
!39 = distinct !{!39, !15, !32, !33}
!40 = distinct !{!40, !15, !33, !32}
!41 = distinct !{!41, !15, !32, !33}
!42 = distinct !{!42, !15}
!43 = distinct !{!43, !15, !32, !33}
!44 = distinct !{!44, !15, !32}
!45 = distinct !{!45, !15, !32, !33}
!46 = distinct !{!46, !15}
