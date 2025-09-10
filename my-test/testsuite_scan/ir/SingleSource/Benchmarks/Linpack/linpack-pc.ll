; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Linpack/linpack-pc.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Linpack/linpack-pc.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@main.aa = internal unnamed_addr global [40000 x float] zeroinitializer, align 4
@main.a = internal global [40200 x float] zeroinitializer, align 4
@main.b = internal global [200 x float] zeroinitializer, align 16
@main.x = internal global [200 x float] zeroinitializer, align 4
@main.ipvt = internal global [200 x i32] zeroinitializer, align 4
@main.j = internal unnamed_addr global i32 0, align 4
@main.ntimes = internal unnamed_addr global i32 0, align 4
@main.info = internal global i32 0, align 4
@.str = private unnamed_addr constant [26 x i8] c"INSERT COMPILER NAME HERE\00", align 1
@.str.1 = private unnamed_addr constant [33 x i8] c"INSERT OPTIMISATION OPTIONS HERE\00", align 1
@stderr = external local_unnamed_addr global ptr, align 8
@.str.2 = private unnamed_addr constant [8 x i8] c"Rolled \00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"Single \00", align 1
@.str.4 = private unnamed_addr constant [54 x i8] c"Precision Linpack Benchmark - PC Version in 'C/C++'\0A\0A\00", align 1
@.str.5 = private unnamed_addr constant [17 x i8] c"Compiler     %s\0A\00", align 1
@.str.6 = private unnamed_addr constant [18 x i8] c"Optimisation %s\0A\0A\00", align 1
@atime = internal unnamed_addr global [9 x [15 x float]] zeroinitializer, align 4
@.str.7 = private unnamed_addr constant [39 x i8] c"norm resid      resid           machep\00", align 1
@.str.8 = private unnamed_addr constant [35 x i8] c"         x[0]-1          x[n-1]-1\0A\00", align 1
@.str.9 = private unnamed_addr constant [33 x i8] c"%6.1f %17.8e%17.8e%17.8e%17.8e\0A\0A\00", align 1
@.str.10 = private unnamed_addr constant [53 x i8] c"Times are reported for matrices of order        %5d\0A\00", align 1
@.str.11 = private unnamed_addr constant [54 x i8] c"1 pass times for array with leading dimension of%5d\0A\0A\00", align 1
@.str.12 = private unnamed_addr constant [56 x i8] c"      dgefa      dgesl      total     Mflops       unit\00", align 1
@.str.13 = private unnamed_addr constant [13 x i8] c"      ratio\0A\00", align 1
@.str.14 = private unnamed_addr constant [30 x i8] c"\0ACalculating matgen overhead\0A\00", align 1
@.str.15 = private unnamed_addr constant [26 x i8] c"%10d times %6.2f seconds\0A\00", align 1
@.str.16 = private unnamed_addr constant [39 x i8] c"Overhead for 1 matgen %12.5f seconds\0A\0A\00", align 1
@.str.17 = private unnamed_addr constant [47 x i8] c"Calculating matgen/dgefa passes for 5 seconds\0A\00", align 1
@.str.18 = private unnamed_addr constant [20 x i8] c"Passes used %10d \0A\0A\00", align 1
@.str.19 = private unnamed_addr constant [47 x i8] c"Times for array with leading dimension of%4d\0A\0A\00", align 1
@.str.20 = private unnamed_addr constant [41 x i8] c"Average                          %11.2f\0A\00", align 1
@.str.21 = private unnamed_addr constant [31 x i8] c"\0ACalculating matgen2 overhead\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local float @second() local_unnamed_addr #0 {
  %1 = tail call i64 @clock() #12
  %2 = sitofp i64 %1 to float
  %3 = fdiv float %2, 1.000000e+06
  ret float %3
}

; Function Attrs: nounwind
declare i64 @clock() local_unnamed_addr #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @what_date() local_unnamed_addr #2 {
  ret void
}

; Function Attrs: cold nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = load ptr, ptr @stderr, align 8, !tbaa !6
  %2 = tail call i64 @fwrite(ptr nonnull @.str.2, i64 7, i64 1, ptr %1) #13
  %3 = load ptr, ptr @stderr, align 8, !tbaa !6
  %4 = tail call i64 @fwrite(ptr nonnull @.str.3, i64 7, i64 1, ptr %3) #13
  %5 = load ptr, ptr @stderr, align 8, !tbaa !6
  %6 = tail call i64 @fwrite(ptr nonnull @.str.4, i64 53, i64 1, ptr %5) #13
  %7 = load ptr, ptr @stderr, align 8, !tbaa !6
  %8 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %7, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str) #14
  %9 = load ptr, ptr @stderr, align 8, !tbaa !6
  %10 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %9, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.1) #14
  br label %11

11:                                               ; preds = %28, %0
  %12 = phi i64 [ 0, %0 ], [ %29, %28 ]
  %13 = phi i32 [ 1325, %0 ], [ %20, %28 ]
  %14 = mul nuw nsw i64 %12, 804
  %15 = getelementptr i8, ptr @main.a, i64 %14
  br label %16

16:                                               ; preds = %16, %11
  %17 = phi i64 [ 0, %11 ], [ %26, %16 ]
  %18 = phi i32 [ %13, %11 ], [ %20, %16 ]
  %19 = mul nuw nsw i32 %18, 3125
  %20 = and i32 %19, 65535
  %21 = add nsw i32 %20, -32768
  %22 = sitofp i32 %21 to double
  %23 = fmul double %22, 0x3F10000000000000
  %24 = fptrunc double %23 to float
  %25 = getelementptr float, ptr %15, i64 %17
  store float %24, ptr %25, align 4, !tbaa !11
  %26 = add nuw nsw i64 %17, 1
  %27 = icmp eq i64 %26, 100
  br i1 %27, label %28, label %16, !llvm.loop !13

28:                                               ; preds = %16
  %29 = add nuw nsw i64 %12, 1
  %30 = icmp eq i64 %29, 100
  br i1 %30, label %31, label %11, !llvm.loop !15

31:                                               ; preds = %28
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(400) @main.b, i8 0, i64 400, i1 false), !tbaa !11
  %32 = load <4 x float>, ptr @main.b, align 16, !tbaa !11
  %33 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  %34 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  %35 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  %36 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  %37 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  %38 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  %39 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  %40 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  %41 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  %42 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  %43 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  %44 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  %45 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  %46 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  %47 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  %48 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  %49 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  %50 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  %51 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  %52 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  %53 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  %54 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  %55 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  %56 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  %57 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  %58 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  %59 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  br label %60

60:                                               ; preds = %60, %31
  %61 = phi float [ %59, %31 ], [ %174, %60 ]
  %62 = phi float [ %58, %31 ], [ %171, %60 ]
  %63 = phi float [ %57, %31 ], [ %168, %60 ]
  %64 = phi float [ %56, %31 ], [ %165, %60 ]
  %65 = phi <4 x float> [ %55, %31 ], [ %162, %60 ]
  %66 = phi <4 x float> [ %54, %31 ], [ %161, %60 ]
  %67 = phi <4 x float> [ %53, %31 ], [ %156, %60 ]
  %68 = phi <4 x float> [ %52, %31 ], [ %155, %60 ]
  %69 = phi <4 x float> [ %51, %31 ], [ %150, %60 ]
  %70 = phi <4 x float> [ %50, %31 ], [ %149, %60 ]
  %71 = phi <4 x float> [ %49, %31 ], [ %144, %60 ]
  %72 = phi <4 x float> [ %48, %31 ], [ %143, %60 ]
  %73 = phi <4 x float> [ %47, %31 ], [ %138, %60 ]
  %74 = phi <4 x float> [ %46, %31 ], [ %137, %60 ]
  %75 = phi <4 x float> [ %45, %31 ], [ %132, %60 ]
  %76 = phi <4 x float> [ %44, %31 ], [ %131, %60 ]
  %77 = phi <4 x float> [ %43, %31 ], [ %126, %60 ]
  %78 = phi <4 x float> [ %42, %31 ], [ %125, %60 ]
  %79 = phi <4 x float> [ %41, %31 ], [ %120, %60 ]
  %80 = phi <4 x float> [ %40, %31 ], [ %119, %60 ]
  %81 = phi <4 x float> [ %39, %31 ], [ %114, %60 ]
  %82 = phi <4 x float> [ %38, %31 ], [ %113, %60 ]
  %83 = phi <4 x float> [ %37, %31 ], [ %108, %60 ]
  %84 = phi <4 x float> [ %36, %31 ], [ %107, %60 ]
  %85 = phi <4 x float> [ %35, %31 ], [ %102, %60 ]
  %86 = phi <4 x float> [ %34, %31 ], [ %101, %60 ]
  %87 = phi <4 x float> [ %33, %31 ], [ %96, %60 ]
  %88 = phi <4 x float> [ %32, %31 ], [ %95, %60 ]
  %89 = phi i64 [ 0, %31 ], [ %175, %60 ]
  %90 = mul nuw nsw i64 %89, 804
  %91 = getelementptr i8, ptr @main.a, i64 %90
  %92 = getelementptr i8, ptr %91, i64 16
  %93 = load <4 x float>, ptr %91, align 4, !tbaa !11
  %94 = load <4 x float>, ptr %92, align 4, !tbaa !11
  %95 = fadd <4 x float> %88, %93
  %96 = fadd <4 x float> %87, %94
  %97 = getelementptr i8, ptr %91, i64 32
  %98 = getelementptr i8, ptr %91, i64 48
  %99 = load <4 x float>, ptr %97, align 4, !tbaa !11
  %100 = load <4 x float>, ptr %98, align 4, !tbaa !11
  %101 = fadd <4 x float> %86, %99
  %102 = fadd <4 x float> %85, %100
  %103 = getelementptr i8, ptr %91, i64 64
  %104 = getelementptr i8, ptr %91, i64 80
  %105 = load <4 x float>, ptr %103, align 4, !tbaa !11
  %106 = load <4 x float>, ptr %104, align 4, !tbaa !11
  %107 = fadd <4 x float> %84, %105
  %108 = fadd <4 x float> %83, %106
  %109 = getelementptr i8, ptr %91, i64 96
  %110 = getelementptr i8, ptr %91, i64 112
  %111 = load <4 x float>, ptr %109, align 4, !tbaa !11
  %112 = load <4 x float>, ptr %110, align 4, !tbaa !11
  %113 = fadd <4 x float> %82, %111
  %114 = fadd <4 x float> %81, %112
  %115 = getelementptr i8, ptr %91, i64 128
  %116 = getelementptr i8, ptr %91, i64 144
  %117 = load <4 x float>, ptr %115, align 4, !tbaa !11
  %118 = load <4 x float>, ptr %116, align 4, !tbaa !11
  %119 = fadd <4 x float> %80, %117
  %120 = fadd <4 x float> %79, %118
  %121 = getelementptr i8, ptr %91, i64 160
  %122 = getelementptr i8, ptr %91, i64 176
  %123 = load <4 x float>, ptr %121, align 4, !tbaa !11
  %124 = load <4 x float>, ptr %122, align 4, !tbaa !11
  %125 = fadd <4 x float> %78, %123
  %126 = fadd <4 x float> %77, %124
  %127 = getelementptr i8, ptr %91, i64 192
  %128 = getelementptr i8, ptr %91, i64 208
  %129 = load <4 x float>, ptr %127, align 4, !tbaa !11
  %130 = load <4 x float>, ptr %128, align 4, !tbaa !11
  %131 = fadd <4 x float> %76, %129
  %132 = fadd <4 x float> %75, %130
  %133 = getelementptr i8, ptr %91, i64 224
  %134 = getelementptr i8, ptr %91, i64 240
  %135 = load <4 x float>, ptr %133, align 4, !tbaa !11
  %136 = load <4 x float>, ptr %134, align 4, !tbaa !11
  %137 = fadd <4 x float> %74, %135
  %138 = fadd <4 x float> %73, %136
  %139 = getelementptr i8, ptr %91, i64 256
  %140 = getelementptr i8, ptr %91, i64 272
  %141 = load <4 x float>, ptr %139, align 4, !tbaa !11
  %142 = load <4 x float>, ptr %140, align 4, !tbaa !11
  %143 = fadd <4 x float> %72, %141
  %144 = fadd <4 x float> %71, %142
  %145 = getelementptr i8, ptr %91, i64 288
  %146 = getelementptr i8, ptr %91, i64 304
  %147 = load <4 x float>, ptr %145, align 4, !tbaa !11
  %148 = load <4 x float>, ptr %146, align 4, !tbaa !11
  %149 = fadd <4 x float> %70, %147
  %150 = fadd <4 x float> %69, %148
  %151 = getelementptr i8, ptr %91, i64 320
  %152 = getelementptr i8, ptr %91, i64 336
  %153 = load <4 x float>, ptr %151, align 4, !tbaa !11
  %154 = load <4 x float>, ptr %152, align 4, !tbaa !11
  %155 = fadd <4 x float> %68, %153
  %156 = fadd <4 x float> %67, %154
  %157 = getelementptr i8, ptr %91, i64 352
  %158 = getelementptr i8, ptr %91, i64 368
  %159 = load <4 x float>, ptr %157, align 4, !tbaa !11
  %160 = load <4 x float>, ptr %158, align 4, !tbaa !11
  %161 = fadd <4 x float> %66, %159
  %162 = fadd <4 x float> %65, %160
  %163 = getelementptr i8, ptr %91, i64 384
  %164 = load float, ptr %163, align 4, !tbaa !11
  %165 = fadd float %64, %164
  %166 = getelementptr i8, ptr %91, i64 388
  %167 = load float, ptr %166, align 4, !tbaa !11
  %168 = fadd float %63, %167
  %169 = getelementptr i8, ptr %91, i64 392
  %170 = load float, ptr %169, align 4, !tbaa !11
  %171 = fadd float %62, %170
  %172 = getelementptr i8, ptr %91, i64 396
  %173 = load float, ptr %172, align 4, !tbaa !11
  %174 = fadd float %61, %173
  %175 = add nuw nsw i64 %89, 1
  %176 = icmp eq i64 %175, 100
  br i1 %176, label %177, label %60, !llvm.loop !16

177:                                              ; preds = %60
  store <4 x float> %95, ptr @main.b, align 16, !tbaa !11
  store <4 x float> %96, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  store <4 x float> %101, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  store <4 x float> %102, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  store <4 x float> %107, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  store <4 x float> %108, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  store <4 x float> %113, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  store <4 x float> %114, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  store <4 x float> %119, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  store <4 x float> %120, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  store <4 x float> %125, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  store <4 x float> %126, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  store <4 x float> %131, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  store <4 x float> %132, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  store <4 x float> %137, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  store <4 x float> %138, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  store <4 x float> %143, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  store <4 x float> %144, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  store <4 x float> %149, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  store <4 x float> %150, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  store <4 x float> %155, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  store <4 x float> %156, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  store <4 x float> %161, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  store <4 x float> %162, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  store float %165, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  store float %168, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  store float %171, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  store float %174, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  %178 = tail call i64 @clock() #12
  %179 = sitofp i64 %178 to float
  %180 = fdiv float %179, 1.000000e+06
  tail call void @dgefa(ptr noundef nonnull @main.a, i32 noundef 201, i32 noundef 100, ptr noundef nonnull @main.ipvt, ptr noundef nonnull @main.info)
  %181 = tail call i64 @clock() #12
  %182 = sitofp i64 %181 to float
  %183 = fdiv float %182, 1.000000e+06
  %184 = fsub float %183, %180
  store float %184, ptr @atime, align 4, !tbaa !11
  %185 = tail call i64 @clock() #12
  br label %186

186:                                              ; preds = %240, %177
  %187 = phi i64 [ 0, %177 ], [ %200, %240 ]
  %188 = sub nsw i64 99, %187
  %189 = getelementptr inbounds nuw i32, ptr @main.ipvt, i64 %187
  %190 = load i32, ptr %189, align 4, !tbaa !17
  %191 = sext i32 %190 to i64
  %192 = getelementptr inbounds float, ptr @main.b, i64 %191
  %193 = load float, ptr %192, align 4, !tbaa !11
  %194 = zext i32 %190 to i64
  %195 = icmp eq i64 %187, %194
  br i1 %195, label %199, label %196

196:                                              ; preds = %186
  %197 = getelementptr inbounds nuw float, ptr @main.b, i64 %187
  %198 = load float, ptr %197, align 4, !tbaa !11
  store float %198, ptr %192, align 4, !tbaa !11
  store float %193, ptr %197, align 4, !tbaa !11
  br label %199

199:                                              ; preds = %196, %186
  %200 = add nuw nsw i64 %187, 1
  %201 = mul nuw nsw i64 %187, 808
  %202 = getelementptr i8, ptr @main.a, i64 %201
  %203 = getelementptr i8, ptr %202, i64 4
  %204 = getelementptr inbounds nuw float, ptr @main.b, i64 %200
  %205 = fcmp oeq float %193, 0.000000e+00
  br i1 %205, label %240, label %206

206:                                              ; preds = %199
  %207 = sub nuw nsw i64 99, %187
  %208 = icmp ult i64 %188, 8
  br i1 %208, label %229, label %209

209:                                              ; preds = %206
  %210 = and i64 %188, -8
  %211 = insertelement <4 x float> poison, float %193, i64 0
  %212 = shufflevector <4 x float> %211, <4 x float> poison, <4 x i32> zeroinitializer
  br label %213

213:                                              ; preds = %213, %209
  %214 = phi i64 [ 0, %209 ], [ %225, %213 ]
  %215 = getelementptr inbounds nuw float, ptr %204, i64 %214
  %216 = getelementptr inbounds nuw i8, ptr %215, i64 16
  %217 = load <4 x float>, ptr %215, align 4, !tbaa !11
  %218 = load <4 x float>, ptr %216, align 4, !tbaa !11
  %219 = getelementptr inbounds nuw float, ptr %203, i64 %214
  %220 = getelementptr inbounds nuw i8, ptr %219, i64 16
  %221 = load <4 x float>, ptr %219, align 4, !tbaa !11
  %222 = load <4 x float>, ptr %220, align 4, !tbaa !11
  %223 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %212, <4 x float> %221, <4 x float> %217)
  %224 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %212, <4 x float> %222, <4 x float> %218)
  store <4 x float> %223, ptr %215, align 4, !tbaa !11
  store <4 x float> %224, ptr %216, align 4, !tbaa !11
  %225 = add nuw i64 %214, 8
  %226 = icmp eq i64 %225, %210
  br i1 %226, label %227, label %213, !llvm.loop !19

227:                                              ; preds = %213
  %228 = icmp eq i64 %188, %210
  br i1 %228, label %240, label %229

229:                                              ; preds = %206, %227
  %230 = phi i64 [ 0, %206 ], [ %210, %227 ]
  br label %231

231:                                              ; preds = %229, %231
  %232 = phi i64 [ %238, %231 ], [ %230, %229 ]
  %233 = getelementptr inbounds nuw float, ptr %204, i64 %232
  %234 = load float, ptr %233, align 4, !tbaa !11
  %235 = getelementptr inbounds nuw float, ptr %203, i64 %232
  %236 = load float, ptr %235, align 4, !tbaa !11
  %237 = tail call float @llvm.fmuladd.f32(float %193, float %236, float %234)
  store float %237, ptr %233, align 4, !tbaa !11
  %238 = add nuw nsw i64 %232, 1
  %239 = icmp eq i64 %238, %207
  br i1 %239, label %240, label %231, !llvm.loop !22

240:                                              ; preds = %231, %227, %199
  %241 = icmp eq i64 %200, 99
  br i1 %241, label %242, label %186, !llvm.loop !23

242:                                              ; preds = %240, %293
  %243 = phi i64 [ %245, %293 ], [ 0, %240 ]
  %244 = sub nsw i64 99, %243
  %245 = add nuw nsw i64 %243, 1
  %246 = sub nuw nsw i64 99, %243
  %247 = getelementptr inbounds nuw float, ptr @main.b, i64 %246
  %248 = load float, ptr %247, align 4, !tbaa !11
  %249 = getelementptr float, ptr @main.a, i64 %246
  %250 = mul nuw nsw i64 %246, 804
  %251 = getelementptr i8, ptr %249, i64 %250
  %252 = load float, ptr %251, align 4, !tbaa !11
  %253 = fdiv float %248, %252
  store float %253, ptr %247, align 4, !tbaa !11
  %254 = fneg float %253
  %255 = mul nuw nsw i64 %246, 804
  %256 = getelementptr inbounds nuw i8, ptr @main.a, i64 %255
  %257 = icmp samesign ugt i64 %243, 98
  %258 = fcmp oeq float %253, 0.000000e+00
  %259 = or i1 %257, %258
  br i1 %259, label %293, label %260

260:                                              ; preds = %242
  %261 = icmp ult i64 %244, 8
  br i1 %261, label %282, label %262

262:                                              ; preds = %260
  %263 = and i64 %244, -8
  %264 = insertelement <4 x float> poison, float %254, i64 0
  %265 = shufflevector <4 x float> %264, <4 x float> poison, <4 x i32> zeroinitializer
  br label %266

266:                                              ; preds = %266, %262
  %267 = phi i64 [ 0, %262 ], [ %278, %266 ]
  %268 = getelementptr inbounds nuw float, ptr @main.b, i64 %267
  %269 = getelementptr inbounds nuw i8, ptr %268, i64 16
  %270 = load <4 x float>, ptr %268, align 16, !tbaa !11
  %271 = load <4 x float>, ptr %269, align 16, !tbaa !11
  %272 = getelementptr inbounds nuw float, ptr %256, i64 %267
  %273 = getelementptr inbounds nuw i8, ptr %272, i64 16
  %274 = load <4 x float>, ptr %272, align 4, !tbaa !11
  %275 = load <4 x float>, ptr %273, align 4, !tbaa !11
  %276 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %265, <4 x float> %274, <4 x float> %270)
  %277 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %265, <4 x float> %275, <4 x float> %271)
  store <4 x float> %276, ptr %268, align 16, !tbaa !11
  store <4 x float> %277, ptr %269, align 16, !tbaa !11
  %278 = add nuw i64 %267, 8
  %279 = icmp eq i64 %278, %263
  br i1 %279, label %280, label %266, !llvm.loop !24

280:                                              ; preds = %266
  %281 = icmp eq i64 %244, %263
  br i1 %281, label %293, label %282

282:                                              ; preds = %260, %280
  %283 = phi i64 [ 0, %260 ], [ %263, %280 ]
  br label %284

284:                                              ; preds = %282, %284
  %285 = phi i64 [ %291, %284 ], [ %283, %282 ]
  %286 = getelementptr inbounds nuw float, ptr @main.b, i64 %285
  %287 = load float, ptr %286, align 4, !tbaa !11
  %288 = getelementptr inbounds nuw float, ptr %256, i64 %285
  %289 = load float, ptr %288, align 4, !tbaa !11
  %290 = tail call float @llvm.fmuladd.f32(float %254, float %289, float %287)
  store float %290, ptr %286, align 4, !tbaa !11
  %291 = add nuw nsw i64 %285, 1
  %292 = icmp eq i64 %291, %246
  br i1 %292, label %293, label %284, !llvm.loop !25

293:                                              ; preds = %284, %280, %242
  %294 = icmp eq i64 %245, 100
  br i1 %294, label %295, label %242, !llvm.loop !26

295:                                              ; preds = %293
  %296 = sitofp i64 %185 to float
  %297 = fdiv float %296, 1.000000e+06
  %298 = tail call i64 @clock() #12
  %299 = sitofp i64 %298 to float
  %300 = fdiv float %299, 1.000000e+06
  %301 = fsub float %300, %297
  store float %301, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 60), align 4, !tbaa !11
  %302 = load float, ptr @atime, align 4, !tbaa !11
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(400) @main.x, ptr noundef nonnull align 16 dereferenceable(400) @main.b, i64 400, i1 false), !tbaa !11
  br label %303

303:                                              ; preds = %295, %324
  %304 = phi float [ %321, %324 ], [ 0.000000e+00, %295 ]
  %305 = phi i64 [ %325, %324 ], [ 0, %295 ]
  %306 = phi i32 [ %314, %324 ], [ 1325, %295 ]
  %307 = mul nuw nsw i64 %305, 804
  %308 = getelementptr i8, ptr @main.a, i64 %307
  br label %309

309:                                              ; preds = %309, %303
  %310 = phi float [ %304, %303 ], [ %321, %309 ]
  %311 = phi i64 [ 0, %303 ], [ %322, %309 ]
  %312 = phi i32 [ %306, %303 ], [ %314, %309 ]
  %313 = mul nuw nsw i32 %312, 3125
  %314 = and i32 %313, 65535
  %315 = add nsw i32 %314, -32768
  %316 = sitofp i32 %315 to double
  %317 = fmul double %316, 0x3F10000000000000
  %318 = fptrunc double %317 to float
  %319 = getelementptr float, ptr %308, i64 %311
  store float %318, ptr %319, align 4, !tbaa !11
  %320 = fcmp olt float %310, %318
  %321 = select i1 %320, float %318, float %310
  %322 = add nuw nsw i64 %311, 1
  %323 = icmp eq i64 %322, 100
  br i1 %323, label %324, label %309, !llvm.loop !13

324:                                              ; preds = %309
  %325 = add nuw nsw i64 %305, 1
  %326 = icmp eq i64 %325, 100
  br i1 %326, label %327, label %303, !llvm.loop !15

327:                                              ; preds = %324
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(400) @main.b, i8 0, i64 400, i1 false), !tbaa !11
  %328 = load <4 x float>, ptr @main.b, align 16, !tbaa !11
  %329 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  %330 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  %331 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  %332 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  %333 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  %334 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  %335 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  %336 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  %337 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  %338 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  %339 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  %340 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  %341 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  %342 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  %343 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  %344 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  %345 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  %346 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  %347 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  %348 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  %349 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  %350 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  %351 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  %352 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  %353 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  %354 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  %355 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  br label %356

356:                                              ; preds = %356, %327
  %357 = phi float [ %355, %327 ], [ %470, %356 ]
  %358 = phi float [ %354, %327 ], [ %467, %356 ]
  %359 = phi float [ %353, %327 ], [ %464, %356 ]
  %360 = phi float [ %352, %327 ], [ %461, %356 ]
  %361 = phi <4 x float> [ %351, %327 ], [ %458, %356 ]
  %362 = phi <4 x float> [ %350, %327 ], [ %457, %356 ]
  %363 = phi <4 x float> [ %349, %327 ], [ %452, %356 ]
  %364 = phi <4 x float> [ %348, %327 ], [ %451, %356 ]
  %365 = phi <4 x float> [ %347, %327 ], [ %446, %356 ]
  %366 = phi <4 x float> [ %346, %327 ], [ %445, %356 ]
  %367 = phi <4 x float> [ %345, %327 ], [ %440, %356 ]
  %368 = phi <4 x float> [ %344, %327 ], [ %439, %356 ]
  %369 = phi <4 x float> [ %343, %327 ], [ %434, %356 ]
  %370 = phi <4 x float> [ %342, %327 ], [ %433, %356 ]
  %371 = phi <4 x float> [ %341, %327 ], [ %428, %356 ]
  %372 = phi <4 x float> [ %340, %327 ], [ %427, %356 ]
  %373 = phi <4 x float> [ %339, %327 ], [ %422, %356 ]
  %374 = phi <4 x float> [ %338, %327 ], [ %421, %356 ]
  %375 = phi <4 x float> [ %337, %327 ], [ %416, %356 ]
  %376 = phi <4 x float> [ %336, %327 ], [ %415, %356 ]
  %377 = phi <4 x float> [ %335, %327 ], [ %410, %356 ]
  %378 = phi <4 x float> [ %334, %327 ], [ %409, %356 ]
  %379 = phi <4 x float> [ %333, %327 ], [ %404, %356 ]
  %380 = phi <4 x float> [ %332, %327 ], [ %403, %356 ]
  %381 = phi <4 x float> [ %331, %327 ], [ %398, %356 ]
  %382 = phi <4 x float> [ %330, %327 ], [ %397, %356 ]
  %383 = phi <4 x float> [ %329, %327 ], [ %392, %356 ]
  %384 = phi <4 x float> [ %328, %327 ], [ %391, %356 ]
  %385 = phi i64 [ 0, %327 ], [ %471, %356 ]
  %386 = mul nuw nsw i64 %385, 804
  %387 = getelementptr i8, ptr @main.a, i64 %386
  %388 = getelementptr i8, ptr %387, i64 16
  %389 = load <4 x float>, ptr %387, align 4, !tbaa !11
  %390 = load <4 x float>, ptr %388, align 4, !tbaa !11
  %391 = fadd <4 x float> %384, %389
  %392 = fadd <4 x float> %383, %390
  %393 = getelementptr i8, ptr %387, i64 32
  %394 = getelementptr i8, ptr %387, i64 48
  %395 = load <4 x float>, ptr %393, align 4, !tbaa !11
  %396 = load <4 x float>, ptr %394, align 4, !tbaa !11
  %397 = fadd <4 x float> %382, %395
  %398 = fadd <4 x float> %381, %396
  %399 = getelementptr i8, ptr %387, i64 64
  %400 = getelementptr i8, ptr %387, i64 80
  %401 = load <4 x float>, ptr %399, align 4, !tbaa !11
  %402 = load <4 x float>, ptr %400, align 4, !tbaa !11
  %403 = fadd <4 x float> %380, %401
  %404 = fadd <4 x float> %379, %402
  %405 = getelementptr i8, ptr %387, i64 96
  %406 = getelementptr i8, ptr %387, i64 112
  %407 = load <4 x float>, ptr %405, align 4, !tbaa !11
  %408 = load <4 x float>, ptr %406, align 4, !tbaa !11
  %409 = fadd <4 x float> %378, %407
  %410 = fadd <4 x float> %377, %408
  %411 = getelementptr i8, ptr %387, i64 128
  %412 = getelementptr i8, ptr %387, i64 144
  %413 = load <4 x float>, ptr %411, align 4, !tbaa !11
  %414 = load <4 x float>, ptr %412, align 4, !tbaa !11
  %415 = fadd <4 x float> %376, %413
  %416 = fadd <4 x float> %375, %414
  %417 = getelementptr i8, ptr %387, i64 160
  %418 = getelementptr i8, ptr %387, i64 176
  %419 = load <4 x float>, ptr %417, align 4, !tbaa !11
  %420 = load <4 x float>, ptr %418, align 4, !tbaa !11
  %421 = fadd <4 x float> %374, %419
  %422 = fadd <4 x float> %373, %420
  %423 = getelementptr i8, ptr %387, i64 192
  %424 = getelementptr i8, ptr %387, i64 208
  %425 = load <4 x float>, ptr %423, align 4, !tbaa !11
  %426 = load <4 x float>, ptr %424, align 4, !tbaa !11
  %427 = fadd <4 x float> %372, %425
  %428 = fadd <4 x float> %371, %426
  %429 = getelementptr i8, ptr %387, i64 224
  %430 = getelementptr i8, ptr %387, i64 240
  %431 = load <4 x float>, ptr %429, align 4, !tbaa !11
  %432 = load <4 x float>, ptr %430, align 4, !tbaa !11
  %433 = fadd <4 x float> %370, %431
  %434 = fadd <4 x float> %369, %432
  %435 = getelementptr i8, ptr %387, i64 256
  %436 = getelementptr i8, ptr %387, i64 272
  %437 = load <4 x float>, ptr %435, align 4, !tbaa !11
  %438 = load <4 x float>, ptr %436, align 4, !tbaa !11
  %439 = fadd <4 x float> %368, %437
  %440 = fadd <4 x float> %367, %438
  %441 = getelementptr i8, ptr %387, i64 288
  %442 = getelementptr i8, ptr %387, i64 304
  %443 = load <4 x float>, ptr %441, align 4, !tbaa !11
  %444 = load <4 x float>, ptr %442, align 4, !tbaa !11
  %445 = fadd <4 x float> %366, %443
  %446 = fadd <4 x float> %365, %444
  %447 = getelementptr i8, ptr %387, i64 320
  %448 = getelementptr i8, ptr %387, i64 336
  %449 = load <4 x float>, ptr %447, align 4, !tbaa !11
  %450 = load <4 x float>, ptr %448, align 4, !tbaa !11
  %451 = fadd <4 x float> %364, %449
  %452 = fadd <4 x float> %363, %450
  %453 = getelementptr i8, ptr %387, i64 352
  %454 = getelementptr i8, ptr %387, i64 368
  %455 = load <4 x float>, ptr %453, align 4, !tbaa !11
  %456 = load <4 x float>, ptr %454, align 4, !tbaa !11
  %457 = fadd <4 x float> %362, %455
  %458 = fadd <4 x float> %361, %456
  %459 = getelementptr i8, ptr %387, i64 384
  %460 = load float, ptr %459, align 4, !tbaa !11
  %461 = fadd float %360, %460
  %462 = getelementptr i8, ptr %387, i64 388
  %463 = load float, ptr %462, align 4, !tbaa !11
  %464 = fadd float %359, %463
  %465 = getelementptr i8, ptr %387, i64 392
  %466 = load float, ptr %465, align 4, !tbaa !11
  %467 = fadd float %358, %466
  %468 = getelementptr i8, ptr %387, i64 396
  %469 = load float, ptr %468, align 4, !tbaa !11
  %470 = fadd float %357, %469
  %471 = add nuw nsw i64 %385, 1
  %472 = icmp eq i64 %471, 100
  br i1 %472, label %473, label %356, !llvm.loop !16

473:                                              ; preds = %356
  store <4 x float> %391, ptr @main.b, align 16, !tbaa !11
  store <4 x float> %392, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  store <4 x float> %397, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  store <4 x float> %398, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  store <4 x float> %403, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  store <4 x float> %404, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  store <4 x float> %409, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  store <4 x float> %410, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  store <4 x float> %415, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  store <4 x float> %416, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  store <4 x float> %421, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  store <4 x float> %422, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  store <4 x float> %427, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  store <4 x float> %428, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  store <4 x float> %433, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  store <4 x float> %434, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  store <4 x float> %439, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  store <4 x float> %440, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  store <4 x float> %445, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  store <4 x float> %446, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  store <4 x float> %451, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  store <4 x float> %452, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  store <4 x float> %457, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  store <4 x float> %458, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  store float %461, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  store float %464, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  store float %467, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  store float %470, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  %474 = load <4 x float>, ptr @main.b, align 16, !tbaa !11
  %475 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  %476 = fneg <4 x float> %474
  %477 = fneg <4 x float> %475
  store <4 x float> %476, ptr @main.b, align 16, !tbaa !11
  store <4 x float> %477, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  %478 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  %479 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  %480 = fneg <4 x float> %478
  %481 = fneg <4 x float> %479
  store <4 x float> %480, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  store <4 x float> %481, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  %482 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  %483 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  %484 = fneg <4 x float> %482
  %485 = fneg <4 x float> %483
  store <4 x float> %484, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  store <4 x float> %485, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  %486 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  %487 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  %488 = fneg <4 x float> %486
  %489 = fneg <4 x float> %487
  store <4 x float> %488, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  store <4 x float> %489, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  %490 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  %491 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  %492 = fneg <4 x float> %490
  %493 = fneg <4 x float> %491
  store <4 x float> %492, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  store <4 x float> %493, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  %494 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  %495 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  %496 = fneg <4 x float> %494
  %497 = fneg <4 x float> %495
  store <4 x float> %496, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  store <4 x float> %497, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  %498 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  %499 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  %500 = fneg <4 x float> %498
  %501 = fneg <4 x float> %499
  store <4 x float> %500, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  store <4 x float> %501, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  %502 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  %503 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  %504 = fneg <4 x float> %502
  %505 = fneg <4 x float> %503
  store <4 x float> %504, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  store <4 x float> %505, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  %506 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  %507 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  %508 = fneg <4 x float> %506
  %509 = fneg <4 x float> %507
  store <4 x float> %508, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  store <4 x float> %509, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  %510 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  %511 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  %512 = fneg <4 x float> %510
  %513 = fneg <4 x float> %511
  store <4 x float> %512, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  store <4 x float> %513, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  %514 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  %515 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  %516 = fneg <4 x float> %514
  %517 = fneg <4 x float> %515
  store <4 x float> %516, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  store <4 x float> %517, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  %518 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  %519 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  %520 = fneg <4 x float> %518
  %521 = fneg <4 x float> %519
  store <4 x float> %520, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  store <4 x float> %521, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  %522 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  %523 = fneg float %522
  store float %523, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  %524 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  %525 = fneg float %524
  store float %525, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  %526 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  %527 = fneg float %526
  store float %527, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  %528 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  %529 = fneg float %528
  store float %529, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  tail call void @dmxpy(i32 noundef 100, ptr noundef nonnull @main.b, i32 noundef 100, i32 noundef 201, ptr noundef nonnull @main.x, ptr noundef nonnull @main.a)
  br label %530

530:                                              ; preds = %473, %530
  %531 = phi i64 [ 0, %473 ], [ %544, %530 ]
  %532 = phi float [ 0.000000e+00, %473 ], [ %543, %530 ]
  %533 = phi float [ 0.000000e+00, %473 ], [ %538, %530 ]
  %534 = getelementptr inbounds nuw float, ptr @main.b, i64 %531
  %535 = load float, ptr %534, align 4, !tbaa !11
  %536 = tail call float @llvm.fabs.f32(float %535)
  %537 = fcmp ogt float %533, %536
  %538 = select i1 %537, float %533, float %536
  %539 = getelementptr inbounds nuw float, ptr @main.x, i64 %531
  %540 = load float, ptr %539, align 4, !tbaa !11
  %541 = tail call float @llvm.fabs.f32(float %540)
  %542 = fcmp ogt float %532, %541
  %543 = select i1 %542, float %532, float %541
  %544 = add nuw nsw i64 %531, 1
  %545 = icmp eq i64 %544, 100
  br i1 %545, label %546, label %530, !llvm.loop !27

546:                                              ; preds = %530
  %547 = fadd float %302, %301
  %548 = fmul float %321, 1.000000e+02
  %549 = fmul float %548, %543
  %550 = fmul float %549, 0x3E80000000000000
  %551 = fdiv float %538, %550
  %552 = load float, ptr @main.x, align 4, !tbaa !11
  %553 = fadd float %552, -1.000000e+00
  %554 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.x, i64 396), align 4, !tbaa !11
  %555 = fadd float %554, -1.000000e+00
  %556 = load ptr, ptr @stderr, align 8, !tbaa !6
  %557 = tail call i64 @fwrite(ptr nonnull @.str.7, i64 38, i64 1, ptr %556) #13
  %558 = load ptr, ptr @stderr, align 8, !tbaa !6
  %559 = tail call i64 @fwrite(ptr nonnull @.str.8, i64 34, i64 1, ptr %558) #13
  %560 = load ptr, ptr @stderr, align 8, !tbaa !6
  %561 = fpext float %551 to double
  %562 = fpext float %538 to double
  %563 = fpext float %553 to double
  %564 = fpext float %555 to double
  %565 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %560, ptr noundef nonnull @.str.9, double noundef %561, double noundef %562, double noundef 0x3E80000000000000, double noundef %563, double noundef %564) #14
  %566 = load ptr, ptr @stderr, align 8, !tbaa !6
  %567 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %566, ptr noundef nonnull @.str.10, i32 noundef 100) #14
  %568 = load ptr, ptr @stderr, align 8, !tbaa !6
  %569 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %568, ptr noundef nonnull @.str.11, i32 noundef 201) #14
  %570 = load ptr, ptr @stderr, align 8, !tbaa !6
  %571 = tail call i64 @fwrite(ptr nonnull @.str.12, i64 55, i64 1, ptr %570) #13
  %572 = load ptr, ptr @stderr, align 8, !tbaa !6
  %573 = tail call i64 @fwrite(ptr nonnull @.str.13, i64 12, i64 1, ptr %572) #13
  store float %547, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 120), align 4, !tbaa !11
  %574 = fcmp ogt float %547, 0.000000e+00
  br i1 %574, label %575, label %581

575:                                              ; preds = %546
  %576 = fpext float %547 to double
  %577 = fmul double %576, 1.000000e+06
  %578 = fdiv double 0x4124F49560000000, %577
  %579 = fptrunc double %578 to float
  %580 = fdiv float 2.000000e+00, %579
  br label %581

581:                                              ; preds = %575, %546
  %582 = phi float [ %579, %575 ], [ 0.000000e+00, %546 ]
  %583 = phi float [ %580, %575 ], [ 0.000000e+00, %546 ]
  store float %582, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 180), align 4, !tbaa !11
  store float %583, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 240), align 4, !tbaa !11
  %584 = fdiv float %547, 0x3FACAC0840000000
  store float %584, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 300), align 4, !tbaa !11
  %585 = load ptr, ptr @stderr, align 8, !tbaa !6
  %586 = tail call i64 @fwrite(ptr nonnull @.str.14, i64 29, i64 1, ptr %585) #13
  %587 = tail call i64 @clock() #12
  br label %588

588:                                              ; preds = %581, %756
  %589 = phi i32 [ %757, %756 ], [ 0, %581 ]
  br label %590

590:                                              ; preds = %588, %607
  %591 = phi i64 [ %608, %607 ], [ 0, %588 ]
  %592 = phi i32 [ %599, %607 ], [ 1325, %588 ]
  %593 = mul nuw nsw i64 %591, 804
  %594 = getelementptr i8, ptr @main.a, i64 %593
  br label %595

595:                                              ; preds = %595, %590
  %596 = phi i64 [ 0, %590 ], [ %605, %595 ]
  %597 = phi i32 [ %592, %590 ], [ %599, %595 ]
  %598 = mul nuw nsw i32 %597, 3125
  %599 = and i32 %598, 65535
  %600 = add nsw i32 %599, -32768
  %601 = sitofp i32 %600 to double
  %602 = fmul double %601, 0x3F10000000000000
  %603 = fptrunc double %602 to float
  %604 = getelementptr float, ptr %594, i64 %596
  store float %603, ptr %604, align 4, !tbaa !11
  %605 = add nuw nsw i64 %596, 1
  %606 = icmp eq i64 %605, 100
  br i1 %606, label %607, label %595, !llvm.loop !13

607:                                              ; preds = %595
  %608 = add nuw nsw i64 %591, 1
  %609 = icmp eq i64 %608, 100
  br i1 %609, label %610, label %590, !llvm.loop !15

610:                                              ; preds = %607
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(400) @main.b, i8 0, i64 400, i1 false), !tbaa !11
  %611 = load <4 x float>, ptr @main.b, align 16, !tbaa !11
  %612 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  %613 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  %614 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  %615 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  %616 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  %617 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  %618 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  %619 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  %620 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  %621 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  %622 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  %623 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  %624 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  %625 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  %626 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  %627 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  %628 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  %629 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  %630 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  %631 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  %632 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  %633 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  %634 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  %635 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  %636 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  %637 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  %638 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  br label %639

639:                                              ; preds = %639, %610
  %640 = phi float [ %638, %610 ], [ %753, %639 ]
  %641 = phi float [ %637, %610 ], [ %750, %639 ]
  %642 = phi float [ %636, %610 ], [ %747, %639 ]
  %643 = phi float [ %635, %610 ], [ %744, %639 ]
  %644 = phi <4 x float> [ %634, %610 ], [ %741, %639 ]
  %645 = phi <4 x float> [ %633, %610 ], [ %740, %639 ]
  %646 = phi <4 x float> [ %632, %610 ], [ %735, %639 ]
  %647 = phi <4 x float> [ %631, %610 ], [ %734, %639 ]
  %648 = phi <4 x float> [ %630, %610 ], [ %729, %639 ]
  %649 = phi <4 x float> [ %629, %610 ], [ %728, %639 ]
  %650 = phi <4 x float> [ %628, %610 ], [ %723, %639 ]
  %651 = phi <4 x float> [ %627, %610 ], [ %722, %639 ]
  %652 = phi <4 x float> [ %626, %610 ], [ %717, %639 ]
  %653 = phi <4 x float> [ %625, %610 ], [ %716, %639 ]
  %654 = phi <4 x float> [ %624, %610 ], [ %711, %639 ]
  %655 = phi <4 x float> [ %623, %610 ], [ %710, %639 ]
  %656 = phi <4 x float> [ %622, %610 ], [ %705, %639 ]
  %657 = phi <4 x float> [ %621, %610 ], [ %704, %639 ]
  %658 = phi <4 x float> [ %620, %610 ], [ %699, %639 ]
  %659 = phi <4 x float> [ %619, %610 ], [ %698, %639 ]
  %660 = phi <4 x float> [ %618, %610 ], [ %693, %639 ]
  %661 = phi <4 x float> [ %617, %610 ], [ %692, %639 ]
  %662 = phi <4 x float> [ %616, %610 ], [ %687, %639 ]
  %663 = phi <4 x float> [ %615, %610 ], [ %686, %639 ]
  %664 = phi <4 x float> [ %614, %610 ], [ %681, %639 ]
  %665 = phi <4 x float> [ %613, %610 ], [ %680, %639 ]
  %666 = phi <4 x float> [ %612, %610 ], [ %675, %639 ]
  %667 = phi <4 x float> [ %611, %610 ], [ %674, %639 ]
  %668 = phi i64 [ 0, %610 ], [ %754, %639 ]
  %669 = mul nuw nsw i64 %668, 804
  %670 = getelementptr i8, ptr @main.a, i64 %669
  %671 = getelementptr i8, ptr %670, i64 16
  %672 = load <4 x float>, ptr %670, align 4, !tbaa !11
  %673 = load <4 x float>, ptr %671, align 4, !tbaa !11
  %674 = fadd <4 x float> %667, %672
  %675 = fadd <4 x float> %666, %673
  %676 = getelementptr i8, ptr %670, i64 32
  %677 = getelementptr i8, ptr %670, i64 48
  %678 = load <4 x float>, ptr %676, align 4, !tbaa !11
  %679 = load <4 x float>, ptr %677, align 4, !tbaa !11
  %680 = fadd <4 x float> %665, %678
  %681 = fadd <4 x float> %664, %679
  %682 = getelementptr i8, ptr %670, i64 64
  %683 = getelementptr i8, ptr %670, i64 80
  %684 = load <4 x float>, ptr %682, align 4, !tbaa !11
  %685 = load <4 x float>, ptr %683, align 4, !tbaa !11
  %686 = fadd <4 x float> %663, %684
  %687 = fadd <4 x float> %662, %685
  %688 = getelementptr i8, ptr %670, i64 96
  %689 = getelementptr i8, ptr %670, i64 112
  %690 = load <4 x float>, ptr %688, align 4, !tbaa !11
  %691 = load <4 x float>, ptr %689, align 4, !tbaa !11
  %692 = fadd <4 x float> %661, %690
  %693 = fadd <4 x float> %660, %691
  %694 = getelementptr i8, ptr %670, i64 128
  %695 = getelementptr i8, ptr %670, i64 144
  %696 = load <4 x float>, ptr %694, align 4, !tbaa !11
  %697 = load <4 x float>, ptr %695, align 4, !tbaa !11
  %698 = fadd <4 x float> %659, %696
  %699 = fadd <4 x float> %658, %697
  %700 = getelementptr i8, ptr %670, i64 160
  %701 = getelementptr i8, ptr %670, i64 176
  %702 = load <4 x float>, ptr %700, align 4, !tbaa !11
  %703 = load <4 x float>, ptr %701, align 4, !tbaa !11
  %704 = fadd <4 x float> %657, %702
  %705 = fadd <4 x float> %656, %703
  %706 = getelementptr i8, ptr %670, i64 192
  %707 = getelementptr i8, ptr %670, i64 208
  %708 = load <4 x float>, ptr %706, align 4, !tbaa !11
  %709 = load <4 x float>, ptr %707, align 4, !tbaa !11
  %710 = fadd <4 x float> %655, %708
  %711 = fadd <4 x float> %654, %709
  %712 = getelementptr i8, ptr %670, i64 224
  %713 = getelementptr i8, ptr %670, i64 240
  %714 = load <4 x float>, ptr %712, align 4, !tbaa !11
  %715 = load <4 x float>, ptr %713, align 4, !tbaa !11
  %716 = fadd <4 x float> %653, %714
  %717 = fadd <4 x float> %652, %715
  %718 = getelementptr i8, ptr %670, i64 256
  %719 = getelementptr i8, ptr %670, i64 272
  %720 = load <4 x float>, ptr %718, align 4, !tbaa !11
  %721 = load <4 x float>, ptr %719, align 4, !tbaa !11
  %722 = fadd <4 x float> %651, %720
  %723 = fadd <4 x float> %650, %721
  %724 = getelementptr i8, ptr %670, i64 288
  %725 = getelementptr i8, ptr %670, i64 304
  %726 = load <4 x float>, ptr %724, align 4, !tbaa !11
  %727 = load <4 x float>, ptr %725, align 4, !tbaa !11
  %728 = fadd <4 x float> %649, %726
  %729 = fadd <4 x float> %648, %727
  %730 = getelementptr i8, ptr %670, i64 320
  %731 = getelementptr i8, ptr %670, i64 336
  %732 = load <4 x float>, ptr %730, align 4, !tbaa !11
  %733 = load <4 x float>, ptr %731, align 4, !tbaa !11
  %734 = fadd <4 x float> %647, %732
  %735 = fadd <4 x float> %646, %733
  %736 = getelementptr i8, ptr %670, i64 352
  %737 = getelementptr i8, ptr %670, i64 368
  %738 = load <4 x float>, ptr %736, align 4, !tbaa !11
  %739 = load <4 x float>, ptr %737, align 4, !tbaa !11
  %740 = fadd <4 x float> %645, %738
  %741 = fadd <4 x float> %644, %739
  %742 = getelementptr i8, ptr %670, i64 384
  %743 = load float, ptr %742, align 4, !tbaa !11
  %744 = fadd float %643, %743
  %745 = getelementptr i8, ptr %670, i64 388
  %746 = load float, ptr %745, align 4, !tbaa !11
  %747 = fadd float %642, %746
  %748 = getelementptr i8, ptr %670, i64 392
  %749 = load float, ptr %748, align 4, !tbaa !11
  %750 = fadd float %641, %749
  %751 = getelementptr i8, ptr %670, i64 396
  %752 = load float, ptr %751, align 4, !tbaa !11
  %753 = fadd float %640, %752
  %754 = add nuw nsw i64 %668, 1
  %755 = icmp eq i64 %754, 100
  br i1 %755, label %756, label %639, !llvm.loop !16

756:                                              ; preds = %639
  store <4 x float> %674, ptr @main.b, align 16, !tbaa !11
  store <4 x float> %675, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  store <4 x float> %680, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  store <4 x float> %681, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  store <4 x float> %686, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  store <4 x float> %687, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  store <4 x float> %692, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  store <4 x float> %693, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  store <4 x float> %698, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  store <4 x float> %699, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  store <4 x float> %704, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  store <4 x float> %705, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  store <4 x float> %710, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  store <4 x float> %711, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  store <4 x float> %716, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  store <4 x float> %717, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  store <4 x float> %722, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  store <4 x float> %723, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  store <4 x float> %728, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  store <4 x float> %729, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  store <4 x float> %734, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  store <4 x float> %735, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  store <4 x float> %740, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  store <4 x float> %741, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  store float %744, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  store float %747, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  store float %750, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  store float %753, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  %757 = add nuw nsw i32 %589, 1
  %758 = icmp eq i32 %757, 100
  br i1 %758, label %759, label %588, !llvm.loop !28

759:                                              ; preds = %756
  %760 = tail call i64 @clock() #12
  %761 = load ptr, ptr @stderr, align 8, !tbaa !6
  %762 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %761, ptr noundef nonnull @.str.15, i32 noundef 100, double noundef 0.000000e+00) #14
  %763 = tail call i64 @clock() #12
  br label %764

764:                                              ; preds = %932, %759
  %765 = phi i32 [ %933, %932 ], [ 0, %759 ]
  br label %766

766:                                              ; preds = %783, %764
  %767 = phi i64 [ %784, %783 ], [ 0, %764 ]
  %768 = phi i32 [ %775, %783 ], [ 1325, %764 ]
  %769 = mul nuw nsw i64 %767, 804
  %770 = getelementptr i8, ptr @main.a, i64 %769
  br label %771

771:                                              ; preds = %771, %766
  %772 = phi i64 [ 0, %766 ], [ %781, %771 ]
  %773 = phi i32 [ %768, %766 ], [ %775, %771 ]
  %774 = mul nuw nsw i32 %773, 3125
  %775 = and i32 %774, 65535
  %776 = add nsw i32 %775, -32768
  %777 = sitofp i32 %776 to double
  %778 = fmul double %777, 0x3F10000000000000
  %779 = fptrunc double %778 to float
  %780 = getelementptr float, ptr %770, i64 %772
  store float %779, ptr %780, align 4, !tbaa !11
  %781 = add nuw nsw i64 %772, 1
  %782 = icmp eq i64 %781, 100
  br i1 %782, label %783, label %771, !llvm.loop !13

783:                                              ; preds = %771
  %784 = add nuw nsw i64 %767, 1
  %785 = icmp eq i64 %784, 100
  br i1 %785, label %786, label %766, !llvm.loop !15

786:                                              ; preds = %783
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(400) @main.b, i8 0, i64 400, i1 false), !tbaa !11
  %787 = load <4 x float>, ptr @main.b, align 16, !tbaa !11
  %788 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  %789 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  %790 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  %791 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  %792 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  %793 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  %794 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  %795 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  %796 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  %797 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  %798 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  %799 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  %800 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  %801 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  %802 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  %803 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  %804 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  %805 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  %806 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  %807 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  %808 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  %809 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  %810 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  %811 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  %812 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  %813 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  %814 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  br label %815

815:                                              ; preds = %815, %786
  %816 = phi float [ %814, %786 ], [ %929, %815 ]
  %817 = phi float [ %813, %786 ], [ %926, %815 ]
  %818 = phi float [ %812, %786 ], [ %923, %815 ]
  %819 = phi float [ %811, %786 ], [ %920, %815 ]
  %820 = phi <4 x float> [ %810, %786 ], [ %917, %815 ]
  %821 = phi <4 x float> [ %809, %786 ], [ %916, %815 ]
  %822 = phi <4 x float> [ %808, %786 ], [ %911, %815 ]
  %823 = phi <4 x float> [ %807, %786 ], [ %910, %815 ]
  %824 = phi <4 x float> [ %806, %786 ], [ %905, %815 ]
  %825 = phi <4 x float> [ %805, %786 ], [ %904, %815 ]
  %826 = phi <4 x float> [ %804, %786 ], [ %899, %815 ]
  %827 = phi <4 x float> [ %803, %786 ], [ %898, %815 ]
  %828 = phi <4 x float> [ %802, %786 ], [ %893, %815 ]
  %829 = phi <4 x float> [ %801, %786 ], [ %892, %815 ]
  %830 = phi <4 x float> [ %800, %786 ], [ %887, %815 ]
  %831 = phi <4 x float> [ %799, %786 ], [ %886, %815 ]
  %832 = phi <4 x float> [ %798, %786 ], [ %881, %815 ]
  %833 = phi <4 x float> [ %797, %786 ], [ %880, %815 ]
  %834 = phi <4 x float> [ %796, %786 ], [ %875, %815 ]
  %835 = phi <4 x float> [ %795, %786 ], [ %874, %815 ]
  %836 = phi <4 x float> [ %794, %786 ], [ %869, %815 ]
  %837 = phi <4 x float> [ %793, %786 ], [ %868, %815 ]
  %838 = phi <4 x float> [ %792, %786 ], [ %863, %815 ]
  %839 = phi <4 x float> [ %791, %786 ], [ %862, %815 ]
  %840 = phi <4 x float> [ %790, %786 ], [ %857, %815 ]
  %841 = phi <4 x float> [ %789, %786 ], [ %856, %815 ]
  %842 = phi <4 x float> [ %788, %786 ], [ %851, %815 ]
  %843 = phi <4 x float> [ %787, %786 ], [ %850, %815 ]
  %844 = phi i64 [ 0, %786 ], [ %930, %815 ]
  %845 = mul nuw nsw i64 %844, 804
  %846 = getelementptr i8, ptr @main.a, i64 %845
  %847 = getelementptr i8, ptr %846, i64 16
  %848 = load <4 x float>, ptr %846, align 4, !tbaa !11
  %849 = load <4 x float>, ptr %847, align 4, !tbaa !11
  %850 = fadd <4 x float> %843, %848
  %851 = fadd <4 x float> %842, %849
  %852 = getelementptr i8, ptr %846, i64 32
  %853 = getelementptr i8, ptr %846, i64 48
  %854 = load <4 x float>, ptr %852, align 4, !tbaa !11
  %855 = load <4 x float>, ptr %853, align 4, !tbaa !11
  %856 = fadd <4 x float> %841, %854
  %857 = fadd <4 x float> %840, %855
  %858 = getelementptr i8, ptr %846, i64 64
  %859 = getelementptr i8, ptr %846, i64 80
  %860 = load <4 x float>, ptr %858, align 4, !tbaa !11
  %861 = load <4 x float>, ptr %859, align 4, !tbaa !11
  %862 = fadd <4 x float> %839, %860
  %863 = fadd <4 x float> %838, %861
  %864 = getelementptr i8, ptr %846, i64 96
  %865 = getelementptr i8, ptr %846, i64 112
  %866 = load <4 x float>, ptr %864, align 4, !tbaa !11
  %867 = load <4 x float>, ptr %865, align 4, !tbaa !11
  %868 = fadd <4 x float> %837, %866
  %869 = fadd <4 x float> %836, %867
  %870 = getelementptr i8, ptr %846, i64 128
  %871 = getelementptr i8, ptr %846, i64 144
  %872 = load <4 x float>, ptr %870, align 4, !tbaa !11
  %873 = load <4 x float>, ptr %871, align 4, !tbaa !11
  %874 = fadd <4 x float> %835, %872
  %875 = fadd <4 x float> %834, %873
  %876 = getelementptr i8, ptr %846, i64 160
  %877 = getelementptr i8, ptr %846, i64 176
  %878 = load <4 x float>, ptr %876, align 4, !tbaa !11
  %879 = load <4 x float>, ptr %877, align 4, !tbaa !11
  %880 = fadd <4 x float> %833, %878
  %881 = fadd <4 x float> %832, %879
  %882 = getelementptr i8, ptr %846, i64 192
  %883 = getelementptr i8, ptr %846, i64 208
  %884 = load <4 x float>, ptr %882, align 4, !tbaa !11
  %885 = load <4 x float>, ptr %883, align 4, !tbaa !11
  %886 = fadd <4 x float> %831, %884
  %887 = fadd <4 x float> %830, %885
  %888 = getelementptr i8, ptr %846, i64 224
  %889 = getelementptr i8, ptr %846, i64 240
  %890 = load <4 x float>, ptr %888, align 4, !tbaa !11
  %891 = load <4 x float>, ptr %889, align 4, !tbaa !11
  %892 = fadd <4 x float> %829, %890
  %893 = fadd <4 x float> %828, %891
  %894 = getelementptr i8, ptr %846, i64 256
  %895 = getelementptr i8, ptr %846, i64 272
  %896 = load <4 x float>, ptr %894, align 4, !tbaa !11
  %897 = load <4 x float>, ptr %895, align 4, !tbaa !11
  %898 = fadd <4 x float> %827, %896
  %899 = fadd <4 x float> %826, %897
  %900 = getelementptr i8, ptr %846, i64 288
  %901 = getelementptr i8, ptr %846, i64 304
  %902 = load <4 x float>, ptr %900, align 4, !tbaa !11
  %903 = load <4 x float>, ptr %901, align 4, !tbaa !11
  %904 = fadd <4 x float> %825, %902
  %905 = fadd <4 x float> %824, %903
  %906 = getelementptr i8, ptr %846, i64 320
  %907 = getelementptr i8, ptr %846, i64 336
  %908 = load <4 x float>, ptr %906, align 4, !tbaa !11
  %909 = load <4 x float>, ptr %907, align 4, !tbaa !11
  %910 = fadd <4 x float> %823, %908
  %911 = fadd <4 x float> %822, %909
  %912 = getelementptr i8, ptr %846, i64 352
  %913 = getelementptr i8, ptr %846, i64 368
  %914 = load <4 x float>, ptr %912, align 4, !tbaa !11
  %915 = load <4 x float>, ptr %913, align 4, !tbaa !11
  %916 = fadd <4 x float> %821, %914
  %917 = fadd <4 x float> %820, %915
  %918 = getelementptr i8, ptr %846, i64 384
  %919 = load float, ptr %918, align 4, !tbaa !11
  %920 = fadd float %819, %919
  %921 = getelementptr i8, ptr %846, i64 388
  %922 = load float, ptr %921, align 4, !tbaa !11
  %923 = fadd float %818, %922
  %924 = getelementptr i8, ptr %846, i64 392
  %925 = load float, ptr %924, align 4, !tbaa !11
  %926 = fadd float %817, %925
  %927 = getelementptr i8, ptr %846, i64 396
  %928 = load float, ptr %927, align 4, !tbaa !11
  %929 = fadd float %816, %928
  %930 = add nuw nsw i64 %844, 1
  %931 = icmp eq i64 %930, 100
  br i1 %931, label %932, label %815, !llvm.loop !16

932:                                              ; preds = %815
  store <4 x float> %850, ptr @main.b, align 16, !tbaa !11
  store <4 x float> %851, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  store <4 x float> %856, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  store <4 x float> %857, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  store <4 x float> %862, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  store <4 x float> %863, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  store <4 x float> %868, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  store <4 x float> %869, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  store <4 x float> %874, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  store <4 x float> %875, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  store <4 x float> %880, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  store <4 x float> %881, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  store <4 x float> %886, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  store <4 x float> %887, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  store <4 x float> %892, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  store <4 x float> %893, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  store <4 x float> %898, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  store <4 x float> %899, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  store <4 x float> %904, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  store <4 x float> %905, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  store <4 x float> %910, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  store <4 x float> %911, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  store <4 x float> %916, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  store <4 x float> %917, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  store float %920, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  store float %923, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  store float %926, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  store float %929, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  %933 = add nuw nsw i32 %765, 1
  %934 = icmp eq i32 %933, 200
  br i1 %934, label %935, label %764, !llvm.loop !28

935:                                              ; preds = %932
  %936 = tail call i64 @clock() #12
  %937 = load ptr, ptr @stderr, align 8, !tbaa !6
  %938 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %937, ptr noundef nonnull @.str.15, i32 noundef 200, double noundef 0.000000e+00) #14
  %939 = tail call i64 @clock() #12
  br label %940

940:                                              ; preds = %1108, %935
  %941 = phi i32 [ %1109, %1108 ], [ 0, %935 ]
  br label %942

942:                                              ; preds = %959, %940
  %943 = phi i64 [ %960, %959 ], [ 0, %940 ]
  %944 = phi i32 [ %951, %959 ], [ 1325, %940 ]
  %945 = mul nuw nsw i64 %943, 804
  %946 = getelementptr i8, ptr @main.a, i64 %945
  br label %947

947:                                              ; preds = %947, %942
  %948 = phi i64 [ 0, %942 ], [ %957, %947 ]
  %949 = phi i32 [ %944, %942 ], [ %951, %947 ]
  %950 = mul nuw nsw i32 %949, 3125
  %951 = and i32 %950, 65535
  %952 = add nsw i32 %951, -32768
  %953 = sitofp i32 %952 to double
  %954 = fmul double %953, 0x3F10000000000000
  %955 = fptrunc double %954 to float
  %956 = getelementptr float, ptr %946, i64 %948
  store float %955, ptr %956, align 4, !tbaa !11
  %957 = add nuw nsw i64 %948, 1
  %958 = icmp eq i64 %957, 100
  br i1 %958, label %959, label %947, !llvm.loop !13

959:                                              ; preds = %947
  %960 = add nuw nsw i64 %943, 1
  %961 = icmp eq i64 %960, 100
  br i1 %961, label %962, label %942, !llvm.loop !15

962:                                              ; preds = %959
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(400) @main.b, i8 0, i64 400, i1 false), !tbaa !11
  %963 = load <4 x float>, ptr @main.b, align 16, !tbaa !11
  %964 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  %965 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  %966 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  %967 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  %968 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  %969 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  %970 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  %971 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  %972 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  %973 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  %974 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  %975 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  %976 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  %977 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  %978 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  %979 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  %980 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  %981 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  %982 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  %983 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  %984 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  %985 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  %986 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  %987 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  %988 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  %989 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  %990 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  br label %991

991:                                              ; preds = %991, %962
  %992 = phi float [ %990, %962 ], [ %1105, %991 ]
  %993 = phi float [ %989, %962 ], [ %1102, %991 ]
  %994 = phi float [ %988, %962 ], [ %1099, %991 ]
  %995 = phi float [ %987, %962 ], [ %1096, %991 ]
  %996 = phi <4 x float> [ %986, %962 ], [ %1093, %991 ]
  %997 = phi <4 x float> [ %985, %962 ], [ %1092, %991 ]
  %998 = phi <4 x float> [ %984, %962 ], [ %1087, %991 ]
  %999 = phi <4 x float> [ %983, %962 ], [ %1086, %991 ]
  %1000 = phi <4 x float> [ %982, %962 ], [ %1081, %991 ]
  %1001 = phi <4 x float> [ %981, %962 ], [ %1080, %991 ]
  %1002 = phi <4 x float> [ %980, %962 ], [ %1075, %991 ]
  %1003 = phi <4 x float> [ %979, %962 ], [ %1074, %991 ]
  %1004 = phi <4 x float> [ %978, %962 ], [ %1069, %991 ]
  %1005 = phi <4 x float> [ %977, %962 ], [ %1068, %991 ]
  %1006 = phi <4 x float> [ %976, %962 ], [ %1063, %991 ]
  %1007 = phi <4 x float> [ %975, %962 ], [ %1062, %991 ]
  %1008 = phi <4 x float> [ %974, %962 ], [ %1057, %991 ]
  %1009 = phi <4 x float> [ %973, %962 ], [ %1056, %991 ]
  %1010 = phi <4 x float> [ %972, %962 ], [ %1051, %991 ]
  %1011 = phi <4 x float> [ %971, %962 ], [ %1050, %991 ]
  %1012 = phi <4 x float> [ %970, %962 ], [ %1045, %991 ]
  %1013 = phi <4 x float> [ %969, %962 ], [ %1044, %991 ]
  %1014 = phi <4 x float> [ %968, %962 ], [ %1039, %991 ]
  %1015 = phi <4 x float> [ %967, %962 ], [ %1038, %991 ]
  %1016 = phi <4 x float> [ %966, %962 ], [ %1033, %991 ]
  %1017 = phi <4 x float> [ %965, %962 ], [ %1032, %991 ]
  %1018 = phi <4 x float> [ %964, %962 ], [ %1027, %991 ]
  %1019 = phi <4 x float> [ %963, %962 ], [ %1026, %991 ]
  %1020 = phi i64 [ 0, %962 ], [ %1106, %991 ]
  %1021 = mul nuw nsw i64 %1020, 804
  %1022 = getelementptr i8, ptr @main.a, i64 %1021
  %1023 = getelementptr i8, ptr %1022, i64 16
  %1024 = load <4 x float>, ptr %1022, align 4, !tbaa !11
  %1025 = load <4 x float>, ptr %1023, align 4, !tbaa !11
  %1026 = fadd <4 x float> %1019, %1024
  %1027 = fadd <4 x float> %1018, %1025
  %1028 = getelementptr i8, ptr %1022, i64 32
  %1029 = getelementptr i8, ptr %1022, i64 48
  %1030 = load <4 x float>, ptr %1028, align 4, !tbaa !11
  %1031 = load <4 x float>, ptr %1029, align 4, !tbaa !11
  %1032 = fadd <4 x float> %1017, %1030
  %1033 = fadd <4 x float> %1016, %1031
  %1034 = getelementptr i8, ptr %1022, i64 64
  %1035 = getelementptr i8, ptr %1022, i64 80
  %1036 = load <4 x float>, ptr %1034, align 4, !tbaa !11
  %1037 = load <4 x float>, ptr %1035, align 4, !tbaa !11
  %1038 = fadd <4 x float> %1015, %1036
  %1039 = fadd <4 x float> %1014, %1037
  %1040 = getelementptr i8, ptr %1022, i64 96
  %1041 = getelementptr i8, ptr %1022, i64 112
  %1042 = load <4 x float>, ptr %1040, align 4, !tbaa !11
  %1043 = load <4 x float>, ptr %1041, align 4, !tbaa !11
  %1044 = fadd <4 x float> %1013, %1042
  %1045 = fadd <4 x float> %1012, %1043
  %1046 = getelementptr i8, ptr %1022, i64 128
  %1047 = getelementptr i8, ptr %1022, i64 144
  %1048 = load <4 x float>, ptr %1046, align 4, !tbaa !11
  %1049 = load <4 x float>, ptr %1047, align 4, !tbaa !11
  %1050 = fadd <4 x float> %1011, %1048
  %1051 = fadd <4 x float> %1010, %1049
  %1052 = getelementptr i8, ptr %1022, i64 160
  %1053 = getelementptr i8, ptr %1022, i64 176
  %1054 = load <4 x float>, ptr %1052, align 4, !tbaa !11
  %1055 = load <4 x float>, ptr %1053, align 4, !tbaa !11
  %1056 = fadd <4 x float> %1009, %1054
  %1057 = fadd <4 x float> %1008, %1055
  %1058 = getelementptr i8, ptr %1022, i64 192
  %1059 = getelementptr i8, ptr %1022, i64 208
  %1060 = load <4 x float>, ptr %1058, align 4, !tbaa !11
  %1061 = load <4 x float>, ptr %1059, align 4, !tbaa !11
  %1062 = fadd <4 x float> %1007, %1060
  %1063 = fadd <4 x float> %1006, %1061
  %1064 = getelementptr i8, ptr %1022, i64 224
  %1065 = getelementptr i8, ptr %1022, i64 240
  %1066 = load <4 x float>, ptr %1064, align 4, !tbaa !11
  %1067 = load <4 x float>, ptr %1065, align 4, !tbaa !11
  %1068 = fadd <4 x float> %1005, %1066
  %1069 = fadd <4 x float> %1004, %1067
  %1070 = getelementptr i8, ptr %1022, i64 256
  %1071 = getelementptr i8, ptr %1022, i64 272
  %1072 = load <4 x float>, ptr %1070, align 4, !tbaa !11
  %1073 = load <4 x float>, ptr %1071, align 4, !tbaa !11
  %1074 = fadd <4 x float> %1003, %1072
  %1075 = fadd <4 x float> %1002, %1073
  %1076 = getelementptr i8, ptr %1022, i64 288
  %1077 = getelementptr i8, ptr %1022, i64 304
  %1078 = load <4 x float>, ptr %1076, align 4, !tbaa !11
  %1079 = load <4 x float>, ptr %1077, align 4, !tbaa !11
  %1080 = fadd <4 x float> %1001, %1078
  %1081 = fadd <4 x float> %1000, %1079
  %1082 = getelementptr i8, ptr %1022, i64 320
  %1083 = getelementptr i8, ptr %1022, i64 336
  %1084 = load <4 x float>, ptr %1082, align 4, !tbaa !11
  %1085 = load <4 x float>, ptr %1083, align 4, !tbaa !11
  %1086 = fadd <4 x float> %999, %1084
  %1087 = fadd <4 x float> %998, %1085
  %1088 = getelementptr i8, ptr %1022, i64 352
  %1089 = getelementptr i8, ptr %1022, i64 368
  %1090 = load <4 x float>, ptr %1088, align 4, !tbaa !11
  %1091 = load <4 x float>, ptr %1089, align 4, !tbaa !11
  %1092 = fadd <4 x float> %997, %1090
  %1093 = fadd <4 x float> %996, %1091
  %1094 = getelementptr i8, ptr %1022, i64 384
  %1095 = load float, ptr %1094, align 4, !tbaa !11
  %1096 = fadd float %995, %1095
  %1097 = getelementptr i8, ptr %1022, i64 388
  %1098 = load float, ptr %1097, align 4, !tbaa !11
  %1099 = fadd float %994, %1098
  %1100 = getelementptr i8, ptr %1022, i64 392
  %1101 = load float, ptr %1100, align 4, !tbaa !11
  %1102 = fadd float %993, %1101
  %1103 = getelementptr i8, ptr %1022, i64 396
  %1104 = load float, ptr %1103, align 4, !tbaa !11
  %1105 = fadd float %992, %1104
  %1106 = add nuw nsw i64 %1020, 1
  %1107 = icmp eq i64 %1106, 100
  br i1 %1107, label %1108, label %991, !llvm.loop !16

1108:                                             ; preds = %991
  store <4 x float> %1026, ptr @main.b, align 16, !tbaa !11
  store <4 x float> %1027, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  store <4 x float> %1032, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  store <4 x float> %1033, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  store <4 x float> %1038, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  store <4 x float> %1039, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  store <4 x float> %1044, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  store <4 x float> %1045, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  store <4 x float> %1050, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  store <4 x float> %1051, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  store <4 x float> %1056, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  store <4 x float> %1057, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  store <4 x float> %1062, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  store <4 x float> %1063, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  store <4 x float> %1068, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  store <4 x float> %1069, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  store <4 x float> %1074, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  store <4 x float> %1075, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  store <4 x float> %1080, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  store <4 x float> %1081, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  store <4 x float> %1086, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  store <4 x float> %1087, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  store <4 x float> %1092, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  store <4 x float> %1093, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  store float %1096, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  store float %1099, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  store float %1102, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  store float %1105, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  %1109 = add nuw nsw i32 %941, 1
  %1110 = icmp eq i32 %1109, 400
  br i1 %1110, label %1111, label %940, !llvm.loop !28

1111:                                             ; preds = %1108
  %1112 = tail call i64 @clock() #12
  %1113 = load ptr, ptr @stderr, align 8, !tbaa !6
  %1114 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %1113, ptr noundef nonnull @.str.15, i32 noundef 400, double noundef 0.000000e+00) #14
  %1115 = load ptr, ptr @stderr, align 8, !tbaa !6
  %1116 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %1115, ptr noundef nonnull @.str.16, double noundef 0.000000e+00) #14
  %1117 = load ptr, ptr @stderr, align 8, !tbaa !6
  %1118 = tail call i64 @fwrite(ptr nonnull @.str.17, i64 46, i64 1, ptr %1117) #13
  br label %1119

1119:                                             ; preds = %1433, %1111
  %1120 = phi i32 [ 100, %1111 ], [ %1435, %1433 ]
  %1121 = phi i32 [ -3, %1111 ], [ %1123, %1433 ]
  store i32 %1120, ptr @main.ntimes, align 4, !tbaa !17
  %1122 = tail call i64 @clock() #12
  %1123 = add nsw i32 %1121, 1
  %1124 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %1125 = icmp sgt i32 %1124, 0
  br i1 %1125, label %1126, label %1427

1126:                                             ; preds = %1119, %1420
  %1127 = phi i32 [ %1421, %1420 ], [ 0, %1119 ]
  br label %1128

1128:                                             ; preds = %1126, %1145
  %1129 = phi i64 [ %1146, %1145 ], [ 0, %1126 ]
  %1130 = phi i32 [ %1137, %1145 ], [ 1325, %1126 ]
  %1131 = mul nuw nsw i64 %1129, 804
  %1132 = getelementptr i8, ptr @main.a, i64 %1131
  br label %1133

1133:                                             ; preds = %1133, %1128
  %1134 = phi i64 [ 0, %1128 ], [ %1143, %1133 ]
  %1135 = phi i32 [ %1130, %1128 ], [ %1137, %1133 ]
  %1136 = mul nuw nsw i32 %1135, 3125
  %1137 = and i32 %1136, 65535
  %1138 = add nsw i32 %1137, -32768
  %1139 = sitofp i32 %1138 to double
  %1140 = fmul double %1139, 0x3F10000000000000
  %1141 = fptrunc double %1140 to float
  %1142 = getelementptr float, ptr %1132, i64 %1134
  store float %1141, ptr %1142, align 4, !tbaa !11
  %1143 = add nuw nsw i64 %1134, 1
  %1144 = icmp eq i64 %1143, 100
  br i1 %1144, label %1145, label %1133, !llvm.loop !13

1145:                                             ; preds = %1133
  %1146 = add nuw nsw i64 %1129, 1
  %1147 = icmp eq i64 %1146, 100
  br i1 %1147, label %1148, label %1128, !llvm.loop !15

1148:                                             ; preds = %1145
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(400) @main.b, i8 0, i64 400, i1 false), !tbaa !11
  %1149 = load <4 x float>, ptr @main.b, align 16, !tbaa !11
  %1150 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  %1151 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  %1152 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  %1153 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  %1154 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  %1155 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  %1156 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  %1157 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  %1158 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  %1159 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  %1160 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  %1161 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  %1162 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  %1163 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  %1164 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  %1165 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  %1166 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  %1167 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  %1168 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  %1169 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  %1170 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  %1171 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  %1172 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  %1173 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  %1174 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  %1175 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  %1176 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  br label %1177

1177:                                             ; preds = %1177, %1148
  %1178 = phi float [ %1176, %1148 ], [ %1291, %1177 ]
  %1179 = phi float [ %1175, %1148 ], [ %1288, %1177 ]
  %1180 = phi float [ %1174, %1148 ], [ %1285, %1177 ]
  %1181 = phi float [ %1173, %1148 ], [ %1282, %1177 ]
  %1182 = phi <4 x float> [ %1172, %1148 ], [ %1279, %1177 ]
  %1183 = phi <4 x float> [ %1171, %1148 ], [ %1278, %1177 ]
  %1184 = phi <4 x float> [ %1170, %1148 ], [ %1273, %1177 ]
  %1185 = phi <4 x float> [ %1169, %1148 ], [ %1272, %1177 ]
  %1186 = phi <4 x float> [ %1168, %1148 ], [ %1267, %1177 ]
  %1187 = phi <4 x float> [ %1167, %1148 ], [ %1266, %1177 ]
  %1188 = phi <4 x float> [ %1166, %1148 ], [ %1261, %1177 ]
  %1189 = phi <4 x float> [ %1165, %1148 ], [ %1260, %1177 ]
  %1190 = phi <4 x float> [ %1164, %1148 ], [ %1255, %1177 ]
  %1191 = phi <4 x float> [ %1163, %1148 ], [ %1254, %1177 ]
  %1192 = phi <4 x float> [ %1162, %1148 ], [ %1249, %1177 ]
  %1193 = phi <4 x float> [ %1161, %1148 ], [ %1248, %1177 ]
  %1194 = phi <4 x float> [ %1160, %1148 ], [ %1243, %1177 ]
  %1195 = phi <4 x float> [ %1159, %1148 ], [ %1242, %1177 ]
  %1196 = phi <4 x float> [ %1158, %1148 ], [ %1237, %1177 ]
  %1197 = phi <4 x float> [ %1157, %1148 ], [ %1236, %1177 ]
  %1198 = phi <4 x float> [ %1156, %1148 ], [ %1231, %1177 ]
  %1199 = phi <4 x float> [ %1155, %1148 ], [ %1230, %1177 ]
  %1200 = phi <4 x float> [ %1154, %1148 ], [ %1225, %1177 ]
  %1201 = phi <4 x float> [ %1153, %1148 ], [ %1224, %1177 ]
  %1202 = phi <4 x float> [ %1152, %1148 ], [ %1219, %1177 ]
  %1203 = phi <4 x float> [ %1151, %1148 ], [ %1218, %1177 ]
  %1204 = phi <4 x float> [ %1150, %1148 ], [ %1213, %1177 ]
  %1205 = phi <4 x float> [ %1149, %1148 ], [ %1212, %1177 ]
  %1206 = phi i64 [ 0, %1148 ], [ %1292, %1177 ]
  %1207 = mul nuw nsw i64 %1206, 804
  %1208 = getelementptr i8, ptr @main.a, i64 %1207
  %1209 = getelementptr i8, ptr %1208, i64 16
  %1210 = load <4 x float>, ptr %1208, align 4, !tbaa !11
  %1211 = load <4 x float>, ptr %1209, align 4, !tbaa !11
  %1212 = fadd <4 x float> %1205, %1210
  %1213 = fadd <4 x float> %1204, %1211
  %1214 = getelementptr i8, ptr %1208, i64 32
  %1215 = getelementptr i8, ptr %1208, i64 48
  %1216 = load <4 x float>, ptr %1214, align 4, !tbaa !11
  %1217 = load <4 x float>, ptr %1215, align 4, !tbaa !11
  %1218 = fadd <4 x float> %1203, %1216
  %1219 = fadd <4 x float> %1202, %1217
  %1220 = getelementptr i8, ptr %1208, i64 64
  %1221 = getelementptr i8, ptr %1208, i64 80
  %1222 = load <4 x float>, ptr %1220, align 4, !tbaa !11
  %1223 = load <4 x float>, ptr %1221, align 4, !tbaa !11
  %1224 = fadd <4 x float> %1201, %1222
  %1225 = fadd <4 x float> %1200, %1223
  %1226 = getelementptr i8, ptr %1208, i64 96
  %1227 = getelementptr i8, ptr %1208, i64 112
  %1228 = load <4 x float>, ptr %1226, align 4, !tbaa !11
  %1229 = load <4 x float>, ptr %1227, align 4, !tbaa !11
  %1230 = fadd <4 x float> %1199, %1228
  %1231 = fadd <4 x float> %1198, %1229
  %1232 = getelementptr i8, ptr %1208, i64 128
  %1233 = getelementptr i8, ptr %1208, i64 144
  %1234 = load <4 x float>, ptr %1232, align 4, !tbaa !11
  %1235 = load <4 x float>, ptr %1233, align 4, !tbaa !11
  %1236 = fadd <4 x float> %1197, %1234
  %1237 = fadd <4 x float> %1196, %1235
  %1238 = getelementptr i8, ptr %1208, i64 160
  %1239 = getelementptr i8, ptr %1208, i64 176
  %1240 = load <4 x float>, ptr %1238, align 4, !tbaa !11
  %1241 = load <4 x float>, ptr %1239, align 4, !tbaa !11
  %1242 = fadd <4 x float> %1195, %1240
  %1243 = fadd <4 x float> %1194, %1241
  %1244 = getelementptr i8, ptr %1208, i64 192
  %1245 = getelementptr i8, ptr %1208, i64 208
  %1246 = load <4 x float>, ptr %1244, align 4, !tbaa !11
  %1247 = load <4 x float>, ptr %1245, align 4, !tbaa !11
  %1248 = fadd <4 x float> %1193, %1246
  %1249 = fadd <4 x float> %1192, %1247
  %1250 = getelementptr i8, ptr %1208, i64 224
  %1251 = getelementptr i8, ptr %1208, i64 240
  %1252 = load <4 x float>, ptr %1250, align 4, !tbaa !11
  %1253 = load <4 x float>, ptr %1251, align 4, !tbaa !11
  %1254 = fadd <4 x float> %1191, %1252
  %1255 = fadd <4 x float> %1190, %1253
  %1256 = getelementptr i8, ptr %1208, i64 256
  %1257 = getelementptr i8, ptr %1208, i64 272
  %1258 = load <4 x float>, ptr %1256, align 4, !tbaa !11
  %1259 = load <4 x float>, ptr %1257, align 4, !tbaa !11
  %1260 = fadd <4 x float> %1189, %1258
  %1261 = fadd <4 x float> %1188, %1259
  %1262 = getelementptr i8, ptr %1208, i64 288
  %1263 = getelementptr i8, ptr %1208, i64 304
  %1264 = load <4 x float>, ptr %1262, align 4, !tbaa !11
  %1265 = load <4 x float>, ptr %1263, align 4, !tbaa !11
  %1266 = fadd <4 x float> %1187, %1264
  %1267 = fadd <4 x float> %1186, %1265
  %1268 = getelementptr i8, ptr %1208, i64 320
  %1269 = getelementptr i8, ptr %1208, i64 336
  %1270 = load <4 x float>, ptr %1268, align 4, !tbaa !11
  %1271 = load <4 x float>, ptr %1269, align 4, !tbaa !11
  %1272 = fadd <4 x float> %1185, %1270
  %1273 = fadd <4 x float> %1184, %1271
  %1274 = getelementptr i8, ptr %1208, i64 352
  %1275 = getelementptr i8, ptr %1208, i64 368
  %1276 = load <4 x float>, ptr %1274, align 4, !tbaa !11
  %1277 = load <4 x float>, ptr %1275, align 4, !tbaa !11
  %1278 = fadd <4 x float> %1183, %1276
  %1279 = fadd <4 x float> %1182, %1277
  %1280 = getelementptr i8, ptr %1208, i64 384
  %1281 = load float, ptr %1280, align 4, !tbaa !11
  %1282 = fadd float %1181, %1281
  %1283 = getelementptr i8, ptr %1208, i64 388
  %1284 = load float, ptr %1283, align 4, !tbaa !11
  %1285 = fadd float %1180, %1284
  %1286 = getelementptr i8, ptr %1208, i64 392
  %1287 = load float, ptr %1286, align 4, !tbaa !11
  %1288 = fadd float %1179, %1287
  %1289 = getelementptr i8, ptr %1208, i64 396
  %1290 = load float, ptr %1289, align 4, !tbaa !11
  %1291 = fadd float %1178, %1290
  %1292 = add nuw nsw i64 %1206, 1
  %1293 = icmp eq i64 %1292, 100
  br i1 %1293, label %1294, label %1177, !llvm.loop !16

1294:                                             ; preds = %1177
  store <4 x float> %1212, ptr @main.b, align 16, !tbaa !11
  store <4 x float> %1213, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  store <4 x float> %1218, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  store <4 x float> %1219, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  store <4 x float> %1224, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  store <4 x float> %1225, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  store <4 x float> %1230, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  store <4 x float> %1231, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  store <4 x float> %1236, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  store <4 x float> %1237, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  store <4 x float> %1242, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  store <4 x float> %1243, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  store <4 x float> %1248, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  store <4 x float> %1249, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  store <4 x float> %1254, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  store <4 x float> %1255, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  store <4 x float> %1260, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  store <4 x float> %1261, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  store <4 x float> %1266, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  store <4 x float> %1267, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  store <4 x float> %1272, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  store <4 x float> %1273, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  store <4 x float> %1278, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  store <4 x float> %1279, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  store float %1282, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  store float %1285, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  store float %1288, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  store float %1291, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  br label %1295

1295:                                             ; preds = %1294, %1416
  %1296 = phi i32 [ %1417, %1416 ], [ 0, %1294 ]
  %1297 = phi i64 [ %1302, %1416 ], [ 0, %1294 ]
  %1298 = phi i64 [ %1418, %1416 ], [ 1, %1294 ]
  %1299 = sub nsw i64 99, %1297
  %1300 = sub nsw i64 99, %1297
  %1301 = trunc i64 %1297 to i32
  %1302 = add nuw nsw i64 %1297, 1
  %1303 = sub nuw nsw i64 100, %1297
  %1304 = getelementptr float, ptr @main.a, i64 %1297
  %1305 = mul nuw nsw i64 %1297, 804
  %1306 = getelementptr i8, ptr %1304, i64 %1305
  %1307 = load float, ptr %1306, align 4, !tbaa !11
  %1308 = tail call float @llvm.fabs.f32(float %1307)
  br label %1309

1309:                                             ; preds = %1309, %1295
  %1310 = phi i64 [ 1, %1295 ], [ %1320, %1309 ]
  %1311 = phi i32 [ 0, %1295 ], [ %1319, %1309 ]
  %1312 = phi float [ %1308, %1295 ], [ %1317, %1309 ]
  %1313 = getelementptr inbounds nuw float, ptr %1306, i64 %1310
  %1314 = load float, ptr %1313, align 4, !tbaa !11
  %1315 = tail call float @llvm.fabs.f32(float %1314)
  %1316 = fcmp ogt float %1315, %1312
  %1317 = select i1 %1316, float %1315, float %1312
  %1318 = trunc nuw nsw i64 %1310 to i32
  %1319 = select i1 %1316, i32 %1318, i32 %1311
  %1320 = add nuw nsw i64 %1310, 1
  %1321 = icmp eq i64 %1320, %1303
  br i1 %1321, label %1322, label %1309, !llvm.loop !29

1322:                                             ; preds = %1309
  %1323 = add nsw i32 %1319, %1301
  %1324 = getelementptr inbounds nuw i32, ptr @main.ipvt, i64 %1297
  store i32 %1323, ptr %1324, align 4, !tbaa !17
  %1325 = sext i32 %1323 to i64
  %1326 = mul i64 %1297, 804
  %1327 = getelementptr i8, ptr @main.a, i64 %1326
  %1328 = getelementptr float, ptr %1327, i64 %1325
  %1329 = load float, ptr %1328, align 4, !tbaa !11
  %1330 = fcmp une float %1329, 0.000000e+00
  br i1 %1330, label %1331, label %1416

1331:                                             ; preds = %1322
  %1332 = icmp eq i32 %1319, 0
  br i1 %1332, label %1334, label %1333

1333:                                             ; preds = %1331
  store float %1307, ptr %1328, align 4, !tbaa !11
  store float %1329, ptr %1306, align 4, !tbaa !11
  br label %1334

1334:                                             ; preds = %1333, %1331
  %1335 = phi float [ %1329, %1333 ], [ %1307, %1331 ]
  %1336 = fdiv float -1.000000e+00, %1335
  %1337 = sub nuw nsw i64 99, %1297
  %1338 = getelementptr i8, ptr %1306, i64 4
  %1339 = icmp ult i64 %1299, 8
  br i1 %1339, label %1356, label %1340

1340:                                             ; preds = %1334
  %1341 = and i64 %1299, -8
  %1342 = insertelement <4 x float> poison, float %1336, i64 0
  %1343 = shufflevector <4 x float> %1342, <4 x float> poison, <4 x i32> zeroinitializer
  br label %1344

1344:                                             ; preds = %1344, %1340
  %1345 = phi i64 [ 0, %1340 ], [ %1352, %1344 ]
  %1346 = getelementptr inbounds nuw float, ptr %1338, i64 %1345
  %1347 = getelementptr inbounds nuw i8, ptr %1346, i64 16
  %1348 = load <4 x float>, ptr %1346, align 4, !tbaa !11
  %1349 = load <4 x float>, ptr %1347, align 4, !tbaa !11
  %1350 = fmul <4 x float> %1343, %1348
  %1351 = fmul <4 x float> %1343, %1349
  store <4 x float> %1350, ptr %1346, align 4, !tbaa !11
  store <4 x float> %1351, ptr %1347, align 4, !tbaa !11
  %1352 = add nuw i64 %1345, 8
  %1353 = icmp eq i64 %1352, %1341
  br i1 %1353, label %1354, label %1344, !llvm.loop !30

1354:                                             ; preds = %1344
  %1355 = icmp eq i64 %1299, %1341
  br i1 %1355, label %1365, label %1356

1356:                                             ; preds = %1334, %1354
  %1357 = phi i64 [ 0, %1334 ], [ %1341, %1354 ]
  br label %1358

1358:                                             ; preds = %1356, %1358
  %1359 = phi i64 [ %1363, %1358 ], [ %1357, %1356 ]
  %1360 = getelementptr inbounds nuw float, ptr %1338, i64 %1359
  %1361 = load float, ptr %1360, align 4, !tbaa !11
  %1362 = fmul float %1336, %1361
  store float %1362, ptr %1360, align 4, !tbaa !11
  %1363 = add nuw nsw i64 %1359, 1
  %1364 = icmp eq i64 %1363, %1337
  br i1 %1364, label %1365, label %1358, !llvm.loop !31

1365:                                             ; preds = %1358, %1354
  %1366 = getelementptr float, ptr @main.a, i64 %1325
  %1367 = icmp ult i64 %1300, 8
  %1368 = and i64 %1300, -8
  %1369 = icmp eq i64 %1300, %1368
  br label %1370

1370:                                             ; preds = %1413, %1365
  %1371 = phi i64 [ %1298, %1365 ], [ %1414, %1413 ]
  %1372 = mul nuw nsw i64 %1371, 201
  %1373 = getelementptr float, ptr %1366, i64 %1372
  %1374 = load float, ptr %1373, align 4, !tbaa !11
  %1375 = add nuw nsw i64 %1372, %1297
  br i1 %1332, label %1379, label %1376

1376:                                             ; preds = %1370
  %1377 = getelementptr inbounds nuw float, ptr @main.a, i64 %1375
  %1378 = load float, ptr %1377, align 4, !tbaa !11
  store float %1378, ptr %1373, align 4, !tbaa !11
  store float %1374, ptr %1377, align 4, !tbaa !11
  br label %1379

1379:                                             ; preds = %1376, %1370
  %1380 = getelementptr float, ptr @main.a, i64 %1375
  %1381 = getelementptr i8, ptr %1380, i64 4
  %1382 = fcmp oeq float %1374, 0.000000e+00
  br i1 %1382, label %1413, label %1383

1383:                                             ; preds = %1379
  br i1 %1367, label %1402, label %1384

1384:                                             ; preds = %1383
  %1385 = insertelement <4 x float> poison, float %1374, i64 0
  %1386 = shufflevector <4 x float> %1385, <4 x float> poison, <4 x i32> zeroinitializer
  br label %1387

1387:                                             ; preds = %1387, %1384
  %1388 = phi i64 [ 0, %1384 ], [ %1399, %1387 ]
  %1389 = getelementptr inbounds nuw float, ptr %1381, i64 %1388
  %1390 = getelementptr inbounds nuw i8, ptr %1389, i64 16
  %1391 = load <4 x float>, ptr %1389, align 4, !tbaa !11
  %1392 = load <4 x float>, ptr %1390, align 4, !tbaa !11
  %1393 = getelementptr inbounds nuw float, ptr %1338, i64 %1388
  %1394 = getelementptr inbounds nuw i8, ptr %1393, i64 16
  %1395 = load <4 x float>, ptr %1393, align 4, !tbaa !11
  %1396 = load <4 x float>, ptr %1394, align 4, !tbaa !11
  %1397 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %1386, <4 x float> %1395, <4 x float> %1391)
  %1398 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %1386, <4 x float> %1396, <4 x float> %1392)
  store <4 x float> %1397, ptr %1389, align 4, !tbaa !11
  store <4 x float> %1398, ptr %1390, align 4, !tbaa !11
  %1399 = add nuw i64 %1388, 8
  %1400 = icmp eq i64 %1399, %1368
  br i1 %1400, label %1401, label %1387, !llvm.loop !32

1401:                                             ; preds = %1387
  br i1 %1369, label %1413, label %1402

1402:                                             ; preds = %1383, %1401
  %1403 = phi i64 [ 0, %1383 ], [ %1368, %1401 ]
  br label %1404

1404:                                             ; preds = %1402, %1404
  %1405 = phi i64 [ %1411, %1404 ], [ %1403, %1402 ]
  %1406 = getelementptr inbounds nuw float, ptr %1381, i64 %1405
  %1407 = load float, ptr %1406, align 4, !tbaa !11
  %1408 = getelementptr inbounds nuw float, ptr %1338, i64 %1405
  %1409 = load float, ptr %1408, align 4, !tbaa !11
  %1410 = tail call float @llvm.fmuladd.f32(float %1374, float %1409, float %1407)
  store float %1410, ptr %1406, align 4, !tbaa !11
  %1411 = add nuw nsw i64 %1405, 1
  %1412 = icmp eq i64 %1411, %1337
  br i1 %1412, label %1413, label %1404, !llvm.loop !33

1413:                                             ; preds = %1404, %1401, %1379
  %1414 = add nuw nsw i64 %1371, 1
  %1415 = icmp eq i64 %1414, 100
  br i1 %1415, label %1416, label %1370, !llvm.loop !34

1416:                                             ; preds = %1413, %1322
  %1417 = phi i32 [ %1301, %1322 ], [ %1296, %1413 ]
  %1418 = add nuw nsw i64 %1298, 1
  %1419 = icmp eq i64 %1302, 99
  br i1 %1419, label %1420, label %1295, !llvm.loop !35

1420:                                             ; preds = %1416
  store i32 99, ptr getelementptr inbounds nuw (i8, ptr @main.ipvt, i64 396), align 4, !tbaa !17
  %1421 = add nuw nsw i32 %1127, 1
  %1422 = icmp eq i32 %1421, %1124
  br i1 %1422, label %1423, label %1126, !llvm.loop !36

1423:                                             ; preds = %1420
  %1424 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.a, i64 79992), align 4, !tbaa !11
  %1425 = fcmp oeq float %1424, 0.000000e+00
  %1426 = select i1 %1425, i32 99, i32 %1417
  store i32 %1426, ptr @main.info, align 4, !tbaa !17
  br label %1427

1427:                                             ; preds = %1423, %1119
  %1428 = tail call i64 @clock() #12
  %1429 = load ptr, ptr @stderr, align 8, !tbaa !6
  %1430 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %1431 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %1429, ptr noundef nonnull @.str.15, i32 noundef %1430, double noundef 0.000000e+00) #14
  %1432 = icmp eq i32 %1121, -1
  br i1 %1432, label %1436, label %1433

1433:                                             ; preds = %1427
  %1434 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %1435 = shl nsw i32 %1434, 1
  br label %1119, !llvm.loop !37

1436:                                             ; preds = %1427
  %1437 = sitofp i64 %939 to float
  %1438 = fdiv float %1437, 1.000000e+06
  %1439 = sitofp i64 %1112 to float
  %1440 = fdiv float %1439, 1.000000e+06
  %1441 = fsub float %1440, %1438
  %1442 = fdiv float %1441, 4.000000e+02
  store i32 1000, ptr @main.ntimes, align 4
  %1443 = load ptr, ptr @stderr, align 8, !tbaa !6
  %1444 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %1443, ptr noundef nonnull @.str.18, i32 noundef 0) #14
  %1445 = load ptr, ptr @stderr, align 8, !tbaa !6
  %1446 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %1445, ptr noundef nonnull @.str.19, i32 noundef 201) #14
  %1447 = load ptr, ptr @stderr, align 8, !tbaa !6
  %1448 = tail call i64 @fwrite(ptr nonnull @.str.12, i64 55, i64 1, ptr %1447) #13
  %1449 = load ptr, ptr @stderr, align 8, !tbaa !6
  %1450 = tail call i64 @fwrite(ptr nonnull @.str.13, i64 12, i64 1, ptr %1449) #13
  %1451 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %1452 = sitofp i32 %1451 to float
  %1453 = fmul float %1442, %1452
  store float 0.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 204), align 4, !tbaa !11
  store i32 1, ptr @main.j, align 4, !tbaa !17
  br label %1454

1454:                                             ; preds = %1436, %1892
  %1455 = tail call i64 @clock() #12
  %1456 = sitofp i64 %1455 to float
  %1457 = fdiv float %1456, 1.000000e+06
  %1458 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %1459 = icmp sgt i32 %1458, 0
  br i1 %1459, label %1460, label %1761

1460:                                             ; preds = %1454, %1754
  %1461 = phi i32 [ %1755, %1754 ], [ 0, %1454 ]
  br label %1462

1462:                                             ; preds = %1460, %1479
  %1463 = phi i64 [ %1480, %1479 ], [ 0, %1460 ]
  %1464 = phi i32 [ %1471, %1479 ], [ 1325, %1460 ]
  %1465 = mul nuw nsw i64 %1463, 804
  %1466 = getelementptr i8, ptr @main.a, i64 %1465
  br label %1467

1467:                                             ; preds = %1467, %1462
  %1468 = phi i64 [ 0, %1462 ], [ %1477, %1467 ]
  %1469 = phi i32 [ %1464, %1462 ], [ %1471, %1467 ]
  %1470 = mul nuw nsw i32 %1469, 3125
  %1471 = and i32 %1470, 65535
  %1472 = add nsw i32 %1471, -32768
  %1473 = sitofp i32 %1472 to double
  %1474 = fmul double %1473, 0x3F10000000000000
  %1475 = fptrunc double %1474 to float
  %1476 = getelementptr float, ptr %1466, i64 %1468
  store float %1475, ptr %1476, align 4, !tbaa !11
  %1477 = add nuw nsw i64 %1468, 1
  %1478 = icmp eq i64 %1477, 100
  br i1 %1478, label %1479, label %1467, !llvm.loop !13

1479:                                             ; preds = %1467
  %1480 = add nuw nsw i64 %1463, 1
  %1481 = icmp eq i64 %1480, 100
  br i1 %1481, label %1482, label %1462, !llvm.loop !15

1482:                                             ; preds = %1479
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(400) @main.b, i8 0, i64 400, i1 false), !tbaa !11
  %1483 = load <4 x float>, ptr @main.b, align 16, !tbaa !11
  %1484 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  %1485 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  %1486 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  %1487 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  %1488 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  %1489 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  %1490 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  %1491 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  %1492 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  %1493 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  %1494 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  %1495 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  %1496 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  %1497 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  %1498 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  %1499 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  %1500 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  %1501 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  %1502 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  %1503 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  %1504 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  %1505 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  %1506 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  %1507 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  %1508 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  %1509 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  %1510 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  br label %1511

1511:                                             ; preds = %1511, %1482
  %1512 = phi float [ %1510, %1482 ], [ %1625, %1511 ]
  %1513 = phi float [ %1509, %1482 ], [ %1622, %1511 ]
  %1514 = phi float [ %1508, %1482 ], [ %1619, %1511 ]
  %1515 = phi float [ %1507, %1482 ], [ %1616, %1511 ]
  %1516 = phi <4 x float> [ %1506, %1482 ], [ %1613, %1511 ]
  %1517 = phi <4 x float> [ %1505, %1482 ], [ %1612, %1511 ]
  %1518 = phi <4 x float> [ %1504, %1482 ], [ %1607, %1511 ]
  %1519 = phi <4 x float> [ %1503, %1482 ], [ %1606, %1511 ]
  %1520 = phi <4 x float> [ %1502, %1482 ], [ %1601, %1511 ]
  %1521 = phi <4 x float> [ %1501, %1482 ], [ %1600, %1511 ]
  %1522 = phi <4 x float> [ %1500, %1482 ], [ %1595, %1511 ]
  %1523 = phi <4 x float> [ %1499, %1482 ], [ %1594, %1511 ]
  %1524 = phi <4 x float> [ %1498, %1482 ], [ %1589, %1511 ]
  %1525 = phi <4 x float> [ %1497, %1482 ], [ %1588, %1511 ]
  %1526 = phi <4 x float> [ %1496, %1482 ], [ %1583, %1511 ]
  %1527 = phi <4 x float> [ %1495, %1482 ], [ %1582, %1511 ]
  %1528 = phi <4 x float> [ %1494, %1482 ], [ %1577, %1511 ]
  %1529 = phi <4 x float> [ %1493, %1482 ], [ %1576, %1511 ]
  %1530 = phi <4 x float> [ %1492, %1482 ], [ %1571, %1511 ]
  %1531 = phi <4 x float> [ %1491, %1482 ], [ %1570, %1511 ]
  %1532 = phi <4 x float> [ %1490, %1482 ], [ %1565, %1511 ]
  %1533 = phi <4 x float> [ %1489, %1482 ], [ %1564, %1511 ]
  %1534 = phi <4 x float> [ %1488, %1482 ], [ %1559, %1511 ]
  %1535 = phi <4 x float> [ %1487, %1482 ], [ %1558, %1511 ]
  %1536 = phi <4 x float> [ %1486, %1482 ], [ %1553, %1511 ]
  %1537 = phi <4 x float> [ %1485, %1482 ], [ %1552, %1511 ]
  %1538 = phi <4 x float> [ %1484, %1482 ], [ %1547, %1511 ]
  %1539 = phi <4 x float> [ %1483, %1482 ], [ %1546, %1511 ]
  %1540 = phi i64 [ 0, %1482 ], [ %1626, %1511 ]
  %1541 = mul nuw nsw i64 %1540, 804
  %1542 = getelementptr i8, ptr @main.a, i64 %1541
  %1543 = getelementptr i8, ptr %1542, i64 16
  %1544 = load <4 x float>, ptr %1542, align 4, !tbaa !11
  %1545 = load <4 x float>, ptr %1543, align 4, !tbaa !11
  %1546 = fadd <4 x float> %1539, %1544
  %1547 = fadd <4 x float> %1538, %1545
  %1548 = getelementptr i8, ptr %1542, i64 32
  %1549 = getelementptr i8, ptr %1542, i64 48
  %1550 = load <4 x float>, ptr %1548, align 4, !tbaa !11
  %1551 = load <4 x float>, ptr %1549, align 4, !tbaa !11
  %1552 = fadd <4 x float> %1537, %1550
  %1553 = fadd <4 x float> %1536, %1551
  %1554 = getelementptr i8, ptr %1542, i64 64
  %1555 = getelementptr i8, ptr %1542, i64 80
  %1556 = load <4 x float>, ptr %1554, align 4, !tbaa !11
  %1557 = load <4 x float>, ptr %1555, align 4, !tbaa !11
  %1558 = fadd <4 x float> %1535, %1556
  %1559 = fadd <4 x float> %1534, %1557
  %1560 = getelementptr i8, ptr %1542, i64 96
  %1561 = getelementptr i8, ptr %1542, i64 112
  %1562 = load <4 x float>, ptr %1560, align 4, !tbaa !11
  %1563 = load <4 x float>, ptr %1561, align 4, !tbaa !11
  %1564 = fadd <4 x float> %1533, %1562
  %1565 = fadd <4 x float> %1532, %1563
  %1566 = getelementptr i8, ptr %1542, i64 128
  %1567 = getelementptr i8, ptr %1542, i64 144
  %1568 = load <4 x float>, ptr %1566, align 4, !tbaa !11
  %1569 = load <4 x float>, ptr %1567, align 4, !tbaa !11
  %1570 = fadd <4 x float> %1531, %1568
  %1571 = fadd <4 x float> %1530, %1569
  %1572 = getelementptr i8, ptr %1542, i64 160
  %1573 = getelementptr i8, ptr %1542, i64 176
  %1574 = load <4 x float>, ptr %1572, align 4, !tbaa !11
  %1575 = load <4 x float>, ptr %1573, align 4, !tbaa !11
  %1576 = fadd <4 x float> %1529, %1574
  %1577 = fadd <4 x float> %1528, %1575
  %1578 = getelementptr i8, ptr %1542, i64 192
  %1579 = getelementptr i8, ptr %1542, i64 208
  %1580 = load <4 x float>, ptr %1578, align 4, !tbaa !11
  %1581 = load <4 x float>, ptr %1579, align 4, !tbaa !11
  %1582 = fadd <4 x float> %1527, %1580
  %1583 = fadd <4 x float> %1526, %1581
  %1584 = getelementptr i8, ptr %1542, i64 224
  %1585 = getelementptr i8, ptr %1542, i64 240
  %1586 = load <4 x float>, ptr %1584, align 4, !tbaa !11
  %1587 = load <4 x float>, ptr %1585, align 4, !tbaa !11
  %1588 = fadd <4 x float> %1525, %1586
  %1589 = fadd <4 x float> %1524, %1587
  %1590 = getelementptr i8, ptr %1542, i64 256
  %1591 = getelementptr i8, ptr %1542, i64 272
  %1592 = load <4 x float>, ptr %1590, align 4, !tbaa !11
  %1593 = load <4 x float>, ptr %1591, align 4, !tbaa !11
  %1594 = fadd <4 x float> %1523, %1592
  %1595 = fadd <4 x float> %1522, %1593
  %1596 = getelementptr i8, ptr %1542, i64 288
  %1597 = getelementptr i8, ptr %1542, i64 304
  %1598 = load <4 x float>, ptr %1596, align 4, !tbaa !11
  %1599 = load <4 x float>, ptr %1597, align 4, !tbaa !11
  %1600 = fadd <4 x float> %1521, %1598
  %1601 = fadd <4 x float> %1520, %1599
  %1602 = getelementptr i8, ptr %1542, i64 320
  %1603 = getelementptr i8, ptr %1542, i64 336
  %1604 = load <4 x float>, ptr %1602, align 4, !tbaa !11
  %1605 = load <4 x float>, ptr %1603, align 4, !tbaa !11
  %1606 = fadd <4 x float> %1519, %1604
  %1607 = fadd <4 x float> %1518, %1605
  %1608 = getelementptr i8, ptr %1542, i64 352
  %1609 = getelementptr i8, ptr %1542, i64 368
  %1610 = load <4 x float>, ptr %1608, align 4, !tbaa !11
  %1611 = load <4 x float>, ptr %1609, align 4, !tbaa !11
  %1612 = fadd <4 x float> %1517, %1610
  %1613 = fadd <4 x float> %1516, %1611
  %1614 = getelementptr i8, ptr %1542, i64 384
  %1615 = load float, ptr %1614, align 4, !tbaa !11
  %1616 = fadd float %1515, %1615
  %1617 = getelementptr i8, ptr %1542, i64 388
  %1618 = load float, ptr %1617, align 4, !tbaa !11
  %1619 = fadd float %1514, %1618
  %1620 = getelementptr i8, ptr %1542, i64 392
  %1621 = load float, ptr %1620, align 4, !tbaa !11
  %1622 = fadd float %1513, %1621
  %1623 = getelementptr i8, ptr %1542, i64 396
  %1624 = load float, ptr %1623, align 4, !tbaa !11
  %1625 = fadd float %1512, %1624
  %1626 = add nuw nsw i64 %1540, 1
  %1627 = icmp eq i64 %1626, 100
  br i1 %1627, label %1628, label %1511, !llvm.loop !16

1628:                                             ; preds = %1511
  store <4 x float> %1546, ptr @main.b, align 16, !tbaa !11
  store <4 x float> %1547, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  store <4 x float> %1552, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  store <4 x float> %1553, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  store <4 x float> %1558, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  store <4 x float> %1559, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  store <4 x float> %1564, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  store <4 x float> %1565, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  store <4 x float> %1570, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  store <4 x float> %1571, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  store <4 x float> %1576, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  store <4 x float> %1577, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  store <4 x float> %1582, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  store <4 x float> %1583, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  store <4 x float> %1588, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  store <4 x float> %1589, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  store <4 x float> %1594, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  store <4 x float> %1595, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  store <4 x float> %1600, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  store <4 x float> %1601, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  store <4 x float> %1606, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  store <4 x float> %1607, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  store <4 x float> %1612, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  store <4 x float> %1613, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  store float %1616, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  store float %1619, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  store float %1622, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  store float %1625, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  br label %1629

1629:                                             ; preds = %1628, %1750
  %1630 = phi i32 [ %1751, %1750 ], [ 0, %1628 ]
  %1631 = phi i64 [ %1636, %1750 ], [ 0, %1628 ]
  %1632 = phi i64 [ %1752, %1750 ], [ 1, %1628 ]
  %1633 = sub nsw i64 99, %1631
  %1634 = sub nsw i64 99, %1631
  %1635 = trunc i64 %1631 to i32
  %1636 = add nuw nsw i64 %1631, 1
  %1637 = sub nuw nsw i64 100, %1631
  %1638 = getelementptr float, ptr @main.a, i64 %1631
  %1639 = mul nuw nsw i64 %1631, 804
  %1640 = getelementptr i8, ptr %1638, i64 %1639
  %1641 = load float, ptr %1640, align 4, !tbaa !11
  %1642 = tail call float @llvm.fabs.f32(float %1641)
  br label %1643

1643:                                             ; preds = %1643, %1629
  %1644 = phi i64 [ 1, %1629 ], [ %1654, %1643 ]
  %1645 = phi i32 [ 0, %1629 ], [ %1653, %1643 ]
  %1646 = phi float [ %1642, %1629 ], [ %1651, %1643 ]
  %1647 = getelementptr inbounds nuw float, ptr %1640, i64 %1644
  %1648 = load float, ptr %1647, align 4, !tbaa !11
  %1649 = tail call float @llvm.fabs.f32(float %1648)
  %1650 = fcmp ogt float %1649, %1646
  %1651 = select i1 %1650, float %1649, float %1646
  %1652 = trunc nuw nsw i64 %1644 to i32
  %1653 = select i1 %1650, i32 %1652, i32 %1645
  %1654 = add nuw nsw i64 %1644, 1
  %1655 = icmp eq i64 %1654, %1637
  br i1 %1655, label %1656, label %1643, !llvm.loop !29

1656:                                             ; preds = %1643
  %1657 = add nsw i32 %1653, %1635
  %1658 = getelementptr inbounds nuw i32, ptr @main.ipvt, i64 %1631
  store i32 %1657, ptr %1658, align 4, !tbaa !17
  %1659 = sext i32 %1657 to i64
  %1660 = mul i64 %1631, 804
  %1661 = getelementptr i8, ptr @main.a, i64 %1660
  %1662 = getelementptr float, ptr %1661, i64 %1659
  %1663 = load float, ptr %1662, align 4, !tbaa !11
  %1664 = fcmp une float %1663, 0.000000e+00
  br i1 %1664, label %1665, label %1750

1665:                                             ; preds = %1656
  %1666 = icmp eq i32 %1653, 0
  br i1 %1666, label %1668, label %1667

1667:                                             ; preds = %1665
  store float %1641, ptr %1662, align 4, !tbaa !11
  store float %1663, ptr %1640, align 4, !tbaa !11
  br label %1668

1668:                                             ; preds = %1667, %1665
  %1669 = phi float [ %1663, %1667 ], [ %1641, %1665 ]
  %1670 = fdiv float -1.000000e+00, %1669
  %1671 = sub nuw nsw i64 99, %1631
  %1672 = getelementptr i8, ptr %1640, i64 4
  %1673 = icmp ult i64 %1633, 8
  br i1 %1673, label %1690, label %1674

1674:                                             ; preds = %1668
  %1675 = and i64 %1633, -8
  %1676 = insertelement <4 x float> poison, float %1670, i64 0
  %1677 = shufflevector <4 x float> %1676, <4 x float> poison, <4 x i32> zeroinitializer
  br label %1678

1678:                                             ; preds = %1678, %1674
  %1679 = phi i64 [ 0, %1674 ], [ %1686, %1678 ]
  %1680 = getelementptr inbounds nuw float, ptr %1672, i64 %1679
  %1681 = getelementptr inbounds nuw i8, ptr %1680, i64 16
  %1682 = load <4 x float>, ptr %1680, align 4, !tbaa !11
  %1683 = load <4 x float>, ptr %1681, align 4, !tbaa !11
  %1684 = fmul <4 x float> %1677, %1682
  %1685 = fmul <4 x float> %1677, %1683
  store <4 x float> %1684, ptr %1680, align 4, !tbaa !11
  store <4 x float> %1685, ptr %1681, align 4, !tbaa !11
  %1686 = add nuw i64 %1679, 8
  %1687 = icmp eq i64 %1686, %1675
  br i1 %1687, label %1688, label %1678, !llvm.loop !38

1688:                                             ; preds = %1678
  %1689 = icmp eq i64 %1633, %1675
  br i1 %1689, label %1699, label %1690

1690:                                             ; preds = %1668, %1688
  %1691 = phi i64 [ 0, %1668 ], [ %1675, %1688 ]
  br label %1692

1692:                                             ; preds = %1690, %1692
  %1693 = phi i64 [ %1697, %1692 ], [ %1691, %1690 ]
  %1694 = getelementptr inbounds nuw float, ptr %1672, i64 %1693
  %1695 = load float, ptr %1694, align 4, !tbaa !11
  %1696 = fmul float %1670, %1695
  store float %1696, ptr %1694, align 4, !tbaa !11
  %1697 = add nuw nsw i64 %1693, 1
  %1698 = icmp eq i64 %1697, %1671
  br i1 %1698, label %1699, label %1692, !llvm.loop !39

1699:                                             ; preds = %1692, %1688
  %1700 = getelementptr float, ptr @main.a, i64 %1659
  %1701 = icmp ult i64 %1634, 8
  %1702 = and i64 %1634, -8
  %1703 = icmp eq i64 %1634, %1702
  br label %1704

1704:                                             ; preds = %1747, %1699
  %1705 = phi i64 [ %1632, %1699 ], [ %1748, %1747 ]
  %1706 = mul nuw nsw i64 %1705, 201
  %1707 = getelementptr float, ptr %1700, i64 %1706
  %1708 = load float, ptr %1707, align 4, !tbaa !11
  %1709 = add nuw nsw i64 %1706, %1631
  br i1 %1666, label %1713, label %1710

1710:                                             ; preds = %1704
  %1711 = getelementptr inbounds nuw float, ptr @main.a, i64 %1709
  %1712 = load float, ptr %1711, align 4, !tbaa !11
  store float %1712, ptr %1707, align 4, !tbaa !11
  store float %1708, ptr %1711, align 4, !tbaa !11
  br label %1713

1713:                                             ; preds = %1710, %1704
  %1714 = getelementptr float, ptr @main.a, i64 %1709
  %1715 = getelementptr i8, ptr %1714, i64 4
  %1716 = fcmp oeq float %1708, 0.000000e+00
  br i1 %1716, label %1747, label %1717

1717:                                             ; preds = %1713
  br i1 %1701, label %1736, label %1718

1718:                                             ; preds = %1717
  %1719 = insertelement <4 x float> poison, float %1708, i64 0
  %1720 = shufflevector <4 x float> %1719, <4 x float> poison, <4 x i32> zeroinitializer
  br label %1721

1721:                                             ; preds = %1721, %1718
  %1722 = phi i64 [ 0, %1718 ], [ %1733, %1721 ]
  %1723 = getelementptr inbounds nuw float, ptr %1715, i64 %1722
  %1724 = getelementptr inbounds nuw i8, ptr %1723, i64 16
  %1725 = load <4 x float>, ptr %1723, align 4, !tbaa !11
  %1726 = load <4 x float>, ptr %1724, align 4, !tbaa !11
  %1727 = getelementptr inbounds nuw float, ptr %1672, i64 %1722
  %1728 = getelementptr inbounds nuw i8, ptr %1727, i64 16
  %1729 = load <4 x float>, ptr %1727, align 4, !tbaa !11
  %1730 = load <4 x float>, ptr %1728, align 4, !tbaa !11
  %1731 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %1720, <4 x float> %1729, <4 x float> %1725)
  %1732 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %1720, <4 x float> %1730, <4 x float> %1726)
  store <4 x float> %1731, ptr %1723, align 4, !tbaa !11
  store <4 x float> %1732, ptr %1724, align 4, !tbaa !11
  %1733 = add nuw i64 %1722, 8
  %1734 = icmp eq i64 %1733, %1702
  br i1 %1734, label %1735, label %1721, !llvm.loop !40

1735:                                             ; preds = %1721
  br i1 %1703, label %1747, label %1736

1736:                                             ; preds = %1717, %1735
  %1737 = phi i64 [ 0, %1717 ], [ %1702, %1735 ]
  br label %1738

1738:                                             ; preds = %1736, %1738
  %1739 = phi i64 [ %1745, %1738 ], [ %1737, %1736 ]
  %1740 = getelementptr inbounds nuw float, ptr %1715, i64 %1739
  %1741 = load float, ptr %1740, align 4, !tbaa !11
  %1742 = getelementptr inbounds nuw float, ptr %1672, i64 %1739
  %1743 = load float, ptr %1742, align 4, !tbaa !11
  %1744 = tail call float @llvm.fmuladd.f32(float %1708, float %1743, float %1741)
  store float %1744, ptr %1740, align 4, !tbaa !11
  %1745 = add nuw nsw i64 %1739, 1
  %1746 = icmp eq i64 %1745, %1671
  br i1 %1746, label %1747, label %1738, !llvm.loop !41

1747:                                             ; preds = %1738, %1735, %1713
  %1748 = add nuw nsw i64 %1705, 1
  %1749 = icmp eq i64 %1748, 100
  br i1 %1749, label %1750, label %1704, !llvm.loop !34

1750:                                             ; preds = %1747, %1656
  %1751 = phi i32 [ %1635, %1656 ], [ %1630, %1747 ]
  %1752 = add nuw nsw i64 %1632, 1
  %1753 = icmp eq i64 %1636, 99
  br i1 %1753, label %1754, label %1629, !llvm.loop !35

1754:                                             ; preds = %1750
  store i32 99, ptr getelementptr inbounds nuw (i8, ptr @main.ipvt, i64 396), align 4, !tbaa !17
  %1755 = add nuw nsw i32 %1461, 1
  %1756 = icmp eq i32 %1755, %1458
  br i1 %1756, label %1757, label %1460, !llvm.loop !42

1757:                                             ; preds = %1754
  %1758 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.a, i64 79992), align 4, !tbaa !11
  %1759 = fcmp oeq float %1758, 0.000000e+00
  %1760 = select i1 %1759, i32 99, i32 %1751
  store i32 %1760, ptr @main.info, align 4, !tbaa !17
  br label %1761

1761:                                             ; preds = %1757, %1454
  %1762 = tail call i64 @clock() #12
  %1763 = sitofp i64 %1762 to float
  %1764 = fdiv float %1763, 1.000000e+06
  %1765 = fsub float %1764, %1457
  %1766 = fsub float %1765, %1453
  %1767 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %1768 = sitofp i32 %1767 to float
  %1769 = fdiv float %1766, %1768
  %1770 = load i32, ptr @main.j, align 4, !tbaa !17
  %1771 = sext i32 %1770 to i64
  %1772 = getelementptr inbounds float, ptr @atime, i64 %1771
  store float %1769, ptr %1772, align 4, !tbaa !11
  %1773 = tail call i64 @clock() #12
  %1774 = sitofp i64 %1773 to float
  %1775 = fdiv float %1774, 1.000000e+06
  %1776 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %1777 = icmp sgt i32 %1776, 0
  br i1 %1777, label %1778, label %1892

1778:                                             ; preds = %1761, %1889
  %1779 = phi i32 [ %1890, %1889 ], [ 0, %1761 ]
  br label %1780

1780:                                             ; preds = %1778, %1834
  %1781 = phi i64 [ %1794, %1834 ], [ 0, %1778 ]
  %1782 = sub nsw i64 99, %1781
  %1783 = getelementptr inbounds nuw i32, ptr @main.ipvt, i64 %1781
  %1784 = load i32, ptr %1783, align 4, !tbaa !17
  %1785 = sext i32 %1784 to i64
  %1786 = getelementptr inbounds float, ptr @main.b, i64 %1785
  %1787 = load float, ptr %1786, align 4, !tbaa !11
  %1788 = zext i32 %1784 to i64
  %1789 = icmp eq i64 %1781, %1788
  br i1 %1789, label %1793, label %1790

1790:                                             ; preds = %1780
  %1791 = getelementptr inbounds nuw float, ptr @main.b, i64 %1781
  %1792 = load float, ptr %1791, align 4, !tbaa !11
  store float %1792, ptr %1786, align 4, !tbaa !11
  store float %1787, ptr %1791, align 4, !tbaa !11
  br label %1793

1793:                                             ; preds = %1790, %1780
  %1794 = add nuw nsw i64 %1781, 1
  %1795 = mul nuw nsw i64 %1781, 808
  %1796 = getelementptr i8, ptr @main.a, i64 %1795
  %1797 = getelementptr i8, ptr %1796, i64 4
  %1798 = getelementptr inbounds nuw float, ptr @main.b, i64 %1794
  %1799 = fcmp oeq float %1787, 0.000000e+00
  br i1 %1799, label %1834, label %1800

1800:                                             ; preds = %1793
  %1801 = sub nuw nsw i64 99, %1781
  %1802 = icmp ult i64 %1782, 8
  br i1 %1802, label %1823, label %1803

1803:                                             ; preds = %1800
  %1804 = and i64 %1782, -8
  %1805 = insertelement <4 x float> poison, float %1787, i64 0
  %1806 = shufflevector <4 x float> %1805, <4 x float> poison, <4 x i32> zeroinitializer
  br label %1807

1807:                                             ; preds = %1807, %1803
  %1808 = phi i64 [ 0, %1803 ], [ %1819, %1807 ]
  %1809 = getelementptr inbounds nuw float, ptr %1798, i64 %1808
  %1810 = getelementptr inbounds nuw i8, ptr %1809, i64 16
  %1811 = load <4 x float>, ptr %1809, align 4, !tbaa !11
  %1812 = load <4 x float>, ptr %1810, align 4, !tbaa !11
  %1813 = getelementptr inbounds nuw float, ptr %1797, i64 %1808
  %1814 = getelementptr inbounds nuw i8, ptr %1813, i64 16
  %1815 = load <4 x float>, ptr %1813, align 4, !tbaa !11
  %1816 = load <4 x float>, ptr %1814, align 4, !tbaa !11
  %1817 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %1806, <4 x float> %1815, <4 x float> %1811)
  %1818 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %1806, <4 x float> %1816, <4 x float> %1812)
  store <4 x float> %1817, ptr %1809, align 4, !tbaa !11
  store <4 x float> %1818, ptr %1810, align 4, !tbaa !11
  %1819 = add nuw i64 %1808, 8
  %1820 = icmp eq i64 %1819, %1804
  br i1 %1820, label %1821, label %1807, !llvm.loop !43

1821:                                             ; preds = %1807
  %1822 = icmp eq i64 %1782, %1804
  br i1 %1822, label %1834, label %1823

1823:                                             ; preds = %1800, %1821
  %1824 = phi i64 [ 0, %1800 ], [ %1804, %1821 ]
  br label %1825

1825:                                             ; preds = %1823, %1825
  %1826 = phi i64 [ %1832, %1825 ], [ %1824, %1823 ]
  %1827 = getelementptr inbounds nuw float, ptr %1798, i64 %1826
  %1828 = load float, ptr %1827, align 4, !tbaa !11
  %1829 = getelementptr inbounds nuw float, ptr %1797, i64 %1826
  %1830 = load float, ptr %1829, align 4, !tbaa !11
  %1831 = tail call float @llvm.fmuladd.f32(float %1787, float %1830, float %1828)
  store float %1831, ptr %1827, align 4, !tbaa !11
  %1832 = add nuw nsw i64 %1826, 1
  %1833 = icmp eq i64 %1832, %1801
  br i1 %1833, label %1834, label %1825, !llvm.loop !44

1834:                                             ; preds = %1825, %1821, %1793
  %1835 = icmp eq i64 %1794, 99
  br i1 %1835, label %1836, label %1780, !llvm.loop !23

1836:                                             ; preds = %1834, %1887
  %1837 = phi i64 [ %1839, %1887 ], [ 0, %1834 ]
  %1838 = sub nsw i64 99, %1837
  %1839 = add nuw nsw i64 %1837, 1
  %1840 = sub nuw nsw i64 99, %1837
  %1841 = getelementptr inbounds nuw float, ptr @main.b, i64 %1840
  %1842 = load float, ptr %1841, align 4, !tbaa !11
  %1843 = getelementptr float, ptr @main.a, i64 %1840
  %1844 = mul nuw nsw i64 %1840, 804
  %1845 = getelementptr i8, ptr %1843, i64 %1844
  %1846 = load float, ptr %1845, align 4, !tbaa !11
  %1847 = fdiv float %1842, %1846
  store float %1847, ptr %1841, align 4, !tbaa !11
  %1848 = fneg float %1847
  %1849 = mul nuw nsw i64 %1840, 804
  %1850 = getelementptr inbounds nuw i8, ptr @main.a, i64 %1849
  %1851 = icmp samesign ugt i64 %1837, 98
  %1852 = fcmp oeq float %1847, 0.000000e+00
  %1853 = or i1 %1851, %1852
  br i1 %1853, label %1887, label %1854

1854:                                             ; preds = %1836
  %1855 = icmp ult i64 %1838, 8
  br i1 %1855, label %1876, label %1856

1856:                                             ; preds = %1854
  %1857 = and i64 %1838, -8
  %1858 = insertelement <4 x float> poison, float %1848, i64 0
  %1859 = shufflevector <4 x float> %1858, <4 x float> poison, <4 x i32> zeroinitializer
  br label %1860

1860:                                             ; preds = %1860, %1856
  %1861 = phi i64 [ 0, %1856 ], [ %1872, %1860 ]
  %1862 = getelementptr inbounds nuw float, ptr @main.b, i64 %1861
  %1863 = getelementptr inbounds nuw i8, ptr %1862, i64 16
  %1864 = load <4 x float>, ptr %1862, align 16, !tbaa !11
  %1865 = load <4 x float>, ptr %1863, align 16, !tbaa !11
  %1866 = getelementptr inbounds nuw float, ptr %1850, i64 %1861
  %1867 = getelementptr inbounds nuw i8, ptr %1866, i64 16
  %1868 = load <4 x float>, ptr %1866, align 4, !tbaa !11
  %1869 = load <4 x float>, ptr %1867, align 4, !tbaa !11
  %1870 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %1859, <4 x float> %1868, <4 x float> %1864)
  %1871 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %1859, <4 x float> %1869, <4 x float> %1865)
  store <4 x float> %1870, ptr %1862, align 16, !tbaa !11
  store <4 x float> %1871, ptr %1863, align 16, !tbaa !11
  %1872 = add nuw i64 %1861, 8
  %1873 = icmp eq i64 %1872, %1857
  br i1 %1873, label %1874, label %1860, !llvm.loop !45

1874:                                             ; preds = %1860
  %1875 = icmp eq i64 %1838, %1857
  br i1 %1875, label %1887, label %1876

1876:                                             ; preds = %1854, %1874
  %1877 = phi i64 [ 0, %1854 ], [ %1857, %1874 ]
  br label %1878

1878:                                             ; preds = %1876, %1878
  %1879 = phi i64 [ %1885, %1878 ], [ %1877, %1876 ]
  %1880 = getelementptr inbounds nuw float, ptr @main.b, i64 %1879
  %1881 = load float, ptr %1880, align 4, !tbaa !11
  %1882 = getelementptr inbounds nuw float, ptr %1850, i64 %1879
  %1883 = load float, ptr %1882, align 4, !tbaa !11
  %1884 = tail call float @llvm.fmuladd.f32(float %1848, float %1883, float %1881)
  store float %1884, ptr %1880, align 4, !tbaa !11
  %1885 = add nuw nsw i64 %1879, 1
  %1886 = icmp eq i64 %1885, %1840
  br i1 %1886, label %1887, label %1878, !llvm.loop !46

1887:                                             ; preds = %1878, %1874, %1836
  %1888 = icmp eq i64 %1839, 100
  br i1 %1888, label %1889, label %1836, !llvm.loop !26

1889:                                             ; preds = %1887
  %1890 = add nuw nsw i32 %1779, 1
  %1891 = icmp eq i32 %1890, %1776
  br i1 %1891, label %1892, label %1778, !llvm.loop !47

1892:                                             ; preds = %1889, %1761
  %1893 = tail call i64 @clock() #12
  %1894 = sitofp i64 %1893 to float
  %1895 = fdiv float %1894, 1.000000e+06
  %1896 = fsub float %1895, %1775
  %1897 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %1898 = sitofp i32 %1897 to float
  %1899 = fdiv float %1896, %1898
  %1900 = load i32, ptr @main.j, align 4, !tbaa !17
  %1901 = sext i32 %1900 to i64
  %1902 = getelementptr inbounds float, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 60), i64 %1901
  store float %1899, ptr %1902, align 4, !tbaa !11
  %1903 = getelementptr inbounds float, ptr @atime, i64 %1901
  %1904 = load float, ptr %1903, align 4, !tbaa !11
  %1905 = fadd float %1904, %1899
  %1906 = getelementptr inbounds float, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 120), i64 %1901
  store float %1905, ptr %1906, align 4, !tbaa !11
  %1907 = fpext float %1905 to double
  %1908 = fmul double %1907, 1.000000e+06
  %1909 = fdiv double 0x4124F49560000000, %1908
  %1910 = fptrunc double %1909 to float
  %1911 = getelementptr inbounds float, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 180), i64 %1901
  store float %1910, ptr %1911, align 4, !tbaa !11
  %1912 = fdiv float 2.000000e+00, %1910
  %1913 = getelementptr inbounds float, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 240), i64 %1901
  store float %1912, ptr %1913, align 4, !tbaa !11
  %1914 = fdiv float %1905, 0x3FACAC0840000000
  %1915 = getelementptr inbounds float, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 300), i64 %1901
  store float %1914, ptr %1915, align 4, !tbaa !11
  %1916 = load float, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 204), align 4, !tbaa !11
  %1917 = fadd float %1916, %1910
  store float %1917, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 204), align 4, !tbaa !11
  %1918 = add nsw i32 %1900, 1
  store i32 %1918, ptr @main.j, align 4, !tbaa !17
  %1919 = icmp slt i32 %1900, 5
  br i1 %1919, label %1454, label %1920, !llvm.loop !48

1920:                                             ; preds = %1892
  %1921 = fdiv float %1917, 5.000000e+00
  store float %1921, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 204), align 4, !tbaa !11
  %1922 = load ptr, ptr @stderr, align 8, !tbaa !6
  %1923 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %1922, ptr noundef nonnull @.str.20, double noundef 0.000000e+00) #14
  %1924 = load ptr, ptr @stderr, align 8, !tbaa !6
  %1925 = tail call i64 @fwrite(ptr nonnull @.str.21, i64 30, i64 1, ptr %1924) #13
  %1926 = tail call i64 @clock() #12
  %1927 = sitofp i64 %1926 to float
  %1928 = fdiv float %1927, 1.000000e+06
  br label %1929

1929:                                             ; preds = %1920, %2097
  %1930 = phi i32 [ %2098, %2097 ], [ 0, %1920 ]
  br label %1931

1931:                                             ; preds = %1929, %1948
  %1932 = phi i64 [ %1949, %1948 ], [ 0, %1929 ]
  %1933 = phi i32 [ %1940, %1948 ], [ 1325, %1929 ]
  %1934 = mul nuw nsw i64 %1932, 800
  %1935 = getelementptr i8, ptr @main.aa, i64 %1934
  br label %1936

1936:                                             ; preds = %1936, %1931
  %1937 = phi i64 [ 0, %1931 ], [ %1946, %1936 ]
  %1938 = phi i32 [ %1933, %1931 ], [ %1940, %1936 ]
  %1939 = mul nuw nsw i32 %1938, 3125
  %1940 = and i32 %1939, 65535
  %1941 = add nsw i32 %1940, -32768
  %1942 = sitofp i32 %1941 to double
  %1943 = fmul double %1942, 0x3F10000000000000
  %1944 = fptrunc double %1943 to float
  %1945 = getelementptr float, ptr %1935, i64 %1937
  store float %1944, ptr %1945, align 4, !tbaa !11
  %1946 = add nuw nsw i64 %1937, 1
  %1947 = icmp eq i64 %1946, 100
  br i1 %1947, label %1948, label %1936, !llvm.loop !13

1948:                                             ; preds = %1936
  %1949 = add nuw nsw i64 %1932, 1
  %1950 = icmp eq i64 %1949, 100
  br i1 %1950, label %1951, label %1931, !llvm.loop !15

1951:                                             ; preds = %1948
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(400) @main.b, i8 0, i64 400, i1 false), !tbaa !11
  %1952 = load <4 x float>, ptr @main.b, align 16, !tbaa !11
  %1953 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  %1954 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  %1955 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  %1956 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  %1957 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  %1958 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  %1959 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  %1960 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  %1961 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  %1962 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  %1963 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  %1964 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  %1965 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  %1966 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  %1967 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  %1968 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  %1969 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  %1970 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  %1971 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  %1972 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  %1973 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  %1974 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  %1975 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  %1976 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  %1977 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  %1978 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  %1979 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  br label %1980

1980:                                             ; preds = %1980, %1951
  %1981 = phi float [ %1979, %1951 ], [ %2094, %1980 ]
  %1982 = phi float [ %1978, %1951 ], [ %2091, %1980 ]
  %1983 = phi float [ %1977, %1951 ], [ %2088, %1980 ]
  %1984 = phi float [ %1976, %1951 ], [ %2085, %1980 ]
  %1985 = phi <4 x float> [ %1975, %1951 ], [ %2082, %1980 ]
  %1986 = phi <4 x float> [ %1974, %1951 ], [ %2081, %1980 ]
  %1987 = phi <4 x float> [ %1973, %1951 ], [ %2076, %1980 ]
  %1988 = phi <4 x float> [ %1972, %1951 ], [ %2075, %1980 ]
  %1989 = phi <4 x float> [ %1971, %1951 ], [ %2070, %1980 ]
  %1990 = phi <4 x float> [ %1970, %1951 ], [ %2069, %1980 ]
  %1991 = phi <4 x float> [ %1969, %1951 ], [ %2064, %1980 ]
  %1992 = phi <4 x float> [ %1968, %1951 ], [ %2063, %1980 ]
  %1993 = phi <4 x float> [ %1967, %1951 ], [ %2058, %1980 ]
  %1994 = phi <4 x float> [ %1966, %1951 ], [ %2057, %1980 ]
  %1995 = phi <4 x float> [ %1965, %1951 ], [ %2052, %1980 ]
  %1996 = phi <4 x float> [ %1964, %1951 ], [ %2051, %1980 ]
  %1997 = phi <4 x float> [ %1963, %1951 ], [ %2046, %1980 ]
  %1998 = phi <4 x float> [ %1962, %1951 ], [ %2045, %1980 ]
  %1999 = phi <4 x float> [ %1961, %1951 ], [ %2040, %1980 ]
  %2000 = phi <4 x float> [ %1960, %1951 ], [ %2039, %1980 ]
  %2001 = phi <4 x float> [ %1959, %1951 ], [ %2034, %1980 ]
  %2002 = phi <4 x float> [ %1958, %1951 ], [ %2033, %1980 ]
  %2003 = phi <4 x float> [ %1957, %1951 ], [ %2028, %1980 ]
  %2004 = phi <4 x float> [ %1956, %1951 ], [ %2027, %1980 ]
  %2005 = phi <4 x float> [ %1955, %1951 ], [ %2022, %1980 ]
  %2006 = phi <4 x float> [ %1954, %1951 ], [ %2021, %1980 ]
  %2007 = phi <4 x float> [ %1953, %1951 ], [ %2016, %1980 ]
  %2008 = phi <4 x float> [ %1952, %1951 ], [ %2015, %1980 ]
  %2009 = phi i64 [ 0, %1951 ], [ %2095, %1980 ]
  %2010 = mul nuw nsw i64 %2009, 800
  %2011 = getelementptr i8, ptr @main.aa, i64 %2010
  %2012 = getelementptr i8, ptr %2011, i64 16
  %2013 = load <4 x float>, ptr %2011, align 4, !tbaa !11
  %2014 = load <4 x float>, ptr %2012, align 4, !tbaa !11
  %2015 = fadd <4 x float> %2008, %2013
  %2016 = fadd <4 x float> %2007, %2014
  %2017 = getelementptr i8, ptr %2011, i64 32
  %2018 = getelementptr i8, ptr %2011, i64 48
  %2019 = load <4 x float>, ptr %2017, align 4, !tbaa !11
  %2020 = load <4 x float>, ptr %2018, align 4, !tbaa !11
  %2021 = fadd <4 x float> %2006, %2019
  %2022 = fadd <4 x float> %2005, %2020
  %2023 = getelementptr i8, ptr %2011, i64 64
  %2024 = getelementptr i8, ptr %2011, i64 80
  %2025 = load <4 x float>, ptr %2023, align 4, !tbaa !11
  %2026 = load <4 x float>, ptr %2024, align 4, !tbaa !11
  %2027 = fadd <4 x float> %2004, %2025
  %2028 = fadd <4 x float> %2003, %2026
  %2029 = getelementptr i8, ptr %2011, i64 96
  %2030 = getelementptr i8, ptr %2011, i64 112
  %2031 = load <4 x float>, ptr %2029, align 4, !tbaa !11
  %2032 = load <4 x float>, ptr %2030, align 4, !tbaa !11
  %2033 = fadd <4 x float> %2002, %2031
  %2034 = fadd <4 x float> %2001, %2032
  %2035 = getelementptr i8, ptr %2011, i64 128
  %2036 = getelementptr i8, ptr %2011, i64 144
  %2037 = load <4 x float>, ptr %2035, align 4, !tbaa !11
  %2038 = load <4 x float>, ptr %2036, align 4, !tbaa !11
  %2039 = fadd <4 x float> %2000, %2037
  %2040 = fadd <4 x float> %1999, %2038
  %2041 = getelementptr i8, ptr %2011, i64 160
  %2042 = getelementptr i8, ptr %2011, i64 176
  %2043 = load <4 x float>, ptr %2041, align 4, !tbaa !11
  %2044 = load <4 x float>, ptr %2042, align 4, !tbaa !11
  %2045 = fadd <4 x float> %1998, %2043
  %2046 = fadd <4 x float> %1997, %2044
  %2047 = getelementptr i8, ptr %2011, i64 192
  %2048 = getelementptr i8, ptr %2011, i64 208
  %2049 = load <4 x float>, ptr %2047, align 4, !tbaa !11
  %2050 = load <4 x float>, ptr %2048, align 4, !tbaa !11
  %2051 = fadd <4 x float> %1996, %2049
  %2052 = fadd <4 x float> %1995, %2050
  %2053 = getelementptr i8, ptr %2011, i64 224
  %2054 = getelementptr i8, ptr %2011, i64 240
  %2055 = load <4 x float>, ptr %2053, align 4, !tbaa !11
  %2056 = load <4 x float>, ptr %2054, align 4, !tbaa !11
  %2057 = fadd <4 x float> %1994, %2055
  %2058 = fadd <4 x float> %1993, %2056
  %2059 = getelementptr i8, ptr %2011, i64 256
  %2060 = getelementptr i8, ptr %2011, i64 272
  %2061 = load <4 x float>, ptr %2059, align 4, !tbaa !11
  %2062 = load <4 x float>, ptr %2060, align 4, !tbaa !11
  %2063 = fadd <4 x float> %1992, %2061
  %2064 = fadd <4 x float> %1991, %2062
  %2065 = getelementptr i8, ptr %2011, i64 288
  %2066 = getelementptr i8, ptr %2011, i64 304
  %2067 = load <4 x float>, ptr %2065, align 4, !tbaa !11
  %2068 = load <4 x float>, ptr %2066, align 4, !tbaa !11
  %2069 = fadd <4 x float> %1990, %2067
  %2070 = fadd <4 x float> %1989, %2068
  %2071 = getelementptr i8, ptr %2011, i64 320
  %2072 = getelementptr i8, ptr %2011, i64 336
  %2073 = load <4 x float>, ptr %2071, align 4, !tbaa !11
  %2074 = load <4 x float>, ptr %2072, align 4, !tbaa !11
  %2075 = fadd <4 x float> %1988, %2073
  %2076 = fadd <4 x float> %1987, %2074
  %2077 = getelementptr i8, ptr %2011, i64 352
  %2078 = getelementptr i8, ptr %2011, i64 368
  %2079 = load <4 x float>, ptr %2077, align 4, !tbaa !11
  %2080 = load <4 x float>, ptr %2078, align 4, !tbaa !11
  %2081 = fadd <4 x float> %1986, %2079
  %2082 = fadd <4 x float> %1985, %2080
  %2083 = getelementptr i8, ptr %2011, i64 384
  %2084 = load float, ptr %2083, align 4, !tbaa !11
  %2085 = fadd float %1984, %2084
  %2086 = getelementptr i8, ptr %2011, i64 388
  %2087 = load float, ptr %2086, align 4, !tbaa !11
  %2088 = fadd float %1983, %2087
  %2089 = getelementptr i8, ptr %2011, i64 392
  %2090 = load float, ptr %2089, align 4, !tbaa !11
  %2091 = fadd float %1982, %2090
  %2092 = getelementptr i8, ptr %2011, i64 396
  %2093 = load float, ptr %2092, align 4, !tbaa !11
  %2094 = fadd float %1981, %2093
  %2095 = add nuw nsw i64 %2009, 1
  %2096 = icmp eq i64 %2095, 100
  br i1 %2096, label %2097, label %1980, !llvm.loop !16

2097:                                             ; preds = %1980
  store <4 x float> %2015, ptr @main.b, align 16, !tbaa !11
  store <4 x float> %2016, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  store <4 x float> %2021, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  store <4 x float> %2022, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  store <4 x float> %2027, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  store <4 x float> %2028, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  store <4 x float> %2033, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  store <4 x float> %2034, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  store <4 x float> %2039, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  store <4 x float> %2040, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  store <4 x float> %2045, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  store <4 x float> %2046, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  store <4 x float> %2051, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  store <4 x float> %2052, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  store <4 x float> %2057, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  store <4 x float> %2058, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  store <4 x float> %2063, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  store <4 x float> %2064, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  store <4 x float> %2069, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  store <4 x float> %2070, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  store <4 x float> %2075, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  store <4 x float> %2076, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  store <4 x float> %2081, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  store <4 x float> %2082, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  store float %2085, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  store float %2088, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  store float %2091, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  store float %2094, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  %2098 = add nuw nsw i32 %1930, 1
  %2099 = icmp eq i32 %2098, 400
  br i1 %2099, label %2100, label %1929, !llvm.loop !49

2100:                                             ; preds = %2097
  %2101 = tail call i64 @clock() #12
  %2102 = sitofp i64 %2101 to float
  %2103 = fdiv float %2102, 1.000000e+06
  %2104 = fsub float %2103, %1928
  %2105 = fdiv float %2104, 4.000000e+02
  %2106 = load ptr, ptr @stderr, align 8, !tbaa !6
  %2107 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %2106, ptr noundef nonnull @.str.16, double noundef 0.000000e+00) #14
  %2108 = load ptr, ptr @stderr, align 8, !tbaa !6
  %2109 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %2108, ptr noundef nonnull @.str.19, i32 noundef 200) #14
  %2110 = load ptr, ptr @stderr, align 8, !tbaa !6
  %2111 = tail call i64 @fwrite(ptr nonnull @.str.12, i64 55, i64 1, ptr %2110) #13
  %2112 = load ptr, ptr @stderr, align 8, !tbaa !6
  %2113 = tail call i64 @fwrite(ptr nonnull @.str.13, i64 12, i64 1, ptr %2112) #13
  %2114 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %2115 = sitofp i32 %2114 to float
  %2116 = fmul float %2105, %2115
  store float 0.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 228), align 4, !tbaa !11
  store i32 7, ptr @main.j, align 4, !tbaa !17
  br label %2117

2117:                                             ; preds = %2100, %2555
  %2118 = tail call i64 @clock() #12
  %2119 = sitofp i64 %2118 to float
  %2120 = fdiv float %2119, 1.000000e+06
  %2121 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %2122 = icmp sgt i32 %2121, 0
  br i1 %2122, label %2123, label %2424

2123:                                             ; preds = %2117, %2417
  %2124 = phi i32 [ %2418, %2417 ], [ 0, %2117 ]
  br label %2125

2125:                                             ; preds = %2123, %2142
  %2126 = phi i64 [ %2143, %2142 ], [ 0, %2123 ]
  %2127 = phi i32 [ %2134, %2142 ], [ 1325, %2123 ]
  %2128 = mul nuw nsw i64 %2126, 800
  %2129 = getelementptr i8, ptr @main.aa, i64 %2128
  br label %2130

2130:                                             ; preds = %2130, %2125
  %2131 = phi i64 [ 0, %2125 ], [ %2140, %2130 ]
  %2132 = phi i32 [ %2127, %2125 ], [ %2134, %2130 ]
  %2133 = mul nuw nsw i32 %2132, 3125
  %2134 = and i32 %2133, 65535
  %2135 = add nsw i32 %2134, -32768
  %2136 = sitofp i32 %2135 to double
  %2137 = fmul double %2136, 0x3F10000000000000
  %2138 = fptrunc double %2137 to float
  %2139 = getelementptr float, ptr %2129, i64 %2131
  store float %2138, ptr %2139, align 4, !tbaa !11
  %2140 = add nuw nsw i64 %2131, 1
  %2141 = icmp eq i64 %2140, 100
  br i1 %2141, label %2142, label %2130, !llvm.loop !13

2142:                                             ; preds = %2130
  %2143 = add nuw nsw i64 %2126, 1
  %2144 = icmp eq i64 %2143, 100
  br i1 %2144, label %2145, label %2125, !llvm.loop !15

2145:                                             ; preds = %2142
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(400) @main.b, i8 0, i64 400, i1 false), !tbaa !11
  %2146 = load <4 x float>, ptr @main.b, align 16, !tbaa !11
  %2147 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  %2148 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  %2149 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  %2150 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  %2151 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  %2152 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  %2153 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  %2154 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  %2155 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  %2156 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  %2157 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  %2158 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  %2159 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  %2160 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  %2161 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  %2162 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  %2163 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  %2164 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  %2165 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  %2166 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  %2167 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  %2168 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  %2169 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  %2170 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  %2171 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  %2172 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  %2173 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  br label %2174

2174:                                             ; preds = %2174, %2145
  %2175 = phi float [ %2173, %2145 ], [ %2288, %2174 ]
  %2176 = phi float [ %2172, %2145 ], [ %2285, %2174 ]
  %2177 = phi float [ %2171, %2145 ], [ %2282, %2174 ]
  %2178 = phi float [ %2170, %2145 ], [ %2279, %2174 ]
  %2179 = phi <4 x float> [ %2169, %2145 ], [ %2276, %2174 ]
  %2180 = phi <4 x float> [ %2168, %2145 ], [ %2275, %2174 ]
  %2181 = phi <4 x float> [ %2167, %2145 ], [ %2270, %2174 ]
  %2182 = phi <4 x float> [ %2166, %2145 ], [ %2269, %2174 ]
  %2183 = phi <4 x float> [ %2165, %2145 ], [ %2264, %2174 ]
  %2184 = phi <4 x float> [ %2164, %2145 ], [ %2263, %2174 ]
  %2185 = phi <4 x float> [ %2163, %2145 ], [ %2258, %2174 ]
  %2186 = phi <4 x float> [ %2162, %2145 ], [ %2257, %2174 ]
  %2187 = phi <4 x float> [ %2161, %2145 ], [ %2252, %2174 ]
  %2188 = phi <4 x float> [ %2160, %2145 ], [ %2251, %2174 ]
  %2189 = phi <4 x float> [ %2159, %2145 ], [ %2246, %2174 ]
  %2190 = phi <4 x float> [ %2158, %2145 ], [ %2245, %2174 ]
  %2191 = phi <4 x float> [ %2157, %2145 ], [ %2240, %2174 ]
  %2192 = phi <4 x float> [ %2156, %2145 ], [ %2239, %2174 ]
  %2193 = phi <4 x float> [ %2155, %2145 ], [ %2234, %2174 ]
  %2194 = phi <4 x float> [ %2154, %2145 ], [ %2233, %2174 ]
  %2195 = phi <4 x float> [ %2153, %2145 ], [ %2228, %2174 ]
  %2196 = phi <4 x float> [ %2152, %2145 ], [ %2227, %2174 ]
  %2197 = phi <4 x float> [ %2151, %2145 ], [ %2222, %2174 ]
  %2198 = phi <4 x float> [ %2150, %2145 ], [ %2221, %2174 ]
  %2199 = phi <4 x float> [ %2149, %2145 ], [ %2216, %2174 ]
  %2200 = phi <4 x float> [ %2148, %2145 ], [ %2215, %2174 ]
  %2201 = phi <4 x float> [ %2147, %2145 ], [ %2210, %2174 ]
  %2202 = phi <4 x float> [ %2146, %2145 ], [ %2209, %2174 ]
  %2203 = phi i64 [ 0, %2145 ], [ %2289, %2174 ]
  %2204 = mul nuw nsw i64 %2203, 800
  %2205 = getelementptr i8, ptr @main.aa, i64 %2204
  %2206 = getelementptr i8, ptr %2205, i64 16
  %2207 = load <4 x float>, ptr %2205, align 4, !tbaa !11
  %2208 = load <4 x float>, ptr %2206, align 4, !tbaa !11
  %2209 = fadd <4 x float> %2202, %2207
  %2210 = fadd <4 x float> %2201, %2208
  %2211 = getelementptr i8, ptr %2205, i64 32
  %2212 = getelementptr i8, ptr %2205, i64 48
  %2213 = load <4 x float>, ptr %2211, align 4, !tbaa !11
  %2214 = load <4 x float>, ptr %2212, align 4, !tbaa !11
  %2215 = fadd <4 x float> %2200, %2213
  %2216 = fadd <4 x float> %2199, %2214
  %2217 = getelementptr i8, ptr %2205, i64 64
  %2218 = getelementptr i8, ptr %2205, i64 80
  %2219 = load <4 x float>, ptr %2217, align 4, !tbaa !11
  %2220 = load <4 x float>, ptr %2218, align 4, !tbaa !11
  %2221 = fadd <4 x float> %2198, %2219
  %2222 = fadd <4 x float> %2197, %2220
  %2223 = getelementptr i8, ptr %2205, i64 96
  %2224 = getelementptr i8, ptr %2205, i64 112
  %2225 = load <4 x float>, ptr %2223, align 4, !tbaa !11
  %2226 = load <4 x float>, ptr %2224, align 4, !tbaa !11
  %2227 = fadd <4 x float> %2196, %2225
  %2228 = fadd <4 x float> %2195, %2226
  %2229 = getelementptr i8, ptr %2205, i64 128
  %2230 = getelementptr i8, ptr %2205, i64 144
  %2231 = load <4 x float>, ptr %2229, align 4, !tbaa !11
  %2232 = load <4 x float>, ptr %2230, align 4, !tbaa !11
  %2233 = fadd <4 x float> %2194, %2231
  %2234 = fadd <4 x float> %2193, %2232
  %2235 = getelementptr i8, ptr %2205, i64 160
  %2236 = getelementptr i8, ptr %2205, i64 176
  %2237 = load <4 x float>, ptr %2235, align 4, !tbaa !11
  %2238 = load <4 x float>, ptr %2236, align 4, !tbaa !11
  %2239 = fadd <4 x float> %2192, %2237
  %2240 = fadd <4 x float> %2191, %2238
  %2241 = getelementptr i8, ptr %2205, i64 192
  %2242 = getelementptr i8, ptr %2205, i64 208
  %2243 = load <4 x float>, ptr %2241, align 4, !tbaa !11
  %2244 = load <4 x float>, ptr %2242, align 4, !tbaa !11
  %2245 = fadd <4 x float> %2190, %2243
  %2246 = fadd <4 x float> %2189, %2244
  %2247 = getelementptr i8, ptr %2205, i64 224
  %2248 = getelementptr i8, ptr %2205, i64 240
  %2249 = load <4 x float>, ptr %2247, align 4, !tbaa !11
  %2250 = load <4 x float>, ptr %2248, align 4, !tbaa !11
  %2251 = fadd <4 x float> %2188, %2249
  %2252 = fadd <4 x float> %2187, %2250
  %2253 = getelementptr i8, ptr %2205, i64 256
  %2254 = getelementptr i8, ptr %2205, i64 272
  %2255 = load <4 x float>, ptr %2253, align 4, !tbaa !11
  %2256 = load <4 x float>, ptr %2254, align 4, !tbaa !11
  %2257 = fadd <4 x float> %2186, %2255
  %2258 = fadd <4 x float> %2185, %2256
  %2259 = getelementptr i8, ptr %2205, i64 288
  %2260 = getelementptr i8, ptr %2205, i64 304
  %2261 = load <4 x float>, ptr %2259, align 4, !tbaa !11
  %2262 = load <4 x float>, ptr %2260, align 4, !tbaa !11
  %2263 = fadd <4 x float> %2184, %2261
  %2264 = fadd <4 x float> %2183, %2262
  %2265 = getelementptr i8, ptr %2205, i64 320
  %2266 = getelementptr i8, ptr %2205, i64 336
  %2267 = load <4 x float>, ptr %2265, align 4, !tbaa !11
  %2268 = load <4 x float>, ptr %2266, align 4, !tbaa !11
  %2269 = fadd <4 x float> %2182, %2267
  %2270 = fadd <4 x float> %2181, %2268
  %2271 = getelementptr i8, ptr %2205, i64 352
  %2272 = getelementptr i8, ptr %2205, i64 368
  %2273 = load <4 x float>, ptr %2271, align 4, !tbaa !11
  %2274 = load <4 x float>, ptr %2272, align 4, !tbaa !11
  %2275 = fadd <4 x float> %2180, %2273
  %2276 = fadd <4 x float> %2179, %2274
  %2277 = getelementptr i8, ptr %2205, i64 384
  %2278 = load float, ptr %2277, align 4, !tbaa !11
  %2279 = fadd float %2178, %2278
  %2280 = getelementptr i8, ptr %2205, i64 388
  %2281 = load float, ptr %2280, align 4, !tbaa !11
  %2282 = fadd float %2177, %2281
  %2283 = getelementptr i8, ptr %2205, i64 392
  %2284 = load float, ptr %2283, align 4, !tbaa !11
  %2285 = fadd float %2176, %2284
  %2286 = getelementptr i8, ptr %2205, i64 396
  %2287 = load float, ptr %2286, align 4, !tbaa !11
  %2288 = fadd float %2175, %2287
  %2289 = add nuw nsw i64 %2203, 1
  %2290 = icmp eq i64 %2289, 100
  br i1 %2290, label %2291, label %2174, !llvm.loop !16

2291:                                             ; preds = %2174
  store <4 x float> %2209, ptr @main.b, align 16, !tbaa !11
  store <4 x float> %2210, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 16), align 16, !tbaa !11
  store <4 x float> %2215, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 32), align 16, !tbaa !11
  store <4 x float> %2216, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 48), align 16, !tbaa !11
  store <4 x float> %2221, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 64), align 16, !tbaa !11
  store <4 x float> %2222, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 80), align 16, !tbaa !11
  store <4 x float> %2227, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 96), align 16, !tbaa !11
  store <4 x float> %2228, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 112), align 16, !tbaa !11
  store <4 x float> %2233, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 128), align 16, !tbaa !11
  store <4 x float> %2234, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 144), align 16, !tbaa !11
  store <4 x float> %2239, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 160), align 16, !tbaa !11
  store <4 x float> %2240, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 176), align 16, !tbaa !11
  store <4 x float> %2245, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 192), align 16, !tbaa !11
  store <4 x float> %2246, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 208), align 16, !tbaa !11
  store <4 x float> %2251, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 224), align 16, !tbaa !11
  store <4 x float> %2252, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 240), align 16, !tbaa !11
  store <4 x float> %2257, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 256), align 16, !tbaa !11
  store <4 x float> %2258, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 272), align 16, !tbaa !11
  store <4 x float> %2263, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 288), align 16, !tbaa !11
  store <4 x float> %2264, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 304), align 16, !tbaa !11
  store <4 x float> %2269, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 320), align 16, !tbaa !11
  store <4 x float> %2270, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 336), align 16, !tbaa !11
  store <4 x float> %2275, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 352), align 16, !tbaa !11
  store <4 x float> %2276, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 368), align 16, !tbaa !11
  store float %2279, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 384), align 16, !tbaa !11
  store float %2282, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 388), align 4, !tbaa !11
  store float %2285, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 392), align 8, !tbaa !11
  store float %2288, ptr getelementptr inbounds nuw (i8, ptr @main.b, i64 396), align 4, !tbaa !11
  br label %2292

2292:                                             ; preds = %2291, %2413
  %2293 = phi i32 [ %2414, %2413 ], [ 0, %2291 ]
  %2294 = phi i64 [ %2299, %2413 ], [ 0, %2291 ]
  %2295 = phi i64 [ %2415, %2413 ], [ 1, %2291 ]
  %2296 = sub nsw i64 99, %2294
  %2297 = sub nsw i64 99, %2294
  %2298 = trunc i64 %2294 to i32
  %2299 = add nuw nsw i64 %2294, 1
  %2300 = sub nuw nsw i64 100, %2294
  %2301 = getelementptr float, ptr @main.aa, i64 %2294
  %2302 = mul nuw nsw i64 %2294, 800
  %2303 = getelementptr i8, ptr %2301, i64 %2302
  %2304 = load float, ptr %2303, align 4, !tbaa !11
  %2305 = tail call float @llvm.fabs.f32(float %2304)
  br label %2306

2306:                                             ; preds = %2306, %2292
  %2307 = phi i64 [ 1, %2292 ], [ %2317, %2306 ]
  %2308 = phi i32 [ 0, %2292 ], [ %2316, %2306 ]
  %2309 = phi float [ %2305, %2292 ], [ %2314, %2306 ]
  %2310 = getelementptr inbounds nuw float, ptr %2303, i64 %2307
  %2311 = load float, ptr %2310, align 4, !tbaa !11
  %2312 = tail call float @llvm.fabs.f32(float %2311)
  %2313 = fcmp ogt float %2312, %2309
  %2314 = select i1 %2313, float %2312, float %2309
  %2315 = trunc nuw nsw i64 %2307 to i32
  %2316 = select i1 %2313, i32 %2315, i32 %2308
  %2317 = add nuw nsw i64 %2307, 1
  %2318 = icmp eq i64 %2317, %2300
  br i1 %2318, label %2319, label %2306, !llvm.loop !29

2319:                                             ; preds = %2306
  %2320 = add nsw i32 %2316, %2298
  %2321 = getelementptr inbounds nuw i32, ptr @main.ipvt, i64 %2294
  store i32 %2320, ptr %2321, align 4, !tbaa !17
  %2322 = sext i32 %2320 to i64
  %2323 = mul i64 %2294, 800
  %2324 = getelementptr i8, ptr @main.aa, i64 %2323
  %2325 = getelementptr float, ptr %2324, i64 %2322
  %2326 = load float, ptr %2325, align 4, !tbaa !11
  %2327 = fcmp une float %2326, 0.000000e+00
  br i1 %2327, label %2328, label %2413

2328:                                             ; preds = %2319
  %2329 = icmp eq i32 %2316, 0
  br i1 %2329, label %2331, label %2330

2330:                                             ; preds = %2328
  store float %2304, ptr %2325, align 4, !tbaa !11
  store float %2326, ptr %2303, align 4, !tbaa !11
  br label %2331

2331:                                             ; preds = %2330, %2328
  %2332 = phi float [ %2326, %2330 ], [ %2304, %2328 ]
  %2333 = fdiv float -1.000000e+00, %2332
  %2334 = sub nuw nsw i64 99, %2294
  %2335 = getelementptr i8, ptr %2303, i64 4
  %2336 = icmp ult i64 %2296, 8
  br i1 %2336, label %2353, label %2337

2337:                                             ; preds = %2331
  %2338 = and i64 %2296, -8
  %2339 = insertelement <4 x float> poison, float %2333, i64 0
  %2340 = shufflevector <4 x float> %2339, <4 x float> poison, <4 x i32> zeroinitializer
  br label %2341

2341:                                             ; preds = %2341, %2337
  %2342 = phi i64 [ 0, %2337 ], [ %2349, %2341 ]
  %2343 = getelementptr inbounds nuw float, ptr %2335, i64 %2342
  %2344 = getelementptr inbounds nuw i8, ptr %2343, i64 16
  %2345 = load <4 x float>, ptr %2343, align 4, !tbaa !11
  %2346 = load <4 x float>, ptr %2344, align 4, !tbaa !11
  %2347 = fmul <4 x float> %2340, %2345
  %2348 = fmul <4 x float> %2340, %2346
  store <4 x float> %2347, ptr %2343, align 4, !tbaa !11
  store <4 x float> %2348, ptr %2344, align 4, !tbaa !11
  %2349 = add nuw i64 %2342, 8
  %2350 = icmp eq i64 %2349, %2338
  br i1 %2350, label %2351, label %2341, !llvm.loop !50

2351:                                             ; preds = %2341
  %2352 = icmp eq i64 %2296, %2338
  br i1 %2352, label %2362, label %2353

2353:                                             ; preds = %2331, %2351
  %2354 = phi i64 [ 0, %2331 ], [ %2338, %2351 ]
  br label %2355

2355:                                             ; preds = %2353, %2355
  %2356 = phi i64 [ %2360, %2355 ], [ %2354, %2353 ]
  %2357 = getelementptr inbounds nuw float, ptr %2335, i64 %2356
  %2358 = load float, ptr %2357, align 4, !tbaa !11
  %2359 = fmul float %2333, %2358
  store float %2359, ptr %2357, align 4, !tbaa !11
  %2360 = add nuw nsw i64 %2356, 1
  %2361 = icmp eq i64 %2360, %2334
  br i1 %2361, label %2362, label %2355, !llvm.loop !51

2362:                                             ; preds = %2355, %2351
  %2363 = getelementptr float, ptr @main.aa, i64 %2322
  %2364 = icmp ult i64 %2297, 8
  %2365 = and i64 %2297, -8
  %2366 = icmp eq i64 %2297, %2365
  br label %2367

2367:                                             ; preds = %2410, %2362
  %2368 = phi i64 [ %2295, %2362 ], [ %2411, %2410 ]
  %2369 = mul nuw nsw i64 %2368, 200
  %2370 = getelementptr float, ptr %2363, i64 %2369
  %2371 = load float, ptr %2370, align 4, !tbaa !11
  %2372 = add nuw nsw i64 %2369, %2294
  br i1 %2329, label %2376, label %2373

2373:                                             ; preds = %2367
  %2374 = getelementptr inbounds nuw float, ptr @main.aa, i64 %2372
  %2375 = load float, ptr %2374, align 4, !tbaa !11
  store float %2375, ptr %2370, align 4, !tbaa !11
  store float %2371, ptr %2374, align 4, !tbaa !11
  br label %2376

2376:                                             ; preds = %2373, %2367
  %2377 = getelementptr float, ptr @main.aa, i64 %2372
  %2378 = getelementptr i8, ptr %2377, i64 4
  %2379 = fcmp oeq float %2371, 0.000000e+00
  br i1 %2379, label %2410, label %2380

2380:                                             ; preds = %2376
  br i1 %2364, label %2399, label %2381

2381:                                             ; preds = %2380
  %2382 = insertelement <4 x float> poison, float %2371, i64 0
  %2383 = shufflevector <4 x float> %2382, <4 x float> poison, <4 x i32> zeroinitializer
  br label %2384

2384:                                             ; preds = %2384, %2381
  %2385 = phi i64 [ 0, %2381 ], [ %2396, %2384 ]
  %2386 = getelementptr inbounds nuw float, ptr %2378, i64 %2385
  %2387 = getelementptr inbounds nuw i8, ptr %2386, i64 16
  %2388 = load <4 x float>, ptr %2386, align 4, !tbaa !11
  %2389 = load <4 x float>, ptr %2387, align 4, !tbaa !11
  %2390 = getelementptr inbounds nuw float, ptr %2335, i64 %2385
  %2391 = getelementptr inbounds nuw i8, ptr %2390, i64 16
  %2392 = load <4 x float>, ptr %2390, align 4, !tbaa !11
  %2393 = load <4 x float>, ptr %2391, align 4, !tbaa !11
  %2394 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %2383, <4 x float> %2392, <4 x float> %2388)
  %2395 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %2383, <4 x float> %2393, <4 x float> %2389)
  store <4 x float> %2394, ptr %2386, align 4, !tbaa !11
  store <4 x float> %2395, ptr %2387, align 4, !tbaa !11
  %2396 = add nuw i64 %2385, 8
  %2397 = icmp eq i64 %2396, %2365
  br i1 %2397, label %2398, label %2384, !llvm.loop !52

2398:                                             ; preds = %2384
  br i1 %2366, label %2410, label %2399

2399:                                             ; preds = %2380, %2398
  %2400 = phi i64 [ 0, %2380 ], [ %2365, %2398 ]
  br label %2401

2401:                                             ; preds = %2399, %2401
  %2402 = phi i64 [ %2408, %2401 ], [ %2400, %2399 ]
  %2403 = getelementptr inbounds nuw float, ptr %2378, i64 %2402
  %2404 = load float, ptr %2403, align 4, !tbaa !11
  %2405 = getelementptr inbounds nuw float, ptr %2335, i64 %2402
  %2406 = load float, ptr %2405, align 4, !tbaa !11
  %2407 = tail call float @llvm.fmuladd.f32(float %2371, float %2406, float %2404)
  store float %2407, ptr %2403, align 4, !tbaa !11
  %2408 = add nuw nsw i64 %2402, 1
  %2409 = icmp eq i64 %2408, %2334
  br i1 %2409, label %2410, label %2401, !llvm.loop !53

2410:                                             ; preds = %2401, %2398, %2376
  %2411 = add nuw nsw i64 %2368, 1
  %2412 = icmp eq i64 %2411, 100
  br i1 %2412, label %2413, label %2367, !llvm.loop !34

2413:                                             ; preds = %2410, %2319
  %2414 = phi i32 [ %2298, %2319 ], [ %2293, %2410 ]
  %2415 = add nuw nsw i64 %2295, 1
  %2416 = icmp eq i64 %2299, 99
  br i1 %2416, label %2417, label %2292, !llvm.loop !35

2417:                                             ; preds = %2413
  store i32 99, ptr getelementptr inbounds nuw (i8, ptr @main.ipvt, i64 396), align 4, !tbaa !17
  %2418 = add nuw nsw i32 %2124, 1
  %2419 = icmp eq i32 %2418, %2121
  br i1 %2419, label %2420, label %2123, !llvm.loop !54

2420:                                             ; preds = %2417
  %2421 = load float, ptr getelementptr inbounds nuw (i8, ptr @main.aa, i64 79596), align 4, !tbaa !11
  %2422 = fcmp oeq float %2421, 0.000000e+00
  %2423 = select i1 %2422, i32 99, i32 %2414
  store i32 %2423, ptr @main.info, align 4, !tbaa !17
  br label %2424

2424:                                             ; preds = %2420, %2117
  %2425 = tail call i64 @clock() #12
  %2426 = sitofp i64 %2425 to float
  %2427 = fdiv float %2426, 1.000000e+06
  %2428 = fsub float %2427, %2120
  %2429 = fsub float %2428, %2116
  %2430 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %2431 = sitofp i32 %2430 to float
  %2432 = fdiv float %2429, %2431
  %2433 = load i32, ptr @main.j, align 4, !tbaa !17
  %2434 = sext i32 %2433 to i64
  %2435 = getelementptr inbounds float, ptr @atime, i64 %2434
  store float %2432, ptr %2435, align 4, !tbaa !11
  %2436 = tail call i64 @clock() #12
  %2437 = sitofp i64 %2436 to float
  %2438 = fdiv float %2437, 1.000000e+06
  %2439 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %2440 = icmp sgt i32 %2439, 0
  br i1 %2440, label %2441, label %2555

2441:                                             ; preds = %2424, %2552
  %2442 = phi i32 [ %2553, %2552 ], [ 0, %2424 ]
  br label %2443

2443:                                             ; preds = %2441, %2497
  %2444 = phi i64 [ %2457, %2497 ], [ 0, %2441 ]
  %2445 = sub nsw i64 99, %2444
  %2446 = getelementptr inbounds nuw i32, ptr @main.ipvt, i64 %2444
  %2447 = load i32, ptr %2446, align 4, !tbaa !17
  %2448 = sext i32 %2447 to i64
  %2449 = getelementptr inbounds float, ptr @main.b, i64 %2448
  %2450 = load float, ptr %2449, align 4, !tbaa !11
  %2451 = zext i32 %2447 to i64
  %2452 = icmp eq i64 %2444, %2451
  br i1 %2452, label %2456, label %2453

2453:                                             ; preds = %2443
  %2454 = getelementptr inbounds nuw float, ptr @main.b, i64 %2444
  %2455 = load float, ptr %2454, align 4, !tbaa !11
  store float %2455, ptr %2449, align 4, !tbaa !11
  store float %2450, ptr %2454, align 4, !tbaa !11
  br label %2456

2456:                                             ; preds = %2453, %2443
  %2457 = add nuw nsw i64 %2444, 1
  %2458 = mul nuw nsw i64 %2444, 804
  %2459 = getelementptr i8, ptr @main.aa, i64 %2458
  %2460 = getelementptr i8, ptr %2459, i64 4
  %2461 = getelementptr inbounds nuw float, ptr @main.b, i64 %2457
  %2462 = fcmp oeq float %2450, 0.000000e+00
  br i1 %2462, label %2497, label %2463

2463:                                             ; preds = %2456
  %2464 = sub nuw nsw i64 99, %2444
  %2465 = icmp ult i64 %2445, 8
  br i1 %2465, label %2486, label %2466

2466:                                             ; preds = %2463
  %2467 = and i64 %2445, -8
  %2468 = insertelement <4 x float> poison, float %2450, i64 0
  %2469 = shufflevector <4 x float> %2468, <4 x float> poison, <4 x i32> zeroinitializer
  br label %2470

2470:                                             ; preds = %2470, %2466
  %2471 = phi i64 [ 0, %2466 ], [ %2482, %2470 ]
  %2472 = getelementptr inbounds nuw float, ptr %2461, i64 %2471
  %2473 = getelementptr inbounds nuw i8, ptr %2472, i64 16
  %2474 = load <4 x float>, ptr %2472, align 4, !tbaa !11
  %2475 = load <4 x float>, ptr %2473, align 4, !tbaa !11
  %2476 = getelementptr inbounds nuw float, ptr %2460, i64 %2471
  %2477 = getelementptr inbounds nuw i8, ptr %2476, i64 16
  %2478 = load <4 x float>, ptr %2476, align 4, !tbaa !11
  %2479 = load <4 x float>, ptr %2477, align 4, !tbaa !11
  %2480 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %2469, <4 x float> %2478, <4 x float> %2474)
  %2481 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %2469, <4 x float> %2479, <4 x float> %2475)
  store <4 x float> %2480, ptr %2472, align 4, !tbaa !11
  store <4 x float> %2481, ptr %2473, align 4, !tbaa !11
  %2482 = add nuw i64 %2471, 8
  %2483 = icmp eq i64 %2482, %2467
  br i1 %2483, label %2484, label %2470, !llvm.loop !55

2484:                                             ; preds = %2470
  %2485 = icmp eq i64 %2445, %2467
  br i1 %2485, label %2497, label %2486

2486:                                             ; preds = %2463, %2484
  %2487 = phi i64 [ 0, %2463 ], [ %2467, %2484 ]
  br label %2488

2488:                                             ; preds = %2486, %2488
  %2489 = phi i64 [ %2495, %2488 ], [ %2487, %2486 ]
  %2490 = getelementptr inbounds nuw float, ptr %2461, i64 %2489
  %2491 = load float, ptr %2490, align 4, !tbaa !11
  %2492 = getelementptr inbounds nuw float, ptr %2460, i64 %2489
  %2493 = load float, ptr %2492, align 4, !tbaa !11
  %2494 = tail call float @llvm.fmuladd.f32(float %2450, float %2493, float %2491)
  store float %2494, ptr %2490, align 4, !tbaa !11
  %2495 = add nuw nsw i64 %2489, 1
  %2496 = icmp eq i64 %2495, %2464
  br i1 %2496, label %2497, label %2488, !llvm.loop !56

2497:                                             ; preds = %2488, %2484, %2456
  %2498 = icmp eq i64 %2457, 99
  br i1 %2498, label %2499, label %2443, !llvm.loop !23

2499:                                             ; preds = %2497, %2550
  %2500 = phi i64 [ %2502, %2550 ], [ 0, %2497 ]
  %2501 = sub nsw i64 99, %2500
  %2502 = add nuw nsw i64 %2500, 1
  %2503 = sub nuw nsw i64 99, %2500
  %2504 = getelementptr inbounds nuw float, ptr @main.b, i64 %2503
  %2505 = load float, ptr %2504, align 4, !tbaa !11
  %2506 = getelementptr float, ptr @main.aa, i64 %2503
  %2507 = mul nuw nsw i64 %2503, 800
  %2508 = getelementptr i8, ptr %2506, i64 %2507
  %2509 = load float, ptr %2508, align 4, !tbaa !11
  %2510 = fdiv float %2505, %2509
  store float %2510, ptr %2504, align 4, !tbaa !11
  %2511 = fneg float %2510
  %2512 = mul nuw nsw i64 %2503, 800
  %2513 = getelementptr inbounds nuw i8, ptr @main.aa, i64 %2512
  %2514 = icmp samesign ugt i64 %2500, 98
  %2515 = fcmp oeq float %2510, 0.000000e+00
  %2516 = or i1 %2514, %2515
  br i1 %2516, label %2550, label %2517

2517:                                             ; preds = %2499
  %2518 = icmp ult i64 %2501, 8
  br i1 %2518, label %2539, label %2519

2519:                                             ; preds = %2517
  %2520 = and i64 %2501, -8
  %2521 = insertelement <4 x float> poison, float %2511, i64 0
  %2522 = shufflevector <4 x float> %2521, <4 x float> poison, <4 x i32> zeroinitializer
  br label %2523

2523:                                             ; preds = %2523, %2519
  %2524 = phi i64 [ 0, %2519 ], [ %2535, %2523 ]
  %2525 = getelementptr inbounds nuw float, ptr @main.b, i64 %2524
  %2526 = getelementptr inbounds nuw i8, ptr %2525, i64 16
  %2527 = load <4 x float>, ptr %2525, align 16, !tbaa !11
  %2528 = load <4 x float>, ptr %2526, align 16, !tbaa !11
  %2529 = getelementptr inbounds nuw float, ptr %2513, i64 %2524
  %2530 = getelementptr inbounds nuw i8, ptr %2529, i64 16
  %2531 = load <4 x float>, ptr %2529, align 4, !tbaa !11
  %2532 = load <4 x float>, ptr %2530, align 4, !tbaa !11
  %2533 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %2522, <4 x float> %2531, <4 x float> %2527)
  %2534 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %2522, <4 x float> %2532, <4 x float> %2528)
  store <4 x float> %2533, ptr %2525, align 16, !tbaa !11
  store <4 x float> %2534, ptr %2526, align 16, !tbaa !11
  %2535 = add nuw i64 %2524, 8
  %2536 = icmp eq i64 %2535, %2520
  br i1 %2536, label %2537, label %2523, !llvm.loop !57

2537:                                             ; preds = %2523
  %2538 = icmp eq i64 %2501, %2520
  br i1 %2538, label %2550, label %2539

2539:                                             ; preds = %2517, %2537
  %2540 = phi i64 [ 0, %2517 ], [ %2520, %2537 ]
  br label %2541

2541:                                             ; preds = %2539, %2541
  %2542 = phi i64 [ %2548, %2541 ], [ %2540, %2539 ]
  %2543 = getelementptr inbounds nuw float, ptr @main.b, i64 %2542
  %2544 = load float, ptr %2543, align 4, !tbaa !11
  %2545 = getelementptr inbounds nuw float, ptr %2513, i64 %2542
  %2546 = load float, ptr %2545, align 4, !tbaa !11
  %2547 = tail call float @llvm.fmuladd.f32(float %2511, float %2546, float %2544)
  store float %2547, ptr %2543, align 4, !tbaa !11
  %2548 = add nuw nsw i64 %2542, 1
  %2549 = icmp eq i64 %2548, %2503
  br i1 %2549, label %2550, label %2541, !llvm.loop !58

2550:                                             ; preds = %2541, %2537, %2499
  %2551 = icmp eq i64 %2502, 100
  br i1 %2551, label %2552, label %2499, !llvm.loop !26

2552:                                             ; preds = %2550
  %2553 = add nuw nsw i32 %2442, 1
  %2554 = icmp eq i32 %2553, %2439
  br i1 %2554, label %2555, label %2441, !llvm.loop !59

2555:                                             ; preds = %2552, %2424
  %2556 = tail call i64 @clock() #12
  %2557 = sitofp i64 %2556 to float
  %2558 = fdiv float %2557, 1.000000e+06
  %2559 = fsub float %2558, %2438
  %2560 = load i32, ptr @main.ntimes, align 4, !tbaa !17
  %2561 = sitofp i32 %2560 to float
  %2562 = fdiv float %2559, %2561
  %2563 = load i32, ptr @main.j, align 4, !tbaa !17
  %2564 = sext i32 %2563 to i64
  %2565 = getelementptr inbounds float, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 60), i64 %2564
  store float %2562, ptr %2565, align 4, !tbaa !11
  %2566 = getelementptr inbounds float, ptr @atime, i64 %2564
  %2567 = load float, ptr %2566, align 4, !tbaa !11
  %2568 = fadd float %2567, %2562
  %2569 = getelementptr inbounds float, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 120), i64 %2564
  store float %2568, ptr %2569, align 4, !tbaa !11
  %2570 = fpext float %2568 to double
  %2571 = fmul double %2570, 1.000000e+06
  %2572 = fdiv double 0x4124F49560000000, %2571
  %2573 = fptrunc double %2572 to float
  %2574 = getelementptr inbounds float, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 180), i64 %2564
  store float %2573, ptr %2574, align 4, !tbaa !11
  %2575 = fdiv float 2.000000e+00, %2573
  %2576 = getelementptr inbounds float, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 240), i64 %2564
  store float %2575, ptr %2576, align 4, !tbaa !11
  %2577 = fdiv float %2568, 0x3FACAC0840000000
  %2578 = getelementptr inbounds float, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 300), i64 %2564
  store float %2577, ptr %2578, align 4, !tbaa !11
  %2579 = load float, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 228), align 4, !tbaa !11
  %2580 = fadd float %2579, %2573
  store float %2580, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 228), align 4, !tbaa !11
  %2581 = add nsw i32 %2563, 1
  store i32 %2581, ptr @main.j, align 4, !tbaa !17
  %2582 = icmp slt i32 %2563, 11
  br i1 %2582, label %2117, label %2583, !llvm.loop !60

2583:                                             ; preds = %2555
  %2584 = fdiv float %2580, 5.000000e+00
  store float %2584, ptr getelementptr inbounds nuw (i8, ptr @atime, i64 228), align 4, !tbaa !11
  %2585 = load ptr, ptr @stderr, align 8, !tbaa !6
  %2586 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %2585, ptr noundef nonnull @.str.20, double noundef 0.000000e+00) #14
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #4

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #5

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @matgen(ptr noundef captures(none) %0, i32 noundef %1, i32 noundef %2, ptr noundef captures(none) %3, ptr noundef captures(none) initializes((0, 4)) %4) local_unnamed_addr #6 {
  store float 0.000000e+00, ptr %4, align 4, !tbaa !11
  %6 = icmp sgt i32 %2, 0
  br i1 %6, label %7, label %87

7:                                                ; preds = %5
  %8 = sext i32 %1 to i64
  %9 = zext nneg i32 %2 to i64
  br label %10

10:                                               ; preds = %7, %30
  %11 = phi i64 [ 0, %7 ], [ %31, %30 ]
  %12 = phi i32 [ 1325, %7 ], [ %19, %30 ]
  %13 = mul nsw i64 %11, %8
  %14 = getelementptr float, ptr %0, i64 %13
  br label %15

15:                                               ; preds = %10, %15
  %16 = phi i64 [ 0, %10 ], [ %28, %15 ]
  %17 = phi i32 [ %12, %10 ], [ %19, %15 ]
  %18 = mul nuw nsw i32 %17, 3125
  %19 = and i32 %18, 65535
  %20 = add nsw i32 %19, -32768
  %21 = sitofp i32 %20 to double
  %22 = fmul double %21, 0x3F10000000000000
  %23 = fptrunc double %22 to float
  %24 = getelementptr float, ptr %14, i64 %16
  store float %23, ptr %24, align 4, !tbaa !11
  %25 = load float, ptr %4, align 4, !tbaa !11
  %26 = fcmp olt float %25, %23
  %27 = select i1 %26, float %23, float %25
  store float %27, ptr %4, align 4, !tbaa !11
  %28 = add nuw nsw i64 %16, 1
  %29 = icmp eq i64 %28, %9
  br i1 %29, label %30, label %15, !llvm.loop !13

30:                                               ; preds = %15
  %31 = add nuw nsw i64 %11, 1
  %32 = icmp eq i64 %31, %9
  br i1 %32, label %33, label %10, !llvm.loop !15

33:                                               ; preds = %30
  %34 = zext nneg i32 %2 to i64
  %35 = shl nuw nsw i64 %34, 2
  tail call void @llvm.memset.p0.i64(ptr align 4 %3, i8 0, i64 %35, i1 false), !tbaa !11
  %36 = sext i32 %1 to i64
  %37 = zext nneg i32 %2 to i64
  %38 = shl nuw nsw i64 %9, 2
  %39 = getelementptr i8, ptr %3, i64 %38
  %40 = add nuw nsw i64 %9, 4611686018427387903
  %41 = mul i64 %40, %36
  %42 = add i64 %41, %9
  %43 = shl i64 %42, 2
  %44 = getelementptr i8, ptr %0, i64 %43
  %45 = icmp ult i32 %2, 8
  %46 = icmp ult ptr %3, %44
  %47 = icmp ult ptr %0, %39
  %48 = and i1 %46, %47
  %49 = icmp slt i32 %1, 0
  %50 = or i1 %48, %49
  %51 = and i64 %9, 2147483640
  %52 = icmp eq i64 %51, %9
  br label %53

53:                                               ; preds = %33, %84
  %54 = phi i64 [ 0, %33 ], [ %85, %84 ]
  %55 = mul nsw i64 %54, %36
  %56 = getelementptr float, ptr %0, i64 %55
  %57 = select i1 %45, i1 true, i1 %50
  br i1 %57, label %73, label %58

58:                                               ; preds = %53, %58
  %59 = phi i64 [ %70, %58 ], [ 0, %53 ]
  %60 = getelementptr inbounds nuw float, ptr %3, i64 %59
  %61 = getelementptr inbounds nuw i8, ptr %60, i64 16
  %62 = load <4 x float>, ptr %60, align 4, !tbaa !11, !alias.scope !61, !noalias !64
  %63 = load <4 x float>, ptr %61, align 4, !tbaa !11, !alias.scope !61, !noalias !64
  %64 = getelementptr float, ptr %56, i64 %59
  %65 = getelementptr i8, ptr %64, i64 16
  %66 = load <4 x float>, ptr %64, align 4, !tbaa !11, !alias.scope !64
  %67 = load <4 x float>, ptr %65, align 4, !tbaa !11, !alias.scope !64
  %68 = fadd <4 x float> %62, %66
  %69 = fadd <4 x float> %63, %67
  store <4 x float> %68, ptr %60, align 4, !tbaa !11, !alias.scope !61, !noalias !64
  store <4 x float> %69, ptr %61, align 4, !tbaa !11, !alias.scope !61, !noalias !64
  %70 = add nuw i64 %59, 8
  %71 = icmp eq i64 %70, %51
  br i1 %71, label %72, label %58, !llvm.loop !66

72:                                               ; preds = %58
  br i1 %52, label %84, label %73

73:                                               ; preds = %53, %72
  %74 = phi i64 [ 0, %53 ], [ %51, %72 ]
  br label %75

75:                                               ; preds = %73, %75
  %76 = phi i64 [ %82, %75 ], [ %74, %73 ]
  %77 = getelementptr inbounds nuw float, ptr %3, i64 %76
  %78 = load float, ptr %77, align 4, !tbaa !11
  %79 = getelementptr float, ptr %56, i64 %76
  %80 = load float, ptr %79, align 4, !tbaa !11
  %81 = fadd float %78, %80
  store float %81, ptr %77, align 4, !tbaa !11
  %82 = add nuw nsw i64 %76, 1
  %83 = icmp eq i64 %82, %37
  br i1 %83, label %84, label %75, !llvm.loop !67

84:                                               ; preds = %75, %72
  %85 = add nuw nsw i64 %54, 1
  %86 = icmp eq i64 %85, %37
  br i1 %86, label %87, label %53, !llvm.loop !16

87:                                               ; preds = %84, %5
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @dgefa(ptr noundef captures(none) %0, i32 noundef %1, i32 noundef %2, ptr noundef writeonly captures(none) %3, ptr noundef writeonly captures(none) initializes((0, 4)) %4) local_unnamed_addr #6 {
  store i32 0, ptr %4, align 4, !tbaa !17
  %6 = add i32 %2, -1
  %7 = icmp sgt i32 %2, 1
  br i1 %7, label %8, label %181

8:                                                ; preds = %5
  %9 = sext i32 %1 to i64
  %10 = zext nneg i32 %2 to i64
  %11 = zext nneg i32 %6 to i64
  %12 = zext nneg i32 %2 to i64
  %13 = shl nsw i64 %9, 2
  %14 = add nsw i64 %13, 4
  %15 = shl nuw nsw i64 %10, 2
  %16 = add nsw i64 %15, -4
  %17 = mul i64 %16, %9
  %18 = getelementptr i8, ptr %0, i64 %17
  %19 = getelementptr i8, ptr %18, i64 %15
  %20 = getelementptr i8, ptr %0, i64 %15
  %21 = icmp slt i32 %1, 0
  br label %22

22:                                               ; preds = %8, %178
  %23 = phi i64 [ 0, %8 ], [ %42, %178 ]
  %24 = phi i64 [ 1, %8 ], [ %179, %178 ]
  %25 = xor i64 %23, -1
  %26 = add nsw i64 %25, %10
  %27 = xor i64 %23, -1
  %28 = add nsw i64 %27, %10
  %29 = add nuw i64 %23, 1
  %30 = mul i64 %14, %29
  %31 = getelementptr i8, ptr %0, i64 %30
  %32 = shl nuw nsw i64 %23, 2
  %33 = getelementptr i8, ptr %0, i64 %32
  %34 = getelementptr i8, ptr %33, i64 4
  %35 = trunc i64 %23 to i32
  %36 = mul i32 %1, %35
  %37 = sext i32 %36 to i64
  %38 = shl nsw i64 %37, 2
  %39 = getelementptr i8, ptr %34, i64 %38
  %40 = getelementptr i8, ptr %20, i64 %38
  %41 = trunc i64 %23 to i32
  %42 = add nuw nsw i64 %23, 1
  %43 = sub nsw i64 %10, %23
  %44 = mul nsw i64 %23, %9
  %45 = mul nsw i32 %1, %41
  %46 = sext i32 %45 to i64
  %47 = getelementptr float, ptr %0, i64 %23
  %48 = getelementptr float, ptr %47, i64 %46
  %49 = icmp eq i64 %43, 1
  br i1 %49, label %74, label %50

50:                                               ; preds = %22
  %51 = load float, ptr %48, align 4, !tbaa !11
  %52 = tail call float @llvm.fabs.f32(float %51)
  br label %53

53:                                               ; preds = %53, %50
  %54 = phi i64 [ 1, %50 ], [ %64, %53 ]
  %55 = phi i32 [ 0, %50 ], [ %63, %53 ]
  %56 = phi float [ %52, %50 ], [ %61, %53 ]
  %57 = getelementptr inbounds nuw float, ptr %48, i64 %54
  %58 = load float, ptr %57, align 4, !tbaa !11
  %59 = tail call float @llvm.fabs.f32(float %58)
  %60 = fcmp ogt float %59, %56
  %61 = select i1 %60, float %59, float %56
  %62 = trunc nuw nsw i64 %54 to i32
  %63 = select i1 %60, i32 %62, i32 %55
  %64 = add nuw nsw i64 %54, 1
  %65 = icmp eq i64 %64, %43
  br i1 %65, label %66, label %53, !llvm.loop !29

66:                                               ; preds = %53
  %67 = add nsw i32 %63, %41
  %68 = getelementptr inbounds nuw i32, ptr %3, i64 %23
  store i32 %67, ptr %68, align 4, !tbaa !17
  %69 = sext i32 %67 to i64
  %70 = getelementptr float, ptr %0, i64 %44
  %71 = getelementptr float, ptr %70, i64 %69
  %72 = load float, ptr %71, align 4, !tbaa !11
  %73 = fcmp une float %72, 0.000000e+00
  br i1 %73, label %84, label %177

74:                                               ; preds = %22
  %75 = getelementptr inbounds nuw i32, ptr %3, i64 %23
  store i32 %41, ptr %75, align 4, !tbaa !17
  %76 = shl i64 %23, 32
  %77 = ashr exact i64 %76, 32
  %78 = getelementptr float, ptr %0, i64 %44
  %79 = getelementptr float, ptr %78, i64 %77
  %80 = load float, ptr %79, align 4, !tbaa !11
  %81 = fcmp une float %80, 0.000000e+00
  br i1 %81, label %82, label %177

82:                                               ; preds = %74
  %83 = load float, ptr %48, align 4, !tbaa !11
  br label %88

84:                                               ; preds = %66
  %85 = icmp eq i32 %63, 0
  %86 = load float, ptr %48, align 4, !tbaa !11
  br i1 %85, label %88, label %87

87:                                               ; preds = %84
  store float %86, ptr %71, align 4, !tbaa !11
  store float %72, ptr %48, align 4, !tbaa !11
  br label %88

88:                                               ; preds = %82, %87, %84
  %89 = phi i1 [ false, %87 ], [ true, %84 ], [ true, %82 ]
  %90 = phi i64 [ %69, %87 ], [ %69, %84 ], [ %77, %82 ]
  %91 = phi float [ %72, %87 ], [ %86, %84 ], [ %83, %82 ]
  %92 = fdiv float -1.000000e+00, %91
  %93 = sub nsw i64 %10, %42
  %94 = getelementptr i8, ptr %48, i64 4
  %95 = icmp ult i64 %26, 8
  br i1 %95, label %112, label %96

96:                                               ; preds = %88
  %97 = and i64 %26, -8
  %98 = insertelement <4 x float> poison, float %92, i64 0
  %99 = shufflevector <4 x float> %98, <4 x float> poison, <4 x i32> zeroinitializer
  br label %100

100:                                              ; preds = %100, %96
  %101 = phi i64 [ 0, %96 ], [ %108, %100 ]
  %102 = getelementptr inbounds nuw float, ptr %94, i64 %101
  %103 = getelementptr inbounds nuw i8, ptr %102, i64 16
  %104 = load <4 x float>, ptr %102, align 4, !tbaa !11
  %105 = load <4 x float>, ptr %103, align 4, !tbaa !11
  %106 = fmul <4 x float> %99, %104
  %107 = fmul <4 x float> %99, %105
  store <4 x float> %106, ptr %102, align 4, !tbaa !11
  store <4 x float> %107, ptr %103, align 4, !tbaa !11
  %108 = add nuw i64 %101, 8
  %109 = icmp eq i64 %108, %97
  br i1 %109, label %110, label %100, !llvm.loop !68

110:                                              ; preds = %100
  %111 = icmp eq i64 %26, %97
  br i1 %111, label %121, label %112

112:                                              ; preds = %88, %110
  %113 = phi i64 [ 0, %88 ], [ %97, %110 ]
  br label %114

114:                                              ; preds = %112, %114
  %115 = phi i64 [ %119, %114 ], [ %113, %112 ]
  %116 = getelementptr inbounds nuw float, ptr %94, i64 %115
  %117 = load float, ptr %116, align 4, !tbaa !11
  %118 = fmul float %92, %117
  store float %118, ptr %116, align 4, !tbaa !11
  %119 = add nuw nsw i64 %115, 1
  %120 = icmp eq i64 %119, %93
  br i1 %120, label %121, label %114, !llvm.loop !69

121:                                              ; preds = %114, %110
  %122 = getelementptr float, ptr %0, i64 %90
  %123 = icmp ult i64 %28, 8
  %124 = icmp ult ptr %31, %40
  %125 = icmp ult ptr %39, %19
  %126 = and i1 %124, %125
  %127 = or i1 %126, %21
  %128 = and i64 %28, -8
  %129 = icmp eq i64 %28, %128
  br label %130

130:                                              ; preds = %121, %174
  %131 = phi i64 [ %24, %121 ], [ %175, %174 ]
  %132 = mul nsw i64 %131, %9
  %133 = getelementptr float, ptr %122, i64 %132
  %134 = load float, ptr %133, align 4, !tbaa !11
  %135 = add nsw i64 %132, %23
  br i1 %89, label %139, label %136

136:                                              ; preds = %130
  %137 = getelementptr inbounds float, ptr %0, i64 %135
  %138 = load float, ptr %137, align 4, !tbaa !11
  store float %138, ptr %133, align 4, !tbaa !11
  store float %134, ptr %137, align 4, !tbaa !11
  br label %139

139:                                              ; preds = %130, %136
  %140 = getelementptr float, ptr %0, i64 %135
  %141 = getelementptr i8, ptr %140, i64 4
  %142 = fcmp oeq float %134, 0.000000e+00
  br i1 %142, label %174, label %143

143:                                              ; preds = %139
  %144 = select i1 %123, i1 true, i1 %127
  br i1 %144, label %163, label %145

145:                                              ; preds = %143
  %146 = insertelement <4 x float> poison, float %134, i64 0
  %147 = shufflevector <4 x float> %146, <4 x float> poison, <4 x i32> zeroinitializer
  br label %148

148:                                              ; preds = %148, %145
  %149 = phi i64 [ 0, %145 ], [ %160, %148 ]
  %150 = getelementptr inbounds nuw float, ptr %141, i64 %149
  %151 = getelementptr inbounds nuw i8, ptr %150, i64 16
  %152 = load <4 x float>, ptr %150, align 4, !tbaa !11, !alias.scope !70, !noalias !73
  %153 = load <4 x float>, ptr %151, align 4, !tbaa !11, !alias.scope !70, !noalias !73
  %154 = getelementptr inbounds nuw float, ptr %94, i64 %149
  %155 = getelementptr inbounds nuw i8, ptr %154, i64 16
  %156 = load <4 x float>, ptr %154, align 4, !tbaa !11, !alias.scope !73
  %157 = load <4 x float>, ptr %155, align 4, !tbaa !11, !alias.scope !73
  %158 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %147, <4 x float> %156, <4 x float> %152)
  %159 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %147, <4 x float> %157, <4 x float> %153)
  store <4 x float> %158, ptr %150, align 4, !tbaa !11, !alias.scope !70, !noalias !73
  store <4 x float> %159, ptr %151, align 4, !tbaa !11, !alias.scope !70, !noalias !73
  %160 = add nuw i64 %149, 8
  %161 = icmp eq i64 %160, %128
  br i1 %161, label %162, label %148, !llvm.loop !75

162:                                              ; preds = %148
  br i1 %129, label %174, label %163

163:                                              ; preds = %143, %162
  %164 = phi i64 [ 0, %143 ], [ %128, %162 ]
  br label %165

165:                                              ; preds = %163, %165
  %166 = phi i64 [ %172, %165 ], [ %164, %163 ]
  %167 = getelementptr inbounds nuw float, ptr %141, i64 %166
  %168 = load float, ptr %167, align 4, !tbaa !11
  %169 = getelementptr inbounds nuw float, ptr %94, i64 %166
  %170 = load float, ptr %169, align 4, !tbaa !11
  %171 = tail call float @llvm.fmuladd.f32(float %134, float %170, float %168)
  store float %171, ptr %167, align 4, !tbaa !11
  %172 = add nuw nsw i64 %166, 1
  %173 = icmp eq i64 %172, %93
  br i1 %173, label %174, label %165, !llvm.loop !76

174:                                              ; preds = %165, %162, %139
  %175 = add nuw nsw i64 %131, 1
  %176 = icmp eq i64 %175, %12
  br i1 %176, label %178, label %130, !llvm.loop !34

177:                                              ; preds = %74, %66
  store i32 %41, ptr %4, align 4, !tbaa !17
  br label %178

178:                                              ; preds = %174, %177
  %179 = add nuw nsw i64 %24, 1
  %180 = icmp eq i64 %42, %11
  br i1 %180, label %181, label %22, !llvm.loop !35

181:                                              ; preds = %178, %5
  %182 = sext i32 %6 to i64
  %183 = getelementptr inbounds i32, ptr %3, i64 %182
  store i32 %6, ptr %183, align 4, !tbaa !17
  %184 = add i32 %1, 1
  %185 = mul i32 %6, %184
  %186 = sext i32 %185 to i64
  %187 = getelementptr inbounds float, ptr %0, i64 %186
  %188 = load float, ptr %187, align 4, !tbaa !11
  %189 = fcmp oeq float %188, 0.000000e+00
  br i1 %189, label %190, label %191

190:                                              ; preds = %181
  store i32 %6, ptr %4, align 4, !tbaa !17
  br label %191

191:                                              ; preds = %190, %181
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @dgesl(ptr noundef readonly captures(none) %0, i32 noundef %1, i32 noundef %2, ptr noundef readonly captures(none) %3, ptr noundef captures(none) %4, i32 noundef %5) local_unnamed_addr #6 {
  %7 = add i32 %2, -1
  %8 = icmp eq i32 %5, 0
  br i1 %8, label %14, label %9

9:                                                ; preds = %6
  %10 = icmp sgt i32 %2, 0
  br i1 %10, label %11, label %304

11:                                               ; preds = %9
  %12 = sext i32 %1 to i64
  %13 = zext nneg i32 %2 to i64
  br label %184

14:                                               ; preds = %6
  %15 = icmp sgt i32 %2, 1
  br i1 %15, label %16, label %98

16:                                               ; preds = %14
  %17 = add i32 %1, 1
  %18 = zext nneg i32 %2 to i64
  %19 = zext nneg i32 %7 to i64
  %20 = shl nuw nsw i64 %18, 2
  %21 = getelementptr i8, ptr %4, i64 %20
  %22 = getelementptr i8, ptr %0, i64 4
  br label %23

23:                                               ; preds = %16, %96
  %24 = phi i64 [ 0, %16 ], [ %50, %96 ]
  %25 = xor i64 %24, -1
  %26 = add nsw i64 %25, %18
  %27 = shl nuw nsw i64 %24, 2
  %28 = getelementptr i8, ptr %4, i64 %27
  %29 = getelementptr i8, ptr %28, i64 4
  %30 = trunc i64 %24 to i32
  %31 = mul i32 %17, %30
  %32 = sext i32 %31 to i64
  %33 = shl nsw i64 %32, 2
  %34 = getelementptr i8, ptr %22, i64 %33
  %35 = sub nsw i64 %18, %24
  %36 = shl i64 %35, 2
  %37 = getelementptr i8, ptr %0, i64 %36
  %38 = getelementptr i8, ptr %37, i64 %33
  %39 = getelementptr inbounds nuw i32, ptr %3, i64 %24
  %40 = load i32, ptr %39, align 4, !tbaa !17
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds float, ptr %4, i64 %41
  %43 = load float, ptr %42, align 4, !tbaa !11
  %44 = zext i32 %40 to i64
  %45 = icmp eq i64 %24, %44
  br i1 %45, label %49, label %46

46:                                               ; preds = %23
  %47 = getelementptr inbounds nuw float, ptr %4, i64 %24
  %48 = load float, ptr %47, align 4, !tbaa !11
  store float %48, ptr %42, align 4, !tbaa !11
  store float %43, ptr %47, align 4, !tbaa !11
  br label %49

49:                                               ; preds = %46, %23
  %50 = add nuw nsw i64 %24, 1
  %51 = trunc nuw nsw i64 %24 to i32
  %52 = mul i32 %17, %51
  %53 = sext i32 %52 to i64
  %54 = getelementptr float, ptr %0, i64 %53
  %55 = getelementptr i8, ptr %54, i64 4
  %56 = getelementptr inbounds nuw float, ptr %4, i64 %50
  %57 = fcmp oeq float %43, 0.000000e+00
  br i1 %57, label %96, label %58

58:                                               ; preds = %49
  %59 = sub nsw i64 %18, %50
  %60 = icmp ult i64 %26, 8
  br i1 %60, label %85, label %61

61:                                               ; preds = %58
  %62 = icmp ult ptr %29, %38
  %63 = icmp ult ptr %34, %21
  %64 = and i1 %62, %63
  br i1 %64, label %85, label %65

65:                                               ; preds = %61
  %66 = and i64 %26, -8
  %67 = insertelement <4 x float> poison, float %43, i64 0
  %68 = shufflevector <4 x float> %67, <4 x float> poison, <4 x i32> zeroinitializer
  br label %69

69:                                               ; preds = %69, %65
  %70 = phi i64 [ 0, %65 ], [ %81, %69 ]
  %71 = getelementptr inbounds nuw float, ptr %56, i64 %70
  %72 = getelementptr inbounds nuw i8, ptr %71, i64 16
  %73 = load <4 x float>, ptr %71, align 4, !tbaa !11, !alias.scope !77, !noalias !80
  %74 = load <4 x float>, ptr %72, align 4, !tbaa !11, !alias.scope !77, !noalias !80
  %75 = getelementptr inbounds nuw float, ptr %55, i64 %70
  %76 = getelementptr inbounds nuw i8, ptr %75, i64 16
  %77 = load <4 x float>, ptr %75, align 4, !tbaa !11, !alias.scope !80
  %78 = load <4 x float>, ptr %76, align 4, !tbaa !11, !alias.scope !80
  %79 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %68, <4 x float> %77, <4 x float> %73)
  %80 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %68, <4 x float> %78, <4 x float> %74)
  store <4 x float> %79, ptr %71, align 4, !tbaa !11, !alias.scope !77, !noalias !80
  store <4 x float> %80, ptr %72, align 4, !tbaa !11, !alias.scope !77, !noalias !80
  %81 = add nuw i64 %70, 8
  %82 = icmp eq i64 %81, %66
  br i1 %82, label %83, label %69, !llvm.loop !82

83:                                               ; preds = %69
  %84 = icmp eq i64 %26, %66
  br i1 %84, label %96, label %85

85:                                               ; preds = %61, %58, %83
  %86 = phi i64 [ 0, %61 ], [ 0, %58 ], [ %66, %83 ]
  br label %87

87:                                               ; preds = %85, %87
  %88 = phi i64 [ %94, %87 ], [ %86, %85 ]
  %89 = getelementptr inbounds nuw float, ptr %56, i64 %88
  %90 = load float, ptr %89, align 4, !tbaa !11
  %91 = getelementptr inbounds nuw float, ptr %55, i64 %88
  %92 = load float, ptr %91, align 4, !tbaa !11
  %93 = tail call float @llvm.fmuladd.f32(float %43, float %92, float %90)
  store float %93, ptr %89, align 4, !tbaa !11
  %94 = add nuw nsw i64 %88, 1
  %95 = icmp eq i64 %94, %59
  br i1 %95, label %96, label %87, !llvm.loop !83

96:                                               ; preds = %87, %83, %49
  %97 = icmp eq i64 %50, %19
  br i1 %97, label %98, label %23, !llvm.loop !23

98:                                               ; preds = %96, %14
  %99 = icmp sgt i32 %2, 0
  br i1 %99, label %100, label %304

100:                                              ; preds = %98
  %101 = zext nneg i32 %2 to i64
  %102 = sext i32 %1 to i64
  %103 = zext nneg i32 %2 to i64
  %104 = shl nuw nsw i64 %101, 2
  %105 = add nsw i64 %104, -4
  %106 = add nuw nsw i64 %101, 4611686018427387903
  %107 = mul i64 %106, %102
  %108 = shl i64 %107, 2
  %109 = mul nsw i64 %102, -4
  %110 = shl nsw i64 %102, 2
  %111 = sub nuw nsw i64 -4, %110
  %112 = getelementptr i8, ptr %0, i64 %108
  %113 = getelementptr i8, ptr %0, i64 %108
  %114 = getelementptr i8, ptr %113, i64 %104
  %115 = getelementptr i8, ptr %114, i64 -4
  br label %116

116:                                              ; preds = %100, %182
  %117 = phi i64 [ 0, %100 ], [ %127, %182 ]
  %118 = xor i64 %117, -1
  %119 = add nsw i64 %118, %101
  %120 = shl i64 %117, 2
  %121 = sub i64 %105, %120
  %122 = getelementptr i8, ptr %4, i64 %121
  %123 = mul i64 %109, %117
  %124 = getelementptr i8, ptr %112, i64 %123
  %125 = mul i64 %111, %117
  %126 = getelementptr i8, ptr %115, i64 %125
  %127 = add nuw nsw i64 %117, 1
  %128 = trunc i64 %127 to i32
  %129 = sub nsw i64 %101, %127
  %130 = sub nsw i32 %2, %128
  %131 = getelementptr inbounds float, ptr %4, i64 %129
  %132 = load float, ptr %131, align 4, !tbaa !11
  %133 = mul nsw i64 %129, %102
  %134 = mul nsw i32 %130, %1
  %135 = sext i32 %134 to i64
  %136 = getelementptr float, ptr %0, i64 %129
  %137 = getelementptr float, ptr %136, i64 %135
  %138 = load float, ptr %137, align 4, !tbaa !11
  %139 = fdiv float %132, %138
  store float %139, ptr %131, align 4, !tbaa !11
  %140 = fneg float %139
  %141 = getelementptr inbounds float, ptr %0, i64 %133
  %142 = icmp slt i64 %129, 1
  %143 = fcmp oeq float %139, 0.000000e+00
  %144 = or i1 %142, %143
  br i1 %144, label %182, label %145

145:                                              ; preds = %116
  %146 = icmp ult i64 %119, 8
  br i1 %146, label %171, label %147

147:                                              ; preds = %145
  %148 = icmp ult ptr %4, %126
  %149 = icmp ult ptr %124, %122
  %150 = and i1 %148, %149
  br i1 %150, label %171, label %151

151:                                              ; preds = %147
  %152 = and i64 %119, -8
  %153 = insertelement <4 x float> poison, float %140, i64 0
  %154 = shufflevector <4 x float> %153, <4 x float> poison, <4 x i32> zeroinitializer
  br label %155

155:                                              ; preds = %155, %151
  %156 = phi i64 [ 0, %151 ], [ %167, %155 ]
  %157 = getelementptr inbounds nuw float, ptr %4, i64 %156
  %158 = getelementptr inbounds nuw i8, ptr %157, i64 16
  %159 = load <4 x float>, ptr %157, align 4, !tbaa !11, !alias.scope !84, !noalias !87
  %160 = load <4 x float>, ptr %158, align 4, !tbaa !11, !alias.scope !84, !noalias !87
  %161 = getelementptr inbounds nuw float, ptr %141, i64 %156
  %162 = getelementptr inbounds nuw i8, ptr %161, i64 16
  %163 = load <4 x float>, ptr %161, align 4, !tbaa !11, !alias.scope !87
  %164 = load <4 x float>, ptr %162, align 4, !tbaa !11, !alias.scope !87
  %165 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %154, <4 x float> %163, <4 x float> %159)
  %166 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %154, <4 x float> %164, <4 x float> %160)
  store <4 x float> %165, ptr %157, align 4, !tbaa !11, !alias.scope !84, !noalias !87
  store <4 x float> %166, ptr %158, align 4, !tbaa !11, !alias.scope !84, !noalias !87
  %167 = add nuw i64 %156, 8
  %168 = icmp eq i64 %167, %152
  br i1 %168, label %169, label %155, !llvm.loop !89

169:                                              ; preds = %155
  %170 = icmp eq i64 %119, %152
  br i1 %170, label %182, label %171

171:                                              ; preds = %147, %145, %169
  %172 = phi i64 [ 0, %147 ], [ 0, %145 ], [ %152, %169 ]
  br label %173

173:                                              ; preds = %171, %173
  %174 = phi i64 [ %180, %173 ], [ %172, %171 ]
  %175 = getelementptr inbounds nuw float, ptr %4, i64 %174
  %176 = load float, ptr %175, align 4, !tbaa !11
  %177 = getelementptr inbounds nuw float, ptr %141, i64 %174
  %178 = load float, ptr %177, align 4, !tbaa !11
  %179 = tail call float @llvm.fmuladd.f32(float %140, float %178, float %176)
  store float %179, ptr %175, align 4, !tbaa !11
  %180 = add nuw nsw i64 %174, 1
  %181 = icmp eq i64 %180, %129
  br i1 %181, label %182, label %173, !llvm.loop !90

182:                                              ; preds = %173, %169, %116
  %183 = icmp eq i64 %127, %103
  br i1 %183, label %304, label %116, !llvm.loop !26

184:                                              ; preds = %11, %227
  %185 = phi i64 [ 0, %11 ], [ %237, %227 ]
  %186 = trunc i64 %185 to i32
  %187 = mul nsw i64 %185, %12
  %188 = mul nsw i32 %1, %186
  %189 = getelementptr inbounds float, ptr %0, i64 %187
  %190 = icmp eq i64 %185, 0
  br i1 %190, label %227, label %191

191:                                              ; preds = %184
  %192 = icmp samesign ult i64 %185, 8
  br i1 %192, label %214, label %193

193:                                              ; preds = %191
  %194 = and i64 %185, 9223372036854775800
  br label %195

195:                                              ; preds = %195, %193
  %196 = phi i64 [ 0, %193 ], [ %210, %195 ]
  %197 = phi float [ 0.000000e+00, %193 ], [ %209, %195 ]
  %198 = getelementptr inbounds nuw float, ptr %189, i64 %196
  %199 = getelementptr inbounds nuw i8, ptr %198, i64 16
  %200 = load <4 x float>, ptr %198, align 4, !tbaa !11
  %201 = load <4 x float>, ptr %199, align 4, !tbaa !11
  %202 = getelementptr inbounds nuw float, ptr %4, i64 %196
  %203 = getelementptr inbounds nuw i8, ptr %202, i64 16
  %204 = load <4 x float>, ptr %202, align 4, !tbaa !11
  %205 = load <4 x float>, ptr %203, align 4, !tbaa !11
  %206 = fmul <4 x float> %200, %204
  %207 = fmul <4 x float> %201, %205
  %208 = tail call float @llvm.vector.reduce.fadd.v4f32(float %197, <4 x float> %206)
  %209 = tail call float @llvm.vector.reduce.fadd.v4f32(float %208, <4 x float> %207)
  %210 = add nuw i64 %196, 8
  %211 = icmp eq i64 %210, %194
  br i1 %211, label %212, label %195, !llvm.loop !91

212:                                              ; preds = %195
  %213 = icmp eq i64 %185, %194
  br i1 %213, label %227, label %214

214:                                              ; preds = %191, %212
  %215 = phi i64 [ 0, %191 ], [ %194, %212 ]
  %216 = phi float [ 0.000000e+00, %191 ], [ %209, %212 ]
  br label %217

217:                                              ; preds = %214, %217
  %218 = phi i64 [ %225, %217 ], [ %215, %214 ]
  %219 = phi float [ %224, %217 ], [ %216, %214 ]
  %220 = getelementptr inbounds nuw float, ptr %189, i64 %218
  %221 = load float, ptr %220, align 4, !tbaa !11
  %222 = getelementptr inbounds nuw float, ptr %4, i64 %218
  %223 = load float, ptr %222, align 4, !tbaa !11
  %224 = tail call float @llvm.fmuladd.f32(float %221, float %223, float %219)
  %225 = add nuw nsw i64 %218, 1
  %226 = icmp eq i64 %225, %185
  br i1 %226, label %227, label %217, !llvm.loop !92

227:                                              ; preds = %217, %212, %184
  %228 = phi float [ 0.000000e+00, %184 ], [ %209, %212 ], [ %224, %217 ]
  %229 = getelementptr inbounds nuw float, ptr %4, i64 %185
  %230 = load float, ptr %229, align 4, !tbaa !11
  %231 = fsub float %230, %228
  %232 = sext i32 %188 to i64
  %233 = getelementptr float, ptr %0, i64 %185
  %234 = getelementptr float, ptr %233, i64 %232
  %235 = load float, ptr %234, align 4, !tbaa !11
  %236 = fdiv float %231, %235
  store float %236, ptr %229, align 4, !tbaa !11
  %237 = add nuw nsw i64 %185, 1
  %238 = icmp eq i64 %237, %13
  br i1 %238, label %239, label %184, !llvm.loop !93

239:                                              ; preds = %227
  %240 = icmp sgt i32 %2, 2
  br i1 %240, label %241, label %304

241:                                              ; preds = %239
  %242 = add i32 %1, 1
  %243 = zext nneg i32 %2 to i64
  %244 = zext nneg i32 %7 to i64
  br label %245

245:                                              ; preds = %241, %302
  %246 = phi i64 [ 1, %241 ], [ %247, %302 ]
  %247 = add nuw nsw i64 %246, 1
  %248 = sub nsw i64 %243, %247
  %249 = getelementptr inbounds float, ptr %4, i64 %248
  %250 = load float, ptr %249, align 4, !tbaa !11
  %251 = trunc nsw i64 %248 to i32
  %252 = mul i32 %242, %251
  %253 = sext i32 %252 to i64
  %254 = getelementptr float, ptr %0, i64 %253
  %255 = getelementptr i8, ptr %254, i64 4
  %256 = getelementptr i8, ptr %249, i64 4
  %257 = icmp samesign ult i64 %246, 8
  br i1 %257, label %279, label %258

258:                                              ; preds = %245
  %259 = and i64 %246, 9223372036854775800
  br label %260

260:                                              ; preds = %260, %258
  %261 = phi i64 [ 0, %258 ], [ %275, %260 ]
  %262 = phi float [ 0.000000e+00, %258 ], [ %274, %260 ]
  %263 = getelementptr inbounds nuw float, ptr %255, i64 %261
  %264 = getelementptr inbounds nuw i8, ptr %263, i64 16
  %265 = load <4 x float>, ptr %263, align 4, !tbaa !11
  %266 = load <4 x float>, ptr %264, align 4, !tbaa !11
  %267 = getelementptr inbounds nuw float, ptr %256, i64 %261
  %268 = getelementptr inbounds nuw i8, ptr %267, i64 16
  %269 = load <4 x float>, ptr %267, align 4, !tbaa !11
  %270 = load <4 x float>, ptr %268, align 4, !tbaa !11
  %271 = fmul <4 x float> %265, %269
  %272 = fmul <4 x float> %266, %270
  %273 = tail call float @llvm.vector.reduce.fadd.v4f32(float %262, <4 x float> %271)
  %274 = tail call float @llvm.vector.reduce.fadd.v4f32(float %273, <4 x float> %272)
  %275 = add nuw i64 %261, 8
  %276 = icmp eq i64 %275, %259
  br i1 %276, label %277, label %260, !llvm.loop !94

277:                                              ; preds = %260
  %278 = icmp eq i64 %246, %259
  br i1 %278, label %292, label %279

279:                                              ; preds = %245, %277
  %280 = phi i64 [ 0, %245 ], [ %259, %277 ]
  %281 = phi float [ 0.000000e+00, %245 ], [ %274, %277 ]
  br label %282

282:                                              ; preds = %279, %282
  %283 = phi i64 [ %290, %282 ], [ %280, %279 ]
  %284 = phi float [ %289, %282 ], [ %281, %279 ]
  %285 = getelementptr inbounds nuw float, ptr %255, i64 %283
  %286 = load float, ptr %285, align 4, !tbaa !11
  %287 = getelementptr inbounds nuw float, ptr %256, i64 %283
  %288 = load float, ptr %287, align 4, !tbaa !11
  %289 = tail call float @llvm.fmuladd.f32(float %286, float %288, float %284)
  %290 = add nuw nsw i64 %283, 1
  %291 = icmp eq i64 %290, %246
  br i1 %291, label %292, label %282, !llvm.loop !95

292:                                              ; preds = %282, %277
  %293 = phi float [ %274, %277 ], [ %289, %282 ]
  %294 = fadd float %250, %293
  store float %294, ptr %249, align 4, !tbaa !11
  %295 = getelementptr inbounds i32, ptr %3, i64 %248
  %296 = load i32, ptr %295, align 4, !tbaa !17
  %297 = icmp eq i32 %296, %251
  br i1 %297, label %302, label %298

298:                                              ; preds = %292
  %299 = sext i32 %296 to i64
  %300 = getelementptr inbounds float, ptr %4, i64 %299
  %301 = load float, ptr %300, align 4, !tbaa !11
  store float %294, ptr %300, align 4, !tbaa !11
  store float %301, ptr %249, align 4, !tbaa !11
  br label %302

302:                                              ; preds = %292, %298
  %303 = icmp eq i64 %247, %244
  br i1 %303, label %304, label %245, !llvm.loop !96

304:                                              ; preds = %302, %182, %9, %98, %239
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @dmxpy(i32 noundef %0, ptr noundef captures(none) %1, i32 noundef %2, i32 noundef %3, ptr noundef readonly captures(none) %4, ptr noundef readonly captures(none) %5) local_unnamed_addr #6 {
  %7 = and i32 %2, -2147483647
  %8 = icmp eq i32 %7, 1
  %9 = icmp sgt i32 %0, 0
  %10 = and i1 %8, %9
  br i1 %10, label %11, label %59

11:                                               ; preds = %6
  %12 = zext nneg i32 %0 to i64
  %13 = icmp ult i32 %0, 8
  br i1 %13, label %47, label %14

14:                                               ; preds = %11
  %15 = shl nuw nsw i64 %12, 2
  %16 = getelementptr i8, ptr %1, i64 %15
  %17 = getelementptr i8, ptr %4, i64 4
  %18 = getelementptr i8, ptr %5, i64 %15
  %19 = icmp ult ptr %1, %17
  %20 = icmp ult ptr %4, %16
  %21 = and i1 %19, %20
  %22 = icmp ult ptr %1, %18
  %23 = icmp ult ptr %5, %16
  %24 = and i1 %22, %23
  %25 = or i1 %21, %24
  br i1 %25, label %47, label %26

26:                                               ; preds = %14
  %27 = and i64 %12, 2147483640
  %28 = load float, ptr %4, align 4, !tbaa !11, !alias.scope !97
  %29 = insertelement <4 x float> poison, float %28, i64 0
  %30 = shufflevector <4 x float> %29, <4 x float> poison, <4 x i32> zeroinitializer
  br label %31

31:                                               ; preds = %31, %26
  %32 = phi i64 [ 0, %26 ], [ %43, %31 ]
  %33 = getelementptr inbounds nuw float, ptr %1, i64 %32
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 16
  %35 = load <4 x float>, ptr %33, align 4, !tbaa !11, !alias.scope !100, !noalias !102
  %36 = load <4 x float>, ptr %34, align 4, !tbaa !11, !alias.scope !100, !noalias !102
  %37 = getelementptr inbounds nuw float, ptr %5, i64 %32
  %38 = getelementptr inbounds nuw i8, ptr %37, i64 16
  %39 = load <4 x float>, ptr %37, align 4, !tbaa !11, !alias.scope !104
  %40 = load <4 x float>, ptr %38, align 4, !tbaa !11, !alias.scope !104
  %41 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %30, <4 x float> %39, <4 x float> %35)
  %42 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %30, <4 x float> %40, <4 x float> %36)
  store <4 x float> %41, ptr %33, align 4, !tbaa !11, !alias.scope !100, !noalias !102
  store <4 x float> %42, ptr %34, align 4, !tbaa !11, !alias.scope !100, !noalias !102
  %43 = add nuw i64 %32, 8
  %44 = icmp eq i64 %43, %27
  br i1 %44, label %45, label %31, !llvm.loop !105

45:                                               ; preds = %31
  %46 = icmp eq i64 %27, %12
  br i1 %46, label %59, label %47

47:                                               ; preds = %14, %11, %45
  %48 = phi i64 [ 0, %14 ], [ 0, %11 ], [ %27, %45 ]
  br label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %57, %49 ], [ %48, %47 ]
  %51 = getelementptr inbounds nuw float, ptr %1, i64 %50
  %52 = load float, ptr %51, align 4, !tbaa !11
  %53 = load float, ptr %4, align 4, !tbaa !11
  %54 = getelementptr inbounds nuw float, ptr %5, i64 %50
  %55 = load float, ptr %54, align 4, !tbaa !11
  %56 = tail call float @llvm.fmuladd.f32(float %53, float %55, float %52)
  store float %56, ptr %51, align 4, !tbaa !11
  %57 = add nuw nsw i64 %50, 1
  %58 = icmp eq i64 %57, %12
  br i1 %58, label %59, label %49, !llvm.loop !106

59:                                               ; preds = %49, %45, %6
  %60 = srem i32 %2, 4
  %61 = icmp sgt i32 %60, 1
  br i1 %61, label %62, label %154

62:                                               ; preds = %59
  br i1 %9, label %63, label %305

63:                                               ; preds = %62
  %64 = add nsw i32 %60, -1
  %65 = add nsw i32 %60, -2
  %66 = zext nneg i32 %65 to i64
  %67 = getelementptr inbounds nuw float, ptr %4, i64 %66
  %68 = mul nuw nsw i32 %65, %3
  %69 = zext nneg i32 %64 to i64
  %70 = getelementptr inbounds nuw float, ptr %4, i64 %69
  %71 = mul nsw i32 %64, %3
  %72 = sext i32 %68 to i64
  %73 = sext i32 %71 to i64
  %74 = zext nneg i32 %0 to i64
  %75 = getelementptr float, ptr %5, i64 %72
  %76 = getelementptr float, ptr %5, i64 %73
  %77 = icmp ult i32 %0, 16
  br i1 %77, label %138, label %78

78:                                               ; preds = %63
  %79 = shl nuw nsw i64 %74, 2
  %80 = getelementptr i8, ptr %1, i64 %79
  %81 = shl nuw nsw i64 %69, 2
  %82 = getelementptr i8, ptr %4, i64 %81
  %83 = getelementptr i8, ptr %82, i64 4
  %84 = shl nuw nsw i64 %66, 2
  %85 = getelementptr i8, ptr %4, i64 %84
  %86 = getelementptr i8, ptr %85, i64 4
  %87 = add nsw i64 %73, %74
  %88 = shl nsw i64 %87, 2
  %89 = getelementptr i8, ptr %5, i64 %88
  %90 = add nsw i64 %72, %74
  %91 = shl nsw i64 %90, 2
  %92 = getelementptr i8, ptr %5, i64 %91
  %93 = icmp ult ptr %1, %83
  %94 = icmp ult ptr %70, %80
  %95 = and i1 %93, %94
  %96 = icmp ult ptr %1, %86
  %97 = icmp ult ptr %67, %80
  %98 = and i1 %96, %97
  %99 = or i1 %95, %98
  %100 = icmp ult ptr %1, %89
  %101 = icmp ult ptr %76, %80
  %102 = and i1 %100, %101
  %103 = or i1 %99, %102
  %104 = icmp ult ptr %1, %92
  %105 = icmp ult ptr %75, %80
  %106 = and i1 %104, %105
  %107 = or i1 %103, %106
  br i1 %107, label %138, label %108

108:                                              ; preds = %78
  %109 = and i64 %74, 2147483640
  %110 = load float, ptr %67, align 4, !tbaa !11, !alias.scope !107
  %111 = insertelement <4 x float> poison, float %110, i64 0
  %112 = shufflevector <4 x float> %111, <4 x float> poison, <4 x i32> zeroinitializer
  %113 = load float, ptr %70, align 4, !tbaa !11, !alias.scope !110
  %114 = insertelement <4 x float> poison, float %113, i64 0
  %115 = shufflevector <4 x float> %114, <4 x float> poison, <4 x i32> zeroinitializer
  br label %116

116:                                              ; preds = %116, %108
  %117 = phi i64 [ 0, %108 ], [ %134, %116 ]
  %118 = getelementptr inbounds nuw float, ptr %1, i64 %117
  %119 = getelementptr inbounds nuw i8, ptr %118, i64 16
  %120 = load <4 x float>, ptr %118, align 4, !tbaa !11, !alias.scope !112, !noalias !114
  %121 = load <4 x float>, ptr %119, align 4, !tbaa !11, !alias.scope !112, !noalias !114
  %122 = getelementptr float, ptr %75, i64 %117
  %123 = getelementptr i8, ptr %122, i64 16
  %124 = load <4 x float>, ptr %122, align 4, !tbaa !11, !alias.scope !117
  %125 = load <4 x float>, ptr %123, align 4, !tbaa !11, !alias.scope !117
  %126 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %112, <4 x float> %124, <4 x float> %120)
  %127 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %112, <4 x float> %125, <4 x float> %121)
  %128 = getelementptr float, ptr %76, i64 %117
  %129 = getelementptr i8, ptr %128, i64 16
  %130 = load <4 x float>, ptr %128, align 4, !tbaa !11, !alias.scope !118
  %131 = load <4 x float>, ptr %129, align 4, !tbaa !11, !alias.scope !118
  %132 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %115, <4 x float> %130, <4 x float> %126)
  %133 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %115, <4 x float> %131, <4 x float> %127)
  store <4 x float> %132, ptr %118, align 4, !tbaa !11, !alias.scope !112, !noalias !114
  store <4 x float> %133, ptr %119, align 4, !tbaa !11, !alias.scope !112, !noalias !114
  %134 = add nuw i64 %117, 8
  %135 = icmp eq i64 %134, %109
  br i1 %135, label %136, label %116, !llvm.loop !119

136:                                              ; preds = %116
  %137 = icmp eq i64 %109, %74
  br i1 %137, label %158, label %138

138:                                              ; preds = %78, %63, %136
  %139 = phi i64 [ 0, %78 ], [ 0, %63 ], [ %109, %136 ]
  br label %140

140:                                              ; preds = %138, %140
  %141 = phi i64 [ %152, %140 ], [ %139, %138 ]
  %142 = getelementptr inbounds nuw float, ptr %1, i64 %141
  %143 = load float, ptr %142, align 4, !tbaa !11
  %144 = load float, ptr %67, align 4, !tbaa !11
  %145 = getelementptr float, ptr %75, i64 %141
  %146 = load float, ptr %145, align 4, !tbaa !11
  %147 = tail call float @llvm.fmuladd.f32(float %144, float %146, float %143)
  %148 = load float, ptr %70, align 4, !tbaa !11
  %149 = getelementptr float, ptr %76, i64 %141
  %150 = load float, ptr %149, align 4, !tbaa !11
  %151 = tail call float @llvm.fmuladd.f32(float %148, float %150, float %147)
  store float %151, ptr %142, align 4, !tbaa !11
  %152 = add nuw nsw i64 %141, 1
  %153 = icmp eq i64 %152, %74
  br i1 %153, label %158, label %140, !llvm.loop !120

154:                                              ; preds = %59
  %155 = srem i32 %2, 8
  %156 = icmp sgt i32 %155, 3
  %157 = and i1 %156, %9
  br i1 %157, label %161, label %305

158:                                              ; preds = %140, %136
  %159 = srem i32 %2, 8
  %160 = icmp sgt i32 %159, 3
  br i1 %160, label %161, label %305

161:                                              ; preds = %154, %158
  %162 = phi i32 [ %159, %158 ], [ %155, %154 ]
  %163 = add nsw i32 %162, -1
  %164 = add nsw i32 %162, -4
  %165 = zext i32 %164 to i64
  %166 = getelementptr float, ptr %4, i64 %165
  %167 = mul i32 %164, %3
  %168 = add nsw i32 %162, -3
  %169 = zext i32 %168 to i64
  %170 = getelementptr float, ptr %4, i64 %169
  %171 = mul i32 %168, %3
  %172 = add nsw i32 %162, -2
  %173 = zext i32 %172 to i64
  %174 = getelementptr inbounds nuw float, ptr %4, i64 %173
  %175 = mul i32 %172, %3
  %176 = zext nneg i32 %163 to i64
  %177 = getelementptr inbounds nuw float, ptr %4, i64 %176
  %178 = mul i32 %163, %3
  %179 = sext i32 %167 to i64
  %180 = sext i32 %171 to i64
  %181 = sext i32 %175 to i64
  %182 = sext i32 %178 to i64
  %183 = zext nneg i32 %0 to i64
  %184 = getelementptr float, ptr %5, i64 %179
  %185 = getelementptr float, ptr %5, i64 %180
  %186 = getelementptr float, ptr %5, i64 %181
  %187 = getelementptr float, ptr %5, i64 %182
  %188 = icmp ult i32 %0, 16
  br i1 %188, label %281, label %189

189:                                              ; preds = %161
  %190 = shl nuw nsw i64 %183, 2
  %191 = getelementptr i8, ptr %1, i64 %190
  %192 = shl nuw nsw i64 %176, 2
  %193 = getelementptr i8, ptr %4, i64 %192
  %194 = getelementptr i8, ptr %193, i64 4
  %195 = shl nuw nsw i64 %173, 2
  %196 = getelementptr i8, ptr %4, i64 %195
  %197 = getelementptr i8, ptr %196, i64 4
  %198 = shl nuw nsw i64 %169, 2
  %199 = getelementptr i8, ptr %4, i64 %198
  %200 = getelementptr i8, ptr %199, i64 4
  %201 = shl nuw nsw i64 %165, 2
  %202 = getelementptr i8, ptr %4, i64 %201
  %203 = getelementptr i8, ptr %202, i64 4
  %204 = add nsw i64 %182, %183
  %205 = shl nsw i64 %204, 2
  %206 = getelementptr i8, ptr %5, i64 %205
  %207 = add nsw i64 %181, %183
  %208 = shl nsw i64 %207, 2
  %209 = getelementptr i8, ptr %5, i64 %208
  %210 = add nsw i64 %180, %183
  %211 = shl nsw i64 %210, 2
  %212 = getelementptr i8, ptr %5, i64 %211
  %213 = add nsw i64 %179, %183
  %214 = shl nsw i64 %213, 2
  %215 = getelementptr i8, ptr %5, i64 %214
  %216 = icmp ult ptr %1, %194
  %217 = icmp ult ptr %177, %191
  %218 = and i1 %216, %217
  %219 = icmp ult ptr %1, %197
  %220 = icmp ult ptr %174, %191
  %221 = and i1 %219, %220
  %222 = or i1 %218, %221
  %223 = icmp ult ptr %1, %200
  %224 = icmp ult ptr %170, %191
  %225 = and i1 %223, %224
  %226 = or i1 %222, %225
  %227 = icmp ult ptr %1, %203
  %228 = icmp ult ptr %166, %191
  %229 = and i1 %227, %228
  %230 = or i1 %226, %229
  %231 = icmp ult ptr %1, %206
  %232 = icmp ult ptr %187, %191
  %233 = and i1 %231, %232
  %234 = or i1 %230, %233
  %235 = icmp ult ptr %1, %209
  %236 = icmp ult ptr %186, %191
  %237 = and i1 %235, %236
  %238 = or i1 %234, %237
  %239 = icmp ult ptr %1, %212
  %240 = icmp ult ptr %185, %191
  %241 = and i1 %239, %240
  %242 = or i1 %238, %241
  %243 = icmp ult ptr %1, %215
  %244 = icmp ult ptr %184, %191
  %245 = and i1 %243, %244
  %246 = or i1 %242, %245
  br i1 %246, label %281, label %247

247:                                              ; preds = %189
  %248 = and i64 %183, 2147483644
  %249 = load float, ptr %166, align 4, !tbaa !11, !alias.scope !121
  %250 = insertelement <4 x float> poison, float %249, i64 0
  %251 = shufflevector <4 x float> %250, <4 x float> poison, <4 x i32> zeroinitializer
  %252 = load float, ptr %170, align 4, !tbaa !11, !alias.scope !124
  %253 = insertelement <4 x float> poison, float %252, i64 0
  %254 = shufflevector <4 x float> %253, <4 x float> poison, <4 x i32> zeroinitializer
  %255 = load float, ptr %174, align 4, !tbaa !11, !alias.scope !126
  %256 = insertelement <4 x float> poison, float %255, i64 0
  %257 = shufflevector <4 x float> %256, <4 x float> poison, <4 x i32> zeroinitializer
  %258 = load float, ptr %177, align 4, !tbaa !11, !alias.scope !128
  %259 = insertelement <4 x float> poison, float %258, i64 0
  %260 = shufflevector <4 x float> %259, <4 x float> poison, <4 x i32> zeroinitializer
  br label %261

261:                                              ; preds = %261, %247
  %262 = phi i64 [ 0, %247 ], [ %277, %261 ]
  %263 = getelementptr inbounds nuw float, ptr %1, i64 %262
  %264 = load <4 x float>, ptr %263, align 4, !tbaa !11, !alias.scope !130, !noalias !132
  %265 = getelementptr float, ptr %184, i64 %262
  %266 = load <4 x float>, ptr %265, align 4, !tbaa !11, !alias.scope !137
  %267 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %251, <4 x float> %266, <4 x float> %264)
  %268 = getelementptr float, ptr %185, i64 %262
  %269 = load <4 x float>, ptr %268, align 4, !tbaa !11, !alias.scope !138
  %270 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %254, <4 x float> %269, <4 x float> %267)
  %271 = getelementptr float, ptr %186, i64 %262
  %272 = load <4 x float>, ptr %271, align 4, !tbaa !11, !alias.scope !139
  %273 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %257, <4 x float> %272, <4 x float> %270)
  %274 = getelementptr float, ptr %187, i64 %262
  %275 = load <4 x float>, ptr %274, align 4, !tbaa !11, !alias.scope !140
  %276 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %260, <4 x float> %275, <4 x float> %273)
  store <4 x float> %276, ptr %263, align 4, !tbaa !11, !alias.scope !130, !noalias !132
  %277 = add nuw i64 %262, 4
  %278 = icmp eq i64 %277, %248
  br i1 %278, label %279, label %261, !llvm.loop !141

279:                                              ; preds = %261
  %280 = icmp eq i64 %248, %183
  br i1 %280, label %305, label %281

281:                                              ; preds = %189, %161, %279
  %282 = phi i64 [ 0, %189 ], [ 0, %161 ], [ %248, %279 ]
  br label %283

283:                                              ; preds = %281, %283
  %284 = phi i64 [ %303, %283 ], [ %282, %281 ]
  %285 = getelementptr inbounds nuw float, ptr %1, i64 %284
  %286 = load float, ptr %285, align 4, !tbaa !11
  %287 = load float, ptr %166, align 4, !tbaa !11
  %288 = getelementptr float, ptr %184, i64 %284
  %289 = load float, ptr %288, align 4, !tbaa !11
  %290 = tail call float @llvm.fmuladd.f32(float %287, float %289, float %286)
  %291 = load float, ptr %170, align 4, !tbaa !11
  %292 = getelementptr float, ptr %185, i64 %284
  %293 = load float, ptr %292, align 4, !tbaa !11
  %294 = tail call float @llvm.fmuladd.f32(float %291, float %293, float %290)
  %295 = load float, ptr %174, align 4, !tbaa !11
  %296 = getelementptr float, ptr %186, i64 %284
  %297 = load float, ptr %296, align 4, !tbaa !11
  %298 = tail call float @llvm.fmuladd.f32(float %295, float %297, float %294)
  %299 = load float, ptr %177, align 4, !tbaa !11
  %300 = getelementptr float, ptr %187, i64 %284
  %301 = load float, ptr %300, align 4, !tbaa !11
  %302 = tail call float @llvm.fmuladd.f32(float %299, float %301, float %298)
  store float %302, ptr %285, align 4, !tbaa !11
  %303 = add nuw nsw i64 %284, 1
  %304 = icmp eq i64 %303, %183
  br i1 %304, label %305, label %283, !llvm.loop !142

305:                                              ; preds = %283, %279, %154, %62, %158
  %306 = srem i32 %2, 16
  %307 = icmp sgt i32 %306, 7
  br i1 %307, label %308, label %572

308:                                              ; preds = %305
  br i1 %9, label %309, label %1059

309:                                              ; preds = %308
  %310 = add nsw i32 %306, -1
  %311 = add nsw i32 %306, -8
  %312 = zext i32 %311 to i64
  %313 = getelementptr float, ptr %4, i64 %312
  %314 = mul i32 %311, %3
  %315 = add nsw i32 %306, -7
  %316 = zext i32 %315 to i64
  %317 = getelementptr float, ptr %4, i64 %316
  %318 = mul i32 %315, %3
  %319 = add nsw i32 %306, -6
  %320 = zext i32 %319 to i64
  %321 = getelementptr float, ptr %4, i64 %320
  %322 = mul i32 %319, %3
  %323 = add nsw i32 %306, -5
  %324 = zext i32 %323 to i64
  %325 = getelementptr float, ptr %4, i64 %324
  %326 = mul i32 %323, %3
  %327 = add nsw i32 %306, -4
  %328 = zext i32 %327 to i64
  %329 = getelementptr float, ptr %4, i64 %328
  %330 = mul i32 %327, %3
  %331 = add nsw i32 %306, -3
  %332 = zext i32 %331 to i64
  %333 = getelementptr float, ptr %4, i64 %332
  %334 = mul i32 %331, %3
  %335 = add nsw i32 %306, -2
  %336 = zext i32 %335 to i64
  %337 = getelementptr float, ptr %4, i64 %336
  %338 = mul i32 %335, %3
  %339 = zext i32 %310 to i64
  %340 = getelementptr float, ptr %4, i64 %339
  %341 = mul i32 %310, %3
  %342 = sext i32 %314 to i64
  %343 = sext i32 %318 to i64
  %344 = sext i32 %322 to i64
  %345 = sext i32 %326 to i64
  %346 = sext i32 %330 to i64
  %347 = sext i32 %334 to i64
  %348 = sext i32 %338 to i64
  %349 = sext i32 %341 to i64
  %350 = zext nneg i32 %0 to i64
  %351 = getelementptr float, ptr %5, i64 %342
  %352 = getelementptr float, ptr %5, i64 %343
  %353 = getelementptr float, ptr %5, i64 %344
  %354 = getelementptr float, ptr %5, i64 %345
  %355 = getelementptr float, ptr %5, i64 %346
  %356 = getelementptr float, ptr %5, i64 %347
  %357 = getelementptr float, ptr %5, i64 %348
  %358 = getelementptr float, ptr %5, i64 %349
  %359 = icmp ult i32 %0, 20
  br i1 %359, label %532, label %360

360:                                              ; preds = %309
  %361 = shl nuw nsw i64 %350, 2
  %362 = getelementptr i8, ptr %1, i64 %361
  %363 = shl nuw nsw i64 %339, 2
  %364 = getelementptr i8, ptr %4, i64 %363
  %365 = getelementptr i8, ptr %364, i64 4
  %366 = shl nuw nsw i64 %336, 2
  %367 = getelementptr i8, ptr %4, i64 %366
  %368 = getelementptr i8, ptr %367, i64 4
  %369 = shl nuw nsw i64 %332, 2
  %370 = getelementptr i8, ptr %4, i64 %369
  %371 = getelementptr i8, ptr %370, i64 4
  %372 = shl nuw nsw i64 %328, 2
  %373 = getelementptr i8, ptr %4, i64 %372
  %374 = getelementptr i8, ptr %373, i64 4
  %375 = shl nuw nsw i64 %324, 2
  %376 = getelementptr i8, ptr %4, i64 %375
  %377 = getelementptr i8, ptr %376, i64 4
  %378 = shl nuw nsw i64 %320, 2
  %379 = getelementptr i8, ptr %4, i64 %378
  %380 = getelementptr i8, ptr %379, i64 4
  %381 = shl nuw nsw i64 %316, 2
  %382 = getelementptr i8, ptr %4, i64 %381
  %383 = getelementptr i8, ptr %382, i64 4
  %384 = shl nuw nsw i64 %312, 2
  %385 = getelementptr i8, ptr %4, i64 %384
  %386 = getelementptr i8, ptr %385, i64 4
  %387 = add nsw i64 %349, %350
  %388 = shl nsw i64 %387, 2
  %389 = getelementptr i8, ptr %5, i64 %388
  %390 = add nsw i64 %348, %350
  %391 = shl nsw i64 %390, 2
  %392 = getelementptr i8, ptr %5, i64 %391
  %393 = add nsw i64 %347, %350
  %394 = shl nsw i64 %393, 2
  %395 = getelementptr i8, ptr %5, i64 %394
  %396 = add nsw i64 %346, %350
  %397 = shl nsw i64 %396, 2
  %398 = getelementptr i8, ptr %5, i64 %397
  %399 = add nsw i64 %345, %350
  %400 = shl nsw i64 %399, 2
  %401 = getelementptr i8, ptr %5, i64 %400
  %402 = add nsw i64 %344, %350
  %403 = shl nsw i64 %402, 2
  %404 = getelementptr i8, ptr %5, i64 %403
  %405 = add nsw i64 %343, %350
  %406 = shl nsw i64 %405, 2
  %407 = getelementptr i8, ptr %5, i64 %406
  %408 = add nsw i64 %342, %350
  %409 = shl nsw i64 %408, 2
  %410 = getelementptr i8, ptr %5, i64 %409
  %411 = icmp ult ptr %1, %365
  %412 = icmp ult ptr %340, %362
  %413 = and i1 %411, %412
  %414 = icmp ult ptr %1, %368
  %415 = icmp ult ptr %337, %362
  %416 = and i1 %414, %415
  %417 = or i1 %413, %416
  %418 = icmp ult ptr %1, %371
  %419 = icmp ult ptr %333, %362
  %420 = and i1 %418, %419
  %421 = or i1 %417, %420
  %422 = icmp ult ptr %1, %374
  %423 = icmp ult ptr %329, %362
  %424 = and i1 %422, %423
  %425 = or i1 %421, %424
  %426 = icmp ult ptr %1, %377
  %427 = icmp ult ptr %325, %362
  %428 = and i1 %426, %427
  %429 = or i1 %425, %428
  %430 = icmp ult ptr %1, %380
  %431 = icmp ult ptr %321, %362
  %432 = and i1 %430, %431
  %433 = or i1 %429, %432
  %434 = icmp ult ptr %1, %383
  %435 = icmp ult ptr %317, %362
  %436 = and i1 %434, %435
  %437 = or i1 %433, %436
  %438 = icmp ult ptr %1, %386
  %439 = icmp ult ptr %313, %362
  %440 = and i1 %438, %439
  %441 = or i1 %437, %440
  %442 = icmp ult ptr %1, %389
  %443 = icmp ult ptr %358, %362
  %444 = and i1 %442, %443
  %445 = or i1 %441, %444
  %446 = icmp ult ptr %1, %392
  %447 = icmp ult ptr %357, %362
  %448 = and i1 %446, %447
  %449 = or i1 %445, %448
  %450 = icmp ult ptr %1, %395
  %451 = icmp ult ptr %356, %362
  %452 = and i1 %450, %451
  %453 = or i1 %449, %452
  %454 = icmp ult ptr %1, %398
  %455 = icmp ult ptr %355, %362
  %456 = and i1 %454, %455
  %457 = or i1 %453, %456
  %458 = icmp ult ptr %1, %401
  %459 = icmp ult ptr %354, %362
  %460 = and i1 %458, %459
  %461 = or i1 %457, %460
  %462 = icmp ult ptr %1, %404
  %463 = icmp ult ptr %353, %362
  %464 = and i1 %462, %463
  %465 = or i1 %461, %464
  %466 = icmp ult ptr %1, %407
  %467 = icmp ult ptr %352, %362
  %468 = and i1 %466, %467
  %469 = or i1 %465, %468
  %470 = icmp ult ptr %1, %410
  %471 = icmp ult ptr %351, %362
  %472 = and i1 %470, %471
  %473 = or i1 %469, %472
  br i1 %473, label %532, label %474

474:                                              ; preds = %360
  %475 = and i64 %350, 2147483644
  %476 = load float, ptr %313, align 4, !tbaa !11, !alias.scope !143
  %477 = insertelement <4 x float> poison, float %476, i64 0
  %478 = shufflevector <4 x float> %477, <4 x float> poison, <4 x i32> zeroinitializer
  %479 = load float, ptr %317, align 4, !tbaa !11, !alias.scope !146
  %480 = insertelement <4 x float> poison, float %479, i64 0
  %481 = shufflevector <4 x float> %480, <4 x float> poison, <4 x i32> zeroinitializer
  %482 = load float, ptr %321, align 4, !tbaa !11, !alias.scope !148
  %483 = insertelement <4 x float> poison, float %482, i64 0
  %484 = shufflevector <4 x float> %483, <4 x float> poison, <4 x i32> zeroinitializer
  %485 = load float, ptr %325, align 4, !tbaa !11, !alias.scope !150
  %486 = insertelement <4 x float> poison, float %485, i64 0
  %487 = shufflevector <4 x float> %486, <4 x float> poison, <4 x i32> zeroinitializer
  %488 = load float, ptr %329, align 4, !tbaa !11, !alias.scope !152
  %489 = insertelement <4 x float> poison, float %488, i64 0
  %490 = shufflevector <4 x float> %489, <4 x float> poison, <4 x i32> zeroinitializer
  %491 = load float, ptr %333, align 4, !tbaa !11, !alias.scope !154
  %492 = insertelement <4 x float> poison, float %491, i64 0
  %493 = shufflevector <4 x float> %492, <4 x float> poison, <4 x i32> zeroinitializer
  %494 = load float, ptr %337, align 4, !tbaa !11, !alias.scope !156
  %495 = insertelement <4 x float> poison, float %494, i64 0
  %496 = shufflevector <4 x float> %495, <4 x float> poison, <4 x i32> zeroinitializer
  %497 = load float, ptr %340, align 4, !tbaa !11, !alias.scope !158
  %498 = insertelement <4 x float> poison, float %497, i64 0
  %499 = shufflevector <4 x float> %498, <4 x float> poison, <4 x i32> zeroinitializer
  br label %500

500:                                              ; preds = %500, %474
  %501 = phi i64 [ 0, %474 ], [ %528, %500 ]
  %502 = getelementptr inbounds nuw float, ptr %1, i64 %501
  %503 = load <4 x float>, ptr %502, align 4, !tbaa !11, !alias.scope !160, !noalias !162
  %504 = getelementptr float, ptr %351, i64 %501
  %505 = load <4 x float>, ptr %504, align 4, !tbaa !11, !alias.scope !171
  %506 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %478, <4 x float> %505, <4 x float> %503)
  %507 = getelementptr float, ptr %352, i64 %501
  %508 = load <4 x float>, ptr %507, align 4, !tbaa !11, !alias.scope !172
  %509 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %481, <4 x float> %508, <4 x float> %506)
  %510 = getelementptr float, ptr %353, i64 %501
  %511 = load <4 x float>, ptr %510, align 4, !tbaa !11, !alias.scope !173
  %512 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %484, <4 x float> %511, <4 x float> %509)
  %513 = getelementptr float, ptr %354, i64 %501
  %514 = load <4 x float>, ptr %513, align 4, !tbaa !11, !alias.scope !174
  %515 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %487, <4 x float> %514, <4 x float> %512)
  %516 = getelementptr float, ptr %355, i64 %501
  %517 = load <4 x float>, ptr %516, align 4, !tbaa !11, !alias.scope !175
  %518 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %490, <4 x float> %517, <4 x float> %515)
  %519 = getelementptr float, ptr %356, i64 %501
  %520 = load <4 x float>, ptr %519, align 4, !tbaa !11, !alias.scope !176
  %521 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %493, <4 x float> %520, <4 x float> %518)
  %522 = getelementptr float, ptr %357, i64 %501
  %523 = load <4 x float>, ptr %522, align 4, !tbaa !11, !alias.scope !177
  %524 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %496, <4 x float> %523, <4 x float> %521)
  %525 = getelementptr float, ptr %358, i64 %501
  %526 = load <4 x float>, ptr %525, align 4, !tbaa !11, !alias.scope !178
  %527 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %499, <4 x float> %526, <4 x float> %524)
  store <4 x float> %527, ptr %502, align 4, !tbaa !11, !alias.scope !160, !noalias !162
  %528 = add nuw i64 %501, 4
  %529 = icmp eq i64 %528, %475
  br i1 %529, label %530, label %500, !llvm.loop !179

530:                                              ; preds = %500
  %531 = icmp eq i64 %475, %350
  br i1 %531, label %576, label %532

532:                                              ; preds = %360, %309, %530
  %533 = phi i64 [ 0, %360 ], [ 0, %309 ], [ %475, %530 ]
  br label %534

534:                                              ; preds = %532, %534
  %535 = phi i64 [ %570, %534 ], [ %533, %532 ]
  %536 = getelementptr inbounds nuw float, ptr %1, i64 %535
  %537 = load float, ptr %536, align 4, !tbaa !11
  %538 = load float, ptr %313, align 4, !tbaa !11
  %539 = getelementptr float, ptr %351, i64 %535
  %540 = load float, ptr %539, align 4, !tbaa !11
  %541 = tail call float @llvm.fmuladd.f32(float %538, float %540, float %537)
  %542 = load float, ptr %317, align 4, !tbaa !11
  %543 = getelementptr float, ptr %352, i64 %535
  %544 = load float, ptr %543, align 4, !tbaa !11
  %545 = tail call float @llvm.fmuladd.f32(float %542, float %544, float %541)
  %546 = load float, ptr %321, align 4, !tbaa !11
  %547 = getelementptr float, ptr %353, i64 %535
  %548 = load float, ptr %547, align 4, !tbaa !11
  %549 = tail call float @llvm.fmuladd.f32(float %546, float %548, float %545)
  %550 = load float, ptr %325, align 4, !tbaa !11
  %551 = getelementptr float, ptr %354, i64 %535
  %552 = load float, ptr %551, align 4, !tbaa !11
  %553 = tail call float @llvm.fmuladd.f32(float %550, float %552, float %549)
  %554 = load float, ptr %329, align 4, !tbaa !11
  %555 = getelementptr float, ptr %355, i64 %535
  %556 = load float, ptr %555, align 4, !tbaa !11
  %557 = tail call float @llvm.fmuladd.f32(float %554, float %556, float %553)
  %558 = load float, ptr %333, align 4, !tbaa !11
  %559 = getelementptr float, ptr %356, i64 %535
  %560 = load float, ptr %559, align 4, !tbaa !11
  %561 = tail call float @llvm.fmuladd.f32(float %558, float %560, float %557)
  %562 = load float, ptr %337, align 4, !tbaa !11
  %563 = getelementptr float, ptr %357, i64 %535
  %564 = load float, ptr %563, align 4, !tbaa !11
  %565 = tail call float @llvm.fmuladd.f32(float %562, float %564, float %561)
  %566 = load float, ptr %340, align 4, !tbaa !11
  %567 = getelementptr float, ptr %358, i64 %535
  %568 = load float, ptr %567, align 4, !tbaa !11
  %569 = tail call float @llvm.fmuladd.f32(float %566, float %568, float %565)
  store float %569, ptr %536, align 4, !tbaa !11
  %570 = add nuw nsw i64 %535, 1
  %571 = icmp eq i64 %570, %350
  br i1 %571, label %576, label %534, !llvm.loop !180

572:                                              ; preds = %305
  %573 = add nsw i32 %306, 15
  %574 = icmp slt i32 %573, %2
  %575 = and i1 %574, %9
  br i1 %575, label %579, label %1059

576:                                              ; preds = %534, %530
  %577 = add nuw nsw i32 %306, 15
  %578 = icmp slt i32 %577, %2
  br i1 %578, label %579, label %1059

579:                                              ; preds = %572, %576
  %580 = add nsw i32 %306, 15
  %581 = zext i32 %580 to i64
  %582 = zext i32 %2 to i64
  %583 = sext i32 %3 to i64
  %584 = zext i32 %0 to i64
  %585 = shl nuw nsw i64 %584, 2
  %586 = getelementptr i8, ptr %1, i64 %585
  %587 = shl nuw nsw i64 %581, 2
  %588 = getelementptr i8, ptr %4, i64 %587
  %589 = getelementptr i8, ptr %588, i64 -60
  %590 = add nuw nsw i64 %581, 16
  %591 = tail call i64 @llvm.umax.i64(i64 %590, i64 %582)
  %592 = xor i64 %581, -1
  %593 = add nsw i64 %591, %592
  %594 = shl nsw i64 %593, 2
  %595 = and i64 %594, -64
  %596 = add nsw i64 %595, %587
  %597 = getelementptr i8, ptr %4, i64 %596
  %598 = getelementptr i8, ptr %597, i64 4
  %599 = mul nsw i64 %583, %581
  %600 = shl i64 %599, 2
  %601 = getelementptr i8, ptr %5, i64 %600
  %602 = mul i64 %596, %583
  %603 = getelementptr i8, ptr %5, i64 %602
  %604 = getelementptr i8, ptr %603, i64 %585
  %605 = add nuw nsw i64 %581, 4611686018427387903
  %606 = mul i64 %605, %583
  %607 = shl i64 %606, 2
  %608 = getelementptr i8, ptr %5, i64 %607
  %609 = add nsw i64 %595, %587
  %610 = add i64 %609, -4
  %611 = mul i64 %610, %583
  %612 = getelementptr i8, ptr %5, i64 %611
  %613 = getelementptr i8, ptr %612, i64 %585
  %614 = add nuw nsw i64 %581, 4611686018427387902
  %615 = mul i64 %614, %583
  %616 = shl i64 %615, 2
  %617 = getelementptr i8, ptr %5, i64 %616
  %618 = add nsw i64 %595, %587
  %619 = add i64 %618, -8
  %620 = mul i64 %619, %583
  %621 = getelementptr i8, ptr %5, i64 %620
  %622 = getelementptr i8, ptr %621, i64 %585
  %623 = add nuw nsw i64 %581, 4611686018427387901
  %624 = mul i64 %623, %583
  %625 = shl i64 %624, 2
  %626 = getelementptr i8, ptr %5, i64 %625
  %627 = add nsw i64 %595, %587
  %628 = add i64 %627, -12
  %629 = mul i64 %628, %583
  %630 = getelementptr i8, ptr %5, i64 %629
  %631 = getelementptr i8, ptr %630, i64 %585
  %632 = add nuw nsw i64 %581, 4611686018427387900
  %633 = mul i64 %632, %583
  %634 = shl i64 %633, 2
  %635 = getelementptr i8, ptr %5, i64 %634
  %636 = add nsw i64 %595, %587
  %637 = add i64 %636, -16
  %638 = mul i64 %637, %583
  %639 = getelementptr i8, ptr %5, i64 %638
  %640 = getelementptr i8, ptr %639, i64 %585
  %641 = add nuw nsw i64 %581, 4611686018427387899
  %642 = mul i64 %641, %583
  %643 = shl i64 %642, 2
  %644 = getelementptr i8, ptr %5, i64 %643
  %645 = add nsw i64 %595, %587
  %646 = add i64 %645, -20
  %647 = mul i64 %646, %583
  %648 = getelementptr i8, ptr %5, i64 %647
  %649 = getelementptr i8, ptr %648, i64 %585
  %650 = add nuw nsw i64 %581, 4611686018427387898
  %651 = mul i64 %650, %583
  %652 = shl i64 %651, 2
  %653 = getelementptr i8, ptr %5, i64 %652
  %654 = add nsw i64 %595, %587
  %655 = add i64 %654, -24
  %656 = mul i64 %655, %583
  %657 = getelementptr i8, ptr %5, i64 %656
  %658 = getelementptr i8, ptr %657, i64 %585
  %659 = add nuw nsw i64 %581, 4611686018427387897
  %660 = mul i64 %659, %583
  %661 = shl i64 %660, 2
  %662 = getelementptr i8, ptr %5, i64 %661
  %663 = add nsw i64 %595, %587
  %664 = add i64 %663, -28
  %665 = mul i64 %664, %583
  %666 = getelementptr i8, ptr %5, i64 %665
  %667 = getelementptr i8, ptr %666, i64 %585
  %668 = add nuw nsw i64 %581, 4611686018427387896
  %669 = mul i64 %668, %583
  %670 = shl i64 %669, 2
  %671 = getelementptr i8, ptr %5, i64 %670
  %672 = add nsw i64 %595, %587
  %673 = add i64 %672, -32
  %674 = mul i64 %673, %583
  %675 = getelementptr i8, ptr %5, i64 %674
  %676 = getelementptr i8, ptr %675, i64 %585
  %677 = add nuw nsw i64 %581, 4611686018427387895
  %678 = mul i64 %677, %583
  %679 = shl i64 %678, 2
  %680 = getelementptr i8, ptr %5, i64 %679
  %681 = add nsw i64 %595, %587
  %682 = add i64 %681, -36
  %683 = mul i64 %682, %583
  %684 = getelementptr i8, ptr %5, i64 %683
  %685 = getelementptr i8, ptr %684, i64 %585
  %686 = add nuw nsw i64 %581, 4611686018427387894
  %687 = mul i64 %686, %583
  %688 = shl i64 %687, 2
  %689 = getelementptr i8, ptr %5, i64 %688
  %690 = add nsw i64 %595, %587
  %691 = add i64 %690, -40
  %692 = mul i64 %691, %583
  %693 = getelementptr i8, ptr %5, i64 %692
  %694 = getelementptr i8, ptr %693, i64 %585
  %695 = add nuw nsw i64 %581, 4611686018427387893
  %696 = mul i64 %695, %583
  %697 = shl i64 %696, 2
  %698 = getelementptr i8, ptr %5, i64 %697
  %699 = add nsw i64 %595, %587
  %700 = add i64 %699, -44
  %701 = mul i64 %700, %583
  %702 = getelementptr i8, ptr %5, i64 %701
  %703 = getelementptr i8, ptr %702, i64 %585
  %704 = add nuw nsw i64 %581, 4611686018427387892
  %705 = mul i64 %704, %583
  %706 = shl i64 %705, 2
  %707 = getelementptr i8, ptr %5, i64 %706
  %708 = add nsw i64 %595, %587
  %709 = add i64 %708, -48
  %710 = mul i64 %709, %583
  %711 = getelementptr i8, ptr %5, i64 %710
  %712 = getelementptr i8, ptr %711, i64 %585
  %713 = add nuw nsw i64 %581, 4611686018427387891
  %714 = mul i64 %713, %583
  %715 = shl i64 %714, 2
  %716 = getelementptr i8, ptr %5, i64 %715
  %717 = add nsw i64 %595, %587
  %718 = add i64 %717, -52
  %719 = mul i64 %718, %583
  %720 = getelementptr i8, ptr %5, i64 %719
  %721 = getelementptr i8, ptr %720, i64 %585
  %722 = add nuw nsw i64 %581, 4611686018427387890
  %723 = mul i64 %722, %583
  %724 = shl i64 %723, 2
  %725 = getelementptr i8, ptr %5, i64 %724
  %726 = add nsw i64 %595, %587
  %727 = add i64 %726, -56
  %728 = mul i64 %727, %583
  %729 = getelementptr i8, ptr %5, i64 %728
  %730 = getelementptr i8, ptr %729, i64 %585
  %731 = add nuw nsw i64 %581, 4611686018427387889
  %732 = mul i64 %731, %583
  %733 = shl i64 %732, 2
  %734 = getelementptr i8, ptr %5, i64 %733
  %735 = add nsw i64 %595, %587
  %736 = add i64 %735, -60
  %737 = mul i64 %736, %583
  %738 = getelementptr i8, ptr %5, i64 %737
  %739 = getelementptr i8, ptr %738, i64 %585
  %740 = icmp ult i32 %0, 8
  %741 = icmp ult ptr %1, %598
  %742 = icmp ult ptr %589, %586
  %743 = and i1 %741, %742
  %744 = icmp ult ptr %1, %604
  %745 = icmp ult ptr %601, %586
  %746 = and i1 %744, %745
  %747 = icmp slt i32 %3, 0
  %748 = or i1 %746, %747
  %749 = or i1 %743, %748
  %750 = icmp ult ptr %1, %613
  %751 = icmp ult ptr %608, %586
  %752 = and i1 %750, %751
  %753 = or i1 %752, %749
  %754 = icmp ult ptr %1, %622
  %755 = icmp ult ptr %617, %586
  %756 = and i1 %754, %755
  %757 = or i1 %756, %753
  %758 = icmp ult ptr %1, %631
  %759 = icmp ult ptr %626, %586
  %760 = and i1 %758, %759
  %761 = or i1 %760, %757
  %762 = icmp ult ptr %1, %640
  %763 = icmp ult ptr %635, %586
  %764 = and i1 %762, %763
  %765 = or i1 %764, %761
  %766 = icmp ult ptr %1, %649
  %767 = icmp ult ptr %644, %586
  %768 = and i1 %766, %767
  %769 = or i1 %768, %765
  %770 = icmp ult ptr %1, %658
  %771 = icmp ult ptr %653, %586
  %772 = and i1 %770, %771
  %773 = or i1 %772, %769
  %774 = icmp ult ptr %1, %667
  %775 = icmp ult ptr %662, %586
  %776 = and i1 %774, %775
  %777 = or i1 %776, %773
  %778 = icmp ult ptr %1, %676
  %779 = icmp ult ptr %671, %586
  %780 = and i1 %778, %779
  %781 = icmp slt i32 %3, 0
  %782 = or i1 %780, %781
  %783 = or i1 %777, %782
  %784 = icmp ult ptr %1, %685
  %785 = icmp ult ptr %680, %586
  %786 = and i1 %784, %785
  %787 = or i1 %786, %783
  %788 = icmp ult ptr %1, %694
  %789 = icmp ult ptr %689, %586
  %790 = and i1 %788, %789
  %791 = or i1 %790, %787
  %792 = icmp ult ptr %1, %703
  %793 = icmp ult ptr %698, %586
  %794 = and i1 %792, %793
  %795 = or i1 %794, %791
  %796 = icmp ult ptr %1, %712
  %797 = icmp ult ptr %707, %586
  %798 = and i1 %796, %797
  %799 = or i1 %798, %795
  %800 = icmp ult ptr %1, %721
  %801 = icmp ult ptr %716, %586
  %802 = and i1 %800, %801
  %803 = or i1 %802, %799
  %804 = icmp ult ptr %1, %730
  %805 = icmp ult ptr %725, %586
  %806 = and i1 %804, %805
  %807 = or i1 %806, %803
  %808 = icmp ult ptr %1, %739
  %809 = icmp ult ptr %734, %586
  %810 = and i1 %808, %809
  %811 = or i1 %810, %807
  %812 = and i64 %584, 4294967292
  %813 = icmp eq i64 %812, %584
  br label %814

814:                                              ; preds = %579, %1056
  %815 = phi i64 [ %581, %579 ], [ %1057, %1056 ]
  %816 = add nsw i64 %815, -15
  %817 = getelementptr inbounds float, ptr %4, i64 %816
  %818 = mul nsw i64 %816, %583
  %819 = add nsw i64 %815, -14
  %820 = getelementptr inbounds float, ptr %4, i64 %819
  %821 = mul nsw i64 %819, %583
  %822 = add nsw i64 %815, -13
  %823 = getelementptr inbounds float, ptr %4, i64 %822
  %824 = mul nsw i64 %822, %583
  %825 = add nsw i64 %815, -12
  %826 = getelementptr inbounds float, ptr %4, i64 %825
  %827 = mul nsw i64 %825, %583
  %828 = add nsw i64 %815, -11
  %829 = getelementptr inbounds float, ptr %4, i64 %828
  %830 = mul nsw i64 %828, %583
  %831 = add nsw i64 %815, -10
  %832 = getelementptr inbounds float, ptr %4, i64 %831
  %833 = mul nsw i64 %831, %583
  %834 = add nsw i64 %815, -9
  %835 = getelementptr inbounds float, ptr %4, i64 %834
  %836 = mul nsw i64 %834, %583
  %837 = add nsw i64 %815, -8
  %838 = getelementptr inbounds float, ptr %4, i64 %837
  %839 = mul nsw i64 %837, %583
  %840 = add nsw i64 %815, -7
  %841 = getelementptr inbounds float, ptr %4, i64 %840
  %842 = mul nsw i64 %840, %583
  %843 = add nsw i64 %815, -6
  %844 = getelementptr inbounds float, ptr %4, i64 %843
  %845 = mul nsw i64 %843, %583
  %846 = add nsw i64 %815, -5
  %847 = getelementptr inbounds float, ptr %4, i64 %846
  %848 = mul nsw i64 %846, %583
  %849 = add nsw i64 %815, -4
  %850 = getelementptr inbounds float, ptr %4, i64 %849
  %851 = mul nsw i64 %849, %583
  %852 = add nsw i64 %815, -3
  %853 = getelementptr inbounds float, ptr %4, i64 %852
  %854 = mul nsw i64 %852, %583
  %855 = add nsw i64 %815, -2
  %856 = getelementptr inbounds float, ptr %4, i64 %855
  %857 = mul nsw i64 %855, %583
  %858 = add nsw i64 %815, -1
  %859 = getelementptr inbounds float, ptr %4, i64 %858
  %860 = mul nsw i64 %858, %583
  %861 = getelementptr inbounds nuw float, ptr %4, i64 %815
  %862 = mul nsw i64 %815, %583
  %863 = getelementptr float, ptr %5, i64 %818
  %864 = getelementptr float, ptr %5, i64 %821
  %865 = getelementptr float, ptr %5, i64 %824
  %866 = getelementptr float, ptr %5, i64 %827
  %867 = getelementptr float, ptr %5, i64 %830
  %868 = getelementptr float, ptr %5, i64 %833
  %869 = getelementptr float, ptr %5, i64 %836
  %870 = getelementptr float, ptr %5, i64 %839
  %871 = getelementptr float, ptr %5, i64 %842
  %872 = getelementptr float, ptr %5, i64 %845
  %873 = getelementptr float, ptr %5, i64 %848
  %874 = getelementptr float, ptr %5, i64 %851
  %875 = getelementptr float, ptr %5, i64 %854
  %876 = getelementptr float, ptr %5, i64 %857
  %877 = getelementptr float, ptr %5, i64 %860
  %878 = getelementptr float, ptr %5, i64 %862
  %879 = select i1 %740, i1 true, i1 %811
  br i1 %879, label %984, label %880

880:                                              ; preds = %814
  %881 = load float, ptr %817, align 4, !tbaa !11, !alias.scope !181
  %882 = insertelement <4 x float> poison, float %881, i64 0
  %883 = shufflevector <4 x float> %882, <4 x float> poison, <4 x i32> zeroinitializer
  %884 = load float, ptr %820, align 4, !tbaa !11, !alias.scope !181
  %885 = insertelement <4 x float> poison, float %884, i64 0
  %886 = shufflevector <4 x float> %885, <4 x float> poison, <4 x i32> zeroinitializer
  %887 = load float, ptr %823, align 4, !tbaa !11, !alias.scope !181
  %888 = insertelement <4 x float> poison, float %887, i64 0
  %889 = shufflevector <4 x float> %888, <4 x float> poison, <4 x i32> zeroinitializer
  %890 = load float, ptr %826, align 4, !tbaa !11, !alias.scope !181
  %891 = insertelement <4 x float> poison, float %890, i64 0
  %892 = shufflevector <4 x float> %891, <4 x float> poison, <4 x i32> zeroinitializer
  %893 = load float, ptr %829, align 4, !tbaa !11, !alias.scope !181
  %894 = insertelement <4 x float> poison, float %893, i64 0
  %895 = shufflevector <4 x float> %894, <4 x float> poison, <4 x i32> zeroinitializer
  %896 = load float, ptr %832, align 4, !tbaa !11, !alias.scope !181
  %897 = insertelement <4 x float> poison, float %896, i64 0
  %898 = shufflevector <4 x float> %897, <4 x float> poison, <4 x i32> zeroinitializer
  %899 = load float, ptr %835, align 4, !tbaa !11, !alias.scope !181
  %900 = insertelement <4 x float> poison, float %899, i64 0
  %901 = shufflevector <4 x float> %900, <4 x float> poison, <4 x i32> zeroinitializer
  %902 = load float, ptr %838, align 4, !tbaa !11, !alias.scope !181
  %903 = insertelement <4 x float> poison, float %902, i64 0
  %904 = shufflevector <4 x float> %903, <4 x float> poison, <4 x i32> zeroinitializer
  %905 = load float, ptr %841, align 4, !tbaa !11, !alias.scope !181
  %906 = insertelement <4 x float> poison, float %905, i64 0
  %907 = shufflevector <4 x float> %906, <4 x float> poison, <4 x i32> zeroinitializer
  %908 = load float, ptr %844, align 4, !tbaa !11, !alias.scope !181
  %909 = insertelement <4 x float> poison, float %908, i64 0
  %910 = shufflevector <4 x float> %909, <4 x float> poison, <4 x i32> zeroinitializer
  %911 = load float, ptr %847, align 4, !tbaa !11, !alias.scope !181
  %912 = insertelement <4 x float> poison, float %911, i64 0
  %913 = shufflevector <4 x float> %912, <4 x float> poison, <4 x i32> zeroinitializer
  %914 = load float, ptr %850, align 4, !tbaa !11, !alias.scope !181
  %915 = insertelement <4 x float> poison, float %914, i64 0
  %916 = shufflevector <4 x float> %915, <4 x float> poison, <4 x i32> zeroinitializer
  %917 = load float, ptr %853, align 4, !tbaa !11, !alias.scope !181
  %918 = insertelement <4 x float> poison, float %917, i64 0
  %919 = shufflevector <4 x float> %918, <4 x float> poison, <4 x i32> zeroinitializer
  %920 = load float, ptr %856, align 4, !tbaa !11, !alias.scope !181
  %921 = insertelement <4 x float> poison, float %920, i64 0
  %922 = shufflevector <4 x float> %921, <4 x float> poison, <4 x i32> zeroinitializer
  %923 = load float, ptr %859, align 4, !tbaa !11, !alias.scope !181
  %924 = insertelement <4 x float> poison, float %923, i64 0
  %925 = shufflevector <4 x float> %924, <4 x float> poison, <4 x i32> zeroinitializer
  %926 = load float, ptr %861, align 4, !tbaa !11, !alias.scope !181
  %927 = insertelement <4 x float> poison, float %926, i64 0
  %928 = shufflevector <4 x float> %927, <4 x float> poison, <4 x i32> zeroinitializer
  br label %929

929:                                              ; preds = %929, %880
  %930 = phi i64 [ 0, %880 ], [ %981, %929 ]
  %931 = getelementptr inbounds nuw float, ptr %1, i64 %930
  %932 = load <4 x float>, ptr %931, align 4, !tbaa !11, !alias.scope !184, !noalias !186
  %933 = getelementptr float, ptr %863, i64 %930
  %934 = load <4 x float>, ptr %933, align 4, !tbaa !11, !alias.scope !203
  %935 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %883, <4 x float> %934, <4 x float> %932)
  %936 = getelementptr float, ptr %864, i64 %930
  %937 = load <4 x float>, ptr %936, align 4, !tbaa !11, !alias.scope !204
  %938 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %886, <4 x float> %937, <4 x float> %935)
  %939 = getelementptr float, ptr %865, i64 %930
  %940 = load <4 x float>, ptr %939, align 4, !tbaa !11, !alias.scope !205
  %941 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %889, <4 x float> %940, <4 x float> %938)
  %942 = getelementptr float, ptr %866, i64 %930
  %943 = load <4 x float>, ptr %942, align 4, !tbaa !11, !alias.scope !206
  %944 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %892, <4 x float> %943, <4 x float> %941)
  %945 = getelementptr float, ptr %867, i64 %930
  %946 = load <4 x float>, ptr %945, align 4, !tbaa !11, !alias.scope !207
  %947 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %895, <4 x float> %946, <4 x float> %944)
  %948 = getelementptr float, ptr %868, i64 %930
  %949 = load <4 x float>, ptr %948, align 4, !tbaa !11, !alias.scope !208
  %950 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %898, <4 x float> %949, <4 x float> %947)
  %951 = getelementptr float, ptr %869, i64 %930
  %952 = load <4 x float>, ptr %951, align 4, !tbaa !11, !alias.scope !209
  %953 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %901, <4 x float> %952, <4 x float> %950)
  %954 = getelementptr float, ptr %870, i64 %930
  %955 = load <4 x float>, ptr %954, align 4, !tbaa !11, !alias.scope !210
  %956 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %904, <4 x float> %955, <4 x float> %953)
  %957 = getelementptr float, ptr %871, i64 %930
  %958 = load <4 x float>, ptr %957, align 4, !tbaa !11, !alias.scope !211
  %959 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %907, <4 x float> %958, <4 x float> %956)
  %960 = getelementptr float, ptr %872, i64 %930
  %961 = load <4 x float>, ptr %960, align 4, !tbaa !11, !alias.scope !212
  %962 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %910, <4 x float> %961, <4 x float> %959)
  %963 = getelementptr float, ptr %873, i64 %930
  %964 = load <4 x float>, ptr %963, align 4, !tbaa !11, !alias.scope !213
  %965 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %913, <4 x float> %964, <4 x float> %962)
  %966 = getelementptr float, ptr %874, i64 %930
  %967 = load <4 x float>, ptr %966, align 4, !tbaa !11, !alias.scope !214
  %968 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %916, <4 x float> %967, <4 x float> %965)
  %969 = getelementptr float, ptr %875, i64 %930
  %970 = load <4 x float>, ptr %969, align 4, !tbaa !11, !alias.scope !215
  %971 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %919, <4 x float> %970, <4 x float> %968)
  %972 = getelementptr float, ptr %876, i64 %930
  %973 = load <4 x float>, ptr %972, align 4, !tbaa !11, !alias.scope !216
  %974 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %922, <4 x float> %973, <4 x float> %971)
  %975 = getelementptr float, ptr %877, i64 %930
  %976 = load <4 x float>, ptr %975, align 4, !tbaa !11, !alias.scope !217
  %977 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %925, <4 x float> %976, <4 x float> %974)
  %978 = getelementptr float, ptr %878, i64 %930
  %979 = load <4 x float>, ptr %978, align 4, !tbaa !11, !alias.scope !218
  %980 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %928, <4 x float> %979, <4 x float> %977)
  store <4 x float> %980, ptr %931, align 4, !tbaa !11, !alias.scope !184, !noalias !186
  %981 = add nuw i64 %930, 4
  %982 = icmp eq i64 %981, %812
  br i1 %982, label %983, label %929, !llvm.loop !219

983:                                              ; preds = %929
  br i1 %813, label %1056, label %984

984:                                              ; preds = %814, %983
  %985 = phi i64 [ 0, %814 ], [ %812, %983 ]
  br label %986

986:                                              ; preds = %984, %986
  %987 = phi i64 [ %1054, %986 ], [ %985, %984 ]
  %988 = getelementptr inbounds nuw float, ptr %1, i64 %987
  %989 = load float, ptr %988, align 4, !tbaa !11
  %990 = load float, ptr %817, align 4, !tbaa !11
  %991 = getelementptr float, ptr %863, i64 %987
  %992 = load float, ptr %991, align 4, !tbaa !11
  %993 = tail call float @llvm.fmuladd.f32(float %990, float %992, float %989)
  %994 = load float, ptr %820, align 4, !tbaa !11
  %995 = getelementptr float, ptr %864, i64 %987
  %996 = load float, ptr %995, align 4, !tbaa !11
  %997 = tail call float @llvm.fmuladd.f32(float %994, float %996, float %993)
  %998 = load float, ptr %823, align 4, !tbaa !11
  %999 = getelementptr float, ptr %865, i64 %987
  %1000 = load float, ptr %999, align 4, !tbaa !11
  %1001 = tail call float @llvm.fmuladd.f32(float %998, float %1000, float %997)
  %1002 = load float, ptr %826, align 4, !tbaa !11
  %1003 = getelementptr float, ptr %866, i64 %987
  %1004 = load float, ptr %1003, align 4, !tbaa !11
  %1005 = tail call float @llvm.fmuladd.f32(float %1002, float %1004, float %1001)
  %1006 = load float, ptr %829, align 4, !tbaa !11
  %1007 = getelementptr float, ptr %867, i64 %987
  %1008 = load float, ptr %1007, align 4, !tbaa !11
  %1009 = tail call float @llvm.fmuladd.f32(float %1006, float %1008, float %1005)
  %1010 = load float, ptr %832, align 4, !tbaa !11
  %1011 = getelementptr float, ptr %868, i64 %987
  %1012 = load float, ptr %1011, align 4, !tbaa !11
  %1013 = tail call float @llvm.fmuladd.f32(float %1010, float %1012, float %1009)
  %1014 = load float, ptr %835, align 4, !tbaa !11
  %1015 = getelementptr float, ptr %869, i64 %987
  %1016 = load float, ptr %1015, align 4, !tbaa !11
  %1017 = tail call float @llvm.fmuladd.f32(float %1014, float %1016, float %1013)
  %1018 = load float, ptr %838, align 4, !tbaa !11
  %1019 = getelementptr float, ptr %870, i64 %987
  %1020 = load float, ptr %1019, align 4, !tbaa !11
  %1021 = tail call float @llvm.fmuladd.f32(float %1018, float %1020, float %1017)
  %1022 = load float, ptr %841, align 4, !tbaa !11
  %1023 = getelementptr float, ptr %871, i64 %987
  %1024 = load float, ptr %1023, align 4, !tbaa !11
  %1025 = tail call float @llvm.fmuladd.f32(float %1022, float %1024, float %1021)
  %1026 = load float, ptr %844, align 4, !tbaa !11
  %1027 = getelementptr float, ptr %872, i64 %987
  %1028 = load float, ptr %1027, align 4, !tbaa !11
  %1029 = tail call float @llvm.fmuladd.f32(float %1026, float %1028, float %1025)
  %1030 = load float, ptr %847, align 4, !tbaa !11
  %1031 = getelementptr float, ptr %873, i64 %987
  %1032 = load float, ptr %1031, align 4, !tbaa !11
  %1033 = tail call float @llvm.fmuladd.f32(float %1030, float %1032, float %1029)
  %1034 = load float, ptr %850, align 4, !tbaa !11
  %1035 = getelementptr float, ptr %874, i64 %987
  %1036 = load float, ptr %1035, align 4, !tbaa !11
  %1037 = tail call float @llvm.fmuladd.f32(float %1034, float %1036, float %1033)
  %1038 = load float, ptr %853, align 4, !tbaa !11
  %1039 = getelementptr float, ptr %875, i64 %987
  %1040 = load float, ptr %1039, align 4, !tbaa !11
  %1041 = tail call float @llvm.fmuladd.f32(float %1038, float %1040, float %1037)
  %1042 = load float, ptr %856, align 4, !tbaa !11
  %1043 = getelementptr float, ptr %876, i64 %987
  %1044 = load float, ptr %1043, align 4, !tbaa !11
  %1045 = tail call float @llvm.fmuladd.f32(float %1042, float %1044, float %1041)
  %1046 = load float, ptr %859, align 4, !tbaa !11
  %1047 = getelementptr float, ptr %877, i64 %987
  %1048 = load float, ptr %1047, align 4, !tbaa !11
  %1049 = tail call float @llvm.fmuladd.f32(float %1046, float %1048, float %1045)
  %1050 = load float, ptr %861, align 4, !tbaa !11
  %1051 = getelementptr float, ptr %878, i64 %987
  %1052 = load float, ptr %1051, align 4, !tbaa !11
  %1053 = tail call float @llvm.fmuladd.f32(float %1050, float %1052, float %1049)
  store float %1053, ptr %988, align 4, !tbaa !11
  %1054 = add nuw nsw i64 %987, 1
  %1055 = icmp eq i64 %1054, %584
  br i1 %1055, label %1056, label %986, !llvm.loop !220

1056:                                             ; preds = %986, %983
  %1057 = add nuw nsw i64 %815, 16
  %1058 = icmp samesign ult i64 %1057, %582
  br i1 %1058, label %814, label %1059, !llvm.loop !221

1059:                                             ; preds = %1056, %572, %308, %576
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef float @epslon(float noundef %0) local_unnamed_addr #2 {
  %2 = tail call float @llvm.fabs.f32(float %0)
  %3 = fmul float %2, 0x3E80000000000000
  ret float %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @print_time(i32 noundef %0) local_unnamed_addr #2 {
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define dso_local i32 @idamax(i32 noundef %0, ptr noundef readonly captures(none) %1, i32 noundef %2) local_unnamed_addr #7 {
  %4 = icmp slt i32 %0, 1
  br i1 %4, label %46, label %5

5:                                                ; preds = %3
  %6 = icmp eq i32 %0, 1
  br i1 %6, label %46, label %7

7:                                                ; preds = %5
  %8 = icmp eq i32 %2, 1
  br i1 %8, label %29, label %9

9:                                                ; preds = %7
  %10 = add i32 %2, 1
  %11 = load float, ptr %1, align 4, !tbaa !11
  %12 = tail call float @llvm.fabs.f32(float %11)
  %13 = sext i32 %10 to i64
  %14 = sext i32 %2 to i64
  br label %15

15:                                               ; preds = %9, %15
  %16 = phi i64 [ %13, %9 ], [ %26, %15 ]
  %17 = phi i32 [ undef, %9 ], [ %25, %15 ]
  %18 = phi i32 [ 1, %9 ], [ %27, %15 ]
  %19 = phi float [ %12, %9 ], [ %24, %15 ]
  %20 = getelementptr inbounds float, ptr %1, i64 %16
  %21 = load float, ptr %20, align 4, !tbaa !11
  %22 = tail call float @llvm.fabs.f32(float %21)
  %23 = fcmp ogt float %22, %19
  %24 = select i1 %23, float %22, float %19
  %25 = select i1 %23, i32 %18, i32 %17
  %26 = add nsw i64 %16, %14
  %27 = add nuw nsw i32 %18, 1
  %28 = icmp eq i32 %27, %0
  br i1 %28, label %46, label %15, !llvm.loop !222

29:                                               ; preds = %7
  %30 = load float, ptr %1, align 4, !tbaa !11
  %31 = tail call float @llvm.fabs.f32(float %30)
  %32 = zext nneg i32 %0 to i64
  br label %33

33:                                               ; preds = %29, %33
  %34 = phi i64 [ 1, %29 ], [ %44, %33 ]
  %35 = phi i32 [ 0, %29 ], [ %43, %33 ]
  %36 = phi float [ %31, %29 ], [ %41, %33 ]
  %37 = getelementptr inbounds nuw float, ptr %1, i64 %34
  %38 = load float, ptr %37, align 4, !tbaa !11
  %39 = tail call float @llvm.fabs.f32(float %38)
  %40 = fcmp ogt float %39, %36
  %41 = select i1 %40, float %39, float %36
  %42 = trunc nuw nsw i64 %34 to i32
  %43 = select i1 %40, i32 %42, i32 %35
  %44 = add nuw nsw i64 %34, 1
  %45 = icmp eq i64 %44, %32
  br i1 %45, label %46, label %33, !llvm.loop !29

46:                                               ; preds = %15, %33, %5, %3
  %47 = phi i32 [ -1, %3 ], [ 0, %5 ], [ %43, %33 ], [ %25, %15 ]
  ret i32 %47
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @dscal(i32 noundef %0, float noundef %1, ptr noundef captures(none) %2, i32 noundef %3) local_unnamed_addr #6 {
  %5 = icmp slt i32 %0, 1
  br i1 %5, label %49, label %6

6:                                                ; preds = %4
  %7 = icmp eq i32 %3, 1
  br i1 %7, label %8, label %29

8:                                                ; preds = %6
  %9 = zext nneg i32 %0 to i64
  %10 = icmp ult i32 %0, 8
  br i1 %10, label %27, label %11

11:                                               ; preds = %8
  %12 = and i64 %9, 2147483640
  %13 = insertelement <4 x float> poison, float %1, i64 0
  %14 = shufflevector <4 x float> %13, <4 x float> poison, <4 x i32> zeroinitializer
  br label %15

15:                                               ; preds = %15, %11
  %16 = phi i64 [ 0, %11 ], [ %23, %15 ]
  %17 = getelementptr inbounds nuw float, ptr %2, i64 %16
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 16
  %19 = load <4 x float>, ptr %17, align 4, !tbaa !11
  %20 = load <4 x float>, ptr %18, align 4, !tbaa !11
  %21 = fmul <4 x float> %14, %19
  %22 = fmul <4 x float> %14, %20
  store <4 x float> %21, ptr %17, align 4, !tbaa !11
  store <4 x float> %22, ptr %18, align 4, !tbaa !11
  %23 = add nuw i64 %16, 8
  %24 = icmp eq i64 %23, %12
  br i1 %24, label %25, label %15, !llvm.loop !223

25:                                               ; preds = %15
  %26 = icmp eq i64 %12, %9
  br i1 %26, label %49, label %27

27:                                               ; preds = %8, %25
  %28 = phi i64 [ 0, %8 ], [ %12, %25 ]
  br label %42

29:                                               ; preds = %6
  %30 = mul nsw i32 %3, %0
  %31 = icmp sgt i32 %30, 0
  br i1 %31, label %32, label %49

32:                                               ; preds = %29
  %33 = sext i32 %3 to i64
  %34 = zext nneg i32 %30 to i64
  br label %35

35:                                               ; preds = %32, %35
  %36 = phi i64 [ 0, %32 ], [ %40, %35 ]
  %37 = getelementptr inbounds float, ptr %2, i64 %36
  %38 = load float, ptr %37, align 4, !tbaa !11
  %39 = fmul float %1, %38
  store float %39, ptr %37, align 4, !tbaa !11
  %40 = add nsw i64 %36, %33
  %41 = icmp slt i64 %40, %34
  br i1 %41, label %35, label %49, !llvm.loop !224

42:                                               ; preds = %27, %42
  %43 = phi i64 [ %47, %42 ], [ %28, %27 ]
  %44 = getelementptr inbounds nuw float, ptr %2, i64 %43
  %45 = load float, ptr %44, align 4, !tbaa !11
  %46 = fmul float %1, %45
  store float %46, ptr %44, align 4, !tbaa !11
  %47 = add nuw nsw i64 %43, 1
  %48 = icmp eq i64 %47, %9
  br i1 %48, label %49, label %42, !llvm.loop !225

49:                                               ; preds = %35, %42, %25, %29, %4
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @daxpy(i32 noundef %0, float noundef %1, ptr noundef readonly captures(none) %2, i32 noundef %3, ptr noundef captures(none) %4, i32 noundef %5) local_unnamed_addr #6 {
  %7 = icmp slt i32 %0, 1
  %8 = fcmp oeq float %1, 0.000000e+00
  %9 = or i1 %7, %8
  br i1 %9, label %136, label %10

10:                                               ; preds = %6
  %11 = icmp ne i32 %3, 1
  %12 = icmp ne i32 %5, 1
  %13 = or i1 %11, %12
  br i1 %13, label %46, label %14

14:                                               ; preds = %10
  %15 = zext nneg i32 %0 to i64
  %16 = icmp ult i32 %0, 8
  br i1 %16, label %44, label %17

17:                                               ; preds = %14
  %18 = shl nuw nsw i64 %15, 2
  %19 = getelementptr i8, ptr %4, i64 %18
  %20 = getelementptr i8, ptr %2, i64 %18
  %21 = icmp ult ptr %4, %20
  %22 = icmp ult ptr %2, %19
  %23 = and i1 %21, %22
  br i1 %23, label %44, label %24

24:                                               ; preds = %17
  %25 = and i64 %15, 2147483640
  %26 = insertelement <4 x float> poison, float %1, i64 0
  %27 = shufflevector <4 x float> %26, <4 x float> poison, <4 x i32> zeroinitializer
  br label %28

28:                                               ; preds = %28, %24
  %29 = phi i64 [ 0, %24 ], [ %40, %28 ]
  %30 = getelementptr inbounds nuw float, ptr %4, i64 %29
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 16
  %32 = load <4 x float>, ptr %30, align 4, !tbaa !11, !alias.scope !226, !noalias !229
  %33 = load <4 x float>, ptr %31, align 4, !tbaa !11, !alias.scope !226, !noalias !229
  %34 = getelementptr inbounds nuw float, ptr %2, i64 %29
  %35 = getelementptr inbounds nuw i8, ptr %34, i64 16
  %36 = load <4 x float>, ptr %34, align 4, !tbaa !11, !alias.scope !229
  %37 = load <4 x float>, ptr %35, align 4, !tbaa !11, !alias.scope !229
  %38 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %27, <4 x float> %36, <4 x float> %32)
  %39 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %27, <4 x float> %37, <4 x float> %33)
  store <4 x float> %38, ptr %30, align 4, !tbaa !11, !alias.scope !226, !noalias !229
  store <4 x float> %39, ptr %31, align 4, !tbaa !11, !alias.scope !226, !noalias !229
  %40 = add nuw i64 %29, 8
  %41 = icmp eq i64 %40, %25
  br i1 %41, label %42, label %28, !llvm.loop !231

42:                                               ; preds = %28
  %43 = icmp eq i64 %25, %15
  br i1 %43, label %136, label %44

44:                                               ; preds = %17, %14, %42
  %45 = phi i64 [ 0, %17 ], [ 0, %14 ], [ %25, %42 ]
  br label %127

46:                                               ; preds = %10
  %47 = icmp slt i32 %5, 0
  %48 = sub nsw i32 1, %0
  %49 = mul nsw i32 %5, %48
  %50 = select i1 %47, i32 %49, i32 0
  %51 = icmp slt i32 %3, 0
  %52 = mul nsw i32 %3, %48
  %53 = select i1 %51, i32 %52, i32 0
  %54 = sext i32 %53 to i64
  %55 = sext i32 %3 to i64
  %56 = sext i32 %50 to i64
  %57 = sext i32 %5 to i64
  %58 = zext nneg i32 %0 to i64
  %59 = icmp ult i32 %0, 12
  br i1 %59, label %110, label %60

60:                                               ; preds = %46
  %61 = icmp ne i32 %5, 1
  %62 = icmp ne i32 %3, 1
  %63 = or i1 %61, %62
  br i1 %63, label %110, label %64

64:                                               ; preds = %60
  %65 = shl nsw i64 %56, 2
  %66 = getelementptr i8, ptr %4, i64 %65
  %67 = add nsw i32 %0, -1
  %68 = zext i32 %67 to i64
  %69 = shl nuw nsw i64 %68, 2
  %70 = getelementptr i8, ptr %4, i64 %65
  %71 = getelementptr i8, ptr %70, i64 %69
  %72 = getelementptr i8, ptr %71, i64 4
  %73 = shl nsw i64 %54, 2
  %74 = getelementptr i8, ptr %2, i64 %73
  %75 = getelementptr i8, ptr %2, i64 %73
  %76 = getelementptr i8, ptr %75, i64 %69
  %77 = getelementptr i8, ptr %76, i64 4
  %78 = icmp ult ptr %66, %77
  %79 = icmp ult ptr %74, %72
  %80 = and i1 %78, %79
  br i1 %80, label %110, label %81

81:                                               ; preds = %64
  %82 = and i64 %58, 2147483640
  %83 = mul nuw nsw i64 %82, %57
  %84 = add nsw i64 %83, %56
  %85 = mul nuw nsw i64 %82, %55
  %86 = add nsw i64 %85, %54
  %87 = trunc nuw nsw i64 %82 to i32
  %88 = insertelement <4 x float> poison, float %1, i64 0
  %89 = shufflevector <4 x float> %88, <4 x float> poison, <4 x i32> zeroinitializer
  %90 = getelementptr float, ptr %4, i64 %56
  %91 = getelementptr float, ptr %2, i64 %54
  br label %92

92:                                               ; preds = %92, %81
  %93 = phi i64 [ 0, %81 ], [ %106, %92 ]
  %94 = mul nuw i64 %93, %57
  %95 = mul nuw i64 %93, %55
  %96 = getelementptr float, ptr %90, i64 %94
  %97 = getelementptr inbounds nuw i8, ptr %96, i64 16
  %98 = load <4 x float>, ptr %96, align 4, !tbaa !11, !alias.scope !232, !noalias !235
  %99 = load <4 x float>, ptr %97, align 4, !tbaa !11, !alias.scope !232, !noalias !235
  %100 = getelementptr float, ptr %91, i64 %95
  %101 = getelementptr inbounds nuw i8, ptr %100, i64 16
  %102 = load <4 x float>, ptr %100, align 4, !tbaa !11, !alias.scope !235
  %103 = load <4 x float>, ptr %101, align 4, !tbaa !11, !alias.scope !235
  %104 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %89, <4 x float> %102, <4 x float> %98)
  %105 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %89, <4 x float> %103, <4 x float> %99)
  store <4 x float> %104, ptr %96, align 4, !tbaa !11, !alias.scope !232, !noalias !235
  store <4 x float> %105, ptr %97, align 4, !tbaa !11, !alias.scope !232, !noalias !235
  %106 = add nuw i64 %93, 8
  %107 = icmp eq i64 %106, %82
  br i1 %107, label %108, label %92, !llvm.loop !237

108:                                              ; preds = %92
  %109 = icmp eq i64 %82, %58
  br i1 %109, label %136, label %110

110:                                              ; preds = %64, %60, %46, %108
  %111 = phi i64 [ %56, %64 ], [ %56, %60 ], [ %56, %46 ], [ %84, %108 ]
  %112 = phi i64 [ %54, %64 ], [ %54, %60 ], [ %54, %46 ], [ %86, %108 ]
  %113 = phi i32 [ 0, %64 ], [ 0, %60 ], [ 0, %46 ], [ %87, %108 ]
  br label %114

114:                                              ; preds = %110, %114
  %115 = phi i64 [ %124, %114 ], [ %111, %110 ]
  %116 = phi i64 [ %123, %114 ], [ %112, %110 ]
  %117 = phi i32 [ %125, %114 ], [ %113, %110 ]
  %118 = getelementptr inbounds float, ptr %4, i64 %115
  %119 = load float, ptr %118, align 4, !tbaa !11
  %120 = getelementptr inbounds float, ptr %2, i64 %116
  %121 = load float, ptr %120, align 4, !tbaa !11
  %122 = tail call float @llvm.fmuladd.f32(float %1, float %121, float %119)
  store float %122, ptr %118, align 4, !tbaa !11
  %123 = add nsw i64 %116, %55
  %124 = add nsw i64 %115, %57
  %125 = add nuw nsw i32 %117, 1
  %126 = icmp eq i32 %125, %0
  br i1 %126, label %136, label %114, !llvm.loop !238

127:                                              ; preds = %44, %127
  %128 = phi i64 [ %134, %127 ], [ %45, %44 ]
  %129 = getelementptr inbounds nuw float, ptr %4, i64 %128
  %130 = load float, ptr %129, align 4, !tbaa !11
  %131 = getelementptr inbounds nuw float, ptr %2, i64 %128
  %132 = load float, ptr %131, align 4, !tbaa !11
  %133 = tail call float @llvm.fmuladd.f32(float %1, float %132, float %130)
  store float %133, ptr %129, align 4, !tbaa !11
  %134 = add nuw nsw i64 %128, 1
  %135 = icmp eq i64 %134, %15
  br i1 %135, label %136, label %127, !llvm.loop !239

136:                                              ; preds = %127, %114, %42, %108, %6
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define dso_local float @ddot(i32 noundef %0, ptr noundef readonly captures(none) %1, i32 noundef %2, ptr noundef readonly captures(none) %3, i32 noundef %4) local_unnamed_addr #7 {
  %6 = icmp slt i32 %0, 1
  br i1 %6, label %115, label %7

7:                                                ; preds = %5
  %8 = icmp ne i32 %2, 1
  %9 = icmp ne i32 %4, 1
  %10 = or i1 %8, %9
  br i1 %10, label %38, label %11

11:                                               ; preds = %7
  %12 = zext nneg i32 %0 to i64
  %13 = icmp ult i32 %0, 8
  br i1 %13, label %35, label %14

14:                                               ; preds = %11
  %15 = and i64 %12, 2147483640
  br label %16

16:                                               ; preds = %16, %14
  %17 = phi i64 [ 0, %14 ], [ %31, %16 ]
  %18 = phi float [ 0.000000e+00, %14 ], [ %30, %16 ]
  %19 = getelementptr inbounds nuw float, ptr %1, i64 %17
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 16
  %21 = load <4 x float>, ptr %19, align 4, !tbaa !11
  %22 = load <4 x float>, ptr %20, align 4, !tbaa !11
  %23 = getelementptr inbounds nuw float, ptr %3, i64 %17
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <4 x float>, ptr %23, align 4, !tbaa !11
  %26 = load <4 x float>, ptr %24, align 4, !tbaa !11
  %27 = fmul <4 x float> %21, %25
  %28 = fmul <4 x float> %22, %26
  %29 = tail call float @llvm.vector.reduce.fadd.v4f32(float %18, <4 x float> %27)
  %30 = tail call float @llvm.vector.reduce.fadd.v4f32(float %29, <4 x float> %28)
  %31 = add nuw i64 %17, 8
  %32 = icmp eq i64 %31, %15
  br i1 %32, label %33, label %16, !llvm.loop !240

33:                                               ; preds = %16
  %34 = icmp eq i64 %15, %12
  br i1 %34, label %115, label %35

35:                                               ; preds = %11, %33
  %36 = phi i64 [ 0, %11 ], [ %15, %33 ]
  %37 = phi float [ 0.000000e+00, %11 ], [ %30, %33 ]
  br label %105

38:                                               ; preds = %7
  %39 = icmp slt i32 %4, 0
  %40 = sub nsw i32 1, %0
  %41 = mul nsw i32 %4, %40
  %42 = select i1 %39, i32 %41, i32 0
  %43 = icmp slt i32 %2, 0
  %44 = mul nsw i32 %2, %40
  %45 = select i1 %43, i32 %44, i32 0
  %46 = sext i32 %42 to i64
  %47 = sext i32 %4 to i64
  %48 = sext i32 %45 to i64
  %49 = sext i32 %2 to i64
  %50 = zext nneg i32 %0 to i64
  %51 = icmp ult i32 %0, 8
  br i1 %51, label %86, label %52

52:                                               ; preds = %38
  %53 = icmp ne i32 %2, 1
  %54 = icmp ne i32 %4, 1
  %55 = or i1 %53, %54
  br i1 %55, label %86, label %56

56:                                               ; preds = %52
  %57 = and i64 %50, 2147483640
  %58 = mul nuw nsw i64 %57, %49
  %59 = add nsw i64 %58, %48
  %60 = mul nuw nsw i64 %57, %47
  %61 = add nsw i64 %60, %46
  %62 = trunc nuw nsw i64 %57 to i32
  %63 = getelementptr float, ptr %1, i64 %48
  %64 = getelementptr float, ptr %3, i64 %46
  br label %65

65:                                               ; preds = %65, %56
  %66 = phi i64 [ 0, %56 ], [ %82, %65 ]
  %67 = phi float [ 0.000000e+00, %56 ], [ %81, %65 ]
  %68 = mul nuw i64 %66, %49
  %69 = mul nuw i64 %66, %47
  %70 = getelementptr float, ptr %63, i64 %68
  %71 = getelementptr inbounds nuw i8, ptr %70, i64 16
  %72 = load <4 x float>, ptr %70, align 4, !tbaa !11
  %73 = load <4 x float>, ptr %71, align 4, !tbaa !11
  %74 = getelementptr float, ptr %64, i64 %69
  %75 = getelementptr inbounds nuw i8, ptr %74, i64 16
  %76 = load <4 x float>, ptr %74, align 4, !tbaa !11
  %77 = load <4 x float>, ptr %75, align 4, !tbaa !11
  %78 = fmul <4 x float> %72, %76
  %79 = fmul <4 x float> %73, %77
  %80 = tail call float @llvm.vector.reduce.fadd.v4f32(float %67, <4 x float> %78)
  %81 = tail call float @llvm.vector.reduce.fadd.v4f32(float %80, <4 x float> %79)
  %82 = add nuw i64 %66, 8
  %83 = icmp eq i64 %82, %57
  br i1 %83, label %84, label %65, !llvm.loop !241

84:                                               ; preds = %65
  %85 = icmp eq i64 %57, %50
  br i1 %85, label %115, label %86

86:                                               ; preds = %52, %38, %84
  %87 = phi i64 [ %48, %52 ], [ %48, %38 ], [ %59, %84 ]
  %88 = phi i64 [ %46, %52 ], [ %46, %38 ], [ %61, %84 ]
  %89 = phi float [ 0.000000e+00, %52 ], [ 0.000000e+00, %38 ], [ %81, %84 ]
  %90 = phi i32 [ 0, %52 ], [ 0, %38 ], [ %62, %84 ]
  br label %91

91:                                               ; preds = %86, %91
  %92 = phi i64 [ %101, %91 ], [ %87, %86 ]
  %93 = phi i64 [ %102, %91 ], [ %88, %86 ]
  %94 = phi float [ %100, %91 ], [ %89, %86 ]
  %95 = phi i32 [ %103, %91 ], [ %90, %86 ]
  %96 = getelementptr inbounds float, ptr %1, i64 %92
  %97 = load float, ptr %96, align 4, !tbaa !11
  %98 = getelementptr inbounds float, ptr %3, i64 %93
  %99 = load float, ptr %98, align 4, !tbaa !11
  %100 = tail call float @llvm.fmuladd.f32(float %97, float %99, float %94)
  %101 = add nsw i64 %92, %49
  %102 = add nsw i64 %93, %47
  %103 = add nuw nsw i32 %95, 1
  %104 = icmp eq i32 %103, %0
  br i1 %104, label %115, label %91, !llvm.loop !242

105:                                              ; preds = %35, %105
  %106 = phi i64 [ %113, %105 ], [ %36, %35 ]
  %107 = phi float [ %112, %105 ], [ %37, %35 ]
  %108 = getelementptr inbounds nuw float, ptr %1, i64 %106
  %109 = load float, ptr %108, align 4, !tbaa !11
  %110 = getelementptr inbounds nuw float, ptr %3, i64 %106
  %111 = load float, ptr %110, align 4, !tbaa !11
  %112 = tail call float @llvm.fmuladd.f32(float %109, float %111, float %107)
  %113 = add nuw nsw i64 %106, 1
  %114 = icmp eq i64 %113, %12
  br i1 %114, label %115, label %105, !llvm.loop !243

115:                                              ; preds = %105, %91, %33, %84, %5
  %116 = phi float [ 0.000000e+00, %5 ], [ %81, %84 ], [ %30, %33 ], [ %100, %91 ], [ %112, %105 ]
  ret float %116
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #8

; Function Attrs: nofree nounwind
declare noundef i64 @fwrite(ptr noundef readonly captures(none), i64 noundef, i64 noundef, ptr noundef captures(none)) local_unnamed_addr #9

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #10

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #11

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>) #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #10

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #5 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nofree norecurse nosync nounwind memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { nofree nounwind }
attributes #10 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #11 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #12 = { nounwind }
attributes #13 = { cold }
attributes #14 = { cold nounwind }

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
!12 = !{!"float", !9, i64 0}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.mustprogress"}
!15 = distinct !{!15, !14}
!16 = distinct !{!16, !14}
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !9, i64 0}
!19 = distinct !{!19, !14, !20, !21}
!20 = !{!"llvm.loop.isvectorized", i32 1}
!21 = !{!"llvm.loop.unroll.runtime.disable"}
!22 = distinct !{!22, !14, !21, !20}
!23 = distinct !{!23, !14}
!24 = distinct !{!24, !14, !20, !21}
!25 = distinct !{!25, !14, !21, !20}
!26 = distinct !{!26, !14}
!27 = distinct !{!27, !14}
!28 = distinct !{!28, !14}
!29 = distinct !{!29, !14}
!30 = distinct !{!30, !14, !20, !21}
!31 = distinct !{!31, !14, !21, !20}
!32 = distinct !{!32, !14, !20, !21}
!33 = distinct !{!33, !14, !21, !20}
!34 = distinct !{!34, !14}
!35 = distinct !{!35, !14}
!36 = distinct !{!36, !14}
!37 = distinct !{!37, !14}
!38 = distinct !{!38, !14, !20, !21}
!39 = distinct !{!39, !14, !21, !20}
!40 = distinct !{!40, !14, !20, !21}
!41 = distinct !{!41, !14, !21, !20}
!42 = distinct !{!42, !14}
!43 = distinct !{!43, !14, !20, !21}
!44 = distinct !{!44, !14, !21, !20}
!45 = distinct !{!45, !14, !20, !21}
!46 = distinct !{!46, !14, !21, !20}
!47 = distinct !{!47, !14}
!48 = distinct !{!48, !14}
!49 = distinct !{!49, !14}
!50 = distinct !{!50, !14, !20, !21}
!51 = distinct !{!51, !14, !21, !20}
!52 = distinct !{!52, !14, !20, !21}
!53 = distinct !{!53, !14, !21, !20}
!54 = distinct !{!54, !14}
!55 = distinct !{!55, !14, !20, !21}
!56 = distinct !{!56, !14, !21, !20}
!57 = distinct !{!57, !14, !20, !21}
!58 = distinct !{!58, !14, !21, !20}
!59 = distinct !{!59, !14}
!60 = distinct !{!60, !14}
!61 = !{!62}
!62 = distinct !{!62, !63}
!63 = distinct !{!63, !"LVerDomain"}
!64 = !{!65}
!65 = distinct !{!65, !63}
!66 = distinct !{!66, !14, !20, !21}
!67 = distinct !{!67, !14, !20}
!68 = distinct !{!68, !14, !20, !21}
!69 = distinct !{!69, !14, !21, !20}
!70 = !{!71}
!71 = distinct !{!71, !72}
!72 = distinct !{!72, !"LVerDomain"}
!73 = !{!74}
!74 = distinct !{!74, !72}
!75 = distinct !{!75, !14, !20, !21}
!76 = distinct !{!76, !14, !20}
!77 = !{!78}
!78 = distinct !{!78, !79}
!79 = distinct !{!79, !"LVerDomain"}
!80 = !{!81}
!81 = distinct !{!81, !79}
!82 = distinct !{!82, !14, !20, !21}
!83 = distinct !{!83, !14, !20}
!84 = !{!85}
!85 = distinct !{!85, !86}
!86 = distinct !{!86, !"LVerDomain"}
!87 = !{!88}
!88 = distinct !{!88, !86}
!89 = distinct !{!89, !14, !20, !21}
!90 = distinct !{!90, !14, !20}
!91 = distinct !{!91, !14, !20, !21}
!92 = distinct !{!92, !14, !21, !20}
!93 = distinct !{!93, !14}
!94 = distinct !{!94, !14, !20, !21}
!95 = distinct !{!95, !14, !21, !20}
!96 = distinct !{!96, !14}
!97 = !{!98}
!98 = distinct !{!98, !99}
!99 = distinct !{!99, !"LVerDomain"}
!100 = !{!101}
!101 = distinct !{!101, !99}
!102 = !{!98, !103}
!103 = distinct !{!103, !99}
!104 = !{!103}
!105 = distinct !{!105, !14, !20, !21}
!106 = distinct !{!106, !14, !20}
!107 = !{!108}
!108 = distinct !{!108, !109}
!109 = distinct !{!109, !"LVerDomain"}
!110 = !{!111}
!111 = distinct !{!111, !109}
!112 = !{!113}
!113 = distinct !{!113, !109}
!114 = !{!111, !108, !115, !116}
!115 = distinct !{!115, !109}
!116 = distinct !{!116, !109}
!117 = !{!116}
!118 = !{!115}
!119 = distinct !{!119, !14, !20, !21}
!120 = distinct !{!120, !14, !20}
!121 = !{!122}
!122 = distinct !{!122, !123}
!123 = distinct !{!123, !"LVerDomain"}
!124 = !{!125}
!125 = distinct !{!125, !123}
!126 = !{!127}
!127 = distinct !{!127, !123}
!128 = !{!129}
!129 = distinct !{!129, !123}
!130 = !{!131}
!131 = distinct !{!131, !123}
!132 = !{!129, !127, !125, !122, !133, !134, !135, !136}
!133 = distinct !{!133, !123}
!134 = distinct !{!134, !123}
!135 = distinct !{!135, !123}
!136 = distinct !{!136, !123}
!137 = !{!136}
!138 = !{!135}
!139 = !{!134}
!140 = !{!133}
!141 = distinct !{!141, !14, !20, !21}
!142 = distinct !{!142, !14, !20}
!143 = !{!144}
!144 = distinct !{!144, !145}
!145 = distinct !{!145, !"LVerDomain"}
!146 = !{!147}
!147 = distinct !{!147, !145}
!148 = !{!149}
!149 = distinct !{!149, !145}
!150 = !{!151}
!151 = distinct !{!151, !145}
!152 = !{!153}
!153 = distinct !{!153, !145}
!154 = !{!155}
!155 = distinct !{!155, !145}
!156 = !{!157}
!157 = distinct !{!157, !145}
!158 = !{!159}
!159 = distinct !{!159, !145}
!160 = !{!161}
!161 = distinct !{!161, !145}
!162 = !{!159, !157, !155, !153, !151, !149, !147, !144, !163, !164, !165, !166, !167, !168, !169, !170}
!163 = distinct !{!163, !145}
!164 = distinct !{!164, !145}
!165 = distinct !{!165, !145}
!166 = distinct !{!166, !145}
!167 = distinct !{!167, !145}
!168 = distinct !{!168, !145}
!169 = distinct !{!169, !145}
!170 = distinct !{!170, !145}
!171 = !{!170}
!172 = !{!169}
!173 = !{!168}
!174 = !{!167}
!175 = !{!166}
!176 = !{!165}
!177 = !{!164}
!178 = !{!163}
!179 = distinct !{!179, !14, !20, !21}
!180 = distinct !{!180, !14, !20}
!181 = !{!182}
!182 = distinct !{!182, !183}
!183 = distinct !{!183, !"LVerDomain"}
!184 = !{!185}
!185 = distinct !{!185, !183}
!186 = !{!182, !187, !188, !189, !190, !191, !192, !193, !194, !195, !196, !197, !198, !199, !200, !201, !202}
!187 = distinct !{!187, !183}
!188 = distinct !{!188, !183}
!189 = distinct !{!189, !183}
!190 = distinct !{!190, !183}
!191 = distinct !{!191, !183}
!192 = distinct !{!192, !183}
!193 = distinct !{!193, !183}
!194 = distinct !{!194, !183}
!195 = distinct !{!195, !183}
!196 = distinct !{!196, !183}
!197 = distinct !{!197, !183}
!198 = distinct !{!198, !183}
!199 = distinct !{!199, !183}
!200 = distinct !{!200, !183}
!201 = distinct !{!201, !183}
!202 = distinct !{!202, !183}
!203 = !{!202}
!204 = !{!201}
!205 = !{!200}
!206 = !{!199}
!207 = !{!198}
!208 = !{!197}
!209 = !{!196}
!210 = !{!195}
!211 = !{!194}
!212 = !{!193}
!213 = !{!192}
!214 = !{!191}
!215 = !{!190}
!216 = !{!189}
!217 = !{!188}
!218 = !{!187}
!219 = distinct !{!219, !14, !20, !21}
!220 = distinct !{!220, !14, !20}
!221 = distinct !{!221, !14}
!222 = distinct !{!222, !14}
!223 = distinct !{!223, !14, !20, !21}
!224 = distinct !{!224, !14, !20}
!225 = distinct !{!225, !14, !21, !20}
!226 = !{!227}
!227 = distinct !{!227, !228}
!228 = distinct !{!228, !"LVerDomain"}
!229 = !{!230}
!230 = distinct !{!230, !228}
!231 = distinct !{!231, !14, !20, !21}
!232 = !{!233}
!233 = distinct !{!233, !234}
!234 = distinct !{!234, !"LVerDomain"}
!235 = !{!236}
!236 = distinct !{!236, !234}
!237 = distinct !{!237, !14, !20, !21}
!238 = distinct !{!238, !14, !20}
!239 = distinct !{!239, !14, !20}
!240 = distinct !{!240, !14, !20, !21}
!241 = distinct !{!241, !14, !20, !21}
!242 = distinct !{!242, !14, !20}
!243 = distinct !{!243, !14, !21, !20}
