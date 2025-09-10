; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/oopack_v1p8.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/oopack_v1p8.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%class.MaxBenchmark = type { %class.Benchmark }
%class.Benchmark = type { ptr }
%class.MatrixBenchmark = type { %class.Benchmark }
%class.IteratorBenchmark = type { %class.Benchmark }
%class.ComplexBenchmark = type { %class.Benchmark }
%class.Complex = type { double, double }

$_ZNK12MaxBenchmark4nameEv = comdat any

$_ZNK15MatrixBenchmark4nameEv = comdat any

$_ZNK17IteratorBenchmark4nameEv = comdat any

$_ZNK16ComplexBenchmark4nameEv = comdat any

$_ZTI9Benchmark = comdat any

$_ZTS9Benchmark = comdat any

@_ZN9Benchmark5countE = dso_local local_unnamed_addr global i32 4, align 4
@_ZN9Benchmark4listE = dso_local local_unnamed_addr global [4 x ptr] [ptr @TheMaxBenchmark, ptr @TheMatrixBenchmark, ptr @TheIteratorBenchmark, ptr @TheComplexBenchmark], align 8
@U = dso_local local_unnamed_addr global [1000 x double] zeroinitializer, align 8
@MaxResult = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@TheMaxBenchmark = dso_local global %class.MaxBenchmark { %class.Benchmark { ptr getelementptr inbounds nuw inrange(-16, 40) (i8, ptr @_ZTV12MaxBenchmark, i64 16) } }, align 8
@C = dso_local local_unnamed_addr global [2500 x double] zeroinitializer, align 8
@D = dso_local local_unnamed_addr global [2500 x double] zeroinitializer, align 8
@E = dso_local local_unnamed_addr global [2500 x double] zeroinitializer, align 8
@TheMatrixBenchmark = dso_local global %class.MatrixBenchmark { %class.Benchmark { ptr getelementptr inbounds nuw inrange(-16, 40) (i8, ptr @_ZTV15MatrixBenchmark, i64 16) } }, align 8
@A = dso_local local_unnamed_addr global [1000 x double] zeroinitializer, align 8
@B = dso_local local_unnamed_addr global [1000 x double] zeroinitializer, align 8
@IteratorResult = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@TheIteratorBenchmark = dso_local global %class.IteratorBenchmark { %class.Benchmark { ptr getelementptr inbounds nuw inrange(-16, 40) (i8, ptr @_ZTV17IteratorBenchmark, i64 16) } }, align 8
@TheComplexBenchmark = dso_local global %class.ComplexBenchmark { %class.Benchmark { ptr getelementptr inbounds nuw inrange(-16, 40) (i8, ptr @_ZTV16ComplexBenchmark, i64 16) } }, align 8
@X = dso_local local_unnamed_addr global [1000 x %class.Complex] zeroinitializer, align 8
@Y = dso_local local_unnamed_addr global [1000 x %class.Complex] zeroinitializer, align 8
@C_Seconds = dso_local local_unnamed_addr global double 1.000000e+00, align 8
@.str = private unnamed_addr constant [75 x i8] c"%-10s: warning: relative checksum error of %g between C (%g) and oop (%g)\0A\00", align 1
@.str.6 = private unnamed_addr constant [12 x i8] c"%-10s %10d\0A\00", align 1
@.str.7 = private unnamed_addr constant [12 x i8] c"Version 1.7\00", align 1
@Version = dso_local local_unnamed_addr global ptr @.str.7, align 8
@.str.8 = private unnamed_addr constant [51 x i8] c"Usage:\09%s test1=iterations1 test2=iterations2 ...\0A\00", align 1
@__const.main.str1 = private unnamed_addr constant [6 x i8] c"a.out\00", align 1
@__const.main.str2 = private unnamed_addr constant [10 x i8] c"Max=15000\00", align 1
@__const.main.str3 = private unnamed_addr constant [11 x i8] c"Matrix=200\00", align 1
@__const.main.str4 = private unnamed_addr constant [13 x i8] c"Complex=2000\00", align 1
@__const.main.str5 = private unnamed_addr constant [15 x i8] c"Iterator=20000\00", align 1
@.str.10 = private unnamed_addr constant [29 x i8] c"%-10s %10s  %11s  %11s  %5s\0A\00", align 1
@.str.11 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.12 = private unnamed_addr constant [10 x i8] c"Seconds  \00", align 1
@.str.13 = private unnamed_addr constant [9 x i8] c"Mflops  \00", align 1
@.str.14 = private unnamed_addr constant [35 x i8] c"%-10s %10s  %5s %5s  %5s %5s  %5s\0A\00", align 1
@.str.15 = private unnamed_addr constant [5 x i8] c"Test\00", align 1
@.str.16 = private unnamed_addr constant [11 x i8] c"Iterations\00", align 1
@.str.17 = private unnamed_addr constant [4 x i8] c" C \00", align 1
@.str.18 = private unnamed_addr constant [4 x i8] c"OOP\00", align 1
@.str.19 = private unnamed_addr constant [6 x i8] c"Ratio\00", align 1
@.str.20 = private unnamed_addr constant [5 x i8] c"----\00", align 1
@.str.21 = private unnamed_addr constant [11 x i8] c"----------\00", align 1
@.str.22 = private unnamed_addr constant [12 x i8] c"-----------\00", align 1
@.str.23 = private unnamed_addr constant [6 x i8] c"-----\00", align 1
@.str.24 = private unnamed_addr constant [2 x i8] c"=\00", align 1
@.str.25 = private unnamed_addr constant [39 x i8] c"missing iteration count for test '%s'\0A\00", align 1
@.str.26 = private unnamed_addr constant [35 x i8] c"skipping non-existent test = '%s'\0A\00", align 1
@_ZTV12MaxBenchmark = dso_local unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr null, ptr @_ZTI12MaxBenchmark, ptr @_ZNK12MaxBenchmark4nameEv, ptr @_ZNK12MaxBenchmark4initEv, ptr @_ZNK12MaxBenchmark7c_styleEv, ptr @_ZNK12MaxBenchmark9oop_styleEv, ptr @_ZNK12MaxBenchmark5checkEiRdS0_] }, align 8
@_ZTI12MaxBenchmark = dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS12MaxBenchmark, ptr @_ZTI9Benchmark }, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
@_ZTS12MaxBenchmark = dso_local constant [15 x i8] c"12MaxBenchmark\00", align 1
@_ZTI9Benchmark = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS9Benchmark }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS9Benchmark = linkonce_odr dso_local constant [11 x i8] c"9Benchmark\00", comdat, align 1
@_ZTV15MatrixBenchmark = dso_local unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr null, ptr @_ZTI15MatrixBenchmark, ptr @_ZNK15MatrixBenchmark4nameEv, ptr @_ZNK15MatrixBenchmark4initEv, ptr @_ZNK15MatrixBenchmark7c_styleEv, ptr @_ZNK15MatrixBenchmark9oop_styleEv, ptr @_ZNK15MatrixBenchmark5checkEiRdS0_] }, align 8
@_ZTI15MatrixBenchmark = dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS15MatrixBenchmark, ptr @_ZTI9Benchmark }, align 8
@_ZTS15MatrixBenchmark = dso_local constant [18 x i8] c"15MatrixBenchmark\00", align 1
@_ZTV17IteratorBenchmark = dso_local unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr null, ptr @_ZTI17IteratorBenchmark, ptr @_ZNK17IteratorBenchmark4nameEv, ptr @_ZNK17IteratorBenchmark4initEv, ptr @_ZNK17IteratorBenchmark7c_styleEv, ptr @_ZNK17IteratorBenchmark9oop_styleEv, ptr @_ZNK17IteratorBenchmark5checkEiRdS0_] }, align 8
@_ZTI17IteratorBenchmark = dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS17IteratorBenchmark, ptr @_ZTI9Benchmark }, align 8
@_ZTS17IteratorBenchmark = dso_local constant [20 x i8] c"17IteratorBenchmark\00", align 1
@_ZTV16ComplexBenchmark = dso_local unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr null, ptr @_ZTI16ComplexBenchmark, ptr @_ZNK16ComplexBenchmark4nameEv, ptr @_ZNK16ComplexBenchmark4initEv, ptr @_ZNK16ComplexBenchmark7c_styleEv, ptr @_ZNK16ComplexBenchmark9oop_styleEv, ptr @_ZNK16ComplexBenchmark5checkEiRdS0_] }, align 8
@_ZTI16ComplexBenchmark = dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS16ComplexBenchmark, ptr @_ZTI9Benchmark }, align 8
@_ZTS16ComplexBenchmark = dso_local constant [19 x i8] c"16ComplexBenchmark\00", align 1
@.str.28 = private unnamed_addr constant [4 x i8] c"Max\00", align 1
@.str.29 = private unnamed_addr constant [7 x i8] c"Matrix\00", align 1
@.str.30 = private unnamed_addr constant [9 x i8] c"Iterator\00", align 1
@.str.31 = private unnamed_addr constant [8 x i8] c"Complex\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer
@str = private unnamed_addr constant [60 x i8] c"E.g.:\09a.out  Max=5000 Matrix=50 Complex=2000  Iterator=5000\00", align 4
@str.32 = private unnamed_addr constant [7 x i8] c"\0ADONE!\00", align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_ZNK12MaxBenchmark7c_styleEv(ptr nonnull readnone align 8 captures(none) %0) unnamed_addr #0 {
  %2 = load double, ptr @U, align 8, !tbaa !6
  br label %4

3:                                                ; preds = %4
  store double %10, ptr @MaxResult, align 8, !tbaa !6
  ret void

4:                                                ; preds = %1, %4
  %5 = phi i64 [ 1, %1 ], [ %11, %4 ]
  %6 = phi double [ %2, %1 ], [ %10, %4 ]
  %7 = getelementptr inbounds nuw double, ptr @U, i64 %5
  %8 = load double, ptr %7, align 8, !tbaa !6
  %9 = fcmp ogt double %8, %6
  %10 = select i1 %9, double %8, double %6
  %11 = add nuw nsw i64 %5, 1
  %12 = icmp eq i64 %11, 1000
  br i1 %12, label %3, label %4, !llvm.loop !10
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_ZNK12MaxBenchmark9oop_styleEv(ptr nonnull readnone align 8 captures(none) %0) unnamed_addr #0 {
  %2 = load double, ptr @U, align 8, !tbaa !6
  br label %4

3:                                                ; preds = %4
  store double %10, ptr @MaxResult, align 8, !tbaa !6
  ret void

4:                                                ; preds = %1, %4
  %5 = phi i64 [ 1, %1 ], [ %11, %4 ]
  %6 = phi double [ %2, %1 ], [ %10, %4 ]
  %7 = getelementptr inbounds nuw double, ptr @U, i64 %5
  %8 = load double, ptr %7, align 8, !tbaa !6
  %9 = fcmp ogt double %8, %6
  %10 = select i1 %9, double %8, double %6
  %11 = add nuw nsw i64 %5, 1
  %12 = icmp eq i64 %11, 1000
  br i1 %12, label %3, label %4, !llvm.loop !12
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_ZNK12MaxBenchmark4initEv(ptr nonnull readnone align 8 captures(none) %0) unnamed_addr #2 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi i64 [ 0, %1 ], [ %18, %2 ]
  %4 = phi <2 x i32> [ <i32 0, i32 1>, %1 ], [ %19, %2 ]
  %5 = add <2 x i32> %4, splat (i32 2)
  %6 = and <2 x i32> %4, splat (i32 1)
  %7 = and <2 x i32> %4, splat (i32 1)
  %8 = icmp eq <2 x i32> %6, zeroinitializer
  %9 = icmp eq <2 x i32> %7, zeroinitializer
  %10 = sub nsw <2 x i32> zeroinitializer, %4
  %11 = sub <2 x i32> splat (i32 -2), %4
  %12 = select <2 x i1> %8, <2 x i32> %4, <2 x i32> %10
  %13 = select <2 x i1> %9, <2 x i32> %5, <2 x i32> %11
  %14 = sitofp <2 x i32> %12 to <2 x double>
  %15 = sitofp <2 x i32> %13 to <2 x double>
  %16 = getelementptr inbounds nuw double, ptr @U, i64 %3
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 16
  store <2 x double> %14, ptr %16, align 8, !tbaa !6
  store <2 x double> %15, ptr %17, align 8, !tbaa !6
  %18 = add nuw i64 %3, 4
  %19 = add <2 x i32> %4, splat (i32 4)
  %20 = icmp eq i64 %18, 1000
  br i1 %20, label %21, label %2, !llvm.loop !13

21:                                               ; preds = %2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none) uwtable
define dso_local void @_ZNK12MaxBenchmark5checkEiRdS0_(ptr nonnull readnone align 8 captures(none) %0, i32 noundef %1, ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(8) initializes((0, 8)) %2, ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(8) initializes((0, 8)) %3) unnamed_addr #3 {
  %5 = sitofp i32 %1 to double
  %6 = fmul double %5, 1.000000e+03
  store double %6, ptr %2, align 8, !tbaa !6
  %7 = load double, ptr @MaxResult, align 8, !tbaa !6
  store double %7, ptr %3, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_ZNK15MatrixBenchmark7c_styleEv(ptr nonnull readnone align 8 captures(none) %0) unnamed_addr #0 {
  br label %2

2:                                                ; preds = %1, %262
  %3 = phi i64 [ 0, %1 ], [ %263, %262 ]
  %4 = mul nuw nsw i64 %3, 50
  %5 = getelementptr inbounds nuw double, ptr @C, i64 %4
  %6 = getelementptr inbounds nuw double, ptr @E, i64 %4
  %7 = load double, ptr %5, align 8, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %9 = load double, ptr %8, align 8, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %11 = load double, ptr %10, align 8, !tbaa !6
  %12 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %13 = load double, ptr %12, align 8, !tbaa !6
  %14 = getelementptr inbounds nuw i8, ptr %5, i64 32
  %15 = load double, ptr %14, align 8, !tbaa !6
  %16 = getelementptr inbounds nuw i8, ptr %5, i64 40
  %17 = load double, ptr %16, align 8, !tbaa !6
  %18 = getelementptr inbounds nuw i8, ptr %5, i64 48
  %19 = load double, ptr %18, align 8, !tbaa !6
  %20 = getelementptr inbounds nuw i8, ptr %5, i64 56
  %21 = load double, ptr %20, align 8, !tbaa !6
  %22 = getelementptr inbounds nuw i8, ptr %5, i64 64
  %23 = load double, ptr %22, align 8, !tbaa !6
  %24 = getelementptr inbounds nuw i8, ptr %5, i64 72
  %25 = load double, ptr %24, align 8, !tbaa !6
  %26 = getelementptr inbounds nuw i8, ptr %5, i64 80
  %27 = load double, ptr %26, align 8, !tbaa !6
  %28 = getelementptr inbounds nuw i8, ptr %5, i64 88
  %29 = load double, ptr %28, align 8, !tbaa !6
  %30 = getelementptr inbounds nuw i8, ptr %5, i64 96
  %31 = load double, ptr %30, align 8, !tbaa !6
  %32 = getelementptr inbounds nuw i8, ptr %5, i64 104
  %33 = load double, ptr %32, align 8, !tbaa !6
  %34 = getelementptr inbounds nuw i8, ptr %5, i64 112
  %35 = load double, ptr %34, align 8, !tbaa !6
  %36 = getelementptr inbounds nuw i8, ptr %5, i64 120
  %37 = load double, ptr %36, align 8, !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %5, i64 128
  %39 = load double, ptr %38, align 8, !tbaa !6
  %40 = getelementptr inbounds nuw i8, ptr %5, i64 136
  %41 = load double, ptr %40, align 8, !tbaa !6
  %42 = getelementptr inbounds nuw i8, ptr %5, i64 144
  %43 = load double, ptr %42, align 8, !tbaa !6
  %44 = getelementptr inbounds nuw i8, ptr %5, i64 152
  %45 = load double, ptr %44, align 8, !tbaa !6
  %46 = getelementptr inbounds nuw i8, ptr %5, i64 160
  %47 = load double, ptr %46, align 8, !tbaa !6
  %48 = getelementptr inbounds nuw i8, ptr %5, i64 168
  %49 = load double, ptr %48, align 8, !tbaa !6
  %50 = getelementptr inbounds nuw i8, ptr %5, i64 176
  %51 = load double, ptr %50, align 8, !tbaa !6
  %52 = getelementptr inbounds nuw i8, ptr %5, i64 184
  %53 = load double, ptr %52, align 8, !tbaa !6
  %54 = getelementptr inbounds nuw i8, ptr %5, i64 192
  %55 = load double, ptr %54, align 8, !tbaa !6
  %56 = getelementptr inbounds nuw i8, ptr %5, i64 200
  %57 = load double, ptr %56, align 8, !tbaa !6
  %58 = getelementptr inbounds nuw i8, ptr %5, i64 208
  %59 = load double, ptr %58, align 8, !tbaa !6
  %60 = getelementptr inbounds nuw i8, ptr %5, i64 216
  %61 = load double, ptr %60, align 8, !tbaa !6
  %62 = getelementptr inbounds nuw i8, ptr %5, i64 224
  %63 = load double, ptr %62, align 8, !tbaa !6
  %64 = getelementptr inbounds nuw i8, ptr %5, i64 232
  %65 = load double, ptr %64, align 8, !tbaa !6
  %66 = getelementptr inbounds nuw i8, ptr %5, i64 240
  %67 = load double, ptr %66, align 8, !tbaa !6
  %68 = getelementptr inbounds nuw i8, ptr %5, i64 248
  %69 = load double, ptr %68, align 8, !tbaa !6
  %70 = getelementptr inbounds nuw i8, ptr %5, i64 256
  %71 = load double, ptr %70, align 8, !tbaa !6
  %72 = getelementptr inbounds nuw i8, ptr %5, i64 264
  %73 = load double, ptr %72, align 8, !tbaa !6
  %74 = getelementptr inbounds nuw i8, ptr %5, i64 272
  %75 = load double, ptr %74, align 8, !tbaa !6
  %76 = getelementptr inbounds nuw i8, ptr %5, i64 280
  %77 = load double, ptr %76, align 8, !tbaa !6
  %78 = getelementptr inbounds nuw i8, ptr %5, i64 288
  %79 = load double, ptr %78, align 8, !tbaa !6
  %80 = getelementptr inbounds nuw i8, ptr %5, i64 296
  %81 = load double, ptr %80, align 8, !tbaa !6
  %82 = getelementptr inbounds nuw i8, ptr %5, i64 304
  %83 = load double, ptr %82, align 8, !tbaa !6
  %84 = getelementptr inbounds nuw i8, ptr %5, i64 312
  %85 = load double, ptr %84, align 8, !tbaa !6
  %86 = getelementptr inbounds nuw i8, ptr %5, i64 320
  %87 = load double, ptr %86, align 8, !tbaa !6
  %88 = getelementptr inbounds nuw i8, ptr %5, i64 328
  %89 = load double, ptr %88, align 8, !tbaa !6
  %90 = getelementptr inbounds nuw i8, ptr %5, i64 336
  %91 = load double, ptr %90, align 8, !tbaa !6
  %92 = getelementptr inbounds nuw i8, ptr %5, i64 344
  %93 = load double, ptr %92, align 8, !tbaa !6
  %94 = getelementptr inbounds nuw i8, ptr %5, i64 352
  %95 = load double, ptr %94, align 8, !tbaa !6
  %96 = getelementptr inbounds nuw i8, ptr %5, i64 360
  %97 = load double, ptr %96, align 8, !tbaa !6
  %98 = getelementptr inbounds nuw i8, ptr %5, i64 368
  %99 = load double, ptr %98, align 8, !tbaa !6
  %100 = getelementptr inbounds nuw i8, ptr %5, i64 376
  %101 = load double, ptr %100, align 8, !tbaa !6
  %102 = getelementptr inbounds nuw i8, ptr %5, i64 384
  %103 = load double, ptr %102, align 8, !tbaa !6
  %104 = getelementptr inbounds nuw i8, ptr %5, i64 392
  %105 = load double, ptr %104, align 8, !tbaa !6
  br label %107

106:                                              ; preds = %262
  ret void

107:                                              ; preds = %2, %107
  %108 = phi i64 [ 0, %2 ], [ %260, %107 ]
  %109 = getelementptr inbounds nuw double, ptr @D, i64 %108
  %110 = load double, ptr %109, align 8, !tbaa !6
  %111 = tail call double @llvm.fmuladd.f64(double %7, double %110, double 0.000000e+00)
  %112 = getelementptr inbounds nuw i8, ptr %109, i64 400
  %113 = load double, ptr %112, align 8, !tbaa !6
  %114 = tail call double @llvm.fmuladd.f64(double %9, double %113, double %111)
  %115 = getelementptr inbounds nuw i8, ptr %109, i64 800
  %116 = load double, ptr %115, align 8, !tbaa !6
  %117 = tail call double @llvm.fmuladd.f64(double %11, double %116, double %114)
  %118 = getelementptr inbounds nuw i8, ptr %109, i64 1200
  %119 = load double, ptr %118, align 8, !tbaa !6
  %120 = tail call double @llvm.fmuladd.f64(double %13, double %119, double %117)
  %121 = getelementptr inbounds nuw i8, ptr %109, i64 1600
  %122 = load double, ptr %121, align 8, !tbaa !6
  %123 = tail call double @llvm.fmuladd.f64(double %15, double %122, double %120)
  %124 = getelementptr inbounds nuw i8, ptr %109, i64 2000
  %125 = load double, ptr %124, align 8, !tbaa !6
  %126 = tail call double @llvm.fmuladd.f64(double %17, double %125, double %123)
  %127 = getelementptr inbounds nuw i8, ptr %109, i64 2400
  %128 = load double, ptr %127, align 8, !tbaa !6
  %129 = tail call double @llvm.fmuladd.f64(double %19, double %128, double %126)
  %130 = getelementptr inbounds nuw i8, ptr %109, i64 2800
  %131 = load double, ptr %130, align 8, !tbaa !6
  %132 = tail call double @llvm.fmuladd.f64(double %21, double %131, double %129)
  %133 = getelementptr inbounds nuw i8, ptr %109, i64 3200
  %134 = load double, ptr %133, align 8, !tbaa !6
  %135 = tail call double @llvm.fmuladd.f64(double %23, double %134, double %132)
  %136 = getelementptr inbounds nuw i8, ptr %109, i64 3600
  %137 = load double, ptr %136, align 8, !tbaa !6
  %138 = tail call double @llvm.fmuladd.f64(double %25, double %137, double %135)
  %139 = getelementptr inbounds nuw i8, ptr %109, i64 4000
  %140 = load double, ptr %139, align 8, !tbaa !6
  %141 = tail call double @llvm.fmuladd.f64(double %27, double %140, double %138)
  %142 = getelementptr inbounds nuw i8, ptr %109, i64 4400
  %143 = load double, ptr %142, align 8, !tbaa !6
  %144 = tail call double @llvm.fmuladd.f64(double %29, double %143, double %141)
  %145 = getelementptr inbounds nuw i8, ptr %109, i64 4800
  %146 = load double, ptr %145, align 8, !tbaa !6
  %147 = tail call double @llvm.fmuladd.f64(double %31, double %146, double %144)
  %148 = getelementptr inbounds nuw i8, ptr %109, i64 5200
  %149 = load double, ptr %148, align 8, !tbaa !6
  %150 = tail call double @llvm.fmuladd.f64(double %33, double %149, double %147)
  %151 = getelementptr inbounds nuw i8, ptr %109, i64 5600
  %152 = load double, ptr %151, align 8, !tbaa !6
  %153 = tail call double @llvm.fmuladd.f64(double %35, double %152, double %150)
  %154 = getelementptr inbounds nuw i8, ptr %109, i64 6000
  %155 = load double, ptr %154, align 8, !tbaa !6
  %156 = tail call double @llvm.fmuladd.f64(double %37, double %155, double %153)
  %157 = getelementptr inbounds nuw i8, ptr %109, i64 6400
  %158 = load double, ptr %157, align 8, !tbaa !6
  %159 = tail call double @llvm.fmuladd.f64(double %39, double %158, double %156)
  %160 = getelementptr inbounds nuw i8, ptr %109, i64 6800
  %161 = load double, ptr %160, align 8, !tbaa !6
  %162 = tail call double @llvm.fmuladd.f64(double %41, double %161, double %159)
  %163 = getelementptr inbounds nuw i8, ptr %109, i64 7200
  %164 = load double, ptr %163, align 8, !tbaa !6
  %165 = tail call double @llvm.fmuladd.f64(double %43, double %164, double %162)
  %166 = getelementptr inbounds nuw i8, ptr %109, i64 7600
  %167 = load double, ptr %166, align 8, !tbaa !6
  %168 = tail call double @llvm.fmuladd.f64(double %45, double %167, double %165)
  %169 = getelementptr inbounds nuw i8, ptr %109, i64 8000
  %170 = load double, ptr %169, align 8, !tbaa !6
  %171 = tail call double @llvm.fmuladd.f64(double %47, double %170, double %168)
  %172 = getelementptr inbounds nuw i8, ptr %109, i64 8400
  %173 = load double, ptr %172, align 8, !tbaa !6
  %174 = tail call double @llvm.fmuladd.f64(double %49, double %173, double %171)
  %175 = getelementptr inbounds nuw i8, ptr %109, i64 8800
  %176 = load double, ptr %175, align 8, !tbaa !6
  %177 = tail call double @llvm.fmuladd.f64(double %51, double %176, double %174)
  %178 = getelementptr inbounds nuw i8, ptr %109, i64 9200
  %179 = load double, ptr %178, align 8, !tbaa !6
  %180 = tail call double @llvm.fmuladd.f64(double %53, double %179, double %177)
  %181 = getelementptr inbounds nuw i8, ptr %109, i64 9600
  %182 = load double, ptr %181, align 8, !tbaa !6
  %183 = tail call double @llvm.fmuladd.f64(double %55, double %182, double %180)
  %184 = getelementptr inbounds nuw i8, ptr %109, i64 10000
  %185 = load double, ptr %184, align 8, !tbaa !6
  %186 = tail call double @llvm.fmuladd.f64(double %57, double %185, double %183)
  %187 = getelementptr inbounds nuw i8, ptr %109, i64 10400
  %188 = load double, ptr %187, align 8, !tbaa !6
  %189 = tail call double @llvm.fmuladd.f64(double %59, double %188, double %186)
  %190 = getelementptr inbounds nuw i8, ptr %109, i64 10800
  %191 = load double, ptr %190, align 8, !tbaa !6
  %192 = tail call double @llvm.fmuladd.f64(double %61, double %191, double %189)
  %193 = getelementptr inbounds nuw i8, ptr %109, i64 11200
  %194 = load double, ptr %193, align 8, !tbaa !6
  %195 = tail call double @llvm.fmuladd.f64(double %63, double %194, double %192)
  %196 = getelementptr inbounds nuw i8, ptr %109, i64 11600
  %197 = load double, ptr %196, align 8, !tbaa !6
  %198 = tail call double @llvm.fmuladd.f64(double %65, double %197, double %195)
  %199 = getelementptr inbounds nuw i8, ptr %109, i64 12000
  %200 = load double, ptr %199, align 8, !tbaa !6
  %201 = tail call double @llvm.fmuladd.f64(double %67, double %200, double %198)
  %202 = getelementptr inbounds nuw i8, ptr %109, i64 12400
  %203 = load double, ptr %202, align 8, !tbaa !6
  %204 = tail call double @llvm.fmuladd.f64(double %69, double %203, double %201)
  %205 = getelementptr inbounds nuw i8, ptr %109, i64 12800
  %206 = load double, ptr %205, align 8, !tbaa !6
  %207 = tail call double @llvm.fmuladd.f64(double %71, double %206, double %204)
  %208 = getelementptr inbounds nuw i8, ptr %109, i64 13200
  %209 = load double, ptr %208, align 8, !tbaa !6
  %210 = tail call double @llvm.fmuladd.f64(double %73, double %209, double %207)
  %211 = getelementptr inbounds nuw i8, ptr %109, i64 13600
  %212 = load double, ptr %211, align 8, !tbaa !6
  %213 = tail call double @llvm.fmuladd.f64(double %75, double %212, double %210)
  %214 = getelementptr inbounds nuw i8, ptr %109, i64 14000
  %215 = load double, ptr %214, align 8, !tbaa !6
  %216 = tail call double @llvm.fmuladd.f64(double %77, double %215, double %213)
  %217 = getelementptr inbounds nuw i8, ptr %109, i64 14400
  %218 = load double, ptr %217, align 8, !tbaa !6
  %219 = tail call double @llvm.fmuladd.f64(double %79, double %218, double %216)
  %220 = getelementptr inbounds nuw i8, ptr %109, i64 14800
  %221 = load double, ptr %220, align 8, !tbaa !6
  %222 = tail call double @llvm.fmuladd.f64(double %81, double %221, double %219)
  %223 = getelementptr inbounds nuw i8, ptr %109, i64 15200
  %224 = load double, ptr %223, align 8, !tbaa !6
  %225 = tail call double @llvm.fmuladd.f64(double %83, double %224, double %222)
  %226 = getelementptr inbounds nuw i8, ptr %109, i64 15600
  %227 = load double, ptr %226, align 8, !tbaa !6
  %228 = tail call double @llvm.fmuladd.f64(double %85, double %227, double %225)
  %229 = getelementptr inbounds nuw i8, ptr %109, i64 16000
  %230 = load double, ptr %229, align 8, !tbaa !6
  %231 = tail call double @llvm.fmuladd.f64(double %87, double %230, double %228)
  %232 = getelementptr inbounds nuw i8, ptr %109, i64 16400
  %233 = load double, ptr %232, align 8, !tbaa !6
  %234 = tail call double @llvm.fmuladd.f64(double %89, double %233, double %231)
  %235 = getelementptr inbounds nuw i8, ptr %109, i64 16800
  %236 = load double, ptr %235, align 8, !tbaa !6
  %237 = tail call double @llvm.fmuladd.f64(double %91, double %236, double %234)
  %238 = getelementptr inbounds nuw i8, ptr %109, i64 17200
  %239 = load double, ptr %238, align 8, !tbaa !6
  %240 = tail call double @llvm.fmuladd.f64(double %93, double %239, double %237)
  %241 = getelementptr inbounds nuw i8, ptr %109, i64 17600
  %242 = load double, ptr %241, align 8, !tbaa !6
  %243 = tail call double @llvm.fmuladd.f64(double %95, double %242, double %240)
  %244 = getelementptr inbounds nuw i8, ptr %109, i64 18000
  %245 = load double, ptr %244, align 8, !tbaa !6
  %246 = tail call double @llvm.fmuladd.f64(double %97, double %245, double %243)
  %247 = getelementptr inbounds nuw i8, ptr %109, i64 18400
  %248 = load double, ptr %247, align 8, !tbaa !6
  %249 = tail call double @llvm.fmuladd.f64(double %99, double %248, double %246)
  %250 = getelementptr inbounds nuw i8, ptr %109, i64 18800
  %251 = load double, ptr %250, align 8, !tbaa !6
  %252 = tail call double @llvm.fmuladd.f64(double %101, double %251, double %249)
  %253 = getelementptr inbounds nuw i8, ptr %109, i64 19200
  %254 = load double, ptr %253, align 8, !tbaa !6
  %255 = tail call double @llvm.fmuladd.f64(double %103, double %254, double %252)
  %256 = getelementptr inbounds nuw i8, ptr %109, i64 19600
  %257 = load double, ptr %256, align 8, !tbaa !6
  %258 = tail call double @llvm.fmuladd.f64(double %105, double %257, double %255)
  %259 = getelementptr inbounds nuw double, ptr %6, i64 %108
  store double %258, ptr %259, align 8, !tbaa !6
  %260 = add nuw nsw i64 %108, 1
  %261 = icmp eq i64 %260, 50
  br i1 %261, label %262, label %107, !llvm.loop !16

262:                                              ; preds = %107
  %263 = add nuw nsw i64 %3, 1
  %264 = icmp eq i64 %263, 50
  br i1 %264, label %106, label %2, !llvm.loop !17
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #4

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_ZNK15MatrixBenchmark9oop_styleEv(ptr nonnull readnone align 8 captures(none) %0) unnamed_addr #0 {
  br label %2

2:                                                ; preds = %1, %262
  %3 = phi i64 [ 0, %1 ], [ %263, %262 ]
  %4 = mul nuw nsw i64 %3, 50
  %5 = getelementptr inbounds nuw double, ptr @C, i64 %4
  %6 = getelementptr inbounds nuw double, ptr @E, i64 %4
  %7 = load double, ptr %5, align 8, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %9 = load double, ptr %8, align 8, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %11 = load double, ptr %10, align 8, !tbaa !6
  %12 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %13 = load double, ptr %12, align 8, !tbaa !6
  %14 = getelementptr inbounds nuw i8, ptr %5, i64 32
  %15 = load double, ptr %14, align 8, !tbaa !6
  %16 = getelementptr inbounds nuw i8, ptr %5, i64 40
  %17 = load double, ptr %16, align 8, !tbaa !6
  %18 = getelementptr inbounds nuw i8, ptr %5, i64 48
  %19 = load double, ptr %18, align 8, !tbaa !6
  %20 = getelementptr inbounds nuw i8, ptr %5, i64 56
  %21 = load double, ptr %20, align 8, !tbaa !6
  %22 = getelementptr inbounds nuw i8, ptr %5, i64 64
  %23 = load double, ptr %22, align 8, !tbaa !6
  %24 = getelementptr inbounds nuw i8, ptr %5, i64 72
  %25 = load double, ptr %24, align 8, !tbaa !6
  %26 = getelementptr inbounds nuw i8, ptr %5, i64 80
  %27 = load double, ptr %26, align 8, !tbaa !6
  %28 = getelementptr inbounds nuw i8, ptr %5, i64 88
  %29 = load double, ptr %28, align 8, !tbaa !6
  %30 = getelementptr inbounds nuw i8, ptr %5, i64 96
  %31 = load double, ptr %30, align 8, !tbaa !6
  %32 = getelementptr inbounds nuw i8, ptr %5, i64 104
  %33 = load double, ptr %32, align 8, !tbaa !6
  %34 = getelementptr inbounds nuw i8, ptr %5, i64 112
  %35 = load double, ptr %34, align 8, !tbaa !6
  %36 = getelementptr inbounds nuw i8, ptr %5, i64 120
  %37 = load double, ptr %36, align 8, !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %5, i64 128
  %39 = load double, ptr %38, align 8, !tbaa !6
  %40 = getelementptr inbounds nuw i8, ptr %5, i64 136
  %41 = load double, ptr %40, align 8, !tbaa !6
  %42 = getelementptr inbounds nuw i8, ptr %5, i64 144
  %43 = load double, ptr %42, align 8, !tbaa !6
  %44 = getelementptr inbounds nuw i8, ptr %5, i64 152
  %45 = load double, ptr %44, align 8, !tbaa !6
  %46 = getelementptr inbounds nuw i8, ptr %5, i64 160
  %47 = load double, ptr %46, align 8, !tbaa !6
  %48 = getelementptr inbounds nuw i8, ptr %5, i64 168
  %49 = load double, ptr %48, align 8, !tbaa !6
  %50 = getelementptr inbounds nuw i8, ptr %5, i64 176
  %51 = load double, ptr %50, align 8, !tbaa !6
  %52 = getelementptr inbounds nuw i8, ptr %5, i64 184
  %53 = load double, ptr %52, align 8, !tbaa !6
  %54 = getelementptr inbounds nuw i8, ptr %5, i64 192
  %55 = load double, ptr %54, align 8, !tbaa !6
  %56 = getelementptr inbounds nuw i8, ptr %5, i64 200
  %57 = load double, ptr %56, align 8, !tbaa !6
  %58 = getelementptr inbounds nuw i8, ptr %5, i64 208
  %59 = load double, ptr %58, align 8, !tbaa !6
  %60 = getelementptr inbounds nuw i8, ptr %5, i64 216
  %61 = load double, ptr %60, align 8, !tbaa !6
  %62 = getelementptr inbounds nuw i8, ptr %5, i64 224
  %63 = load double, ptr %62, align 8, !tbaa !6
  %64 = getelementptr inbounds nuw i8, ptr %5, i64 232
  %65 = load double, ptr %64, align 8, !tbaa !6
  %66 = getelementptr inbounds nuw i8, ptr %5, i64 240
  %67 = load double, ptr %66, align 8, !tbaa !6
  %68 = getelementptr inbounds nuw i8, ptr %5, i64 248
  %69 = load double, ptr %68, align 8, !tbaa !6
  %70 = getelementptr inbounds nuw i8, ptr %5, i64 256
  %71 = load double, ptr %70, align 8, !tbaa !6
  %72 = getelementptr inbounds nuw i8, ptr %5, i64 264
  %73 = load double, ptr %72, align 8, !tbaa !6
  %74 = getelementptr inbounds nuw i8, ptr %5, i64 272
  %75 = load double, ptr %74, align 8, !tbaa !6
  %76 = getelementptr inbounds nuw i8, ptr %5, i64 280
  %77 = load double, ptr %76, align 8, !tbaa !6
  %78 = getelementptr inbounds nuw i8, ptr %5, i64 288
  %79 = load double, ptr %78, align 8, !tbaa !6
  %80 = getelementptr inbounds nuw i8, ptr %5, i64 296
  %81 = load double, ptr %80, align 8, !tbaa !6
  %82 = getelementptr inbounds nuw i8, ptr %5, i64 304
  %83 = load double, ptr %82, align 8, !tbaa !6
  %84 = getelementptr inbounds nuw i8, ptr %5, i64 312
  %85 = load double, ptr %84, align 8, !tbaa !6
  %86 = getelementptr inbounds nuw i8, ptr %5, i64 320
  %87 = load double, ptr %86, align 8, !tbaa !6
  %88 = getelementptr inbounds nuw i8, ptr %5, i64 328
  %89 = load double, ptr %88, align 8, !tbaa !6
  %90 = getelementptr inbounds nuw i8, ptr %5, i64 336
  %91 = load double, ptr %90, align 8, !tbaa !6
  %92 = getelementptr inbounds nuw i8, ptr %5, i64 344
  %93 = load double, ptr %92, align 8, !tbaa !6
  %94 = getelementptr inbounds nuw i8, ptr %5, i64 352
  %95 = load double, ptr %94, align 8, !tbaa !6
  %96 = getelementptr inbounds nuw i8, ptr %5, i64 360
  %97 = load double, ptr %96, align 8, !tbaa !6
  %98 = getelementptr inbounds nuw i8, ptr %5, i64 368
  %99 = load double, ptr %98, align 8, !tbaa !6
  %100 = getelementptr inbounds nuw i8, ptr %5, i64 376
  %101 = load double, ptr %100, align 8, !tbaa !6
  %102 = getelementptr inbounds nuw i8, ptr %5, i64 384
  %103 = load double, ptr %102, align 8, !tbaa !6
  %104 = getelementptr inbounds nuw i8, ptr %5, i64 392
  %105 = load double, ptr %104, align 8, !tbaa !6
  br label %107

106:                                              ; preds = %262
  ret void

107:                                              ; preds = %2, %107
  %108 = phi i64 [ 0, %2 ], [ %260, %107 ]
  %109 = getelementptr inbounds nuw double, ptr @D, i64 %108
  %110 = load double, ptr %109, align 8, !tbaa !6
  %111 = tail call double @llvm.fmuladd.f64(double %7, double %110, double 0.000000e+00)
  %112 = getelementptr inbounds nuw i8, ptr %109, i64 400
  %113 = load double, ptr %112, align 8, !tbaa !6
  %114 = tail call double @llvm.fmuladd.f64(double %9, double %113, double %111)
  %115 = getelementptr inbounds nuw i8, ptr %109, i64 800
  %116 = load double, ptr %115, align 8, !tbaa !6
  %117 = tail call double @llvm.fmuladd.f64(double %11, double %116, double %114)
  %118 = getelementptr inbounds nuw i8, ptr %109, i64 1200
  %119 = load double, ptr %118, align 8, !tbaa !6
  %120 = tail call double @llvm.fmuladd.f64(double %13, double %119, double %117)
  %121 = getelementptr inbounds nuw i8, ptr %109, i64 1600
  %122 = load double, ptr %121, align 8, !tbaa !6
  %123 = tail call double @llvm.fmuladd.f64(double %15, double %122, double %120)
  %124 = getelementptr inbounds nuw i8, ptr %109, i64 2000
  %125 = load double, ptr %124, align 8, !tbaa !6
  %126 = tail call double @llvm.fmuladd.f64(double %17, double %125, double %123)
  %127 = getelementptr inbounds nuw i8, ptr %109, i64 2400
  %128 = load double, ptr %127, align 8, !tbaa !6
  %129 = tail call double @llvm.fmuladd.f64(double %19, double %128, double %126)
  %130 = getelementptr inbounds nuw i8, ptr %109, i64 2800
  %131 = load double, ptr %130, align 8, !tbaa !6
  %132 = tail call double @llvm.fmuladd.f64(double %21, double %131, double %129)
  %133 = getelementptr inbounds nuw i8, ptr %109, i64 3200
  %134 = load double, ptr %133, align 8, !tbaa !6
  %135 = tail call double @llvm.fmuladd.f64(double %23, double %134, double %132)
  %136 = getelementptr inbounds nuw i8, ptr %109, i64 3600
  %137 = load double, ptr %136, align 8, !tbaa !6
  %138 = tail call double @llvm.fmuladd.f64(double %25, double %137, double %135)
  %139 = getelementptr inbounds nuw i8, ptr %109, i64 4000
  %140 = load double, ptr %139, align 8, !tbaa !6
  %141 = tail call double @llvm.fmuladd.f64(double %27, double %140, double %138)
  %142 = getelementptr inbounds nuw i8, ptr %109, i64 4400
  %143 = load double, ptr %142, align 8, !tbaa !6
  %144 = tail call double @llvm.fmuladd.f64(double %29, double %143, double %141)
  %145 = getelementptr inbounds nuw i8, ptr %109, i64 4800
  %146 = load double, ptr %145, align 8, !tbaa !6
  %147 = tail call double @llvm.fmuladd.f64(double %31, double %146, double %144)
  %148 = getelementptr inbounds nuw i8, ptr %109, i64 5200
  %149 = load double, ptr %148, align 8, !tbaa !6
  %150 = tail call double @llvm.fmuladd.f64(double %33, double %149, double %147)
  %151 = getelementptr inbounds nuw i8, ptr %109, i64 5600
  %152 = load double, ptr %151, align 8, !tbaa !6
  %153 = tail call double @llvm.fmuladd.f64(double %35, double %152, double %150)
  %154 = getelementptr inbounds nuw i8, ptr %109, i64 6000
  %155 = load double, ptr %154, align 8, !tbaa !6
  %156 = tail call double @llvm.fmuladd.f64(double %37, double %155, double %153)
  %157 = getelementptr inbounds nuw i8, ptr %109, i64 6400
  %158 = load double, ptr %157, align 8, !tbaa !6
  %159 = tail call double @llvm.fmuladd.f64(double %39, double %158, double %156)
  %160 = getelementptr inbounds nuw i8, ptr %109, i64 6800
  %161 = load double, ptr %160, align 8, !tbaa !6
  %162 = tail call double @llvm.fmuladd.f64(double %41, double %161, double %159)
  %163 = getelementptr inbounds nuw i8, ptr %109, i64 7200
  %164 = load double, ptr %163, align 8, !tbaa !6
  %165 = tail call double @llvm.fmuladd.f64(double %43, double %164, double %162)
  %166 = getelementptr inbounds nuw i8, ptr %109, i64 7600
  %167 = load double, ptr %166, align 8, !tbaa !6
  %168 = tail call double @llvm.fmuladd.f64(double %45, double %167, double %165)
  %169 = getelementptr inbounds nuw i8, ptr %109, i64 8000
  %170 = load double, ptr %169, align 8, !tbaa !6
  %171 = tail call double @llvm.fmuladd.f64(double %47, double %170, double %168)
  %172 = getelementptr inbounds nuw i8, ptr %109, i64 8400
  %173 = load double, ptr %172, align 8, !tbaa !6
  %174 = tail call double @llvm.fmuladd.f64(double %49, double %173, double %171)
  %175 = getelementptr inbounds nuw i8, ptr %109, i64 8800
  %176 = load double, ptr %175, align 8, !tbaa !6
  %177 = tail call double @llvm.fmuladd.f64(double %51, double %176, double %174)
  %178 = getelementptr inbounds nuw i8, ptr %109, i64 9200
  %179 = load double, ptr %178, align 8, !tbaa !6
  %180 = tail call double @llvm.fmuladd.f64(double %53, double %179, double %177)
  %181 = getelementptr inbounds nuw i8, ptr %109, i64 9600
  %182 = load double, ptr %181, align 8, !tbaa !6
  %183 = tail call double @llvm.fmuladd.f64(double %55, double %182, double %180)
  %184 = getelementptr inbounds nuw i8, ptr %109, i64 10000
  %185 = load double, ptr %184, align 8, !tbaa !6
  %186 = tail call double @llvm.fmuladd.f64(double %57, double %185, double %183)
  %187 = getelementptr inbounds nuw i8, ptr %109, i64 10400
  %188 = load double, ptr %187, align 8, !tbaa !6
  %189 = tail call double @llvm.fmuladd.f64(double %59, double %188, double %186)
  %190 = getelementptr inbounds nuw i8, ptr %109, i64 10800
  %191 = load double, ptr %190, align 8, !tbaa !6
  %192 = tail call double @llvm.fmuladd.f64(double %61, double %191, double %189)
  %193 = getelementptr inbounds nuw i8, ptr %109, i64 11200
  %194 = load double, ptr %193, align 8, !tbaa !6
  %195 = tail call double @llvm.fmuladd.f64(double %63, double %194, double %192)
  %196 = getelementptr inbounds nuw i8, ptr %109, i64 11600
  %197 = load double, ptr %196, align 8, !tbaa !6
  %198 = tail call double @llvm.fmuladd.f64(double %65, double %197, double %195)
  %199 = getelementptr inbounds nuw i8, ptr %109, i64 12000
  %200 = load double, ptr %199, align 8, !tbaa !6
  %201 = tail call double @llvm.fmuladd.f64(double %67, double %200, double %198)
  %202 = getelementptr inbounds nuw i8, ptr %109, i64 12400
  %203 = load double, ptr %202, align 8, !tbaa !6
  %204 = tail call double @llvm.fmuladd.f64(double %69, double %203, double %201)
  %205 = getelementptr inbounds nuw i8, ptr %109, i64 12800
  %206 = load double, ptr %205, align 8, !tbaa !6
  %207 = tail call double @llvm.fmuladd.f64(double %71, double %206, double %204)
  %208 = getelementptr inbounds nuw i8, ptr %109, i64 13200
  %209 = load double, ptr %208, align 8, !tbaa !6
  %210 = tail call double @llvm.fmuladd.f64(double %73, double %209, double %207)
  %211 = getelementptr inbounds nuw i8, ptr %109, i64 13600
  %212 = load double, ptr %211, align 8, !tbaa !6
  %213 = tail call double @llvm.fmuladd.f64(double %75, double %212, double %210)
  %214 = getelementptr inbounds nuw i8, ptr %109, i64 14000
  %215 = load double, ptr %214, align 8, !tbaa !6
  %216 = tail call double @llvm.fmuladd.f64(double %77, double %215, double %213)
  %217 = getelementptr inbounds nuw i8, ptr %109, i64 14400
  %218 = load double, ptr %217, align 8, !tbaa !6
  %219 = tail call double @llvm.fmuladd.f64(double %79, double %218, double %216)
  %220 = getelementptr inbounds nuw i8, ptr %109, i64 14800
  %221 = load double, ptr %220, align 8, !tbaa !6
  %222 = tail call double @llvm.fmuladd.f64(double %81, double %221, double %219)
  %223 = getelementptr inbounds nuw i8, ptr %109, i64 15200
  %224 = load double, ptr %223, align 8, !tbaa !6
  %225 = tail call double @llvm.fmuladd.f64(double %83, double %224, double %222)
  %226 = getelementptr inbounds nuw i8, ptr %109, i64 15600
  %227 = load double, ptr %226, align 8, !tbaa !6
  %228 = tail call double @llvm.fmuladd.f64(double %85, double %227, double %225)
  %229 = getelementptr inbounds nuw i8, ptr %109, i64 16000
  %230 = load double, ptr %229, align 8, !tbaa !6
  %231 = tail call double @llvm.fmuladd.f64(double %87, double %230, double %228)
  %232 = getelementptr inbounds nuw i8, ptr %109, i64 16400
  %233 = load double, ptr %232, align 8, !tbaa !6
  %234 = tail call double @llvm.fmuladd.f64(double %89, double %233, double %231)
  %235 = getelementptr inbounds nuw i8, ptr %109, i64 16800
  %236 = load double, ptr %235, align 8, !tbaa !6
  %237 = tail call double @llvm.fmuladd.f64(double %91, double %236, double %234)
  %238 = getelementptr inbounds nuw i8, ptr %109, i64 17200
  %239 = load double, ptr %238, align 8, !tbaa !6
  %240 = tail call double @llvm.fmuladd.f64(double %93, double %239, double %237)
  %241 = getelementptr inbounds nuw i8, ptr %109, i64 17600
  %242 = load double, ptr %241, align 8, !tbaa !6
  %243 = tail call double @llvm.fmuladd.f64(double %95, double %242, double %240)
  %244 = getelementptr inbounds nuw i8, ptr %109, i64 18000
  %245 = load double, ptr %244, align 8, !tbaa !6
  %246 = tail call double @llvm.fmuladd.f64(double %97, double %245, double %243)
  %247 = getelementptr inbounds nuw i8, ptr %109, i64 18400
  %248 = load double, ptr %247, align 8, !tbaa !6
  %249 = tail call double @llvm.fmuladd.f64(double %99, double %248, double %246)
  %250 = getelementptr inbounds nuw i8, ptr %109, i64 18800
  %251 = load double, ptr %250, align 8, !tbaa !6
  %252 = tail call double @llvm.fmuladd.f64(double %101, double %251, double %249)
  %253 = getelementptr inbounds nuw i8, ptr %109, i64 19200
  %254 = load double, ptr %253, align 8, !tbaa !6
  %255 = tail call double @llvm.fmuladd.f64(double %103, double %254, double %252)
  %256 = getelementptr inbounds nuw i8, ptr %109, i64 19600
  %257 = load double, ptr %256, align 8, !tbaa !6
  %258 = tail call double @llvm.fmuladd.f64(double %105, double %257, double %255)
  %259 = getelementptr inbounds nuw double, ptr %6, i64 %108
  store double %258, ptr %259, align 8, !tbaa !6
  %260 = add nuw nsw i64 %108, 1
  %261 = icmp eq i64 %260, 50
  br i1 %261, label %262, label %107, !llvm.loop !18

262:                                              ; preds = %107
  %263 = add nuw nsw i64 %3, 1
  %264 = icmp eq i64 %263, 50
  br i1 %264, label %106, label %2, !llvm.loop !19
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_ZNK15MatrixBenchmark4initEv(ptr nonnull readnone align 8 captures(none) %0) unnamed_addr #2 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi i64 [ 0, %1 ], [ %17, %2 ]
  %4 = phi <2 x i64> [ <i64 0, i64 1>, %1 ], [ %18, %2 ]
  %5 = trunc <2 x i64> %4 to <2 x i32>
  %6 = add <2 x i32> %5, splat (i32 1)
  %7 = trunc <2 x i64> %4 to <2 x i32>
  %8 = add <2 x i32> %7, splat (i32 3)
  %9 = uitofp nneg <2 x i32> %6 to <2 x double>
  %10 = uitofp nneg <2 x i32> %8 to <2 x double>
  %11 = getelementptr inbounds nuw double, ptr @C, i64 %3
  %12 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store <2 x double> %9, ptr %11, align 8, !tbaa !6
  store <2 x double> %10, ptr %12, align 8, !tbaa !6
  %13 = fdiv <2 x double> splat (double 1.000000e+00), %9
  %14 = fdiv <2 x double> splat (double 1.000000e+00), %10
  %15 = getelementptr inbounds nuw double, ptr @D, i64 %3
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 16
  store <2 x double> %13, ptr %15, align 8, !tbaa !6
  store <2 x double> %14, ptr %16, align 8, !tbaa !6
  %17 = add nuw i64 %3, 4
  %18 = add <2 x i64> %4, splat (i64 4)
  %19 = icmp eq i64 %17, 2500
  br i1 %19, label %20, label %2, !llvm.loop !20

20:                                               ; preds = %2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(read, argmem: write, inaccessiblemem: none) uwtable
define dso_local void @_ZNK15MatrixBenchmark5checkEiRdS0_(ptr nonnull readnone align 8 captures(none) %0, i32 noundef %1, ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(8) %3) unnamed_addr #5 {
  br label %5

5:                                                ; preds = %5, %4
  %6 = phi i64 [ 0, %4 ], [ %14, %5 ]
  %7 = phi double [ 0.000000e+00, %4 ], [ %13, %5 ]
  %8 = getelementptr inbounds nuw double, ptr @E, i64 %6
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %10 = load <2 x double>, ptr %8, align 8, !tbaa !6
  %11 = load <2 x double>, ptr %9, align 8, !tbaa !6
  %12 = tail call double @llvm.vector.reduce.fadd.v2f64(double %7, <2 x double> %10)
  %13 = tail call double @llvm.vector.reduce.fadd.v2f64(double %12, <2 x double> %11)
  %14 = add nuw i64 %6, 4
  %15 = icmp eq i64 %14, 2500
  br i1 %15, label %16, label %5, !llvm.loop !21

16:                                               ; preds = %5
  store double %13, ptr %3, align 8, !tbaa !6
  %17 = sitofp i32 %1 to double
  %18 = fmul double %17, 2.500000e+05
  store double %18, ptr %2, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_ZNK17IteratorBenchmark7c_styleEv(ptr nonnull readnone align 8 captures(none) %0) unnamed_addr #0 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi i64 [ 0, %1 ], [ %17, %2 ]
  %4 = phi double [ 0.000000e+00, %1 ], [ %16, %2 ]
  %5 = getelementptr inbounds nuw double, ptr @A, i64 %3
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %7 = load <2 x double>, ptr %5, align 8, !tbaa !6
  %8 = load <2 x double>, ptr %6, align 8, !tbaa !6
  %9 = getelementptr inbounds nuw double, ptr @B, i64 %3
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %11 = load <2 x double>, ptr %9, align 8, !tbaa !6
  %12 = load <2 x double>, ptr %10, align 8, !tbaa !6
  %13 = fmul <2 x double> %7, %11
  %14 = fmul <2 x double> %8, %12
  %15 = tail call double @llvm.vector.reduce.fadd.v2f64(double %4, <2 x double> %13)
  %16 = tail call double @llvm.vector.reduce.fadd.v2f64(double %15, <2 x double> %14)
  %17 = add nuw i64 %3, 4
  %18 = icmp eq i64 %17, 1000
  br i1 %18, label %19, label %2, !llvm.loop !22

19:                                               ; preds = %2
  store double %16, ptr @IteratorResult, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_ZNK17IteratorBenchmark9oop_styleEv(ptr nonnull readnone align 8 captures(none) %0) unnamed_addr #0 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi i64 [ 0, %1 ], [ %17, %2 ]
  %4 = phi double [ 0.000000e+00, %1 ], [ %16, %2 ]
  %5 = getelementptr inbounds nuw double, ptr @A, i64 %3
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %7 = load <2 x double>, ptr %5, align 8, !tbaa !6
  %8 = load <2 x double>, ptr %6, align 8, !tbaa !6
  %9 = getelementptr inbounds nuw double, ptr @B, i64 %3
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %11 = load <2 x double>, ptr %9, align 8, !tbaa !6
  %12 = load <2 x double>, ptr %10, align 8, !tbaa !6
  %13 = fmul <2 x double> %7, %11
  %14 = fmul <2 x double> %8, %12
  %15 = tail call double @llvm.vector.reduce.fadd.v2f64(double %4, <2 x double> %13)
  %16 = tail call double @llvm.vector.reduce.fadd.v2f64(double %15, <2 x double> %14)
  %17 = add nuw i64 %3, 4
  %18 = icmp eq i64 %17, 1000
  br i1 %18, label %19, label %2, !llvm.loop !23

19:                                               ; preds = %2
  store double %16, ptr @IteratorResult, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_ZNK17IteratorBenchmark4initEv(ptr nonnull readnone align 8 captures(none) %0) unnamed_addr #2 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi i64 [ 0, %1 ], [ %17, %2 ]
  %4 = phi <2 x i64> [ <i64 0, i64 1>, %1 ], [ %18, %2 ]
  %5 = trunc <2 x i64> %4 to <2 x i32>
  %6 = add <2 x i32> %5, splat (i32 1)
  %7 = trunc <2 x i64> %4 to <2 x i32>
  %8 = add <2 x i32> %7, splat (i32 3)
  %9 = uitofp nneg <2 x i32> %6 to <2 x double>
  %10 = uitofp nneg <2 x i32> %8 to <2 x double>
  %11 = getelementptr inbounds nuw double, ptr @A, i64 %3
  %12 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store <2 x double> %9, ptr %11, align 8, !tbaa !6
  store <2 x double> %10, ptr %12, align 8, !tbaa !6
  %13 = fdiv <2 x double> splat (double 1.000000e+00), %9
  %14 = fdiv <2 x double> splat (double 1.000000e+00), %10
  %15 = getelementptr inbounds nuw double, ptr @B, i64 %3
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 16
  store <2 x double> %13, ptr %15, align 8, !tbaa !6
  store <2 x double> %14, ptr %16, align 8, !tbaa !6
  %17 = add nuw i64 %3, 4
  %18 = add <2 x i64> %4, splat (i64 4)
  %19 = icmp eq i64 %17, 1000
  br i1 %19, label %20, label %2, !llvm.loop !24

20:                                               ; preds = %2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none) uwtable
define dso_local void @_ZNK17IteratorBenchmark5checkEiRdS0_(ptr nonnull readnone align 8 captures(none) %0, i32 noundef %1, ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(8) initializes((0, 8)) %2, ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(8) initializes((0, 8)) %3) unnamed_addr #3 {
  %5 = mul nsw i32 %1, 2000
  %6 = sitofp i32 %5 to double
  store double %6, ptr %2, align 8, !tbaa !6
  %7 = load double, ptr @IteratorResult, align 8, !tbaa !6
  store double %7, ptr %3, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_ZNK16ComplexBenchmark7c_styleEv(ptr nonnull readnone align 8 captures(none) %0) unnamed_addr #0 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi i64 [ 0, %1 ], [ %17, %2 ]
  %4 = getelementptr inbounds nuw %class.Complex, ptr @Y, i64 %3
  %5 = load <4 x double>, ptr %4, align 8, !tbaa !6
  %6 = shufflevector <4 x double> %5, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %7 = shufflevector <4 x double> %5, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %8 = getelementptr inbounds nuw %class.Complex, ptr @X, i64 %3
  %9 = load <4 x double>, ptr %8, align 8, !tbaa !6
  %10 = shufflevector <4 x double> %9, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %11 = shufflevector <4 x double> %9, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %12 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %10, <2 x double> splat (double 5.000000e-01), <2 x double> %6)
  %13 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %11, <2 x double> splat (double 0xBFEBB67AE8584CAA), <2 x double> %12)
  %14 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %11, <2 x double> splat (double 5.000000e-01), <2 x double> %7)
  %15 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %10, <2 x double> splat (double 0x3FEBB67AE8584CAA), <2 x double> %14)
  %16 = shufflevector <2 x double> %13, <2 x double> %15, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %16, ptr %4, align 8, !tbaa !6
  %17 = add nuw i64 %3, 2
  %18 = icmp eq i64 %17, 1000
  br i1 %18, label %19, label %2, !llvm.loop !25

19:                                               ; preds = %2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_ZNK16ComplexBenchmark9oop_styleEv(ptr nonnull readnone align 8 captures(none) %0) unnamed_addr #0 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi i64 [ 0, %1 ], [ %19, %2 ]
  %4 = getelementptr inbounds nuw %class.Complex, ptr @Y, i64 %3
  %5 = load <4 x double>, ptr %4, align 8, !tbaa !6
  %6 = shufflevector <4 x double> %5, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %7 = shufflevector <4 x double> %5, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %8 = getelementptr inbounds nuw %class.Complex, ptr @X, i64 %3
  %9 = load <4 x double>, ptr %8, align 8, !tbaa !6
  %10 = shufflevector <4 x double> %9, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %11 = shufflevector <4 x double> %9, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %12 = fmul <2 x double> %11, splat (double 0xBFEBB67AE8584CAA)
  %13 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %10, <2 x double> splat (double 5.000000e-01), <2 x double> %12)
  %14 = fmul <2 x double> %10, splat (double 0x3FEBB67AE8584CAA)
  %15 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %11, <2 x double> splat (double 5.000000e-01), <2 x double> %14)
  %16 = fadd <2 x double> %6, %13
  %17 = fadd <2 x double> %7, %15
  %18 = shufflevector <2 x double> %16, <2 x double> %17, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %18, ptr %4, align 8, !tbaa !6
  %19 = add nuw i64 %3, 2
  %20 = icmp eq i64 %19, 1000
  br i1 %20, label %21, label %2, !llvm.loop !26

21:                                               ; preds = %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #6

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_ZNK16ComplexBenchmark4initEv(ptr nonnull readnone align 8 captures(none) %0) unnamed_addr #2 {
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) @Y, i8 0, i64 16000, i1 false), !tbaa !6
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi i64 [ 0, %1 ], [ %18, %2 ]
  %4 = phi <2 x i64> [ <i64 0, i64 1>, %1 ], [ %19, %2 ]
  %5 = trunc <2 x i64> %4 to <2 x i32>
  %6 = add <2 x i32> %5, splat (i32 1)
  %7 = trunc <2 x i64> %4 to <2 x i32>
  %8 = add <2 x i32> %7, splat (i32 3)
  %9 = uitofp nneg <2 x i32> %6 to <2 x double>
  %10 = uitofp nneg <2 x i32> %8 to <2 x double>
  %11 = fdiv <2 x double> splat (double 1.000000e+00), %9
  %12 = fdiv <2 x double> splat (double 1.000000e+00), %10
  %13 = getelementptr inbounds nuw %class.Complex, ptr @X, i64 %3
  %14 = getelementptr inbounds nuw %class.Complex, ptr @X, i64 %3
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 32
  %16 = shufflevector <2 x double> %9, <2 x double> %11, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %16, ptr %13, align 8, !tbaa !6
  %17 = shufflevector <2 x double> %10, <2 x double> %12, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %17, ptr %15, align 8, !tbaa !6
  %18 = add nuw i64 %3, 4
  %19 = add <2 x i64> %4, splat (i64 4)
  %20 = icmp eq i64 %18, 1000
  br i1 %20, label %21, label %2, !llvm.loop !27

21:                                               ; preds = %2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(read, argmem: write, inaccessiblemem: none) uwtable
define dso_local void @_ZNK16ComplexBenchmark5checkEiRdS0_(ptr nonnull readnone align 8 captures(none) %0, i32 noundef %1, ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(8) %3) unnamed_addr #5 {
  br label %5

5:                                                ; preds = %5, %4
  %6 = phi i64 [ 0, %4 ], [ %21, %5 ]
  %7 = phi double [ 0.000000e+00, %4 ], [ %20, %5 ]
  %8 = getelementptr inbounds nuw %class.Complex, ptr @Y, i64 %6
  %9 = getelementptr inbounds nuw %class.Complex, ptr @Y, i64 %6
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 32
  %11 = load <4 x double>, ptr %8, align 8, !tbaa !6
  %12 = shufflevector <4 x double> %11, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %13 = shufflevector <4 x double> %11, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %14 = load <4 x double>, ptr %10, align 8, !tbaa !6
  %15 = shufflevector <4 x double> %14, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %16 = shufflevector <4 x double> %14, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %17 = fadd <2 x double> %12, %13
  %18 = fadd <2 x double> %15, %16
  %19 = tail call double @llvm.vector.reduce.fadd.v2f64(double %7, <2 x double> %17)
  %20 = tail call double @llvm.vector.reduce.fadd.v2f64(double %19, <2 x double> %18)
  %21 = add nuw i64 %6, 4
  %22 = icmp eq i64 %21, 1000
  br i1 %22, label %23, label %5, !llvm.loop !28

23:                                               ; preds = %5
  store double %20, ptr %3, align 8, !tbaa !6
  %24 = mul nsw i32 %1, 8000
  %25 = sitofp i32 %24 to double
  store double %25, ptr %2, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local void @_ZNK9Benchmark8time_oneEMS_KFvvEiRdS2_S2_(ptr noundef nonnull align 8 dereferenceable(8) %0, [2 x i64] %1, i32 noundef %2, ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5) local_unnamed_addr #7 {
  %7 = alloca double, align 8
  %8 = extractvalue [2 x i64] %1, 0
  %9 = extractvalue [2 x i64] %1, 1
  %10 = load ptr, ptr %0, align 8, !tbaa !29
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %12 = load ptr, ptr %11, align 8
  tail call void %12(ptr noundef nonnull align 8 dereferenceable(8) %0)
  %13 = ashr i64 %9, 1
  %14 = getelementptr inbounds i8, ptr %0, i64 %13
  %15 = and i64 %9, 1
  %16 = icmp eq i64 %15, 0
  br i1 %16, label %21, label %17

17:                                               ; preds = %6
  %18 = load ptr, ptr %14, align 8, !tbaa !29
  %19 = getelementptr i8, ptr %18, i64 %8, !nosanitize !31
  %20 = load ptr, ptr %19, align 8, !nosanitize !31
  br label %23

21:                                               ; preds = %6
  %22 = inttoptr i64 %8 to ptr
  br label %23

23:                                               ; preds = %21, %17
  %24 = phi ptr [ %20, %17 ], [ %22, %21 ]
  tail call void %24(ptr noundef nonnull align 8 dereferenceable(8) %14)
  %25 = load ptr, ptr %0, align 8, !tbaa !29
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 8
  %27 = load ptr, ptr %26, align 8
  tail call void %27(ptr noundef nonnull align 8 dereferenceable(8) %0)
  %28 = tail call i64 @clock() #21
  %29 = icmp sgt i32 %2, 0
  br i1 %29, label %30, label %36

30:                                               ; preds = %23
  %31 = inttoptr i64 %8 to ptr
  br i1 %16, label %32, label %47

32:                                               ; preds = %30, %32
  %33 = phi i32 [ %34, %32 ], [ 0, %30 ]
  tail call void %31(ptr noundef nonnull align 8 dereferenceable(8) %14)
  %34 = add nuw nsw i32 %33, 1
  %35 = icmp eq i32 %34, %2
  br i1 %35, label %36, label %32, !llvm.loop !32

36:                                               ; preds = %47, %32, %23
  %37 = tail call i64 @clock() #21
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #21
  %38 = load ptr, ptr %0, align 8, !tbaa !29
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 32
  %40 = load ptr, ptr %39, align 8
  call void %40(ptr noundef nonnull align 8 dereferenceable(8) %0, i32 noundef %2, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %5)
  %41 = sub nsw i64 %37, %28
  %42 = sitofp i64 %41 to double
  %43 = fdiv double %42, 1.000000e+06
  store double %43, ptr %3, align 8, !tbaa !6
  %44 = load double, ptr %7, align 8, !tbaa !6
  %45 = fdiv double %44, %43
  %46 = fmul double %45, 0x3EB0C6F7A0B5ED8D
  store double %46, ptr %4, align 8, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #21
  ret void

47:                                               ; preds = %30, %47
  %48 = phi i32 [ %52, %47 ], [ 0, %30 ]
  %49 = load ptr, ptr %14, align 8, !tbaa !29
  %50 = getelementptr i8, ptr %49, i64 %8, !nosanitize !31
  %51 = load ptr, ptr %50, align 8, !nosanitize !31
  tail call void %51(ptr noundef nonnull align 8 dereferenceable(8) %14)
  %52 = add nuw nsw i32 %48, 1
  %53 = icmp eq i32 %52, %2
  br i1 %53, label %36, label %47, !llvm.loop !32
}

; Function Attrs: nounwind
declare i64 @clock() local_unnamed_addr #8

; Function Attrs: mustprogress uwtable
define dso_local noundef ptr @_ZN9Benchmark4findEPKc(ptr noundef readonly captures(none) %0) local_unnamed_addr #7 {
  %2 = load i32, ptr @_ZN9Benchmark5countE, align 4, !tbaa !33
  %3 = icmp sgt i32 %2, 0
  br i1 %3, label %9, label %20

4:                                                ; preds = %9
  %5 = add nuw nsw i64 %10, 1
  %6 = load i32, ptr @_ZN9Benchmark5countE, align 4, !tbaa !33
  %7 = sext i32 %6 to i64
  %8 = icmp slt i64 %5, %7
  br i1 %8, label %9, label %20, !llvm.loop !35

9:                                                ; preds = %1, %4
  %10 = phi i64 [ %5, %4 ], [ 0, %1 ]
  %11 = getelementptr inbounds nuw ptr, ptr @_ZN9Benchmark4listE, i64 %10
  %12 = load ptr, ptr %11, align 8, !tbaa !36
  %13 = load ptr, ptr %12, align 8, !tbaa !29
  %14 = load ptr, ptr %13, align 8
  %15 = tail call noundef ptr %14(ptr noundef nonnull align 8 dereferenceable(8) %12)
  %16 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %0, ptr noundef nonnull dereferenceable(1) %15) #22
  %17 = icmp eq i32 %16, 0
  br i1 %17, label %18, label %4

18:                                               ; preds = %9
  %19 = load ptr, ptr %11, align 8, !tbaa !36
  br label %20

20:                                               ; preds = %4, %1, %18
  %21 = phi ptr [ %19, %18 ], [ null, %1 ], [ null, %4 ]
  ret ptr %21
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @strcmp(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #9

; Function Attrs: mustprogress uwtable
define dso_local void @_ZNK9Benchmark9time_bothEi(ptr noundef nonnull align 8 dereferenceable(8) %0, i32 noundef %1) local_unnamed_addr #7 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca double, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #21
  %7 = load ptr, ptr %0, align 8, !tbaa !29
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %9 = load ptr, ptr %8, align 8
  tail call void %9(ptr noundef nonnull align 8 dereferenceable(8) %0)
  %10 = load ptr, ptr %0, align 8, !tbaa !29
  %11 = getelementptr i8, ptr %10, i64 16, !nosanitize !31
  %12 = load ptr, ptr %11, align 8, !nosanitize !31
  tail call void %12(ptr noundef nonnull align 8 dereferenceable(8) %0)
  %13 = load ptr, ptr %0, align 8, !tbaa !29
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 8
  %15 = load ptr, ptr %14, align 8
  tail call void %15(ptr noundef nonnull align 8 dereferenceable(8) %0)
  %16 = tail call i64 @clock() #21
  %17 = icmp sgt i32 %1, 0
  br i1 %17, label %18, label %25

18:                                               ; preds = %2, %18
  %19 = phi i32 [ %23, %18 ], [ 0, %2 ]
  %20 = load ptr, ptr %0, align 8, !tbaa !29
  %21 = getelementptr i8, ptr %20, i64 16, !nosanitize !31
  %22 = load ptr, ptr %21, align 8, !nosanitize !31
  tail call void %22(ptr noundef nonnull align 8 dereferenceable(8) %0)
  %23 = add nuw nsw i32 %19, 1
  %24 = icmp eq i32 %23, %1
  br i1 %24, label %25, label %18, !llvm.loop !32

25:                                               ; preds = %18, %2
  %26 = tail call i64 @clock() #21
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #21
  %27 = load ptr, ptr %0, align 8, !tbaa !29
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 32
  %29 = load ptr, ptr %28, align 8
  call void %29(ptr noundef nonnull align 8 dereferenceable(8) %0, i32 noundef %1, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #21
  %30 = load ptr, ptr %0, align 8, !tbaa !29
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 8
  %32 = load ptr, ptr %31, align 8
  call void %32(ptr noundef nonnull align 8 dereferenceable(8) %0)
  %33 = load ptr, ptr %0, align 8, !tbaa !29
  %34 = getelementptr i8, ptr %33, i64 24, !nosanitize !31
  %35 = load ptr, ptr %34, align 8, !nosanitize !31
  call void %35(ptr noundef nonnull align 8 dereferenceable(8) %0)
  %36 = load ptr, ptr %0, align 8, !tbaa !29
  %37 = getelementptr inbounds nuw i8, ptr %36, i64 8
  %38 = load ptr, ptr %37, align 8
  call void %38(ptr noundef nonnull align 8 dereferenceable(8) %0)
  %39 = call i64 @clock() #21
  br i1 %17, label %40, label %47

40:                                               ; preds = %25, %40
  %41 = phi i32 [ %45, %40 ], [ 0, %25 ]
  %42 = load ptr, ptr %0, align 8, !tbaa !29
  %43 = getelementptr i8, ptr %42, i64 24, !nosanitize !31
  %44 = load ptr, ptr %43, align 8, !nosanitize !31
  call void %44(ptr noundef nonnull align 8 dereferenceable(8) %0)
  %45 = add nuw nsw i32 %41, 1
  %46 = icmp eq i32 %45, %1
  br i1 %46, label %47, label %40, !llvm.loop !32

47:                                               ; preds = %40, %25
  %48 = call i64 @clock() #21
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #21
  %49 = load ptr, ptr %0, align 8, !tbaa !29
  %50 = getelementptr inbounds nuw i8, ptr %49, i64 32
  %51 = load ptr, ptr %50, align 8
  call void %51(ptr noundef nonnull align 8 dereferenceable(8) %0, i32 noundef %1, ptr noundef nonnull align 8 dereferenceable(8) %3, ptr noundef nonnull align 8 dereferenceable(8) %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #21
  %52 = load double, ptr %5, align 8, !tbaa !6
  %53 = load double, ptr %6, align 8, !tbaa !6
  %54 = fsub double %52, %53
  %55 = fcmp olt double %52, %53
  %56 = select i1 %55, double %52, double %53
  %57 = fdiv double %54, %56
  %58 = call double @llvm.fabs.f64(double %57)
  %59 = fcmp ogt double %58, 0x3D10000000000000
  br i1 %59, label %60, label %67

60:                                               ; preds = %47
  %61 = load ptr, ptr %0, align 8, !tbaa !29
  %62 = load ptr, ptr %61, align 8
  %63 = call noundef ptr %62(ptr noundef nonnull align 8 dereferenceable(8) %0)
  %64 = load double, ptr %5, align 8, !tbaa !6
  %65 = load double, ptr %6, align 8, !tbaa !6
  %66 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef %63, double noundef %57, double noundef %64, double noundef %65)
  br label %67

67:                                               ; preds = %47, %60
  %68 = load ptr, ptr %0, align 8, !tbaa !29
  %69 = load ptr, ptr %68, align 8
  %70 = call noundef ptr %69(ptr noundef nonnull align 8 dereferenceable(8) %0)
  %71 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, ptr noundef %70, i32 noundef %1)
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #21
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #21
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #10

; Function Attrs: cold mustprogress nofree noreturn nounwind uwtable
define dso_local void @_Z5UsageiPPc(i32 %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #11 {
  %3 = load ptr, ptr %1, align 8, !tbaa !39
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, ptr noundef %3)
  %5 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @exit(i32 noundef 1) #23
  unreachable
}

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #12

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #13 {
  %3 = alloca [6 x i8], align 1
  %4 = alloca [10 x i8], align 1
  %5 = alloca [11 x i8], align 1
  %6 = alloca [13 x i8], align 1
  %7 = alloca [15 x i8], align 1
  %8 = alloca [6 x ptr], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #21
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %3, ptr noundef nonnull align 1 dereferenceable(6) @__const.main.str1, i64 6, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #21
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(10) %4, ptr noundef nonnull align 1 dereferenceable(10) @__const.main.str2, i64 10, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #21
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(11) %5, ptr noundef nonnull align 1 dereferenceable(11) @__const.main.str3, i64 11, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #21
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(13) %6, ptr noundef nonnull align 1 dereferenceable(13) @__const.main.str4, i64 13, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #21
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(15) %7, ptr noundef nonnull align 1 dereferenceable(15) @__const.main.str5, i64 15, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #21
  store ptr %3, ptr %8, align 8, !tbaa !39
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 8
  store ptr %4, ptr %9, align 8, !tbaa !39
  %10 = getelementptr inbounds nuw i8, ptr %8, i64 16
  store ptr %5, ptr %10, align 8, !tbaa !39
  %11 = getelementptr inbounds nuw i8, ptr %8, i64 24
  store ptr %6, ptr %11, align 8, !tbaa !39
  %12 = getelementptr inbounds nuw i8, ptr %8, i64 32
  store ptr %7, ptr %12, align 8, !tbaa !39
  %13 = getelementptr inbounds nuw i8, ptr %8, i64 40
  store ptr null, ptr %13, align 8, !tbaa !39
  %14 = tail call ptr @__ctype_b_loc() #24
  %15 = load ptr, ptr %14, align 8, !tbaa !41
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 154
  %17 = load i16, ptr %16, align 2, !tbaa !43
  %18 = and i16 %17, 1024
  %19 = icmp eq i16 %18, 0
  br i1 %19, label %37, label %20

20:                                               ; preds = %2
  %21 = getelementptr inbounds nuw i8, ptr %15, i64 134
  %22 = load i16, ptr %21, align 2, !tbaa !43
  %23 = and i16 %22, 1024
  %24 = icmp eq i16 %23, 0
  br i1 %24, label %37, label %25

25:                                               ; preds = %20
  %26 = getelementptr inbounds nuw i8, ptr %15, i64 146
  %27 = load i16, ptr %26, align 2, !tbaa !43
  %28 = and i16 %27, 1024
  %29 = icmp eq i16 %28, 0
  br i1 %29, label %37, label %30

30:                                               ; preds = %25
  %31 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, ptr noundef nonnull @.str.11, ptr noundef nonnull @.str.11, ptr noundef nonnull @.str.12, ptr noundef nonnull @.str.13, ptr noundef nonnull @.str.11)
  %32 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14, ptr noundef nonnull @.str.15, ptr noundef nonnull @.str.16, ptr noundef nonnull @.str.17, ptr noundef nonnull @.str.18, ptr noundef nonnull @.str.17, ptr noundef nonnull @.str.18, ptr noundef nonnull @.str.19)
  %33 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, ptr noundef nonnull @.str.20, ptr noundef nonnull @.str.21, ptr noundef nonnull @.str.22, ptr noundef nonnull @.str.22, ptr noundef nonnull @.str.23)
  %34 = call ptr @strtok(ptr noundef nonnull %4, ptr noundef nonnull @.str.24) #21
  %35 = call ptr @strtok(ptr noundef null, ptr noundef nonnull @.str.11) #21
  %36 = icmp eq ptr %35, null
  br i1 %36, label %38, label %40

37:                                               ; preds = %25, %20, %2
  call void @_Z5UsageiPPc(i32 poison, ptr noundef nonnull %8)
  unreachable

38:                                               ; preds = %30
  %39 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.25, ptr noundef %34)
  br label %66

40:                                               ; preds = %30
  %41 = call i64 @__isoc23_strtol(ptr noundef nonnull %35, ptr noundef null, i32 noundef 0) #21
  %42 = load i32, ptr @_ZN9Benchmark5countE, align 4, !tbaa !33
  %43 = icmp sgt i32 %42, 0
  br i1 %43, label %49, label %61

44:                                               ; preds = %49
  %45 = add nuw nsw i64 %50, 1
  %46 = load i32, ptr @_ZN9Benchmark5countE, align 4, !tbaa !33
  %47 = sext i32 %46 to i64
  %48 = icmp slt i64 %45, %47
  br i1 %48, label %49, label %61, !llvm.loop !35

49:                                               ; preds = %40, %44
  %50 = phi i64 [ %45, %44 ], [ 0, %40 ]
  %51 = getelementptr inbounds nuw ptr, ptr @_ZN9Benchmark4listE, i64 %50
  %52 = load ptr, ptr %51, align 8, !tbaa !36
  %53 = load ptr, ptr %52, align 8, !tbaa !29
  %54 = load ptr, ptr %53, align 8
  %55 = call noundef ptr %54(ptr noundef nonnull align 8 dereferenceable(8) %52)
  %56 = call i32 @strcmp(ptr noundef nonnull readonly dereferenceable(1) %34, ptr noundef nonnull dereferenceable(1) %55) #22
  %57 = icmp eq i32 %56, 0
  br i1 %57, label %58, label %44

58:                                               ; preds = %49
  %59 = load ptr, ptr %51, align 8, !tbaa !36
  %60 = icmp eq ptr %59, null
  br i1 %60, label %61, label %64

61:                                               ; preds = %44, %83, %112, %141, %40, %58, %70, %88, %99, %117, %128, %146
  %62 = phi ptr [ %34, %58 ], [ %34, %40 ], [ %67, %70 ], [ %67, %88 ], [ %96, %99 ], [ %96, %117 ], [ %125, %128 ], [ %125, %146 ], [ %125, %141 ], [ %96, %112 ], [ %67, %83 ], [ %34, %44 ]
  %63 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.26, ptr noundef %62)
  call void @abort() #23
  unreachable

64:                                               ; preds = %58
  %65 = trunc i64 %41 to i32
  call void @_ZNK9Benchmark9time_bothEi(ptr noundef nonnull align 8 dereferenceable(8) %59, i32 noundef %65)
  br label %66

66:                                               ; preds = %64, %38
  %67 = call ptr @strtok(ptr noundef nonnull %5, ptr noundef nonnull @.str.24) #21
  %68 = call ptr @strtok(ptr noundef null, ptr noundef nonnull @.str.11) #21
  %69 = icmp eq ptr %68, null
  br i1 %69, label %93, label %70

70:                                               ; preds = %66
  %71 = call i64 @__isoc23_strtol(ptr noundef nonnull %68, ptr noundef null, i32 noundef 0) #21
  %72 = load i32, ptr @_ZN9Benchmark5countE, align 4, !tbaa !33
  %73 = icmp sgt i32 %72, 0
  br i1 %73, label %74, label %61

74:                                               ; preds = %70, %83
  %75 = phi i64 [ %84, %83 ], [ 0, %70 ]
  %76 = getelementptr inbounds nuw ptr, ptr @_ZN9Benchmark4listE, i64 %75
  %77 = load ptr, ptr %76, align 8, !tbaa !36
  %78 = load ptr, ptr %77, align 8, !tbaa !29
  %79 = load ptr, ptr %78, align 8
  %80 = call noundef ptr %79(ptr noundef nonnull align 8 dereferenceable(8) %77)
  %81 = call i32 @strcmp(ptr noundef nonnull readonly dereferenceable(1) %67, ptr noundef nonnull dereferenceable(1) %80) #22
  %82 = icmp eq i32 %81, 0
  br i1 %82, label %88, label %83

83:                                               ; preds = %74
  %84 = add nuw nsw i64 %75, 1
  %85 = load i32, ptr @_ZN9Benchmark5countE, align 4, !tbaa !33
  %86 = sext i32 %85 to i64
  %87 = icmp slt i64 %84, %86
  br i1 %87, label %74, label %61, !llvm.loop !35

88:                                               ; preds = %74
  %89 = load ptr, ptr %76, align 8, !tbaa !36
  %90 = icmp eq ptr %89, null
  br i1 %90, label %61, label %91

91:                                               ; preds = %88
  %92 = trunc i64 %71 to i32
  call void @_ZNK9Benchmark9time_bothEi(ptr noundef nonnull align 8 dereferenceable(8) %89, i32 noundef %92)
  br label %95

93:                                               ; preds = %66
  %94 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.25, ptr noundef %67)
  br label %95

95:                                               ; preds = %93, %91
  %96 = call ptr @strtok(ptr noundef nonnull %6, ptr noundef nonnull @.str.24) #21
  %97 = call ptr @strtok(ptr noundef null, ptr noundef nonnull @.str.11) #21
  %98 = icmp eq ptr %97, null
  br i1 %98, label %122, label %99

99:                                               ; preds = %95
  %100 = call i64 @__isoc23_strtol(ptr noundef nonnull %97, ptr noundef null, i32 noundef 0) #21
  %101 = load i32, ptr @_ZN9Benchmark5countE, align 4, !tbaa !33
  %102 = icmp sgt i32 %101, 0
  br i1 %102, label %103, label %61

103:                                              ; preds = %99, %112
  %104 = phi i64 [ %113, %112 ], [ 0, %99 ]
  %105 = getelementptr inbounds nuw ptr, ptr @_ZN9Benchmark4listE, i64 %104
  %106 = load ptr, ptr %105, align 8, !tbaa !36
  %107 = load ptr, ptr %106, align 8, !tbaa !29
  %108 = load ptr, ptr %107, align 8
  %109 = call noundef ptr %108(ptr noundef nonnull align 8 dereferenceable(8) %106)
  %110 = call i32 @strcmp(ptr noundef nonnull readonly dereferenceable(1) %96, ptr noundef nonnull dereferenceable(1) %109) #22
  %111 = icmp eq i32 %110, 0
  br i1 %111, label %117, label %112

112:                                              ; preds = %103
  %113 = add nuw nsw i64 %104, 1
  %114 = load i32, ptr @_ZN9Benchmark5countE, align 4, !tbaa !33
  %115 = sext i32 %114 to i64
  %116 = icmp slt i64 %113, %115
  br i1 %116, label %103, label %61, !llvm.loop !35

117:                                              ; preds = %103
  %118 = load ptr, ptr %105, align 8, !tbaa !36
  %119 = icmp eq ptr %118, null
  br i1 %119, label %61, label %120

120:                                              ; preds = %117
  %121 = trunc i64 %100 to i32
  call void @_ZNK9Benchmark9time_bothEi(ptr noundef nonnull align 8 dereferenceable(8) %118, i32 noundef %121)
  br label %124

122:                                              ; preds = %95
  %123 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.25, ptr noundef %96)
  br label %124

124:                                              ; preds = %122, %120
  %125 = call ptr @strtok(ptr noundef nonnull %7, ptr noundef nonnull @.str.24) #21
  %126 = call ptr @strtok(ptr noundef null, ptr noundef nonnull @.str.11) #21
  %127 = icmp eq ptr %126, null
  br i1 %127, label %151, label %128

128:                                              ; preds = %124
  %129 = call i64 @__isoc23_strtol(ptr noundef nonnull %126, ptr noundef null, i32 noundef 0) #21
  %130 = load i32, ptr @_ZN9Benchmark5countE, align 4, !tbaa !33
  %131 = icmp sgt i32 %130, 0
  br i1 %131, label %132, label %61

132:                                              ; preds = %128, %141
  %133 = phi i64 [ %142, %141 ], [ 0, %128 ]
  %134 = getelementptr inbounds nuw ptr, ptr @_ZN9Benchmark4listE, i64 %133
  %135 = load ptr, ptr %134, align 8, !tbaa !36
  %136 = load ptr, ptr %135, align 8, !tbaa !29
  %137 = load ptr, ptr %136, align 8
  %138 = call noundef ptr %137(ptr noundef nonnull align 8 dereferenceable(8) %135)
  %139 = call i32 @strcmp(ptr noundef nonnull readonly dereferenceable(1) %125, ptr noundef nonnull dereferenceable(1) %138) #22
  %140 = icmp eq i32 %139, 0
  br i1 %140, label %146, label %141

141:                                              ; preds = %132
  %142 = add nuw nsw i64 %133, 1
  %143 = load i32, ptr @_ZN9Benchmark5countE, align 4, !tbaa !33
  %144 = sext i32 %143 to i64
  %145 = icmp slt i64 %142, %144
  br i1 %145, label %132, label %61, !llvm.loop !35

146:                                              ; preds = %132
  %147 = load ptr, ptr %134, align 8, !tbaa !36
  %148 = icmp eq ptr %147, null
  br i1 %148, label %61, label %149

149:                                              ; preds = %146
  %150 = trunc i64 %129 to i32
  call void @_ZNK9Benchmark9time_bothEi(ptr noundef nonnull align 8 dereferenceable(8) %147, i32 noundef %150)
  br label %153

151:                                              ; preds = %124
  %152 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.25, ptr noundef %125)
  br label %153

153:                                              ; preds = %151, %149
  %154 = call i32 @puts(ptr nonnull dereferenceable(1) @str.32)
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #21
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #21
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #21
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #21
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #21
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #21
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare ptr @strtok(ptr noundef, ptr noundef readonly captures(none)) local_unnamed_addr #14

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #8

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #15

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @_ZNK12MaxBenchmark4nameEv(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #16 comdat {
  ret ptr @.str.28
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @_ZNK15MatrixBenchmark4nameEv(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #16 comdat {
  ret ptr @.str.29
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @_ZNK17IteratorBenchmark4nameEv(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #16 comdat {
  ret ptr @.str.30
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef ptr @_ZNK16ComplexBenchmark4nameEv(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #16 comdat {
  ret ptr @.str.31
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare ptr @__ctype_b_loc() local_unnamed_addr #17

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #18

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #19

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #20

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.vector.reduce.fadd.v2f64(double, <2 x double>) #18

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #18

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { mustprogress nofree norecurse nosync nounwind memory(read, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { cold mustprogress nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #14 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #15 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #16 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #17 = { mustprogress nofree nosync nounwind willreturn memory(none) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #18 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #19 = { nofree nounwind }
attributes #20 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #21 = { nounwind }
attributes #22 = { nounwind willreturn memory(read) }
attributes #23 = { cold noreturn nounwind }
attributes #24 = { nounwind willreturn memory(none) }

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
!9 = !{!"Simple C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = distinct !{!12, !11}
!13 = distinct !{!13, !11, !14, !15}
!14 = !{!"llvm.loop.isvectorized", i32 1}
!15 = !{!"llvm.loop.unroll.runtime.disable"}
!16 = distinct !{!16, !11}
!17 = distinct !{!17, !11}
!18 = distinct !{!18, !11}
!19 = distinct !{!19, !11}
!20 = distinct !{!20, !11, !14, !15}
!21 = distinct !{!21, !11, !14, !15}
!22 = distinct !{!22, !11, !14, !15}
!23 = distinct !{!23, !11, !14, !15}
!24 = distinct !{!24, !11, !14, !15}
!25 = distinct !{!25, !11, !14, !15}
!26 = distinct !{!26, !11, !14, !15}
!27 = distinct !{!27, !11, !14, !15}
!28 = distinct !{!28, !11, !14, !15}
!29 = !{!30, !30, i64 0}
!30 = !{!"vtable pointer", !9, i64 0}
!31 = !{}
!32 = distinct !{!32, !11}
!33 = !{!34, !34, i64 0}
!34 = !{!"int", !8, i64 0}
!35 = distinct !{!35, !11}
!36 = !{!37, !37, i64 0}
!37 = !{!"p1 _ZTS9Benchmark", !38, i64 0}
!38 = !{!"any pointer", !8, i64 0}
!39 = !{!40, !40, i64 0}
!40 = !{!"p1 omnipotent char", !38, i64 0}
!41 = !{!42, !42, i64 0}
!42 = !{!"p1 short", !38, i64 0}
!43 = !{!44, !44, i64 0}
!44 = !{!"short", !8, i64 0}
