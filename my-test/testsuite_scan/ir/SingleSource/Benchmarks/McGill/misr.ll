; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/McGill/misr.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/McGill/misr.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.cells = type { i32, i32, ptr }

@reg_len = dso_local local_unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [4 x i8] c"%hu\00", align 1
@.str.1 = private unnamed_addr constant [30 x i8] c"Register too long; Max. = %d\0A\00", align 1
@.str.4 = private unnamed_addr constant [75 x i8] c"reg_len\09#_vect\09prob      #_tms\09struct\09seed1\09seed2\09seed3\09Prob same output\0A \00", align 1
@.str.5 = private unnamed_addr constant [32 x i8] c"%d\09%d\09%.3e %d\09%s\09%d\09%d\09%d\09%.8e\0A\00", align 1
@str = private unnamed_addr constant [42 x i8] c"Structure does not match Register length:\00", align 4

; Function Attrs: nounwind uwtable
define dso_local range(i32 0, 5) i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = alloca %struct.cells, align 8
  %4 = alloca [100 x i8], align 4
  %5 = alloca [3 x i16], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #13
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #13
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #13
  store i32 10, ptr @reg_len, align 4, !tbaa !6
  %6 = icmp sgt i32 %0, 6
  br i1 %6, label %10, label %7

7:                                                ; preds = %2
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 1
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(9) %8, i8 48, i64 9, i1 false), !tbaa !10
  store i8 49, ptr %4, align 4, !tbaa !10
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 10
  store i8 0, ptr %9, align 2, !tbaa !10
  br label %15

10:                                               ; preds = %2
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %12 = load ptr, ptr %11, align 8, !tbaa !11
  %13 = call ptr @strcpy(ptr noundef nonnull dereferenceable(1) %4, ptr noundef nonnull dereferenceable(1) %12) #13
  %14 = icmp eq i32 %0, 7
  br i1 %14, label %15, label %16

15:                                               ; preds = %7, %10
  store i16 1, ptr %5, align 4, !tbaa !14
  br label %21

16:                                               ; preds = %10
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %18 = load ptr, ptr %17, align 8, !tbaa !11
  %19 = call i32 (ptr, ptr, ...) @__isoc99_sscanf(ptr noundef %18, ptr noundef nonnull @.str, ptr noundef nonnull %5) #13
  %20 = icmp eq i32 %0, 8
  br i1 %20, label %21, label %23

21:                                               ; preds = %16, %15
  %22 = getelementptr inbounds nuw i8, ptr %5, i64 2
  store i16 0, ptr %22, align 2, !tbaa !14
  br label %34

23:                                               ; preds = %16
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %25 = load ptr, ptr %24, align 8, !tbaa !11
  %26 = getelementptr inbounds nuw i8, ptr %5, i64 2
  %27 = call i32 (ptr, ptr, ...) @__isoc99_sscanf(ptr noundef %25, ptr noundef nonnull @.str, ptr noundef nonnull %26) #13
  %28 = icmp samesign ugt i32 %0, 9
  br i1 %28, label %29, label %34

29:                                               ; preds = %23
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %31 = load ptr, ptr %30, align 8, !tbaa !11
  %32 = getelementptr inbounds nuw i8, ptr %5, i64 4
  %33 = call i32 (ptr, ptr, ...) @__isoc99_sscanf(ptr noundef %31, ptr noundef nonnull @.str, ptr noundef nonnull %32) #13
  br label %36

34:                                               ; preds = %21, %23
  %35 = getelementptr inbounds nuw i8, ptr %5, i64 4
  store i16 0, ptr %35, align 4, !tbaa !14
  br label %36

36:                                               ; preds = %34, %29
  %37 = load i32, ptr @reg_len, align 4, !tbaa !6
  %38 = icmp sgt i32 %37, 100
  br i1 %38, label %39, label %41

39:                                               ; preds = %36
  %40 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 100)
  br label %94

41:                                               ; preds = %36
  %42 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %4) #14
  %43 = sext i32 %37 to i64
  %44 = icmp eq i64 %42, %43
  br i1 %44, label %47, label %45

45:                                               ; preds = %41
  %46 = call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %94

47:                                               ; preds = %41
  %48 = call ptr @seed48(ptr noundef nonnull %5) #13
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, i8 0, i64 16, i1 false)
  %49 = load i32, ptr @reg_len, align 4, !tbaa !6
  %50 = icmp slt i32 %49, 0
  br i1 %50, label %59, label %51

51:                                               ; preds = %47, %51
  %52 = phi ptr [ %54, %51 ], [ %3, %47 ]
  %53 = phi i32 [ %57, %51 ], [ 0, %47 ]
  %54 = call noalias dereferenceable_or_null(16) ptr @malloc(i64 noundef 16) #15
  store <2 x i32> splat (i32 1), ptr %54, align 8, !tbaa !6
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 8
  store ptr null, ptr %55, align 8, !tbaa !16
  %56 = getelementptr inbounds nuw i8, ptr %52, i64 8
  store ptr %54, ptr %56, align 8, !tbaa !16
  %57 = add nuw i32 %53, 1
  %58 = icmp eq i32 %53, %49
  br i1 %58, label %59, label %51, !llvm.loop !19

59:                                               ; preds = %51, %47
  %60 = getelementptr inbounds nuw i8, ptr %3, i64 8
  br label %61

61:                                               ; preds = %59, %74
  %62 = phi i32 [ 0, %59 ], [ %77, %74 ]
  %63 = phi i32 [ 0, %59 ], [ %76, %74 ]
  %64 = load ptr, ptr %60, align 8, !tbaa !16
  %65 = icmp eq ptr %64, null
  br i1 %65, label %74, label %66

66:                                               ; preds = %61, %66
  %67 = phi ptr [ %72, %66 ], [ %64, %61 ]
  %68 = phi ptr [ %67, %66 ], [ %3, %61 ]
  %69 = load i32, ptr %68, align 8, !tbaa !21
  %70 = getelementptr inbounds nuw i8, ptr %68, i64 4
  store i32 %69, ptr %70, align 4, !tbaa !22
  %71 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %72 = load ptr, ptr %71, align 8, !tbaa !16
  %73 = icmp eq ptr %72, null
  br i1 %73, label %74, label %66, !llvm.loop !23

74:                                               ; preds = %66, %61
  %75 = call i32 @simulate(i32 noundef 10, ptr noundef nonnull %3, double noundef 2.500000e-01, ptr noundef nonnull %4)
  %76 = add nuw nsw i32 %75, %63
  %77 = add nuw nsw i32 %62, 1
  %78 = icmp eq i32 %77, 100000
  br i1 %78, label %79, label %61, !llvm.loop !24

79:                                               ; preds = %74
  %80 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4)
  %81 = load i32, ptr @reg_len, align 4, !tbaa !6
  %82 = load i16, ptr %5, align 4, !tbaa !14
  %83 = zext i16 %82 to i32
  %84 = getelementptr inbounds nuw i8, ptr %5, i64 2
  %85 = load i16, ptr %84, align 2, !tbaa !14
  %86 = zext i16 %85 to i32
  %87 = getelementptr inbounds nuw i8, ptr %5, i64 4
  %88 = load i16, ptr %87, align 4, !tbaa !14
  %89 = zext i16 %88 to i32
  %90 = sub nsw i32 100000, %76
  %91 = sitofp i32 %90 to double
  %92 = fdiv double %91, 1.000000e+05
  %93 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %81, i32 noundef 10, double noundef 2.500000e-01, i32 noundef 100000, ptr noundef nonnull %4, i32 noundef %83, i32 noundef %86, i32 noundef %89, double noundef %92)
  br label %94

94:                                               ; preds = %79, %45, %39
  %95 = phi i32 [ 2, %39 ], [ 4, %45 ], [ 0, %79 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #13
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #13
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #13
  ret i32 %95
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare ptr @strcpy(ptr noalias noundef returned writeonly, ptr noalias noundef readonly captures(none)) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @__isoc99_sscanf(ptr noundef readonly captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #4

; Function Attrs: nounwind
declare ptr @seed48(ptr noundef) local_unnamed_addr #5

; Function Attrs: nofree nounwind memory(readwrite, argmem: write) uwtable
define dso_local void @create_link_list(ptr noundef writeonly captures(none) initializes((0, 16)) %0) local_unnamed_addr #6 {
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %0, i8 0, i64 16, i1 false)
  %2 = load i32, ptr @reg_len, align 4, !tbaa !6
  %3 = icmp slt i32 %2, 0
  br i1 %3, label %12, label %4

4:                                                ; preds = %1, %4
  %5 = phi ptr [ %7, %4 ], [ %0, %1 ]
  %6 = phi i32 [ %10, %4 ], [ 0, %1 ]
  %7 = tail call noalias dereferenceable_or_null(16) ptr @malloc(i64 noundef 16) #15
  store <2 x i32> splat (i32 1), ptr %7, align 8, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store ptr null, ptr %8, align 8, !tbaa !16
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store ptr %7, ptr %9, align 8, !tbaa !16
  %10 = add nuw i32 %6, 1
  %11 = icmp eq i32 %6, %2
  br i1 %11, label %12, label %4, !llvm.loop !19

12:                                               ; preds = %4, %1
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @init(ptr noundef captures(none) %0) local_unnamed_addr #7 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load ptr, ptr %2, align 8, !tbaa !16
  %4 = icmp eq ptr %3, null
  br i1 %4, label %13, label %5

5:                                                ; preds = %1, %5
  %6 = phi ptr [ %11, %5 ], [ %3, %1 ]
  %7 = phi ptr [ %6, %5 ], [ %0, %1 ]
  %8 = load i32, ptr %7, align 8, !tbaa !21
  %9 = getelementptr inbounds nuw i8, ptr %7, i64 4
  store i32 %8, ptr %9, align 4, !tbaa !22
  %10 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %11 = load ptr, ptr %10, align 8, !tbaa !16
  %12 = icmp eq ptr %11, null
  br i1 %12, label %13, label %5, !llvm.loop !23

13:                                               ; preds = %5, %1
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local range(i32 0, 2) i32 @simulate(i32 noundef %0, ptr noundef captures(none) %1, double noundef %2, ptr noundef readonly captures(none) %3) local_unnamed_addr #0 {
  %5 = load i32, ptr @reg_len, align 4, !tbaa !6
  %6 = icmp sgt i32 %0, 0
  br i1 %6, label %7, label %25

7:                                                ; preds = %4
  %8 = add nsw i32 %5, -1
  %9 = freeze i32 %8
  %10 = sdiv i32 %9, 31
  %11 = mul i32 %10, 31
  %12 = sub i32 %9, %11
  %13 = icmp sgt i32 %5, 31
  %14 = icmp sgt i32 %12, 0
  %15 = mul nsw i32 %10, 31
  %16 = tail call i32 @llvm.smax.i32(i32 %10, i32 1)
  %17 = sext i32 %15 to i64
  %18 = zext nneg i32 %16 to i64
  %19 = zext nneg i32 %12 to i64
  %20 = getelementptr i8, ptr %3, i64 %17
  br label %21

21:                                               ; preds = %7, %127
  %22 = phi i32 [ 0, %7 ], [ %145, %127 ]
  br i1 %13, label %28, label %74

23:                                               ; preds = %127
  %24 = load i32, ptr @reg_len, align 4, !tbaa !6
  br label %25

25:                                               ; preds = %23, %4
  %26 = phi i32 [ %24, %23 ], [ %5, %4 ]
  %27 = icmp sgt i32 %26, 0
  br i1 %27, label %147, label %160

28:                                               ; preds = %21, %71
  %29 = phi i64 [ %72, %71 ], [ 0, %21 ]
  %30 = phi ptr [ %61, %71 ], [ %1, %21 ]
  %31 = phi <2 x i32> [ %47, %71 ], [ zeroinitializer, %21 ]
  %32 = tail call i64 @lrand48() #13
  %33 = mul nuw nsw i64 %29, 31
  %34 = getelementptr inbounds nuw i8, ptr %3, i64 %33
  br label %35

35:                                               ; preds = %28, %46
  %36 = phi i64 [ 0, %28 ], [ %69, %46 ]
  %37 = phi i64 [ %32, %28 ], [ %68, %46 ]
  %38 = phi ptr [ %30, %28 ], [ %61, %46 ]
  %39 = phi <2 x i32> [ %31, %28 ], [ %47, %46 ]
  %40 = getelementptr inbounds nuw i8, ptr %34, i64 %36
  %41 = load i8, ptr %40, align 1, !tbaa !10
  %42 = icmp eq i8 %41, 49
  br i1 %42, label %43, label %46

43:                                               ; preds = %35
  %44 = load <2 x i32>, ptr %38, align 8, !tbaa !6
  %45 = add nsw <2 x i32> %44, %39
  br label %46

46:                                               ; preds = %43, %35
  %47 = phi <2 x i32> [ %45, %43 ], [ %39, %35 ]
  %48 = getelementptr inbounds nuw i8, ptr %38, i64 8
  %49 = load ptr, ptr %48, align 8, !tbaa !16
  %50 = load i32, ptr %49, align 8, !tbaa !21
  %51 = trunc i64 %37 to i32
  %52 = add i32 %50, %51
  %53 = and i32 %52, 1
  store i32 %53, ptr %38, align 8, !tbaa !21
  %54 = tail call i64 @lrand48() #13
  %55 = srem i64 %54, 1000
  %56 = sitofp i64 %55 to double
  %57 = fdiv double %56, 1.000000e+03
  %58 = fcmp ogt double %2, %57
  %59 = zext i1 %58 to i64
  %60 = xor i64 %37, %59
  %61 = load ptr, ptr %48, align 8, !tbaa !16
  %62 = getelementptr inbounds nuw i8, ptr %61, i64 4
  %63 = load i32, ptr %62, align 4, !tbaa !22
  %64 = trunc i64 %60 to i32
  %65 = add i32 %63, %64
  %66 = and i32 %65, 1
  %67 = getelementptr inbounds nuw i8, ptr %38, i64 4
  store i32 %66, ptr %67, align 4, !tbaa !22
  %68 = ashr i64 %37, 1
  %69 = add nuw nsw i64 %36, 1
  %70 = icmp eq i64 %69, 31
  br i1 %70, label %71, label %35, !llvm.loop !25

71:                                               ; preds = %46
  %72 = add nuw nsw i64 %29, 1
  %73 = icmp eq i64 %72, %18
  br i1 %73, label %74, label %28, !llvm.loop !26

74:                                               ; preds = %71, %21
  %75 = phi ptr [ %1, %21 ], [ %61, %71 ]
  %76 = phi <2 x i32> [ zeroinitializer, %21 ], [ %47, %71 ]
  %77 = tail call i64 @lrand48() #13
  br i1 %14, label %78, label %114

78:                                               ; preds = %74, %89
  %79 = phi i64 [ %112, %89 ], [ 0, %74 ]
  %80 = phi i64 [ %111, %89 ], [ %77, %74 ]
  %81 = phi ptr [ %104, %89 ], [ %75, %74 ]
  %82 = phi <2 x i32> [ %90, %89 ], [ %76, %74 ]
  %83 = getelementptr i8, ptr %20, i64 %79
  %84 = load i8, ptr %83, align 1, !tbaa !10
  %85 = icmp eq i8 %84, 49
  br i1 %85, label %86, label %89

86:                                               ; preds = %78
  %87 = load <2 x i32>, ptr %81, align 8, !tbaa !6
  %88 = add nsw <2 x i32> %87, %82
  br label %89

89:                                               ; preds = %86, %78
  %90 = phi <2 x i32> [ %88, %86 ], [ %82, %78 ]
  %91 = getelementptr inbounds nuw i8, ptr %81, i64 8
  %92 = load ptr, ptr %91, align 8, !tbaa !16
  %93 = load i32, ptr %92, align 8, !tbaa !21
  %94 = trunc i64 %80 to i32
  %95 = add i32 %93, %94
  %96 = and i32 %95, 1
  store i32 %96, ptr %81, align 8, !tbaa !21
  %97 = tail call i64 @lrand48() #13
  %98 = srem i64 %97, 1000
  %99 = sitofp i64 %98 to double
  %100 = fdiv double %99, 1.000000e+03
  %101 = fcmp ogt double %2, %100
  %102 = zext i1 %101 to i64
  %103 = xor i64 %80, %102
  %104 = load ptr, ptr %91, align 8, !tbaa !16
  %105 = getelementptr inbounds nuw i8, ptr %104, i64 4
  %106 = load i32, ptr %105, align 4, !tbaa !22
  %107 = trunc i64 %103 to i32
  %108 = add i32 %106, %107
  %109 = and i32 %108, 1
  %110 = getelementptr inbounds nuw i8, ptr %81, i64 4
  store i32 %109, ptr %110, align 4, !tbaa !22
  %111 = ashr i64 %80, 1
  %112 = add nuw nsw i64 %79, 1
  %113 = icmp eq i64 %112, %19
  br i1 %113, label %114, label %78, !llvm.loop !27

114:                                              ; preds = %89, %74
  %115 = phi ptr [ %75, %74 ], [ %104, %89 ]
  %116 = phi <2 x i32> [ %76, %74 ], [ %90, %89 ]
  %117 = tail call i64 @lrand48() #13
  %118 = load i32, ptr @reg_len, align 4, !tbaa !6
  %119 = sext i32 %118 to i64
  %120 = getelementptr i8, ptr %3, i64 %119
  %121 = getelementptr i8, ptr %120, i64 -1
  %122 = load i8, ptr %121, align 1, !tbaa !10
  %123 = icmp eq i8 %122, 49
  br i1 %123, label %124, label %127

124:                                              ; preds = %114
  %125 = load <2 x i32>, ptr %115, align 8, !tbaa !6
  %126 = add nsw <2 x i32> %125, %116
  br label %127

127:                                              ; preds = %124, %114
  %128 = phi <2 x i32> [ %126, %124 ], [ %116, %114 ]
  %129 = trunc i64 %117 to i32
  %130 = extractelement <2 x i32> %128, i64 0
  %131 = add i32 %130, %129
  %132 = and i32 %131, 1
  store i32 %132, ptr %115, align 8, !tbaa !21
  %133 = tail call i64 @lrand48() #13
  %134 = srem i64 %133, 10000
  %135 = sitofp i64 %134 to double
  %136 = fdiv double %135, 1.000000e+04
  %137 = fcmp ogt double %2, %136
  %138 = zext i1 %137 to i64
  %139 = xor i64 %117, %138
  %140 = trunc i64 %139 to i32
  %141 = extractelement <2 x i32> %128, i64 1
  %142 = add i32 %141, %140
  %143 = and i32 %142, 1
  %144 = getelementptr inbounds nuw i8, ptr %115, i64 4
  store i32 %143, ptr %144, align 4, !tbaa !22
  %145 = add nuw nsw i32 %22, 1
  %146 = icmp eq i32 %145, %0
  br i1 %146, label %23, label %21, !llvm.loop !28

147:                                              ; preds = %25, %147
  %148 = phi i32 [ %158, %147 ], [ 0, %25 ]
  %149 = phi ptr [ %157, %147 ], [ %1, %25 ]
  %150 = phi i32 [ %155, %147 ], [ 0, %25 ]
  %151 = load i32, ptr %149, align 8, !tbaa !21
  %152 = getelementptr inbounds nuw i8, ptr %149, i64 4
  %153 = load i32, ptr %152, align 4, !tbaa !22
  %154 = icmp eq i32 %151, %153
  %155 = select i1 %154, i32 %150, i32 1
  %156 = getelementptr inbounds nuw i8, ptr %149, i64 8
  %157 = load ptr, ptr %156, align 8, !tbaa !16
  %158 = add nuw nsw i32 %148, 1
  %159 = icmp eq i32 %158, %26
  br i1 %159, label %160, label %147, !llvm.loop !29

160:                                              ; preds = %147, %25
  %161 = phi i32 [ 0, %25 ], [ %155, %147 ]
  ret i32 %161
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #8

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #9

; Function Attrs: nounwind uwtable
define dso_local void @kill_list(ptr noundef captures(address_is_null) %0) local_unnamed_addr #0 {
  %2 = icmp eq ptr %0, null
  br i1 %2, label %8, label %3

3:                                                ; preds = %1, %3
  %4 = phi ptr [ %6, %3 ], [ %0, %1 ]
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !16
  tail call void @free(ptr noundef nonnull %4) #13
  %7 = icmp eq ptr %6, null
  br i1 %7, label %8, label %3, !llvm.loop !30

8:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #10

; Function Attrs: nounwind
declare i64 @lrand48() local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #11

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #12

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nounwind memory(readwrite, argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #9 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { nofree nounwind }
attributes #12 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #13 = { nounwind }
attributes #14 = { nounwind willreturn memory(read) }
attributes #15 = { nounwind allocsize(0) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!8, !8, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"p1 omnipotent char", !13, i64 0}
!13 = !{!"any pointer", !8, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"short", !8, i64 0}
!16 = !{!17, !18, i64 8}
!17 = !{!"cells", !7, i64 0, !7, i64 4, !18, i64 8}
!18 = !{!"p1 _ZTS5cells", !13, i64 0}
!19 = distinct !{!19, !20}
!20 = !{!"llvm.loop.mustprogress"}
!21 = !{!17, !7, i64 0}
!22 = !{!17, !7, i64 4}
!23 = distinct !{!23, !20}
!24 = distinct !{!24, !20}
!25 = distinct !{!25, !20}
!26 = distinct !{!26, !20}
!27 = distinct !{!27, !20}
!28 = distinct !{!28, !20}
!29 = distinct !{!29, !20}
!30 = distinct !{!30, !20}
