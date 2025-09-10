; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/Queens.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/Queens.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.element = type { i32, i32 }
%struct.complex = type { float, float }

@seed = dso_local local_unnamed_addr global i64 0, align 8
@.str.1 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
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
@z = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 4
@w = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 4
@e = dso_local local_unnamed_addr global [130 x %struct.complex] zeroinitializer, align 4
@zr = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@zi = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@str = private unnamed_addr constant [18 x i8] c" Error in Queens.\00", align 4

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

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @Try(i32 noundef %0, ptr noundef captures(none) initializes((0, 4)) %1, ptr noundef captures(none) %2, ptr noundef captures(none) %3, ptr noundef captures(none) %4, ptr noundef writeonly captures(none) %5) local_unnamed_addr #2 {
  %7 = sext i32 %0 to i64
  %8 = getelementptr inbounds i32, ptr %5, i64 %7
  %9 = icmp slt i32 %0, 8
  %10 = add nsw i32 %0, 1
  br i1 %9, label %11, label %161

11:                                               ; preds = %6
  %12 = getelementptr i32, ptr %2, i64 %7
  store i32 0, ptr %1, align 4, !tbaa !10
  %13 = getelementptr inbounds nuw i8, ptr %3, i64 4
  %14 = load i32, ptr %13, align 4, !tbaa !10
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %31, label %16

16:                                               ; preds = %11
  %17 = getelementptr i8, ptr %12, i64 4
  %18 = load i32, ptr %17, align 4, !tbaa !10
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %31, label %20

20:                                               ; preds = %16
  %21 = getelementptr i32, ptr %4, i64 %7
  %22 = getelementptr i8, ptr %21, i64 24
  %23 = load i32, ptr %22, align 4, !tbaa !10
  %24 = icmp eq i32 %23, 0
  br i1 %24, label %31, label %25

25:                                               ; preds = %20
  store i32 1, ptr %8, align 4, !tbaa !10
  store i32 0, ptr %13, align 4, !tbaa !10
  store i32 0, ptr %17, align 4, !tbaa !10
  store i32 0, ptr %22, align 4, !tbaa !10
  tail call void @Try(i32 noundef %10, ptr noundef nonnull %1, ptr noundef nonnull %2, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %26 = load i32, ptr %1, align 4, !tbaa !10
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %28, label %288

28:                                               ; preds = %25
  store i32 1, ptr %13, align 4, !tbaa !10
  store i32 1, ptr %17, align 4, !tbaa !10
  store i32 1, ptr %22, align 4, !tbaa !10
  %29 = load i32, ptr %1, align 4, !tbaa !10
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %31, label %288, !llvm.loop !12

31:                                               ; preds = %20, %16, %11, %28
  store i32 0, ptr %1, align 4, !tbaa !10
  %32 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %33 = load i32, ptr %32, align 4, !tbaa !10
  %34 = icmp eq i32 %33, 0
  br i1 %34, label %50, label %35

35:                                               ; preds = %31
  %36 = getelementptr i8, ptr %12, i64 8
  %37 = load i32, ptr %36, align 4, !tbaa !10
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %50, label %39

39:                                               ; preds = %35
  %40 = getelementptr i32, ptr %4, i64 %7
  %41 = getelementptr i8, ptr %40, i64 20
  %42 = load i32, ptr %41, align 4, !tbaa !10
  %43 = icmp eq i32 %42, 0
  br i1 %43, label %50, label %44

44:                                               ; preds = %39
  store i32 2, ptr %8, align 4, !tbaa !10
  store i32 0, ptr %32, align 4, !tbaa !10
  store i32 0, ptr %36, align 4, !tbaa !10
  store i32 0, ptr %41, align 4, !tbaa !10
  tail call void @Try(i32 noundef %10, ptr noundef nonnull %1, ptr noundef nonnull %2, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %45 = load i32, ptr %1, align 4, !tbaa !10
  %46 = icmp eq i32 %45, 0
  br i1 %46, label %47, label %288

47:                                               ; preds = %44
  store i32 1, ptr %32, align 4, !tbaa !10
  store i32 1, ptr %36, align 4, !tbaa !10
  store i32 1, ptr %41, align 4, !tbaa !10
  %48 = load i32, ptr %1, align 4, !tbaa !10
  %49 = icmp eq i32 %48, 0
  br i1 %49, label %50, label %288, !llvm.loop !12

50:                                               ; preds = %39, %35, %31, %47
  store i32 0, ptr %1, align 4, !tbaa !10
  %51 = getelementptr inbounds nuw i8, ptr %3, i64 12
  %52 = load i32, ptr %51, align 4, !tbaa !10
  %53 = icmp eq i32 %52, 0
  br i1 %53, label %69, label %54

54:                                               ; preds = %50
  %55 = getelementptr i8, ptr %12, i64 12
  %56 = load i32, ptr %55, align 4, !tbaa !10
  %57 = icmp eq i32 %56, 0
  br i1 %57, label %69, label %58

58:                                               ; preds = %54
  %59 = getelementptr i32, ptr %4, i64 %7
  %60 = getelementptr i8, ptr %59, i64 16
  %61 = load i32, ptr %60, align 4, !tbaa !10
  %62 = icmp eq i32 %61, 0
  br i1 %62, label %69, label %63

63:                                               ; preds = %58
  store i32 3, ptr %8, align 4, !tbaa !10
  store i32 0, ptr %51, align 4, !tbaa !10
  store i32 0, ptr %55, align 4, !tbaa !10
  store i32 0, ptr %60, align 4, !tbaa !10
  tail call void @Try(i32 noundef %10, ptr noundef nonnull %1, ptr noundef nonnull %2, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %64 = load i32, ptr %1, align 4, !tbaa !10
  %65 = icmp eq i32 %64, 0
  br i1 %65, label %66, label %288

66:                                               ; preds = %63
  store i32 1, ptr %51, align 4, !tbaa !10
  store i32 1, ptr %55, align 4, !tbaa !10
  store i32 1, ptr %60, align 4, !tbaa !10
  %67 = load i32, ptr %1, align 4, !tbaa !10
  %68 = icmp eq i32 %67, 0
  br i1 %68, label %69, label %288, !llvm.loop !12

69:                                               ; preds = %58, %54, %50, %66
  store i32 0, ptr %1, align 4, !tbaa !10
  %70 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %71 = load i32, ptr %70, align 4, !tbaa !10
  %72 = icmp eq i32 %71, 0
  br i1 %72, label %88, label %73

73:                                               ; preds = %69
  %74 = getelementptr i8, ptr %12, i64 16
  %75 = load i32, ptr %74, align 4, !tbaa !10
  %76 = icmp eq i32 %75, 0
  br i1 %76, label %88, label %77

77:                                               ; preds = %73
  %78 = getelementptr i32, ptr %4, i64 %7
  %79 = getelementptr i8, ptr %78, i64 12
  %80 = load i32, ptr %79, align 4, !tbaa !10
  %81 = icmp eq i32 %80, 0
  br i1 %81, label %88, label %82

82:                                               ; preds = %77
  store i32 4, ptr %8, align 4, !tbaa !10
  store i32 0, ptr %70, align 4, !tbaa !10
  store i32 0, ptr %74, align 4, !tbaa !10
  store i32 0, ptr %79, align 4, !tbaa !10
  tail call void @Try(i32 noundef %10, ptr noundef nonnull %1, ptr noundef nonnull %2, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %83 = load i32, ptr %1, align 4, !tbaa !10
  %84 = icmp eq i32 %83, 0
  br i1 %84, label %85, label %288

85:                                               ; preds = %82
  store i32 1, ptr %70, align 4, !tbaa !10
  store i32 1, ptr %74, align 4, !tbaa !10
  store i32 1, ptr %79, align 4, !tbaa !10
  %86 = load i32, ptr %1, align 4, !tbaa !10
  %87 = icmp eq i32 %86, 0
  br i1 %87, label %88, label %288, !llvm.loop !12

88:                                               ; preds = %77, %73, %69, %85
  store i32 0, ptr %1, align 4, !tbaa !10
  %89 = getelementptr inbounds nuw i8, ptr %3, i64 20
  %90 = load i32, ptr %89, align 4, !tbaa !10
  %91 = icmp eq i32 %90, 0
  br i1 %91, label %107, label %92

92:                                               ; preds = %88
  %93 = getelementptr i8, ptr %12, i64 20
  %94 = load i32, ptr %93, align 4, !tbaa !10
  %95 = icmp eq i32 %94, 0
  br i1 %95, label %107, label %96

96:                                               ; preds = %92
  %97 = getelementptr i32, ptr %4, i64 %7
  %98 = getelementptr i8, ptr %97, i64 8
  %99 = load i32, ptr %98, align 4, !tbaa !10
  %100 = icmp eq i32 %99, 0
  br i1 %100, label %107, label %101

101:                                              ; preds = %96
  store i32 5, ptr %8, align 4, !tbaa !10
  store i32 0, ptr %89, align 4, !tbaa !10
  store i32 0, ptr %93, align 4, !tbaa !10
  store i32 0, ptr %98, align 4, !tbaa !10
  tail call void @Try(i32 noundef %10, ptr noundef nonnull %1, ptr noundef nonnull %2, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %102 = load i32, ptr %1, align 4, !tbaa !10
  %103 = icmp eq i32 %102, 0
  br i1 %103, label %104, label %288

104:                                              ; preds = %101
  store i32 1, ptr %89, align 4, !tbaa !10
  store i32 1, ptr %93, align 4, !tbaa !10
  store i32 1, ptr %98, align 4, !tbaa !10
  %105 = load i32, ptr %1, align 4, !tbaa !10
  %106 = icmp eq i32 %105, 0
  br i1 %106, label %107, label %288, !llvm.loop !12

107:                                              ; preds = %96, %92, %88, %104
  store i32 0, ptr %1, align 4, !tbaa !10
  %108 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %109 = load i32, ptr %108, align 4, !tbaa !10
  %110 = icmp eq i32 %109, 0
  br i1 %110, label %126, label %111

111:                                              ; preds = %107
  %112 = getelementptr i8, ptr %12, i64 24
  %113 = load i32, ptr %112, align 4, !tbaa !10
  %114 = icmp eq i32 %113, 0
  br i1 %114, label %126, label %115

115:                                              ; preds = %111
  %116 = getelementptr i32, ptr %4, i64 %7
  %117 = getelementptr i8, ptr %116, i64 4
  %118 = load i32, ptr %117, align 4, !tbaa !10
  %119 = icmp eq i32 %118, 0
  br i1 %119, label %126, label %120

120:                                              ; preds = %115
  store i32 6, ptr %8, align 4, !tbaa !10
  store i32 0, ptr %108, align 4, !tbaa !10
  store i32 0, ptr %112, align 4, !tbaa !10
  store i32 0, ptr %117, align 4, !tbaa !10
  tail call void @Try(i32 noundef %10, ptr noundef nonnull %1, ptr noundef nonnull %2, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %121 = load i32, ptr %1, align 4, !tbaa !10
  %122 = icmp eq i32 %121, 0
  br i1 %122, label %123, label %288

123:                                              ; preds = %120
  store i32 1, ptr %108, align 4, !tbaa !10
  store i32 1, ptr %112, align 4, !tbaa !10
  store i32 1, ptr %117, align 4, !tbaa !10
  %124 = load i32, ptr %1, align 4, !tbaa !10
  %125 = icmp eq i32 %124, 0
  br i1 %125, label %126, label %288, !llvm.loop !12

126:                                              ; preds = %115, %111, %107, %123
  store i32 0, ptr %1, align 4, !tbaa !10
  %127 = getelementptr inbounds nuw i8, ptr %3, i64 28
  %128 = load i32, ptr %127, align 4, !tbaa !10
  %129 = icmp eq i32 %128, 0
  br i1 %129, label %144, label %130

130:                                              ; preds = %126
  %131 = getelementptr i8, ptr %12, i64 28
  %132 = load i32, ptr %131, align 4, !tbaa !10
  %133 = icmp eq i32 %132, 0
  br i1 %133, label %144, label %134

134:                                              ; preds = %130
  %135 = getelementptr i32, ptr %4, i64 %7
  %136 = load i32, ptr %135, align 4, !tbaa !10
  %137 = icmp eq i32 %136, 0
  br i1 %137, label %144, label %138

138:                                              ; preds = %134
  store i32 7, ptr %8, align 4, !tbaa !10
  store i32 0, ptr %127, align 4, !tbaa !10
  store i32 0, ptr %131, align 4, !tbaa !10
  store i32 0, ptr %135, align 4, !tbaa !10
  tail call void @Try(i32 noundef %10, ptr noundef nonnull %1, ptr noundef nonnull %2, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %139 = load i32, ptr %1, align 4, !tbaa !10
  %140 = icmp eq i32 %139, 0
  br i1 %140, label %141, label %288

141:                                              ; preds = %138
  store i32 1, ptr %127, align 4, !tbaa !10
  store i32 1, ptr %131, align 4, !tbaa !10
  store i32 1, ptr %135, align 4, !tbaa !10
  %142 = load i32, ptr %1, align 4, !tbaa !10
  %143 = icmp eq i32 %142, 0
  br i1 %143, label %144, label %288, !llvm.loop !12

144:                                              ; preds = %134, %130, %126, %141
  store i32 0, ptr %1, align 4, !tbaa !10
  %145 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %146 = load i32, ptr %145, align 4, !tbaa !10
  %147 = icmp eq i32 %146, 0
  br i1 %147, label %288, label %148

148:                                              ; preds = %144
  %149 = getelementptr i8, ptr %12, i64 32
  %150 = load i32, ptr %149, align 4, !tbaa !10
  %151 = icmp eq i32 %150, 0
  br i1 %151, label %288, label %152

152:                                              ; preds = %148
  %153 = getelementptr i32, ptr %4, i64 %7
  %154 = getelementptr i8, ptr %153, i64 -4
  %155 = load i32, ptr %154, align 4, !tbaa !10
  %156 = icmp eq i32 %155, 0
  br i1 %156, label %288, label %157

157:                                              ; preds = %152
  store i32 8, ptr %8, align 4, !tbaa !10
  store i32 0, ptr %145, align 4, !tbaa !10
  store i32 0, ptr %149, align 4, !tbaa !10
  store i32 0, ptr %154, align 4, !tbaa !10
  tail call void @Try(i32 noundef %10, ptr noundef nonnull %1, ptr noundef nonnull %2, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %158 = load i32, ptr %1, align 4, !tbaa !10
  %159 = icmp eq i32 %158, 0
  br i1 %159, label %160, label %288

160:                                              ; preds = %157
  store i32 1, ptr %145, align 4, !tbaa !10
  store i32 1, ptr %149, align 4, !tbaa !10
  store i32 1, ptr %154, align 4, !tbaa !10
  br label %288

161:                                              ; preds = %6
  store i32 0, ptr %1, align 4, !tbaa !10
  %162 = getelementptr inbounds nuw i8, ptr %3, i64 4
  %163 = load i32, ptr %162, align 4, !tbaa !10
  %164 = icmp eq i32 %163, 0
  br i1 %164, label %184, label %165

165:                                              ; preds = %161
  %166 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %167 = getelementptr inbounds nuw i32, ptr %166, i64 %7
  %168 = load i32, ptr %167, align 4, !tbaa !10
  %169 = icmp eq i32 %168, 0
  br i1 %169, label %184, label %170

170:                                              ; preds = %165
  %171 = getelementptr i32, ptr %4, i64 %7
  %172 = getelementptr i8, ptr %171, i64 -4
  %173 = getelementptr i8, ptr %171, i64 24
  %174 = load i32, ptr %173, align 4, !tbaa !10
  %175 = icmp eq i32 %174, 0
  br i1 %175, label %184, label %176

176:                                              ; preds = %282, %268, %253, %238, %223, %208, %193, %170
  %177 = phi i64 [ 1, %170 ], [ 2, %193 ], [ 3, %208 ], [ 4, %223 ], [ 5, %238 ], [ 6, %253 ], [ 7, %268 ], [ 8, %282 ]
  %178 = phi ptr [ %172, %170 ], [ %195, %193 ], [ %210, %208 ], [ %225, %223 ], [ %240, %238 ], [ %255, %253 ], [ %270, %268 ], [ %284, %282 ]
  %179 = phi ptr [ %166, %170 ], [ %189, %193 ], [ %204, %208 ], [ %219, %223 ], [ %234, %238 ], [ %249, %253 ], [ %264, %268 ], [ %278, %282 ]
  %180 = getelementptr inbounds nuw i32, ptr %3, i64 %177
  %181 = getelementptr inbounds nuw i32, ptr %179, i64 %7
  %182 = getelementptr i8, ptr %178, i64 28
  %183 = trunc nuw nsw i64 %177 to i32
  store i32 %183, ptr %8, align 4, !tbaa !10
  store i32 0, ptr %180, align 4, !tbaa !10
  store i32 0, ptr %181, align 4, !tbaa !10
  store i32 0, ptr %182, align 4, !tbaa !10
  store i32 1, ptr %1, align 4, !tbaa !10
  br label %288

184:                                              ; preds = %170, %165, %161
  store i32 0, ptr %1, align 4, !tbaa !10
  %185 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %186 = load i32, ptr %185, align 4, !tbaa !10
  %187 = icmp eq i32 %186, 0
  br i1 %187, label %199, label %188

188:                                              ; preds = %184
  %189 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %190 = getelementptr inbounds nuw i32, ptr %189, i64 %7
  %191 = load i32, ptr %190, align 4, !tbaa !10
  %192 = icmp eq i32 %191, 0
  br i1 %192, label %199, label %193

193:                                              ; preds = %188
  %194 = getelementptr i32, ptr %4, i64 %7
  %195 = getelementptr i8, ptr %194, i64 -8
  %196 = getelementptr i8, ptr %194, i64 20
  %197 = load i32, ptr %196, align 4, !tbaa !10
  %198 = icmp eq i32 %197, 0
  br i1 %198, label %199, label %176

199:                                              ; preds = %193, %188, %184
  store i32 0, ptr %1, align 4, !tbaa !10
  %200 = getelementptr inbounds nuw i8, ptr %3, i64 12
  %201 = load i32, ptr %200, align 4, !tbaa !10
  %202 = icmp eq i32 %201, 0
  br i1 %202, label %214, label %203

203:                                              ; preds = %199
  %204 = getelementptr inbounds nuw i8, ptr %2, i64 12
  %205 = getelementptr inbounds nuw i32, ptr %204, i64 %7
  %206 = load i32, ptr %205, align 4, !tbaa !10
  %207 = icmp eq i32 %206, 0
  br i1 %207, label %214, label %208

208:                                              ; preds = %203
  %209 = getelementptr i32, ptr %4, i64 %7
  %210 = getelementptr i8, ptr %209, i64 -12
  %211 = getelementptr i8, ptr %209, i64 16
  %212 = load i32, ptr %211, align 4, !tbaa !10
  %213 = icmp eq i32 %212, 0
  br i1 %213, label %214, label %176

214:                                              ; preds = %208, %203, %199
  store i32 0, ptr %1, align 4, !tbaa !10
  %215 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %216 = load i32, ptr %215, align 4, !tbaa !10
  %217 = icmp eq i32 %216, 0
  br i1 %217, label %229, label %218

218:                                              ; preds = %214
  %219 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %220 = getelementptr inbounds nuw i32, ptr %219, i64 %7
  %221 = load i32, ptr %220, align 4, !tbaa !10
  %222 = icmp eq i32 %221, 0
  br i1 %222, label %229, label %223

223:                                              ; preds = %218
  %224 = getelementptr i32, ptr %4, i64 %7
  %225 = getelementptr i8, ptr %224, i64 -16
  %226 = getelementptr i8, ptr %224, i64 12
  %227 = load i32, ptr %226, align 4, !tbaa !10
  %228 = icmp eq i32 %227, 0
  br i1 %228, label %229, label %176

229:                                              ; preds = %223, %218, %214
  store i32 0, ptr %1, align 4, !tbaa !10
  %230 = getelementptr inbounds nuw i8, ptr %3, i64 20
  %231 = load i32, ptr %230, align 4, !tbaa !10
  %232 = icmp eq i32 %231, 0
  br i1 %232, label %244, label %233

233:                                              ; preds = %229
  %234 = getelementptr inbounds nuw i8, ptr %2, i64 20
  %235 = getelementptr inbounds nuw i32, ptr %234, i64 %7
  %236 = load i32, ptr %235, align 4, !tbaa !10
  %237 = icmp eq i32 %236, 0
  br i1 %237, label %244, label %238

238:                                              ; preds = %233
  %239 = getelementptr i32, ptr %4, i64 %7
  %240 = getelementptr i8, ptr %239, i64 -20
  %241 = getelementptr i8, ptr %239, i64 8
  %242 = load i32, ptr %241, align 4, !tbaa !10
  %243 = icmp eq i32 %242, 0
  br i1 %243, label %244, label %176

244:                                              ; preds = %238, %233, %229
  store i32 0, ptr %1, align 4, !tbaa !10
  %245 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %246 = load i32, ptr %245, align 4, !tbaa !10
  %247 = icmp eq i32 %246, 0
  br i1 %247, label %259, label %248

248:                                              ; preds = %244
  %249 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %250 = getelementptr inbounds nuw i32, ptr %249, i64 %7
  %251 = load i32, ptr %250, align 4, !tbaa !10
  %252 = icmp eq i32 %251, 0
  br i1 %252, label %259, label %253

253:                                              ; preds = %248
  %254 = getelementptr i32, ptr %4, i64 %7
  %255 = getelementptr i8, ptr %254, i64 -24
  %256 = getelementptr i8, ptr %254, i64 4
  %257 = load i32, ptr %256, align 4, !tbaa !10
  %258 = icmp eq i32 %257, 0
  br i1 %258, label %259, label %176

259:                                              ; preds = %253, %248, %244
  store i32 0, ptr %1, align 4, !tbaa !10
  %260 = getelementptr inbounds nuw i8, ptr %3, i64 28
  %261 = load i32, ptr %260, align 4, !tbaa !10
  %262 = icmp eq i32 %261, 0
  br i1 %262, label %273, label %263

263:                                              ; preds = %259
  %264 = getelementptr inbounds nuw i8, ptr %2, i64 28
  %265 = getelementptr inbounds nuw i32, ptr %264, i64 %7
  %266 = load i32, ptr %265, align 4, !tbaa !10
  %267 = icmp eq i32 %266, 0
  br i1 %267, label %273, label %268

268:                                              ; preds = %263
  %269 = getelementptr i32, ptr %4, i64 %7
  %270 = getelementptr i8, ptr %269, i64 -28
  %271 = load i32, ptr %269, align 4, !tbaa !10
  %272 = icmp eq i32 %271, 0
  br i1 %272, label %273, label %176

273:                                              ; preds = %268, %263, %259
  store i32 0, ptr %1, align 4, !tbaa !10
  %274 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %275 = load i32, ptr %274, align 4, !tbaa !10
  %276 = icmp eq i32 %275, 0
  br i1 %276, label %288, label %277

277:                                              ; preds = %273
  %278 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %279 = getelementptr inbounds nuw i32, ptr %278, i64 %7
  %280 = load i32, ptr %279, align 4, !tbaa !10
  %281 = icmp eq i32 %280, 0
  br i1 %281, label %288, label %282

282:                                              ; preds = %277
  %283 = getelementptr i32, ptr %4, i64 %7
  %284 = getelementptr i8, ptr %283, i64 -32
  %285 = getelementptr i8, ptr %283, i64 -4
  %286 = load i32, ptr %285, align 4, !tbaa !10
  %287 = icmp eq i32 %286, 0
  br i1 %287, label %288, label %176

288:                                              ; preds = %25, %28, %44, %47, %63, %66, %82, %85, %101, %104, %120, %123, %138, %141, %157, %160, %152, %148, %144, %273, %277, %282, %176
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: nofree nounwind uwtable
define dso_local void @Doit() local_unnamed_addr #4 {
  %1 = alloca i32, align 4
  %2 = alloca [9 x i32], align 4
  %3 = alloca [17 x i32], align 4
  %4 = alloca [15 x i32], align 4
  %5 = alloca [9 x i32], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #7
  br label %6

6:                                                ; preds = %0, %21
  %7 = phi i64 [ -7, %0 ], [ %22, %21 ]
  %8 = trunc i64 %7 to i32
  %9 = add i32 %8, -1
  %10 = icmp ult i32 %9, 8
  br i1 %10, label %11, label %13

11:                                               ; preds = %6
  %12 = getelementptr inbounds nuw i32, ptr %2, i64 %7
  store i32 1, ptr %12, align 4, !tbaa !10
  br label %13

13:                                               ; preds = %11, %6
  %14 = icmp sgt i64 %7, 1
  br i1 %14, label %15, label %18

15:                                               ; preds = %13
  %16 = getelementptr inbounds nuw i32, ptr %3, i64 %7
  store i32 1, ptr %16, align 4, !tbaa !10
  %17 = icmp samesign ult i64 %7, 8
  br i1 %17, label %18, label %21

18:                                               ; preds = %13, %15
  %19 = getelementptr i32, ptr %4, i64 %7
  %20 = getelementptr i8, ptr %19, i64 28
  store i32 1, ptr %20, align 4, !tbaa !10
  br label %21

21:                                               ; preds = %18, %15
  %22 = add nsw i64 %7, 1
  %23 = icmp eq i64 %22, 17
  br i1 %23, label %24, label %6, !llvm.loop !14

24:                                               ; preds = %21
  call void @Try(i32 noundef 1, ptr noundef nonnull %1, ptr noundef nonnull %3, ptr noundef nonnull %2, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %25 = load i32, ptr %1, align 4, !tbaa !10
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %27, label %29

27:                                               ; preds = %24
  %28 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %29

29:                                               ; preds = %27, %24
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #5

; Function Attrs: nofree nounwind uwtable
define dso_local void @Queens(i32 noundef %0) local_unnamed_addr #4 {
  %2 = alloca i32, align 4
  %3 = alloca [9 x i32], align 4
  %4 = alloca [17 x i32], align 4
  %5 = alloca [15 x i32], align 4
  %6 = alloca [9 x i32], align 4
  br label %7

7:                                                ; preds = %1, %32
  %8 = phi i32 [ 1, %1 ], [ %33, %32 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #7
  br label %9

9:                                                ; preds = %24, %7
  %10 = phi i64 [ -7, %7 ], [ %25, %24 ]
  %11 = trunc i64 %10 to i32
  %12 = add i32 %11, -1
  %13 = icmp ult i32 %12, 8
  br i1 %13, label %14, label %16

14:                                               ; preds = %9
  %15 = getelementptr inbounds nuw i32, ptr %3, i64 %10
  store i32 1, ptr %15, align 4, !tbaa !10
  br label %16

16:                                               ; preds = %14, %9
  %17 = icmp sgt i64 %10, 1
  br i1 %17, label %18, label %21

18:                                               ; preds = %16
  %19 = getelementptr inbounds nuw i32, ptr %4, i64 %10
  store i32 1, ptr %19, align 4, !tbaa !10
  %20 = icmp samesign ult i64 %10, 8
  br i1 %20, label %21, label %24

21:                                               ; preds = %18, %16
  %22 = getelementptr i32, ptr %5, i64 %10
  %23 = getelementptr i8, ptr %22, i64 28
  store i32 1, ptr %23, align 4, !tbaa !10
  br label %24

24:                                               ; preds = %21, %18
  %25 = add nsw i64 %10, 1
  %26 = icmp eq i64 %25, 17
  br i1 %26, label %27, label %9, !llvm.loop !14

27:                                               ; preds = %24
  call void @Try(i32 noundef 1, ptr noundef nonnull %2, ptr noundef nonnull %4, ptr noundef nonnull %3, ptr noundef nonnull %5, ptr noundef nonnull %6)
  %28 = load i32, ptr %2, align 4, !tbaa !10
  %29 = icmp eq i32 %28, 0
  br i1 %29, label %30, label %32

30:                                               ; preds = %27
  %31 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %32

32:                                               ; preds = %27, %30
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  %33 = add nuw nsw i32 %8, 1
  %34 = icmp eq i32 %33, 51
  br i1 %34, label %35, label %7, !llvm.loop !15

35:                                               ; preds = %32
  %36 = add nsw i32 %0, 1
  %37 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %36)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  %1 = alloca i32, align 4
  %2 = alloca [9 x i32], align 4
  %3 = alloca [17 x i32], align 4
  %4 = alloca [15 x i32], align 4
  %5 = alloca [9 x i32], align 4
  br label %6

6:                                                ; preds = %0, %36
  %7 = phi i32 [ 0, %0 ], [ %37, %36 ]
  br label %8

8:                                                ; preds = %6, %33
  %9 = phi i32 [ %34, %33 ], [ 1, %6 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #7
  br label %10

10:                                               ; preds = %25, %8
  %11 = phi i64 [ -7, %8 ], [ %26, %25 ]
  %12 = trunc i64 %11 to i32
  %13 = add i32 %12, -1
  %14 = icmp ult i32 %13, 8
  br i1 %14, label %15, label %17

15:                                               ; preds = %10
  %16 = getelementptr inbounds nuw i32, ptr %2, i64 %11
  store i32 1, ptr %16, align 4, !tbaa !10
  br label %17

17:                                               ; preds = %15, %10
  %18 = icmp sgt i64 %11, 1
  br i1 %18, label %19, label %22

19:                                               ; preds = %17
  %20 = getelementptr inbounds nuw i32, ptr %3, i64 %11
  store i32 1, ptr %20, align 4, !tbaa !10
  %21 = icmp samesign ult i64 %11, 8
  br i1 %21, label %22, label %25

22:                                               ; preds = %19, %17
  %23 = getelementptr i32, ptr %4, i64 %11
  %24 = getelementptr i8, ptr %23, i64 28
  store i32 1, ptr %24, align 4, !tbaa !10
  br label %25

25:                                               ; preds = %22, %19
  %26 = add nsw i64 %11, 1
  %27 = icmp eq i64 %26, 17
  br i1 %27, label %28, label %10, !llvm.loop !14

28:                                               ; preds = %25
  call void @Try(i32 noundef 1, ptr noundef nonnull %1, ptr noundef nonnull %3, ptr noundef nonnull %2, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %29 = load i32, ptr %1, align 4, !tbaa !10
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %31, label %33

31:                                               ; preds = %28
  %32 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %33

33:                                               ; preds = %31, %28
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  %34 = add nuw nsw i32 %9, 1
  %35 = icmp eq i32 %34, 51
  br i1 %35, label %36, label %8, !llvm.loop !15

36:                                               ; preds = %33
  %37 = add nuw nsw i32 %7, 1
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %37)
  %39 = icmp eq i32 %37, 100
  br i1 %39, label %40, label %6, !llvm.loop !16

40:                                               ; preds = %36
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nounwind }
attributes #7 = { nounwind }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13}
!15 = distinct !{!15, !13}
!16 = distinct !{!16, !13}
