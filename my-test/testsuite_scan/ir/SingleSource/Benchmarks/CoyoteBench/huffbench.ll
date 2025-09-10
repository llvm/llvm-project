; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/CoyoteBench/huffbench.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/CoyoteBench/huffbench.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [33 x i8] c"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345\00", align 1
@stderr = external local_unnamed_addr global ptr, align 8
@.str.1 = private unnamed_addr constant [26 x i8] c"error: bit code overflow\0A\00", align 1
@.str.2 = private unnamed_addr constant [33 x i8] c"error: file has only one value!\0A\00", align 1
@.str.3 = private unnamed_addr constant [23 x i8] c"error: no compression\0A\00", align 1
@.str.4 = private unnamed_addr constant [4 x i8] c"-ga\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8
@.str.5 = private unnamed_addr constant [3 x i8] c"%f\00", align 1
@.str.6 = private unnamed_addr constant [35 x i8] c"\0Ahuffbench (Std. C) run time: %f\0A\0A\00", align 1
@seed = internal unnamed_addr global i64 1325, align 8

; Function Attrs: nofree nounwind memory(readwrite, argmem: none) uwtable
define dso_local noalias noundef ptr @generate_test_data(i64 noundef %0) local_unnamed_addr #0 {
  %2 = tail call noalias ptr @malloc(i64 noundef %0) #12
  %3 = icmp eq i64 %0, 0
  br i1 %3, label %28, label %4

4:                                                ; preds = %1
  %5 = load i64, ptr @seed, align 8
  %6 = xor i64 %5, 123459876
  br label %7

7:                                                ; preds = %4, %7
  %8 = phi i64 [ 0, %4 ], [ %24, %7 ]
  %9 = phi ptr [ %2, %4 ], [ %23, %7 ]
  %10 = phi i64 [ %6, %4 ], [ %19, %7 ]
  %11 = sdiv i64 %10, 127773
  %12 = mul nsw i64 %11, -127773
  %13 = add i64 %12, %10
  %14 = mul nsw i64 %13, 16807
  %15 = mul nsw i64 %11, -2836
  %16 = add i64 %14, %15
  %17 = icmp slt i64 %16, 0
  %18 = add nsw i64 %16, 2147483647
  %19 = select i1 %17, i64 %18, i64 %16
  %20 = srem i64 %19, 32
  %21 = getelementptr inbounds nuw i8, ptr @.str, i64 %20
  %22 = load i8, ptr %21, align 1, !tbaa !6
  store i8 %22, ptr %9, align 1, !tbaa !6
  %23 = getelementptr inbounds nuw i8, ptr %9, i64 1
  %24 = add nuw nsw i64 %8, 1
  %25 = icmp eq i64 %24, %0
  br i1 %25, label %26, label %7, !llvm.loop !9

26:                                               ; preds = %7
  %27 = xor i64 %19, 123459876
  store i64 %27, ptr @seed, align 8, !tbaa !11
  br label %28

28:                                               ; preds = %26, %1
  ret ptr %2
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind uwtable
define dso_local void @compdecomp(ptr noundef captures(none) %0, i64 noundef %1) local_unnamed_addr #3 {
  %3 = alloca [512 x i64], align 8
  %4 = alloca [256 x i64], align 8
  %5 = alloca [512 x i32], align 4
  %6 = alloca [256 x i64], align 8
  %7 = alloca [256 x i8], align 1
  %8 = alloca [256 x i64], align 8
  %9 = alloca [256 x i8], align 1
  %10 = add i64 %1, 1
  %11 = tail call ptr @calloc(i64 1, i64 %10)
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #13
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #13
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #13
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #13
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #13
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(4096) %3, i8 0, i64 4096, i1 false)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(2048) %4, i8 0, i64 2048, i1 false)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(2048) %5, i8 0, i64 2048, i1 false)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(2048) %6, i8 0, i64 2048, i1 false)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(256) %7, i8 0, i64 256, i1 false)
  %12 = icmp eq i64 %1, 0
  br i1 %12, label %13, label %14

13:                                               ; preds = %14, %2
  br label %30

14:                                               ; preds = %2, %14
  %15 = phi i64 [ %23, %14 ], [ 0, %2 ]
  %16 = phi ptr [ %22, %14 ], [ %0, %2 ]
  %17 = load i8, ptr %16, align 1, !tbaa !6
  %18 = zext i8 %17 to i64
  %19 = getelementptr inbounds nuw i64, ptr %3, i64 %18
  %20 = load i64, ptr %19, align 8, !tbaa !11
  %21 = add i64 %20, 1
  store i64 %21, ptr %19, align 8, !tbaa !11
  %22 = getelementptr inbounds nuw i8, ptr %16, i64 1
  %23 = add nuw i64 %15, 1
  %24 = icmp eq i64 %23, %1
  br i1 %24, label %13, label %14, !llvm.loop !13

25:                                               ; preds = %39
  %26 = icmp eq i64 %40, 0
  br i1 %26, label %202, label %27

27:                                               ; preds = %25
  %28 = trunc i64 %40 to i32
  %29 = sdiv i32 %28, 2
  br label %45

30:                                               ; preds = %13, %39
  %31 = phi i64 [ %41, %39 ], [ 0, %13 ]
  %32 = phi i64 [ %40, %39 ], [ 0, %13 ]
  %33 = getelementptr inbounds nuw i64, ptr %3, i64 %31
  %34 = load i64, ptr %33, align 8, !tbaa !11
  %35 = icmp eq i64 %34, 0
  br i1 %35, label %39, label %36

36:                                               ; preds = %30
  %37 = getelementptr inbounds nuw i64, ptr %4, i64 %32
  store i64 %31, ptr %37, align 8, !tbaa !11
  %38 = add i64 %32, 1
  br label %39

39:                                               ; preds = %30, %36
  %40 = phi i64 [ %38, %36 ], [ %32, %30 ]
  %41 = add nuw nsw i64 %31, 1
  %42 = icmp eq i64 %41, 256
  br i1 %42, label %25, label %30, !llvm.loop !14

43:                                               ; preds = %90
  %44 = icmp eq i64 %40, 1
  br i1 %44, label %202, label %97

45:                                               ; preds = %27, %90
  %46 = phi i64 [ %40, %27 ], [ %95, %90 ]
  %47 = trunc i64 %46 to i32
  %48 = shl i64 %46, 32
  %49 = ashr exact i64 %48, 32
  %50 = getelementptr i64, ptr %4, i64 %49
  %51 = getelementptr i8, ptr %50, i64 -8
  %52 = load i64, ptr %51, align 8, !tbaa !11
  %53 = icmp slt i32 %29, %47
  %54 = shl i64 %52, 32
  br i1 %53, label %90, label %55

55:                                               ; preds = %45
  %56 = ashr exact i64 %54, 29
  %57 = getelementptr inbounds i8, ptr %3, i64 %56
  %58 = load i64, ptr %57, align 8, !tbaa !11
  br label %59

59:                                               ; preds = %86, %55
  %60 = phi i32 [ %47, %55 ], [ %77, %86 ]
  %61 = shl nsw i32 %60, 1
  %62 = icmp slt i32 %61, %28
  br i1 %62, label %63, label %76

63:                                               ; preds = %59
  %64 = sext i32 %61 to i64
  %65 = getelementptr i64, ptr %4, i64 %64
  %66 = getelementptr i8, ptr %65, i64 -8
  %67 = load i64, ptr %66, align 8, !tbaa !11
  %68 = getelementptr inbounds nuw i64, ptr %3, i64 %67
  %69 = load i64, ptr %68, align 8, !tbaa !11
  %70 = load i64, ptr %65, align 8, !tbaa !11
  %71 = getelementptr inbounds nuw i64, ptr %3, i64 %70
  %72 = load i64, ptr %71, align 8, !tbaa !11
  %73 = icmp ugt i64 %69, %72
  %74 = zext i1 %73 to i32
  %75 = or disjoint i32 %61, %74
  br label %76

76:                                               ; preds = %63, %59
  %77 = phi i32 [ %61, %59 ], [ %75, %63 ]
  %78 = sext i32 %77 to i64
  %79 = getelementptr i64, ptr %4, i64 %78
  %80 = getelementptr i8, ptr %79, i64 -8
  %81 = load i64, ptr %80, align 8, !tbaa !11
  %82 = getelementptr inbounds nuw i64, ptr %3, i64 %81
  %83 = load i64, ptr %82, align 8, !tbaa !11
  %84 = icmp ult i64 %58, %83
  %85 = sext i32 %60 to i64
  br i1 %84, label %90, label %86

86:                                               ; preds = %76
  %87 = getelementptr i64, ptr %4, i64 %85
  %88 = getelementptr i8, ptr %87, i64 -8
  store i64 %81, ptr %88, align 8, !tbaa !11
  %89 = icmp sgt i32 %77, %29
  br i1 %89, label %90, label %59, !llvm.loop !15

90:                                               ; preds = %86, %76, %45
  %91 = phi i64 [ %49, %45 ], [ %78, %86 ], [ %85, %76 ]
  %92 = ashr exact i64 %54, 32
  %93 = getelementptr i64, ptr %4, i64 %91
  %94 = getelementptr i8, ptr %93, i64 -8
  store i64 %92, ptr %94, align 8, !tbaa !11
  %95 = add i64 %46, -1
  %96 = icmp eq i64 %95, 0
  br i1 %96, label %43, label %45, !llvm.loop !16

97:                                               ; preds = %43, %196
  %98 = phi i64 [ %99, %196 ], [ %40, %43 ]
  %99 = add i64 %98, -1
  %100 = load i64, ptr %4, align 8, !tbaa !11
  %101 = getelementptr inbounds nuw i64, ptr %4, i64 %99
  %102 = load i64, ptr %101, align 8, !tbaa !11
  store i64 %102, ptr %4, align 8, !tbaa !11
  %103 = trunc i64 %99 to i32
  %104 = sdiv i32 %103, 2
  %105 = icmp slt i32 %103, 2
  %106 = shl i64 %102, 32
  br i1 %105, label %142, label %107

107:                                              ; preds = %97
  %108 = ashr exact i64 %106, 29
  %109 = getelementptr inbounds i8, ptr %3, i64 %108
  %110 = load i64, ptr %109, align 8, !tbaa !11
  br label %111

111:                                              ; preds = %138, %107
  %112 = phi i32 [ 1, %107 ], [ %129, %138 ]
  %113 = shl nsw i32 %112, 1
  %114 = icmp slt i32 %113, %103
  br i1 %114, label %115, label %128

115:                                              ; preds = %111
  %116 = sext i32 %113 to i64
  %117 = getelementptr i64, ptr %4, i64 %116
  %118 = getelementptr i8, ptr %117, i64 -8
  %119 = load i64, ptr %118, align 8, !tbaa !11
  %120 = getelementptr inbounds nuw i64, ptr %3, i64 %119
  %121 = load i64, ptr %120, align 8, !tbaa !11
  %122 = load i64, ptr %117, align 8, !tbaa !11
  %123 = getelementptr inbounds nuw i64, ptr %3, i64 %122
  %124 = load i64, ptr %123, align 8, !tbaa !11
  %125 = icmp ugt i64 %121, %124
  %126 = zext i1 %125 to i32
  %127 = or disjoint i32 %113, %126
  br label %128

128:                                              ; preds = %115, %111
  %129 = phi i32 [ %113, %111 ], [ %127, %115 ]
  %130 = sext i32 %129 to i64
  %131 = getelementptr i64, ptr %4, i64 %130
  %132 = getelementptr i8, ptr %131, i64 -8
  %133 = load i64, ptr %132, align 8, !tbaa !11
  %134 = getelementptr inbounds nuw i64, ptr %3, i64 %133
  %135 = load i64, ptr %134, align 8, !tbaa !11
  %136 = icmp ult i64 %110, %135
  %137 = sext i32 %112 to i64
  br i1 %136, label %142, label %138

138:                                              ; preds = %128
  %139 = getelementptr i64, ptr %4, i64 %137
  %140 = getelementptr i8, ptr %139, i64 -8
  store i64 %133, ptr %140, align 8, !tbaa !11
  %141 = icmp sgt i32 %129, %104
  br i1 %141, label %142, label %111, !llvm.loop !15

142:                                              ; preds = %138, %128, %97
  %143 = phi i64 [ 1, %97 ], [ %130, %138 ], [ %137, %128 ]
  %144 = ashr exact i64 %106, 32
  %145 = getelementptr i64, ptr %4, i64 %143
  %146 = getelementptr i8, ptr %145, i64 -8
  store i64 %144, ptr %146, align 8, !tbaa !11
  %147 = load i64, ptr %4, align 8, !tbaa !11
  %148 = getelementptr inbounds nuw i64, ptr %3, i64 %147
  %149 = load i64, ptr %148, align 8, !tbaa !11
  %150 = getelementptr inbounds nuw i64, ptr %3, i64 %100
  %151 = load i64, ptr %150, align 8, !tbaa !11
  %152 = add i64 %151, %149
  %153 = add i64 %98, 255
  %154 = getelementptr inbounds nuw i64, ptr %3, i64 %153
  store i64 %152, ptr %154, align 8, !tbaa !11
  %155 = trunc i64 %153 to i32
  %156 = getelementptr inbounds nuw i32, ptr %5, i64 %100
  store i32 %155, ptr %156, align 4, !tbaa !17
  %157 = trunc i64 %98 to i32
  %158 = sub i32 -255, %157
  %159 = getelementptr inbounds nuw i32, ptr %5, i64 %147
  store i32 %158, ptr %159, align 4, !tbaa !17
  store i64 %153, ptr %4, align 8, !tbaa !11
  %160 = shl i64 %153, 32
  br i1 %105, label %196, label %161

161:                                              ; preds = %142
  %162 = ashr exact i64 %160, 29
  %163 = getelementptr inbounds i8, ptr %3, i64 %162
  %164 = load i64, ptr %163, align 8, !tbaa !11
  br label %165

165:                                              ; preds = %192, %161
  %166 = phi i32 [ 1, %161 ], [ %183, %192 ]
  %167 = shl nsw i32 %166, 1
  %168 = icmp slt i32 %167, %103
  br i1 %168, label %169, label %182

169:                                              ; preds = %165
  %170 = sext i32 %167 to i64
  %171 = getelementptr i64, ptr %4, i64 %170
  %172 = getelementptr i8, ptr %171, i64 -8
  %173 = load i64, ptr %172, align 8, !tbaa !11
  %174 = getelementptr inbounds nuw i64, ptr %3, i64 %173
  %175 = load i64, ptr %174, align 8, !tbaa !11
  %176 = load i64, ptr %171, align 8, !tbaa !11
  %177 = getelementptr inbounds nuw i64, ptr %3, i64 %176
  %178 = load i64, ptr %177, align 8, !tbaa !11
  %179 = icmp ugt i64 %175, %178
  %180 = zext i1 %179 to i32
  %181 = or disjoint i32 %167, %180
  br label %182

182:                                              ; preds = %169, %165
  %183 = phi i32 [ %167, %165 ], [ %181, %169 ]
  %184 = sext i32 %183 to i64
  %185 = getelementptr i64, ptr %4, i64 %184
  %186 = getelementptr i8, ptr %185, i64 -8
  %187 = load i64, ptr %186, align 8, !tbaa !11
  %188 = getelementptr inbounds nuw i64, ptr %3, i64 %187
  %189 = load i64, ptr %188, align 8, !tbaa !11
  %190 = icmp ult i64 %164, %189
  %191 = sext i32 %166 to i64
  br i1 %190, label %196, label %192

192:                                              ; preds = %182
  %193 = getelementptr i64, ptr %4, i64 %191
  %194 = getelementptr i8, ptr %193, i64 -8
  store i64 %187, ptr %194, align 8, !tbaa !11
  %195 = icmp sgt i32 %183, %104
  br i1 %195, label %196, label %165, !llvm.loop !15

196:                                              ; preds = %192, %182, %142
  %197 = phi i64 [ 1, %142 ], [ %184, %192 ], [ %191, %182 ]
  %198 = ashr exact i64 %160, 32
  %199 = getelementptr i64, ptr %4, i64 %197
  %200 = getelementptr i8, ptr %199, i64 -8
  store i64 %198, ptr %200, align 8, !tbaa !11
  %201 = icmp ugt i64 %99, 1
  br i1 %201, label %97, label %202, !llvm.loop !19

202:                                              ; preds = %196, %25, %43
  %203 = phi i64 [ %40, %43 ], [ 0, %25 ], [ 1, %196 ]
  %204 = getelementptr inbounds nuw i32, ptr %5, i64 %203
  %205 = getelementptr inbounds nuw i8, ptr %204, i64 1024
  store i32 0, ptr %205, align 4, !tbaa !17
  br label %206

206:                                              ; preds = %202, %243
  %207 = phi i64 [ 0, %202 ], [ %245, %243 ]
  %208 = phi i64 [ 0, %202 ], [ %244, %243 ]
  %209 = phi i64 [ 0, %202 ], [ %246, %243 ]
  %210 = getelementptr inbounds nuw i64, ptr %3, i64 %209
  %211 = load i64, ptr %210, align 8, !tbaa !11
  %212 = icmp eq i64 %211, 0
  br i1 %212, label %217, label %213

213:                                              ; preds = %206
  %214 = getelementptr inbounds nuw i32, ptr %5, i64 %209
  %215 = load i32, ptr %214, align 4, !tbaa !17
  %216 = icmp eq i32 %215, 0
  br i1 %216, label %235, label %220

217:                                              ; preds = %206
  %218 = getelementptr inbounds nuw i64, ptr %6, i64 %209
  store i64 0, ptr %218, align 8, !tbaa !11
  %219 = getelementptr inbounds nuw i8, ptr %7, i64 %209
  store i8 0, ptr %219, align 1, !tbaa !6
  br label %243

220:                                              ; preds = %213, %220
  %221 = phi i32 [ %233, %220 ], [ %215, %213 ]
  %222 = phi i64 [ %227, %220 ], [ 0, %213 ]
  %223 = phi i64 [ %231, %220 ], [ 0, %213 ]
  %224 = phi i64 [ %230, %220 ], [ 1, %213 ]
  %225 = icmp slt i32 %221, 0
  %226 = select i1 %225, i64 %224, i64 0
  %227 = add i64 %226, %222
  %228 = tail call i32 @llvm.abs.i32(i32 %221, i1 true)
  %229 = zext nneg i32 %228 to i64
  %230 = shl i64 %224, 1
  %231 = add i64 %223, 1
  %232 = getelementptr inbounds nuw i32, ptr %5, i64 %229
  %233 = load i32, ptr %232, align 4, !tbaa !17
  %234 = icmp eq i32 %233, 0
  br i1 %234, label %235, label %220, !llvm.loop !20

235:                                              ; preds = %220, %213
  %236 = phi i64 [ 0, %213 ], [ %231, %220 ]
  %237 = phi i64 [ 0, %213 ], [ %227, %220 ]
  %238 = getelementptr inbounds nuw i64, ptr %6, i64 %209
  store i64 %237, ptr %238, align 8, !tbaa !11
  %239 = trunc i64 %236 to i8
  %240 = getelementptr inbounds nuw i8, ptr %7, i64 %209
  store i8 %239, ptr %240, align 1, !tbaa !6
  %241 = tail call i64 @llvm.umax.i64(i64 %237, i64 %208)
  %242 = tail call i64 @llvm.umax.i64(i64 %236, i64 %207)
  br label %243

243:                                              ; preds = %235, %217
  %244 = phi i64 [ %208, %217 ], [ %241, %235 ]
  %245 = phi i64 [ %207, %217 ], [ %242, %235 ]
  %246 = add nuw nsw i64 %209, 1
  %247 = icmp eq i64 %246, 256
  br i1 %247, label %248, label %206, !llvm.loop !21

248:                                              ; preds = %243
  %249 = icmp ugt i64 %245, 64
  br i1 %249, label %250, label %253

250:                                              ; preds = %248
  %251 = load ptr, ptr @stderr, align 8, !tbaa !22
  %252 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 25, i64 1, ptr %251) #14
  tail call void @exit(i32 noundef 1) #15
  unreachable

253:                                              ; preds = %248
  %254 = icmp eq i64 %244, 0
  br i1 %254, label %256, label %255

255:                                              ; preds = %253
  br i1 %12, label %318, label %259

256:                                              ; preds = %253
  %257 = load ptr, ptr @stderr, align 8, !tbaa !22
  %258 = tail call i64 @fwrite(ptr nonnull @.str.2, i64 32, i64 1, ptr %257) #14
  tail call void @exit(i32 noundef 1) #15
  unreachable

259:                                              ; preds = %255, %306
  %260 = phi i32 [ %309, %306 ], [ -1, %255 ]
  %261 = phi i8 [ %308, %306 ], [ 0, %255 ]
  %262 = phi i64 [ %307, %306 ], [ 0, %255 ]
  %263 = phi i64 [ %311, %306 ], [ 0, %255 ]
  %264 = phi ptr [ %310, %306 ], [ %0, %255 ]
  %265 = load i8, ptr %264, align 1, !tbaa !6
  %266 = zext i8 %265 to i64
  %267 = getelementptr inbounds nuw i8, ptr %7, i64 %266
  %268 = load i8, ptr %267, align 1, !tbaa !6
  %269 = zext i8 %268 to i64
  %270 = icmp eq i8 %268, 0
  br i1 %270, label %306, label %271

271:                                              ; preds = %259
  %272 = zext i8 %268 to i32
  %273 = add nsw i32 %272, -1
  %274 = shl nuw i32 1, %273
  %275 = sext i32 %274 to i64
  %276 = getelementptr inbounds nuw i64, ptr %6, i64 %266
  br label %277

277:                                              ; preds = %271, %294
  %278 = phi i32 [ %260, %271 ], [ %297, %294 ]
  %279 = phi i8 [ %261, %271 ], [ %302, %294 ]
  %280 = phi i64 [ %262, %271 ], [ %295, %294 ]
  %281 = phi i64 [ 0, %271 ], [ %304, %294 ]
  %282 = phi i64 [ %275, %271 ], [ %303, %294 ]
  %283 = icmp eq i32 %278, 7
  br i1 %283, label %284, label %291

284:                                              ; preds = %277
  %285 = getelementptr inbounds nuw i8, ptr %11, i64 %280
  store i8 %279, ptr %285, align 1, !tbaa !6
  %286 = add i64 %280, 1
  %287 = icmp eq i64 %286, %1
  br i1 %287, label %288, label %294

288:                                              ; preds = %284
  %289 = load ptr, ptr @stderr, align 8, !tbaa !22
  %290 = tail call i64 @fwrite(ptr nonnull @.str.3, i64 22, i64 1, ptr %289) #14
  tail call void @exit(i32 noundef 1) #15
  unreachable

291:                                              ; preds = %277
  %292 = add nsw i32 %278, 1
  %293 = shl i8 %279, 1
  br label %294

294:                                              ; preds = %284, %291
  %295 = phi i64 [ %280, %291 ], [ %286, %284 ]
  %296 = phi i8 [ %293, %291 ], [ 0, %284 ]
  %297 = phi i32 [ %292, %291 ], [ 0, %284 ]
  %298 = load i64, ptr %276, align 8, !tbaa !11
  %299 = and i64 %298, %282
  %300 = icmp ne i64 %299, 0
  %301 = zext i1 %300 to i8
  %302 = or disjoint i8 %296, %301
  %303 = lshr i64 %282, 1
  %304 = add nuw nsw i64 %281, 1
  %305 = icmp eq i64 %304, %269
  br i1 %305, label %306, label %277, !llvm.loop !25

306:                                              ; preds = %294, %259
  %307 = phi i64 [ %262, %259 ], [ %295, %294 ]
  %308 = phi i8 [ %261, %259 ], [ %302, %294 ]
  %309 = phi i32 [ %260, %259 ], [ %297, %294 ]
  %310 = getelementptr inbounds nuw i8, ptr %264, i64 1
  %311 = add nuw i64 %263, 1
  %312 = icmp eq i64 %311, %1
  br i1 %312, label %313, label %259, !llvm.loop !26

313:                                              ; preds = %306
  %314 = sub nsw i32 7, %309
  %315 = zext i8 %308 to i32
  %316 = shl i32 %315, %314
  %317 = trunc i32 %316 to i8
  br label %318

318:                                              ; preds = %313, %255
  %319 = phi i64 [ 0, %255 ], [ %307, %313 ]
  %320 = phi i8 [ 0, %255 ], [ %317, %313 ]
  %321 = getelementptr inbounds nuw i8, ptr %11, i64 %319
  store i8 %320, ptr %321, align 1, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #13
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #13
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(2048) %8, i8 0, i64 2048, i1 false)
  br label %322

322:                                              ; preds = %318, %357
  %323 = phi ptr [ %9, %318 ], [ %326, %357 ]
  %324 = phi i64 [ 0, %318 ], [ %358, %357 ]
  %325 = trunc nuw i64 %324 to i8
  store i8 %325, ptr %323, align 1, !tbaa !6
  %326 = getelementptr inbounds nuw i8, ptr %323, i64 1
  %327 = getelementptr inbounds nuw i64, ptr %6, i64 %324
  %328 = load i64, ptr %327, align 8, !tbaa !11
  %329 = getelementptr inbounds nuw i8, ptr %7, i64 %324
  %330 = load i8, ptr %329, align 1, !tbaa !6
  %331 = zext i8 %330 to i64
  %332 = or i64 %328, %331
  %333 = icmp eq i64 %332, 0
  br i1 %333, label %357, label %334

334:                                              ; preds = %322
  %335 = icmp eq i8 %330, 0
  br i1 %335, label %354, label %336

336:                                              ; preds = %334
  %337 = zext i8 %330 to i32
  %338 = add nsw i32 %337, -1
  %339 = shl nuw i32 1, %338
  %340 = sext i32 %339 to i64
  br label %341

341:                                              ; preds = %336, %341
  %342 = phi i64 [ %352, %341 ], [ 0, %336 ]
  %343 = phi i64 [ %351, %341 ], [ %340, %336 ]
  %344 = phi i64 [ %350, %341 ], [ 0, %336 ]
  %345 = shl i64 %344, 1
  %346 = or disjoint i64 %345, 1
  %347 = and i64 %343, %328
  %348 = icmp eq i64 %347, 0
  %349 = add i64 %345, 2
  %350 = select i1 %348, i64 %346, i64 %349
  %351 = lshr i64 %343, 1
  %352 = add nuw nsw i64 %342, 1
  %353 = icmp eq i64 %352, %331
  br i1 %353, label %354, label %341, !llvm.loop !27

354:                                              ; preds = %341, %334
  %355 = phi i64 [ 0, %334 ], [ %350, %341 ]
  %356 = getelementptr inbounds nuw i64, ptr %8, i64 %324
  store i64 %355, ptr %356, align 8, !tbaa !11
  br label %357

357:                                              ; preds = %322, %354
  %358 = add nuw nsw i64 %324, 1
  %359 = icmp eq i64 %358, 256
  br i1 %359, label %360, label %322, !llvm.loop !28

360:                                              ; preds = %357, %378
  %361 = phi i64 [ %382, %378 ], [ 1, %357 ]
  %362 = getelementptr inbounds nuw i64, ptr %8, i64 %361
  %363 = load i64, ptr %362, align 8, !tbaa !11
  %364 = getelementptr inbounds nuw i8, ptr %9, i64 %361
  %365 = load i8, ptr %364, align 1, !tbaa !6
  br label %366

366:                                              ; preds = %360, %372
  %367 = phi i64 [ %361, %360 ], [ %368, %372 ]
  %368 = add nsw i64 %367, -1
  %369 = getelementptr inbounds nuw i64, ptr %8, i64 %368
  %370 = load i64, ptr %369, align 8, !tbaa !11
  %371 = icmp ugt i64 %370, %363
  br i1 %371, label %372, label %378

372:                                              ; preds = %366
  %373 = getelementptr inbounds nuw i64, ptr %8, i64 %367
  store i64 %370, ptr %373, align 8, !tbaa !11
  %374 = getelementptr inbounds nuw i8, ptr %9, i64 %368
  %375 = load i8, ptr %374, align 1, !tbaa !6
  %376 = getelementptr inbounds nuw i8, ptr %9, i64 %367
  store i8 %375, ptr %376, align 1, !tbaa !6
  %377 = icmp eq i64 %368, 0
  br i1 %377, label %378, label %366, !llvm.loop !29

378:                                              ; preds = %372, %366
  %379 = phi i64 [ 0, %372 ], [ %367, %366 ]
  %380 = getelementptr inbounds nuw i64, ptr %8, i64 %379
  store i64 %363, ptr %380, align 8, !tbaa !11
  %381 = getelementptr inbounds nuw i8, ptr %9, i64 %379
  store i8 %365, ptr %381, align 1, !tbaa !6
  %382 = add nuw nsw i64 %361, 1
  %383 = icmp eq i64 %382, 256
  br i1 %383, label %384, label %360, !llvm.loop !30

384:                                              ; preds = %378, %384
  %385 = phi i64 [ %389, %384 ], [ 0, %378 ]
  %386 = getelementptr inbounds nuw i64, ptr %8, i64 %385
  %387 = load i64, ptr %386, align 8, !tbaa !11
  %388 = icmp eq i64 %387, 0
  %389 = add i64 %385, 1
  br i1 %388, label %384, label %390, !llvm.loop !31

390:                                              ; preds = %384
  br i1 %12, label %430, label %391

391:                                              ; preds = %390, %419
  %392 = phi i64 [ %423, %419 ], [ %385, %390 ]
  %393 = phi ptr [ %422, %419 ], [ %0, %390 ]
  %394 = phi ptr [ %428, %419 ], [ %11, %390 ]
  %395 = phi i64 [ %421, %419 ], [ 0, %390 ]
  %396 = phi i64 [ %426, %419 ], [ 128, %390 ]
  %397 = phi i64 [ %420, %419 ], [ 0, %390 ]
  %398 = shl i64 %397, 1
  %399 = or disjoint i64 %398, 1
  %400 = load i8, ptr %394, align 1, !tbaa !6
  %401 = zext i8 %400 to i64
  %402 = and i64 %396, %401
  %403 = icmp eq i64 %402, 0
  %404 = add i64 %398, 2
  %405 = select i1 %403, i64 %399, i64 %404
  br label %406

406:                                              ; preds = %406, %391
  %407 = phi i64 [ %392, %391 ], [ %411, %406 ]
  %408 = getelementptr inbounds nuw i64, ptr %8, i64 %407
  %409 = load i64, ptr %408, align 8, !tbaa !11
  %410 = icmp ult i64 %409, %405
  %411 = add i64 %407, 1
  br i1 %410, label %406, label %412, !llvm.loop !32

412:                                              ; preds = %406
  %413 = icmp eq i64 %405, %409
  br i1 %413, label %414, label %419

414:                                              ; preds = %412
  %415 = getelementptr inbounds nuw i8, ptr %9, i64 %407
  %416 = load i8, ptr %415, align 1, !tbaa !6
  store i8 %416, ptr %393, align 1, !tbaa !6
  %417 = getelementptr inbounds nuw i8, ptr %393, i64 1
  %418 = add nuw i64 %395, 1
  br label %419

419:                                              ; preds = %414, %412
  %420 = phi i64 [ 0, %414 ], [ %405, %412 ]
  %421 = phi i64 [ %418, %414 ], [ %395, %412 ]
  %422 = phi ptr [ %417, %414 ], [ %393, %412 ]
  %423 = phi i64 [ %385, %414 ], [ %407, %412 ]
  %424 = icmp ult i64 %396, 2
  %425 = lshr i64 %396, 1
  %426 = select i1 %424, i64 128, i64 %425
  %427 = zext i1 %424 to i64
  %428 = getelementptr inbounds nuw i8, ptr %394, i64 %427
  %429 = icmp ult i64 %421, %1
  br i1 %429, label %391, label %430, !llvm.loop !33

430:                                              ; preds = %419, %390
  tail call void @free(ptr noundef %11) #13
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #13
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #13
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #13
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #13
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #13
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #13
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #13
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #4

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #5

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #6

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #7

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #3 {
  %3 = icmp sgt i32 %0, 1
  br i1 %3, label %4, label %10

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !34
  %7 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %6, ptr noundef nonnull dereferenceable(4) @.str.4) #16
  %8 = icmp eq i32 %7, 0
  %9 = select i1 %8, ptr @.str.5, ptr @.str.6
  br label %10

10:                                               ; preds = %4, %2
  %11 = phi ptr [ @.str.6, %2 ], [ %9, %4 ]
  %12 = tail call noalias dereferenceable_or_null(10000000) ptr @malloc(i64 noundef 10000000) #12
  %13 = load i64, ptr @seed, align 8
  %14 = xor i64 %13, 123459876
  br label %15

15:                                               ; preds = %15, %10
  %16 = phi i64 [ 0, %10 ], [ %32, %15 ]
  %17 = phi ptr [ %12, %10 ], [ %31, %15 ]
  %18 = phi i64 [ %14, %10 ], [ %27, %15 ]
  %19 = sdiv i64 %18, 127773
  %20 = mul nsw i64 %19, -127773
  %21 = add i64 %20, %18
  %22 = mul nsw i64 %21, 16807
  %23 = mul nsw i64 %19, -2836
  %24 = add i64 %22, %23
  %25 = icmp slt i64 %24, 0
  %26 = add nsw i64 %24, 2147483647
  %27 = select i1 %25, i64 %26, i64 %24
  %28 = srem i64 %27, 32
  %29 = getelementptr inbounds nuw i8, ptr @.str, i64 %28
  %30 = load i8, ptr %29, align 1, !tbaa !6
  store i8 %30, ptr %17, align 1, !tbaa !6
  %31 = getelementptr inbounds nuw i8, ptr %17, i64 1
  %32 = add nuw nsw i64 %16, 1
  %33 = icmp eq i64 %32, 10000000
  br i1 %33, label %34, label %15, !llvm.loop !9

34:                                               ; preds = %15
  %35 = xor i64 %27, 123459876
  store i64 %35, ptr @seed, align 8, !tbaa !11
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @compdecomp(ptr noundef %12, i64 noundef 10000000)
  tail call void @free(ptr noundef %12) #13
  %36 = load ptr, ptr @stdout, align 8, !tbaa !22
  %37 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %36, ptr noundef nonnull %11, double noundef 0.000000e+00) #13
  %38 = load ptr, ptr @stdout, align 8, !tbaa !22
  %39 = tail call i32 @fflush(ptr noundef %38)
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @strcmp(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #8

; Function Attrs: nofree nounwind
declare noundef i32 @fflush(ptr noundef captures(none)) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare noundef i64 @fwrite(ptr noundef readonly captures(none), i64 noundef, i64 noundef, ptr noundef captures(none)) local_unnamed_addr #9

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.abs.i32(i32, i1 immarg) #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #10

; Function Attrs: nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) local_unnamed_addr #11

attributes #0 = { nofree nounwind memory(readwrite, argmem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #5 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nofree nounwind }
attributes #10 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #11 = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
attributes #12 = { nounwind allocsize(0) }
attributes #13 = { nounwind }
attributes #14 = { cold }
attributes #15 = { cold noreturn nounwind }
attributes #16 = { nounwind willreturn memory(read) }

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
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!12, !12, i64 0}
!12 = !{!"long", !7, i64 0}
!13 = distinct !{!13, !10}
!14 = distinct !{!14, !10}
!15 = distinct !{!15, !10}
!16 = distinct !{!16, !10}
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !7, i64 0}
!19 = distinct !{!19, !10}
!20 = distinct !{!20, !10}
!21 = distinct !{!21, !10}
!22 = !{!23, !23, i64 0}
!23 = !{!"p1 _ZTS8_IO_FILE", !24, i64 0}
!24 = !{!"any pointer", !7, i64 0}
!25 = distinct !{!25, !10}
!26 = distinct !{!26, !10}
!27 = distinct !{!27, !10}
!28 = distinct !{!28, !10}
!29 = distinct !{!29, !10}
!30 = distinct !{!30, !10}
!31 = distinct !{!31, !10}
!32 = distinct !{!32, !10}
!33 = distinct !{!33, !10}
!34 = !{!35, !35, i64 0}
!35 = !{!"p1 omnipotent char", !24, i64 0}
