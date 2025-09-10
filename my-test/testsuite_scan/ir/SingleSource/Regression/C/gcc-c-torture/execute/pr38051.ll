; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr38051.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr38051.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@buf = dso_local global [256 x i8] zeroinitializer, align 1
@.str = private unnamed_addr constant [16 x i8] c"\017\82\A7UI\9D\BF\F8D\B6U\17\8E\F9\00", align 1
@.str.1 = private unnamed_addr constant [16 x i8] c"\017\82\A7UI\D0\F3\B7*m#qIj\00", align 1

; Function Attrs: nofree noinline norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable
define dso_local range(i32 -255, 256) i32 @mymemcmp(ptr noundef %0, ptr noundef %1, i64 noundef %2) local_unnamed_addr #0 {
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  %10 = alloca i64, align 8
  %11 = alloca i64, align 8
  %12 = alloca i64, align 8
  %13 = alloca i64, align 8
  %14 = alloca i64, align 8
  %15 = alloca i64, align 8
  %16 = alloca i64, align 8
  %17 = alloca i64, align 8
  %18 = alloca i64, align 8
  %19 = alloca i64, align 8
  %20 = alloca i64, align 8
  %21 = alloca i64, align 8
  %22 = alloca i64, align 8
  %23 = alloca i64, align 8
  %24 = ptrtoint ptr %0 to i64
  %25 = ptrtoint ptr %1 to i64
  %26 = and i64 %24, 7
  %27 = icmp eq i64 %26, 0
  %28 = lshr i64 %2, 3
  br i1 %27, label %29, label %191

29:                                               ; preds = %3
  %30 = and i64 %28, 3
  switch i64 %30, label %31 [
    i64 2, label %32
    i64 3, label %36
    i64 0, label %42
    i64 1, label %44
  ]

31:                                               ; preds = %191, %29
  unreachable

32:                                               ; preds = %29
  %33 = add i64 %24, -16
  %34 = add i64 %25, -16
  %35 = add nuw nsw i64 %28, 2
  br label %133

36:                                               ; preds = %29
  %37 = add i64 %24, -8
  %38 = add i64 %25, -8
  %39 = add nuw nsw i64 %28, 1
  %40 = inttoptr i64 %37 to ptr
  %41 = inttoptr i64 %38 to ptr
  br label %103

42:                                               ; preds = %29
  %43 = icmp ult i64 %2, 8
  br i1 %43, label %410, label %75

44:                                               ; preds = %29
  %45 = load i64, ptr %0, align 8, !tbaa !6
  %46 = load i64, ptr %1, align 8, !tbaa !6
  %47 = add i64 %24, 8
  %48 = add i64 %25, 8
  %49 = add nsw i64 %28, -1
  %50 = icmp eq i64 %49, 0
  br i1 %50, label %170, label %51

51:                                               ; preds = %159, %44
  %52 = phi i64 [ %166, %159 ], [ %47, %44 ]
  %53 = phi i64 [ %167, %159 ], [ %48, %44 ]
  %54 = phi i64 [ %168, %159 ], [ %49, %44 ]
  %55 = phi i64 [ %165, %159 ], [ %45, %44 ]
  %56 = phi i64 [ %162, %159 ], [ %46, %44 ]
  %57 = icmp eq i64 %55, %56
  br i1 %57, label %75, label %58

58:                                               ; preds = %51
  call void @llvm.lifetime.start.p0(ptr nonnull %22)
  call void @llvm.lifetime.start.p0(ptr nonnull %23)
  store i64 %55, ptr %22, align 8, !tbaa !6
  store i64 %56, ptr %23, align 8, !tbaa !6
  %59 = ptrtoint ptr %22 to i64
  %60 = ptrtoint ptr %23 to i64
  br label %61

61:                                               ; preds = %61, %58
  %62 = phi i64 [ %60, %58 ], [ %69, %61 ]
  %63 = phi i64 [ %59, %58 ], [ %68, %61 ]
  %64 = inttoptr i64 %63 to ptr
  %65 = load i8, ptr %64, align 1, !tbaa !10
  %66 = inttoptr i64 %62 to ptr
  %67 = load i8, ptr %66, align 1, !tbaa !10
  %68 = add nsw i64 %63, 1
  %69 = add nsw i64 %62, 1
  %70 = icmp eq i8 %65, %67
  br i1 %70, label %61, label %71, !llvm.loop !11

71:                                               ; preds = %61
  %72 = zext i8 %67 to i32
  %73 = zext i8 %65 to i32
  %74 = sub nsw i32 %73, %72
  call void @llvm.lifetime.end.p0(ptr nonnull %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %23)
  br label %410

75:                                               ; preds = %51, %42
  %76 = phi i64 [ %52, %51 ], [ %24, %42 ]
  %77 = phi i64 [ %53, %51 ], [ %25, %42 ]
  %78 = phi i64 [ %54, %51 ], [ %28, %42 ]
  %79 = inttoptr i64 %77 to ptr
  %80 = load i64, ptr %79, align 8, !tbaa !6
  %81 = inttoptr i64 %76 to ptr
  %82 = load i64, ptr %81, align 8, !tbaa !6
  %83 = getelementptr inbounds nuw i8, ptr %81, i64 8
  %84 = getelementptr inbounds nuw i8, ptr %79, i64 8
  %85 = icmp eq i64 %82, %80
  br i1 %85, label %103, label %86

86:                                               ; preds = %75
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  store i64 %82, ptr %20, align 8, !tbaa !6
  store i64 %80, ptr %21, align 8, !tbaa !6
  %87 = ptrtoint ptr %20 to i64
  %88 = ptrtoint ptr %21 to i64
  br label %89

89:                                               ; preds = %89, %86
  %90 = phi i64 [ %88, %86 ], [ %97, %89 ]
  %91 = phi i64 [ %87, %86 ], [ %96, %89 ]
  %92 = inttoptr i64 %91 to ptr
  %93 = load i8, ptr %92, align 1, !tbaa !10
  %94 = inttoptr i64 %90 to ptr
  %95 = load i8, ptr %94, align 1, !tbaa !10
  %96 = add nsw i64 %91, 1
  %97 = add nsw i64 %90, 1
  %98 = icmp eq i8 %93, %95
  br i1 %98, label %89, label %99, !llvm.loop !11

99:                                               ; preds = %89
  %100 = zext i8 %95 to i32
  %101 = zext i8 %93 to i32
  %102 = sub nsw i32 %101, %100
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %21)
  br label %410

103:                                              ; preds = %75, %36
  %104 = phi ptr [ %79, %75 ], [ %41, %36 ]
  %105 = phi ptr [ %81, %75 ], [ %40, %36 ]
  %106 = phi i64 [ %76, %75 ], [ %37, %36 ]
  %107 = phi i64 [ %77, %75 ], [ %38, %36 ]
  %108 = phi i64 [ %78, %75 ], [ %39, %36 ]
  %109 = phi ptr [ %83, %75 ], [ %0, %36 ]
  %110 = phi ptr [ %84, %75 ], [ %1, %36 ]
  %111 = load i64, ptr %110, align 8, !tbaa !6
  %112 = load i64, ptr %109, align 8, !tbaa !6
  %113 = getelementptr inbounds nuw i8, ptr %105, i64 16
  %114 = getelementptr inbounds nuw i8, ptr %104, i64 16
  %115 = icmp eq i64 %112, %111
  br i1 %115, label %133, label %116

116:                                              ; preds = %103
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  store i64 %112, ptr %18, align 8, !tbaa !6
  store i64 %111, ptr %19, align 8, !tbaa !6
  %117 = ptrtoint ptr %18 to i64
  %118 = ptrtoint ptr %19 to i64
  br label %119

119:                                              ; preds = %119, %116
  %120 = phi i64 [ %118, %116 ], [ %127, %119 ]
  %121 = phi i64 [ %117, %116 ], [ %126, %119 ]
  %122 = inttoptr i64 %121 to ptr
  %123 = load i8, ptr %122, align 1, !tbaa !10
  %124 = inttoptr i64 %120 to ptr
  %125 = load i8, ptr %124, align 1, !tbaa !10
  %126 = add nsw i64 %121, 1
  %127 = add nsw i64 %120, 1
  %128 = icmp eq i8 %123, %125
  br i1 %128, label %119, label %129, !llvm.loop !11

129:                                              ; preds = %119
  %130 = zext i8 %125 to i32
  %131 = zext i8 %123 to i32
  %132 = sub nsw i32 %131, %130
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  br label %410

133:                                              ; preds = %103, %32
  %134 = phi i64 [ %33, %32 ], [ %106, %103 ]
  %135 = phi i64 [ %34, %32 ], [ %107, %103 ]
  %136 = phi i64 [ %35, %32 ], [ %108, %103 ]
  %137 = phi ptr [ %0, %32 ], [ %113, %103 ]
  %138 = phi ptr [ %1, %32 ], [ %114, %103 ]
  %139 = load i64, ptr %138, align 8, !tbaa !6
  %140 = load i64, ptr %137, align 8, !tbaa !6
  %141 = icmp eq i64 %140, %139
  br i1 %141, label %159, label %142

142:                                              ; preds = %133
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  store i64 %140, ptr %16, align 8, !tbaa !6
  store i64 %139, ptr %17, align 8, !tbaa !6
  %143 = ptrtoint ptr %16 to i64
  %144 = ptrtoint ptr %17 to i64
  br label %145

145:                                              ; preds = %145, %142
  %146 = phi i64 [ %144, %142 ], [ %153, %145 ]
  %147 = phi i64 [ %143, %142 ], [ %152, %145 ]
  %148 = inttoptr i64 %147 to ptr
  %149 = load i8, ptr %148, align 1, !tbaa !10
  %150 = inttoptr i64 %146 to ptr
  %151 = load i8, ptr %150, align 1, !tbaa !10
  %152 = add nsw i64 %147, 1
  %153 = add nsw i64 %146, 1
  %154 = icmp eq i8 %149, %151
  br i1 %154, label %145, label %155, !llvm.loop !11

155:                                              ; preds = %145
  %156 = zext i8 %151 to i32
  %157 = zext i8 %149 to i32
  %158 = sub nsw i32 %157, %156
  call void @llvm.lifetime.end.p0(ptr nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  br label %410

159:                                              ; preds = %133
  %160 = inttoptr i64 %135 to ptr
  %161 = getelementptr inbounds nuw i8, ptr %160, i64 24
  %162 = load i64, ptr %161, align 8, !tbaa !6
  %163 = inttoptr i64 %134 to ptr
  %164 = getelementptr inbounds nuw i8, ptr %163, i64 24
  %165 = load i64, ptr %164, align 8, !tbaa !6
  %166 = add i64 %134, 32
  %167 = add i64 %135, 32
  %168 = add i64 %136, -4
  %169 = icmp eq i64 %168, 0
  br i1 %169, label %170, label %51, !llvm.loop !13

170:                                              ; preds = %159, %44
  %171 = phi i64 [ %165, %159 ], [ %45, %44 ]
  %172 = phi i64 [ %162, %159 ], [ %46, %44 ]
  %173 = icmp eq i64 %171, %172
  br i1 %173, label %410, label %174

174:                                              ; preds = %170
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store i64 %171, ptr %14, align 8, !tbaa !6
  store i64 %172, ptr %15, align 8, !tbaa !6
  %175 = ptrtoint ptr %14 to i64
  %176 = ptrtoint ptr %15 to i64
  br label %177

177:                                              ; preds = %177, %174
  %178 = phi i64 [ %176, %174 ], [ %185, %177 ]
  %179 = phi i64 [ %175, %174 ], [ %184, %177 ]
  %180 = inttoptr i64 %179 to ptr
  %181 = load i8, ptr %180, align 1, !tbaa !10
  %182 = inttoptr i64 %178 to ptr
  %183 = load i8, ptr %182, align 1, !tbaa !10
  %184 = add nsw i64 %179, 1
  %185 = add nsw i64 %178, 1
  %186 = icmp eq i8 %181, %183
  br i1 %186, label %177, label %187, !llvm.loop !11

187:                                              ; preds = %177
  %188 = zext i8 %183 to i32
  %189 = zext i8 %181 to i32
  %190 = sub nsw i32 %189, %188
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  br label %410

191:                                              ; preds = %3
  %192 = trunc i64 %24 to i32
  %193 = shl i32 %192, 3
  %194 = and i32 %193, 56
  %195 = sub nuw nsw i32 64, %194
  %196 = and i64 %24, -8
  %197 = and i64 %28, 3
  switch i64 %197, label %31 [
    i64 2, label %198
    i64 3, label %205
    i64 0, label %213
    i64 1, label %223
  ]

198:                                              ; preds = %191
  %199 = inttoptr i64 %196 to ptr
  %200 = load i64, ptr %199, align 8, !tbaa !6
  %201 = getelementptr inbounds nuw i8, ptr %199, i64 8
  %202 = add i64 %196, -8
  %203 = add i64 %25, -16
  %204 = add nuw nsw i64 %28, 2
  br label %340

205:                                              ; preds = %191
  %206 = inttoptr i64 %196 to ptr
  %207 = load i64, ptr %206, align 8, !tbaa !6
  %208 = add i64 %25, -8
  %209 = add nuw nsw i64 %28, 1
  %210 = inttoptr i64 %208 to ptr
  %211 = zext nneg i32 %194 to i64
  %212 = zext nneg i32 %195 to i64
  br label %304

213:                                              ; preds = %191
  %214 = icmp ult i64 %2, 8
  br i1 %214, label %410, label %215

215:                                              ; preds = %213
  %216 = inttoptr i64 %196 to ptr
  %217 = load i64, ptr %216, align 8, !tbaa !6
  %218 = getelementptr inbounds nuw i8, ptr %216, i64 8
  %219 = add i64 %196, 8
  %220 = inttoptr i64 %219 to ptr
  %221 = zext nneg i32 %194 to i64
  %222 = zext nneg i32 %195 to i64
  br label %270

223:                                              ; preds = %191
  %224 = inttoptr i64 %196 to ptr
  %225 = load i64, ptr %224, align 8, !tbaa !6
  %226 = getelementptr inbounds nuw i8, ptr %224, i64 8
  %227 = load i64, ptr %226, align 8, !tbaa !6
  %228 = load i64, ptr %1, align 8, !tbaa !6
  %229 = add nsw i64 %28, -1
  %230 = icmp eq i64 %229, 0
  br i1 %230, label %231, label %234

231:                                              ; preds = %223
  %232 = zext nneg i32 %194 to i64
  %233 = zext nneg i32 %195 to i64
  br label %383

234:                                              ; preds = %223
  %235 = add i64 %25, 8
  %236 = add i64 %196, 16
  %237 = zext nneg i32 %194 to i64
  %238 = zext nneg i32 %195 to i64
  br label %239

239:                                              ; preds = %372, %234
  %240 = phi i64 [ %238, %234 ], [ %351, %372 ]
  %241 = phi i64 [ %237, %234 ], [ %349, %372 ]
  %242 = phi i64 [ %235, %234 ], [ %380, %372 ]
  %243 = phi i64 [ %229, %234 ], [ %381, %372 ]
  %244 = phi i64 [ %225, %234 ], [ %348, %372 ]
  %245 = phi i64 [ %227, %234 ], [ %378, %372 ]
  %246 = phi i64 [ %228, %234 ], [ %375, %372 ]
  %247 = phi i64 [ %236, %234 ], [ %379, %372 ]
  %248 = inttoptr i64 %247 to ptr
  %249 = lshr i64 %244, %241
  %250 = shl i64 %245, %240
  %251 = or i64 %250, %249
  %252 = icmp eq i64 %251, %246
  br i1 %252, label %270, label %253

253:                                              ; preds = %239
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  store i64 %251, ptr %12, align 8, !tbaa !6
  store i64 %246, ptr %13, align 8, !tbaa !6
  %254 = ptrtoint ptr %12 to i64
  %255 = ptrtoint ptr %13 to i64
  br label %256

256:                                              ; preds = %256, %253
  %257 = phi i64 [ %255, %253 ], [ %264, %256 ]
  %258 = phi i64 [ %254, %253 ], [ %263, %256 ]
  %259 = inttoptr i64 %258 to ptr
  %260 = load i8, ptr %259, align 1, !tbaa !10
  %261 = inttoptr i64 %257 to ptr
  %262 = load i8, ptr %261, align 1, !tbaa !10
  %263 = add nsw i64 %258, 1
  %264 = add nsw i64 %257, 1
  %265 = icmp eq i8 %260, %262
  br i1 %265, label %256, label %266, !llvm.loop !11

266:                                              ; preds = %256
  %267 = zext i8 %262 to i32
  %268 = zext i8 %260 to i32
  %269 = sub nsw i32 %268, %267
  call void @llvm.lifetime.end.p0(ptr nonnull %12)
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  br label %410

270:                                              ; preds = %239, %215
  %271 = phi i64 [ %240, %239 ], [ %222, %215 ]
  %272 = phi i64 [ %241, %239 ], [ %221, %215 ]
  %273 = phi ptr [ %248, %239 ], [ %220, %215 ]
  %274 = phi i64 [ %242, %239 ], [ %25, %215 ]
  %275 = phi i64 [ %243, %239 ], [ %28, %215 ]
  %276 = phi ptr [ %248, %239 ], [ %218, %215 ]
  %277 = phi i64 [ %245, %239 ], [ %217, %215 ]
  %278 = phi i64 [ %247, %239 ], [ %219, %215 ]
  %279 = inttoptr i64 %274 to ptr
  %280 = load i64, ptr %279, align 8, !tbaa !6
  %281 = load i64, ptr %276, align 8, !tbaa !6
  %282 = getelementptr inbounds nuw i8, ptr %279, i64 8
  %283 = lshr i64 %277, %272
  %284 = shl i64 %281, %271
  %285 = or i64 %284, %283
  %286 = icmp eq i64 %285, %280
  br i1 %286, label %304, label %287

287:                                              ; preds = %270
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  store i64 %285, ptr %10, align 8, !tbaa !6
  store i64 %280, ptr %11, align 8, !tbaa !6
  %288 = ptrtoint ptr %10 to i64
  %289 = ptrtoint ptr %11 to i64
  br label %290

290:                                              ; preds = %290, %287
  %291 = phi i64 [ %289, %287 ], [ %298, %290 ]
  %292 = phi i64 [ %288, %287 ], [ %297, %290 ]
  %293 = inttoptr i64 %292 to ptr
  %294 = load i8, ptr %293, align 1, !tbaa !10
  %295 = inttoptr i64 %291 to ptr
  %296 = load i8, ptr %295, align 1, !tbaa !10
  %297 = add nsw i64 %292, 1
  %298 = add nsw i64 %291, 1
  %299 = icmp eq i8 %294, %296
  br i1 %299, label %290, label %300, !llvm.loop !11

300:                                              ; preds = %290
  %301 = zext i8 %296 to i32
  %302 = zext i8 %294 to i32
  %303 = sub nsw i32 %302, %301
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %11)
  br label %410

304:                                              ; preds = %270, %205
  %305 = phi i64 [ %271, %270 ], [ %212, %205 ]
  %306 = phi i64 [ %272, %270 ], [ %211, %205 ]
  %307 = phi ptr [ %279, %270 ], [ %210, %205 ]
  %308 = phi i64 [ %274, %270 ], [ %208, %205 ]
  %309 = phi i64 [ %275, %270 ], [ %209, %205 ]
  %310 = phi i64 [ %281, %270 ], [ %207, %205 ]
  %311 = phi ptr [ %273, %270 ], [ %206, %205 ]
  %312 = phi ptr [ %282, %270 ], [ %1, %205 ]
  %313 = phi i64 [ %278, %270 ], [ %196, %205 ]
  %314 = load i64, ptr %312, align 8, !tbaa !6
  %315 = getelementptr inbounds nuw i8, ptr %311, i64 8
  %316 = load i64, ptr %315, align 8, !tbaa !6
  %317 = getelementptr inbounds nuw i8, ptr %311, i64 16
  %318 = getelementptr inbounds nuw i8, ptr %307, i64 16
  %319 = lshr i64 %310, %306
  %320 = shl i64 %316, %305
  %321 = or i64 %320, %319
  %322 = icmp eq i64 %321, %314
  br i1 %322, label %340, label %323

323:                                              ; preds = %304
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store i64 %321, ptr %8, align 8, !tbaa !6
  store i64 %314, ptr %9, align 8, !tbaa !6
  %324 = ptrtoint ptr %8 to i64
  %325 = ptrtoint ptr %9 to i64
  br label %326

326:                                              ; preds = %326, %323
  %327 = phi i64 [ %325, %323 ], [ %334, %326 ]
  %328 = phi i64 [ %324, %323 ], [ %333, %326 ]
  %329 = inttoptr i64 %328 to ptr
  %330 = load i8, ptr %329, align 1, !tbaa !10
  %331 = inttoptr i64 %327 to ptr
  %332 = load i8, ptr %331, align 1, !tbaa !10
  %333 = add nsw i64 %328, 1
  %334 = add nsw i64 %327, 1
  %335 = icmp eq i8 %330, %332
  br i1 %335, label %326, label %336, !llvm.loop !11

336:                                              ; preds = %326
  %337 = zext i8 %332 to i32
  %338 = zext i8 %330 to i32
  %339 = sub nsw i32 %338, %337
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  br label %410

340:                                              ; preds = %304, %198
  %341 = phi i64 [ %203, %198 ], [ %308, %304 ]
  %342 = phi i64 [ %204, %198 ], [ %309, %304 ]
  %343 = phi i64 [ %200, %198 ], [ %316, %304 ]
  %344 = phi ptr [ %201, %198 ], [ %317, %304 ]
  %345 = phi ptr [ %1, %198 ], [ %318, %304 ]
  %346 = phi i64 [ %202, %198 ], [ %313, %304 ]
  %347 = load i64, ptr %345, align 8, !tbaa !6
  %348 = load i64, ptr %344, align 8, !tbaa !6
  %349 = zext nneg i32 %194 to i64
  %350 = lshr i64 %343, %349
  %351 = zext nneg i32 %195 to i64
  %352 = shl i64 %348, %351
  %353 = or i64 %352, %350
  %354 = icmp eq i64 %353, %347
  br i1 %354, label %372, label %355

355:                                              ; preds = %340
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  store i64 %353, ptr %6, align 8, !tbaa !6
  store i64 %347, ptr %7, align 8, !tbaa !6
  %356 = ptrtoint ptr %6 to i64
  %357 = ptrtoint ptr %7 to i64
  br label %358

358:                                              ; preds = %358, %355
  %359 = phi i64 [ %357, %355 ], [ %366, %358 ]
  %360 = phi i64 [ %356, %355 ], [ %365, %358 ]
  %361 = inttoptr i64 %360 to ptr
  %362 = load i8, ptr %361, align 1, !tbaa !10
  %363 = inttoptr i64 %359 to ptr
  %364 = load i8, ptr %363, align 1, !tbaa !10
  %365 = add nsw i64 %360, 1
  %366 = add nsw i64 %359, 1
  %367 = icmp eq i8 %362, %364
  br i1 %367, label %358, label %368, !llvm.loop !11

368:                                              ; preds = %358
  %369 = zext i8 %364 to i32
  %370 = zext i8 %362 to i32
  %371 = sub nsw i32 %370, %369
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  br label %410

372:                                              ; preds = %340
  %373 = inttoptr i64 %341 to ptr
  %374 = getelementptr inbounds nuw i8, ptr %373, i64 24
  %375 = load i64, ptr %374, align 8, !tbaa !6
  %376 = inttoptr i64 %346 to ptr
  %377 = getelementptr inbounds nuw i8, ptr %376, i64 24
  %378 = load i64, ptr %377, align 8, !tbaa !6
  %379 = add i64 %346, 32
  %380 = add i64 %341, 32
  %381 = add i64 %342, -4
  %382 = icmp eq i64 %381, 0
  br i1 %382, label %383, label %239, !llvm.loop !14

383:                                              ; preds = %372, %231
  %384 = phi i64 [ %233, %231 ], [ %351, %372 ]
  %385 = phi i64 [ %232, %231 ], [ %349, %372 ]
  %386 = phi i64 [ %225, %231 ], [ %348, %372 ]
  %387 = phi i64 [ %227, %231 ], [ %378, %372 ]
  %388 = phi i64 [ %228, %231 ], [ %375, %372 ]
  %389 = lshr i64 %386, %385
  %390 = shl i64 %387, %384
  %391 = or i64 %390, %389
  %392 = icmp eq i64 %391, %388
  br i1 %392, label %410, label %393

393:                                              ; preds = %383
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  store i64 %391, ptr %4, align 8, !tbaa !6
  store i64 %388, ptr %5, align 8, !tbaa !6
  %394 = ptrtoint ptr %4 to i64
  %395 = ptrtoint ptr %5 to i64
  br label %396

396:                                              ; preds = %396, %393
  %397 = phi i64 [ %395, %393 ], [ %404, %396 ]
  %398 = phi i64 [ %394, %393 ], [ %403, %396 ]
  %399 = inttoptr i64 %398 to ptr
  %400 = load i8, ptr %399, align 1, !tbaa !10
  %401 = inttoptr i64 %397 to ptr
  %402 = load i8, ptr %401, align 1, !tbaa !10
  %403 = add nsw i64 %398, 1
  %404 = add nsw i64 %397, 1
  %405 = icmp eq i8 %400, %402
  br i1 %405, label %396, label %406, !llvm.loop !11

406:                                              ; preds = %396
  %407 = zext i8 %402 to i32
  %408 = zext i8 %400 to i32
  %409 = sub nsw i32 %408, %407
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  br label %410

410:                                              ; preds = %406, %383, %368, %336, %300, %266, %213, %187, %170, %155, %129, %99, %71, %42
  %411 = phi i32 [ %158, %155 ], [ %74, %71 ], [ %102, %99 ], [ %132, %129 ], [ %190, %187 ], [ 0, %42 ], [ 0, %170 ], [ %371, %368 ], [ %269, %266 ], [ %303, %300 ], [ %339, %336 ], [ %409, %406 ], [ 0, %213 ], [ 0, %383 ]
  ret i32 %411
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = and i64 ptrtoint (ptr @buf to i64), 15
  %2 = sub nsw i64 0, %1
  %3 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @buf, i64 16), i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 9
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(15) %4, ptr noundef nonnull align 1 dereferenceable(15) @.str, i64 15, i1 false)
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 152
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(15) %5, ptr noundef nonnull align 1 dereferenceable(15) @.str.1, i64 15, i1 false)
  %6 = tail call i32 @mymemcmp(ptr noundef nonnull %4, ptr noundef nonnull %5, i64 noundef 33)
  %7 = icmp eq i32 %6, -51
  br i1 %7, label %9, label %8

8:                                                ; preds = %0
  tail call void @abort() #5
  unreachable

9:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #4

attributes #0 = { nofree noinline norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { noreturn nounwind }

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
!10 = !{!8, !8, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = distinct !{!13, !12}
!14 = distinct !{!14, !12}
