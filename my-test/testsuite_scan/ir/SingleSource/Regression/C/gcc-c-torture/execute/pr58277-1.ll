; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58277-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58277-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@e = dso_local global ptr null, align 8
@i = dso_local local_unnamed_addr global ptr @e, align 8
@l = dso_local local_unnamed_addr global i32 1, align 4
@u = dso_local local_unnamed_addr global i8 0, align 4
@m = dso_local local_unnamed_addr constant i32 0, align 4
@a = internal unnamed_addr global [2 x i32] zeroinitializer, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@n = dso_local local_unnamed_addr global i32 0, align 4
@j = internal global ptr @e, align 8
@b = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i32 0, align 4
@k = dso_local local_unnamed_addr global i32 0, align 4
@o = dso_local local_unnamed_addr global i32 0, align 4
@p = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @bar() local_unnamed_addr #0 {
  store i8 0, ptr @u, align 4, !tbaa !6
  ret i32 0
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @baz() local_unnamed_addr #1 {
  tail call void asm sideeffect "", ""() #6, !srcloc !9
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca [5 x i32], align 4
  store i32 1, ptr @a, align 4, !tbaa !10
  store i32 1, ptr @n, align 4, !tbaa !10
  %2 = load ptr, ptr @i, align 8
  %3 = load i32, ptr @l, align 4, !tbaa !10
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %8, label %5

5:                                                ; preds = %0
  %6 = load volatile ptr, ptr @j, align 8, !tbaa !12
  store ptr null, ptr %6, align 8, !tbaa !16
  %7 = load volatile ptr, ptr @j, align 8, !tbaa !12
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  store ptr %1, ptr %2, align 8, !tbaa !16
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  store i32 0, ptr @g, align 4, !tbaa !10
  store i32 0, ptr @d, align 4, !tbaa !10
  store i32 0, ptr @n, align 4, !tbaa !10
  br label %26

8:                                                ; preds = %0, %21
  %9 = phi i32 [ %24, %21 ], [ 1, %0 ]
  %10 = phi i32 [ %23, %21 ], [ 0, %0 ]
  store i32 0, ptr @g, align 4, !tbaa !10
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %15, label %12

12:                                               ; preds = %8
  %13 = load volatile ptr, ptr @j, align 8, !tbaa !12
  store ptr null, ptr %13, align 8, !tbaa !16
  store i32 0, ptr @d, align 4, !tbaa !10
  %14 = load volatile ptr, ptr @j, align 8, !tbaa !12
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  store ptr %1, ptr %2, align 8, !tbaa !16
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  br label %21

15:                                               ; preds = %8
  store ptr null, ptr %2, align 8, !tbaa !16
  %16 = load ptr, ptr @e, align 8, !tbaa !16
  store i32 0, ptr %16, align 4, !tbaa !10
  store i32 0, ptr @o, align 4, !tbaa !10
  %17 = load i32, ptr @p, align 4, !tbaa !10
  %18 = icmp ne i32 %17, 0
  call void @llvm.assume(i1 %18)
  store i32 0, ptr @f, align 4, !tbaa !10
  %19 = load i32, ptr @l, align 4, !tbaa !10
  %20 = load i32, ptr @n, align 4, !tbaa !10
  br label %21

21:                                               ; preds = %15, %12
  %22 = phi i32 [ %20, %15 ], [ %9, %12 ]
  %23 = phi i32 [ %19, %15 ], [ %10, %12 ]
  %24 = add nsw i32 %22, -1
  store i32 %24, ptr @n, align 4, !tbaa !10
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %26, label %8, !llvm.loop !18

26:                                               ; preds = %21, %5
  store i8 0, ptr @u, align 4, !tbaa !6
  %27 = load i32, ptr @b, align 4, !tbaa !10
  %28 = icmp eq i32 %27, 0
  br i1 %28, label %277, label %29

29:                                               ; preds = %26
  %30 = load i32, ptr @c, align 4
  br label %31

31:                                               ; preds = %29, %31
  %32 = phi i32 [ %274, %31 ], [ %27, %29 ]
  %33 = phi i32 [ %273, %31 ], [ %30, %29 ]
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds i32, ptr @a, i64 %34
  %36 = load i32, ptr %35, align 4, !tbaa !10
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds i32, ptr @a, i64 %37
  %39 = load i32, ptr %38, align 4, !tbaa !10
  %40 = sext i32 %39 to i64
  %41 = getelementptr inbounds i32, ptr @a, i64 %40
  %42 = load i32, ptr %41, align 4, !tbaa !10
  %43 = sext i32 %42 to i64
  %44 = getelementptr inbounds i32, ptr @a, i64 %43
  %45 = load i32, ptr %44, align 4, !tbaa !10
  %46 = sext i32 %45 to i64
  %47 = getelementptr inbounds i32, ptr @a, i64 %46
  %48 = load i32, ptr %47, align 4, !tbaa !10
  %49 = sext i32 %48 to i64
  %50 = getelementptr inbounds i32, ptr @a, i64 %49
  %51 = load i32, ptr %50, align 4, !tbaa !10
  %52 = sext i32 %51 to i64
  %53 = getelementptr inbounds i32, ptr @a, i64 %52
  %54 = load i32, ptr %53, align 4, !tbaa !10
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds i32, ptr @a, i64 %55
  %57 = load i32, ptr %56, align 4, !tbaa !10
  %58 = sext i32 %57 to i64
  %59 = getelementptr inbounds i32, ptr @a, i64 %58
  %60 = load i32, ptr %59, align 4, !tbaa !10
  %61 = sext i32 %60 to i64
  %62 = getelementptr inbounds i32, ptr @a, i64 %61
  %63 = load i32, ptr %62, align 4, !tbaa !10
  %64 = sext i32 %63 to i64
  %65 = getelementptr inbounds i32, ptr @a, i64 %64
  %66 = load i32, ptr %65, align 4, !tbaa !10
  %67 = sext i32 %66 to i64
  %68 = getelementptr inbounds i32, ptr @a, i64 %67
  %69 = load i32, ptr %68, align 4, !tbaa !10
  %70 = sext i32 %69 to i64
  %71 = getelementptr inbounds i32, ptr @a, i64 %70
  %72 = load i32, ptr %71, align 4, !tbaa !10
  %73 = sext i32 %72 to i64
  %74 = getelementptr inbounds i32, ptr @a, i64 %73
  %75 = load i32, ptr %74, align 4, !tbaa !10
  %76 = sext i32 %75 to i64
  %77 = getelementptr inbounds i32, ptr @a, i64 %76
  %78 = load i32, ptr %77, align 4, !tbaa !10
  %79 = sext i32 %78 to i64
  %80 = getelementptr inbounds i32, ptr @a, i64 %79
  %81 = load i32, ptr %80, align 4, !tbaa !10
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds i32, ptr @a, i64 %82
  %84 = load i32, ptr %83, align 4, !tbaa !10
  %85 = sext i32 %84 to i64
  %86 = getelementptr inbounds i32, ptr @a, i64 %85
  %87 = load i32, ptr %86, align 4, !tbaa !10
  %88 = sext i32 %87 to i64
  %89 = getelementptr inbounds i32, ptr @a, i64 %88
  %90 = load i32, ptr %89, align 4, !tbaa !10
  %91 = sext i32 %90 to i64
  %92 = getelementptr inbounds i32, ptr @a, i64 %91
  %93 = load i32, ptr %92, align 4, !tbaa !10
  %94 = sext i32 %93 to i64
  %95 = getelementptr inbounds i32, ptr @a, i64 %94
  %96 = load i32, ptr %95, align 4, !tbaa !10
  %97 = sext i32 %96 to i64
  %98 = getelementptr inbounds i32, ptr @a, i64 %97
  %99 = load i32, ptr %98, align 4, !tbaa !10
  %100 = sext i32 %99 to i64
  %101 = getelementptr inbounds i32, ptr @a, i64 %100
  %102 = load i32, ptr %101, align 4, !tbaa !10
  %103 = sext i32 %102 to i64
  %104 = getelementptr inbounds i32, ptr @a, i64 %103
  %105 = load i32, ptr %104, align 4, !tbaa !10
  %106 = sext i32 %105 to i64
  %107 = getelementptr inbounds i32, ptr @a, i64 %106
  %108 = load i32, ptr %107, align 4, !tbaa !10
  %109 = sext i32 %108 to i64
  %110 = getelementptr inbounds i32, ptr @a, i64 %109
  %111 = load i32, ptr %110, align 4, !tbaa !10
  %112 = sext i32 %111 to i64
  %113 = getelementptr inbounds i32, ptr @a, i64 %112
  %114 = load i32, ptr %113, align 4, !tbaa !10
  %115 = sext i32 %114 to i64
  %116 = getelementptr inbounds i32, ptr @a, i64 %115
  %117 = load i32, ptr %116, align 4, !tbaa !10
  %118 = sext i32 %117 to i64
  %119 = getelementptr inbounds i32, ptr @a, i64 %118
  %120 = load i32, ptr %119, align 4, !tbaa !10
  %121 = sext i32 %120 to i64
  %122 = getelementptr inbounds i32, ptr @a, i64 %121
  %123 = load i32, ptr %122, align 4, !tbaa !10
  %124 = sext i32 %123 to i64
  %125 = getelementptr inbounds i32, ptr @a, i64 %124
  %126 = load i32, ptr %125, align 4, !tbaa !10
  %127 = sext i32 %126 to i64
  %128 = getelementptr inbounds i32, ptr @a, i64 %127
  %129 = load i32, ptr %128, align 4, !tbaa !10
  %130 = sext i32 %129 to i64
  %131 = getelementptr inbounds i32, ptr @a, i64 %130
  %132 = load i32, ptr %131, align 4, !tbaa !10
  %133 = sext i32 %132 to i64
  %134 = getelementptr inbounds i32, ptr @a, i64 %133
  %135 = load i32, ptr %134, align 4, !tbaa !10
  %136 = sext i32 %135 to i64
  %137 = getelementptr inbounds i32, ptr @a, i64 %136
  %138 = load i32, ptr %137, align 4, !tbaa !10
  %139 = sext i32 %138 to i64
  %140 = getelementptr inbounds i32, ptr @a, i64 %139
  %141 = load i32, ptr %140, align 4, !tbaa !10
  %142 = sext i32 %141 to i64
  %143 = getelementptr inbounds i32, ptr @a, i64 %142
  %144 = load i32, ptr %143, align 4, !tbaa !10
  %145 = sext i32 %144 to i64
  %146 = getelementptr inbounds i32, ptr @a, i64 %145
  %147 = load i32, ptr %146, align 4, !tbaa !10
  %148 = sext i32 %147 to i64
  %149 = getelementptr inbounds i32, ptr @a, i64 %148
  %150 = load i32, ptr %149, align 4, !tbaa !10
  %151 = sext i32 %150 to i64
  %152 = getelementptr inbounds i32, ptr @a, i64 %151
  %153 = load i32, ptr %152, align 4, !tbaa !10
  %154 = sext i32 %153 to i64
  %155 = getelementptr inbounds i32, ptr @a, i64 %154
  %156 = load i32, ptr %155, align 4, !tbaa !10
  %157 = sext i32 %156 to i64
  %158 = getelementptr inbounds i32, ptr @a, i64 %157
  %159 = load i32, ptr %158, align 4, !tbaa !10
  %160 = sext i32 %159 to i64
  %161 = getelementptr inbounds i32, ptr @a, i64 %160
  %162 = load i32, ptr %161, align 4, !tbaa !10
  %163 = sext i32 %162 to i64
  %164 = getelementptr inbounds i32, ptr @a, i64 %163
  %165 = load i32, ptr %164, align 4, !tbaa !10
  %166 = sext i32 %165 to i64
  %167 = getelementptr inbounds i32, ptr @a, i64 %166
  %168 = load i32, ptr %167, align 4, !tbaa !10
  %169 = sext i32 %168 to i64
  %170 = getelementptr inbounds i32, ptr @a, i64 %169
  %171 = load i32, ptr %170, align 4, !tbaa !10
  %172 = sext i32 %171 to i64
  %173 = getelementptr inbounds i32, ptr @a, i64 %172
  %174 = load i32, ptr %173, align 4, !tbaa !10
  %175 = sext i32 %174 to i64
  %176 = getelementptr inbounds i32, ptr @a, i64 %175
  %177 = load i32, ptr %176, align 4, !tbaa !10
  %178 = sext i32 %177 to i64
  %179 = getelementptr inbounds i32, ptr @a, i64 %178
  %180 = load i32, ptr %179, align 4, !tbaa !10
  %181 = sext i32 %180 to i64
  %182 = getelementptr inbounds i32, ptr @a, i64 %181
  %183 = load i32, ptr %182, align 4, !tbaa !10
  %184 = sext i32 %183 to i64
  %185 = getelementptr inbounds i32, ptr @a, i64 %184
  %186 = load i32, ptr %185, align 4, !tbaa !10
  %187 = sext i32 %186 to i64
  %188 = getelementptr inbounds i32, ptr @a, i64 %187
  %189 = load i32, ptr %188, align 4, !tbaa !10
  %190 = sext i32 %189 to i64
  %191 = getelementptr inbounds i32, ptr @a, i64 %190
  %192 = load i32, ptr %191, align 4, !tbaa !10
  %193 = sext i32 %192 to i64
  %194 = getelementptr inbounds i32, ptr @a, i64 %193
  %195 = load i32, ptr %194, align 4, !tbaa !10
  %196 = sext i32 %195 to i64
  %197 = getelementptr inbounds i32, ptr @a, i64 %196
  %198 = load i32, ptr %197, align 4, !tbaa !10
  %199 = sext i32 %198 to i64
  %200 = getelementptr inbounds i32, ptr @a, i64 %199
  %201 = load i32, ptr %200, align 4, !tbaa !10
  %202 = sext i32 %201 to i64
  %203 = getelementptr inbounds i32, ptr @a, i64 %202
  %204 = load i32, ptr %203, align 4, !tbaa !10
  %205 = sext i32 %204 to i64
  %206 = getelementptr inbounds i32, ptr @a, i64 %205
  %207 = load i32, ptr %206, align 4, !tbaa !10
  %208 = sext i32 %207 to i64
  %209 = getelementptr inbounds i32, ptr @a, i64 %208
  %210 = load i32, ptr %209, align 4, !tbaa !10
  %211 = sext i32 %210 to i64
  %212 = getelementptr inbounds i32, ptr @a, i64 %211
  %213 = load i32, ptr %212, align 4, !tbaa !10
  %214 = sext i32 %213 to i64
  %215 = getelementptr inbounds i32, ptr @a, i64 %214
  %216 = load i32, ptr %215, align 4, !tbaa !10
  %217 = sext i32 %216 to i64
  %218 = getelementptr inbounds i32, ptr @a, i64 %217
  %219 = load i32, ptr %218, align 4, !tbaa !10
  %220 = sext i32 %219 to i64
  %221 = getelementptr inbounds i32, ptr @a, i64 %220
  %222 = load i32, ptr %221, align 4, !tbaa !10
  %223 = sext i32 %222 to i64
  %224 = getelementptr inbounds i32, ptr @a, i64 %223
  %225 = load i32, ptr %224, align 4, !tbaa !10
  %226 = sext i32 %225 to i64
  %227 = getelementptr inbounds i32, ptr @a, i64 %226
  %228 = load i32, ptr %227, align 4, !tbaa !10
  %229 = sext i32 %228 to i64
  %230 = getelementptr inbounds i32, ptr @a, i64 %229
  %231 = load i32, ptr %230, align 4, !tbaa !10
  %232 = sext i32 %231 to i64
  %233 = getelementptr inbounds i32, ptr @a, i64 %232
  %234 = load i32, ptr %233, align 4, !tbaa !10
  %235 = sext i32 %234 to i64
  %236 = getelementptr inbounds i32, ptr @a, i64 %235
  %237 = load i32, ptr %236, align 4, !tbaa !10
  %238 = sext i32 %237 to i64
  %239 = getelementptr inbounds i32, ptr @a, i64 %238
  %240 = load i32, ptr %239, align 4, !tbaa !10
  %241 = sext i32 %240 to i64
  %242 = getelementptr inbounds i32, ptr @a, i64 %241
  %243 = load i32, ptr %242, align 4, !tbaa !10
  %244 = sext i32 %243 to i64
  %245 = getelementptr inbounds i32, ptr @a, i64 %244
  %246 = load i32, ptr %245, align 4, !tbaa !10
  %247 = sext i32 %246 to i64
  %248 = getelementptr inbounds i32, ptr @a, i64 %247
  %249 = load i32, ptr %248, align 4, !tbaa !10
  %250 = sext i32 %249 to i64
  %251 = getelementptr inbounds i32, ptr @a, i64 %250
  %252 = load i32, ptr %251, align 4, !tbaa !10
  %253 = sext i32 %252 to i64
  %254 = getelementptr inbounds i32, ptr @a, i64 %253
  %255 = load i32, ptr %254, align 4, !tbaa !10
  %256 = sext i32 %255 to i64
  %257 = getelementptr inbounds i32, ptr @a, i64 %256
  %258 = load i32, ptr %257, align 4, !tbaa !10
  %259 = sext i32 %258 to i64
  %260 = getelementptr inbounds i32, ptr @a, i64 %259
  %261 = load i32, ptr %260, align 4, !tbaa !10
  %262 = sext i32 %261 to i64
  %263 = getelementptr inbounds i32, ptr @a, i64 %262
  %264 = load i32, ptr %263, align 4, !tbaa !10
  %265 = sext i32 %264 to i64
  %266 = getelementptr inbounds i32, ptr @a, i64 %265
  %267 = load i32, ptr %266, align 4, !tbaa !10
  %268 = sext i32 %267 to i64
  %269 = getelementptr inbounds i32, ptr @a, i64 %268
  %270 = load i32, ptr %269, align 4, !tbaa !10
  %271 = sext i32 %270 to i64
  %272 = getelementptr inbounds i32, ptr @a, i64 %271
  %273 = load i32, ptr %272, align 4, !tbaa !10
  %274 = add nsw i32 %32, 1
  %275 = icmp eq i32 %274, 0
  br i1 %275, label %276, label %31, !llvm.loop !21

276:                                              ; preds = %31
  store i32 %273, ptr @c, align 4, !tbaa !10
  store i32 0, ptr @b, align 4, !tbaa !10
  br label %277

277:                                              ; preds = %276, %26
  call void @baz()
  %278 = load i8, ptr @u, align 4, !tbaa !6
  %279 = zext i8 %278 to i64
  %280 = getelementptr inbounds nuw i32, ptr @a, i64 %279
  %281 = load i32, ptr %280, align 4, !tbaa !10
  %282 = sext i32 %281 to i64
  %283 = getelementptr inbounds i32, ptr @a, i64 %282
  %284 = load i32, ptr %283, align 4, !tbaa !10
  %285 = sext i32 %284 to i64
  %286 = getelementptr inbounds i32, ptr @a, i64 %285
  %287 = load i32, ptr %286, align 4, !tbaa !10
  %288 = sext i32 %287 to i64
  %289 = getelementptr inbounds i32, ptr @a, i64 %288
  %290 = load i32, ptr %289, align 4, !tbaa !10
  %291 = sext i32 %290 to i64
  %292 = getelementptr inbounds i32, ptr @a, i64 %291
  %293 = load i32, ptr %292, align 4, !tbaa !10
  %294 = sext i32 %293 to i64
  %295 = getelementptr inbounds i32, ptr @a, i64 %294
  %296 = load i32, ptr %295, align 4, !tbaa !10
  %297 = sext i32 %296 to i64
  %298 = getelementptr inbounds i32, ptr @a, i64 %297
  %299 = load i32, ptr %298, align 4, !tbaa !10
  %300 = sext i32 %299 to i64
  %301 = getelementptr inbounds i32, ptr @a, i64 %300
  %302 = load i32, ptr %301, align 4, !tbaa !10
  %303 = sext i32 %302 to i64
  %304 = getelementptr inbounds i32, ptr @a, i64 %303
  %305 = load i32, ptr %304, align 4, !tbaa !10
  %306 = sext i32 %305 to i64
  %307 = getelementptr inbounds i32, ptr @a, i64 %306
  %308 = load i32, ptr %307, align 4, !tbaa !10
  %309 = sext i32 %308 to i64
  %310 = getelementptr inbounds i32, ptr @a, i64 %309
  %311 = load i32, ptr %310, align 4, !tbaa !10
  %312 = sext i32 %311 to i64
  %313 = getelementptr inbounds i32, ptr @a, i64 %312
  %314 = load i32, ptr %313, align 4, !tbaa !10
  %315 = sext i32 %314 to i64
  %316 = getelementptr inbounds i32, ptr @a, i64 %315
  %317 = load i32, ptr %316, align 4, !tbaa !10
  %318 = sext i32 %317 to i64
  %319 = getelementptr inbounds i32, ptr @a, i64 %318
  %320 = load i32, ptr %319, align 4, !tbaa !10
  %321 = sext i32 %320 to i64
  %322 = getelementptr inbounds i32, ptr @a, i64 %321
  %323 = load i32, ptr %322, align 4, !tbaa !10
  %324 = icmp eq i32 %323, 0
  br i1 %324, label %325, label %326

325:                                              ; preds = %277
  call void @abort() #7
  unreachable

326:                                              ; preds = %277
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #5

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }

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
!9 = !{i64 279}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !7, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"p2 int", !14, i64 0}
!14 = !{!"any p2 pointer", !15, i64 0}
!15 = !{!"any pointer", !7, i64 0}
!16 = !{!17, !17, i64 0}
!17 = !{!"p1 int", !15, i64 0}
!18 = distinct !{!18, !19, !20}
!19 = !{!"llvm.loop.mustprogress"}
!20 = !{!"llvm.loop.unswitch.partial.disable"}
!21 = distinct !{!21, !19}
