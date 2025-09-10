; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.be.data32.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.be.data32.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@main.sample = internal unnamed_addr constant [8 x i32] [i32 0, i32 1, i32 11, i32 16, i32 129, i32 142, i32 196, i32 255], align 4
@CRCTable = internal unnamed_addr global [256 x i16] zeroinitializer, align 16

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #0 {
  %1 = load i16, ptr @CRCTable, align 16
  %2 = insertelement <8 x i16> poison, i16 %1, i64 0
  %3 = shufflevector <8 x i16> %2, <8 x i16> poison, <8 x i32> zeroinitializer
  %4 = xor <8 x i16> %3, <i16 4129, i16 8258, i16 12387, i16 16516, i16 20645, i16 24774, i16 28903, i16 -32504>
  %5 = xor <8 x i16> %3, <i16 -28375, i16 -24246, i16 -20117, i16 -15988, i16 -11859, i16 -7730, i16 -3601, i16 4657>
  %6 = xor <8 x i16> %3, <i16 528, i16 12915, i16 8786, i16 21173, i16 17044, i16 29431, i16 25302, i16 -27847>
  %7 = xor <8 x i16> %3, <i16 -31976, i16 -19589, i16 -23718, i16 -11331, i16 -15460, i16 -3073, i16 -7202, i16 9314>
  %8 = xor <8 x i16> %3, <i16 13379, i16 1056, i16 5121, i16 25830, i16 29895, i16 17572, i16 21637, i16 -23190>
  %9 = xor <8 x i16> %3, <i16 -19125, i16 -31448, i16 -27383, i16 -6674, i16 -2609, i16 -14932, i16 -10867, i16 13907>
  %10 = xor <8 x i16> %3, <i16 9842, i16 5649, i16 1584, i16 30423, i16 26358, i16 22165, i16 18100, i16 -18597>
  %11 = insertelement <4 x i16> poison, i16 %1, i64 0
  %12 = shufflevector <4 x i16> %11, <4 x i16> poison, <4 x i32> zeroinitializer
  %13 = xor <4 x i16> %12, <i16 -22662, i16 -26855, i16 -30920, i16 -2081>
  %14 = xor i16 %1, -6146
  %15 = xor i16 %1, -10339
  %16 = xor i16 %1, -14404
  br label %18

17:                                               ; preds = %70
  ret i32 %262

18:                                               ; preds = %0, %70
  %19 = phi i64 [ 0, %0 ], [ %263, %70 ]
  %20 = phi i32 [ 0, %0 ], [ %262, %70 ]
  %21 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %19
  %22 = load i32, ptr %21, align 4, !tbaa !6
  %23 = trunc i32 %22 to i16
  %24 = sub nuw nsw i64 7, %19
  %25 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %24
  %26 = load i32, ptr %25, align 4, !tbaa !6
  %27 = load i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 510), align 2, !tbaa !10
  %28 = icmp eq i16 %27, 0
  br i1 %28, label %29, label %70

29:                                               ; preds = %18
  store <8 x i16> %4, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), align 2, !tbaa !10
  store <8 x i16> %5, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 18), align 2, !tbaa !10
  store <8 x i16> %6, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 34), align 2, !tbaa !10
  store <8 x i16> %7, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 50), align 2, !tbaa !10
  store <8 x i16> %8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 66), align 2, !tbaa !10
  store <8 x i16> %9, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 82), align 2, !tbaa !10
  store <8 x i16> %10, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 98), align 2, !tbaa !10
  store <4 x i16> %13, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 114), align 2, !tbaa !10
  store i16 %14, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 122), align 2, !tbaa !10
  store i16 %15, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 124), align 4, !tbaa !10
  store i16 %16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 126), align 2, !tbaa !10
  %30 = load <8 x i16>, ptr @CRCTable, align 16, !tbaa !10
  %31 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !10
  %32 = xor <8 x i16> %30, splat (i16 18628)
  %33 = xor <8 x i16> %31, splat (i16 18628)
  store <8 x i16> %32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 16, !tbaa !10
  store <8 x i16> %33, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 144), align 16, !tbaa !10
  %34 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !10
  %35 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !10
  %36 = xor <8 x i16> %34, splat (i16 18628)
  %37 = xor <8 x i16> %35, splat (i16 18628)
  store <8 x i16> %36, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 16, !tbaa !10
  store <8 x i16> %37, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 176), align 16, !tbaa !10
  %38 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 16, !tbaa !10
  %39 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !10
  %40 = xor <8 x i16> %38, splat (i16 18628)
  %41 = xor <8 x i16> %39, splat (i16 18628)
  store <8 x i16> %40, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 16, !tbaa !10
  store <8 x i16> %41, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 16, !tbaa !10
  %42 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 16, !tbaa !10
  %43 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !10
  %44 = xor <8 x i16> %42, splat (i16 18628)
  %45 = xor <8 x i16> %43, splat (i16 18628)
  store <8 x i16> %44, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 16, !tbaa !10
  store <8 x i16> %45, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !10
  %46 = load <8 x i16>, ptr @CRCTable, align 16, !tbaa !10
  %47 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !10
  %48 = xor <8 x i16> %46, splat (i16 -28280)
  %49 = xor <8 x i16> %47, splat (i16 -28280)
  store <8 x i16> %48, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 256), align 16, !tbaa !10
  store <8 x i16> %49, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 272), align 16, !tbaa !10
  %50 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !10
  %51 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !10
  %52 = xor <8 x i16> %50, splat (i16 -28280)
  %53 = xor <8 x i16> %51, splat (i16 -28280)
  store <8 x i16> %52, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 288), align 16, !tbaa !10
  store <8 x i16> %53, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 304), align 16, !tbaa !10
  %54 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 16, !tbaa !10
  %55 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !10
  %56 = xor <8 x i16> %54, splat (i16 -28280)
  %57 = xor <8 x i16> %55, splat (i16 -28280)
  store <8 x i16> %56, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 320), align 16, !tbaa !10
  store <8 x i16> %57, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 336), align 16, !tbaa !10
  %58 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 16, !tbaa !10
  %59 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !10
  %60 = xor <8 x i16> %58, splat (i16 -28280)
  %61 = xor <8 x i16> %59, splat (i16 -28280)
  store <8 x i16> %60, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 352), align 16, !tbaa !10
  store <8 x i16> %61, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 368), align 16, !tbaa !10
  %62 = xor <8 x i16> %32, splat (i16 -28280)
  %63 = xor <8 x i16> %33, splat (i16 -28280)
  store <8 x i16> %62, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 384), align 16, !tbaa !10
  store <8 x i16> %63, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 400), align 16, !tbaa !10
  %64 = xor <8 x i16> %36, splat (i16 -28280)
  %65 = xor <8 x i16> %37, splat (i16 -28280)
  store <8 x i16> %64, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 416), align 16, !tbaa !10
  store <8 x i16> %65, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 432), align 16, !tbaa !10
  %66 = xor <8 x i16> %40, splat (i16 -28280)
  %67 = xor <8 x i16> %41, splat (i16 -28280)
  store <8 x i16> %66, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 448), align 16, !tbaa !10
  store <8 x i16> %67, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 464), align 16, !tbaa !10
  %68 = xor <8 x i16> %44, splat (i16 -28280)
  %69 = xor <8 x i16> %45, splat (i16 -28280)
  store <8 x i16> %68, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 480), align 16, !tbaa !10
  store <8 x i16> %69, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 496), align 16, !tbaa !10
  br label %70

70:                                               ; preds = %29, %18
  %71 = xor i32 %26, %22
  %72 = lshr i32 %71, 8
  %73 = shl i16 %23, 8
  %74 = and i32 %72, 255
  %75 = zext nneg i32 %74 to i64
  %76 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %75
  %77 = load i16, ptr %76, align 2, !tbaa !10
  %78 = xor i16 %77, %73
  %79 = zext i16 %78 to i32
  %80 = shl i32 %26, 8
  %81 = xor i32 %80, %79
  %82 = lshr i32 %81, 8
  %83 = shl i16 %77, 8
  %84 = and i32 %82, 255
  %85 = zext nneg i32 %84 to i64
  %86 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %85
  %87 = load i16, ptr %86, align 2, !tbaa !10
  %88 = xor i16 %87, %83
  %89 = lshr i16 %88, 8
  %90 = shl i16 %87, 8
  %91 = zext nneg i16 %89 to i64
  %92 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %91
  %93 = load i16, ptr %92, align 2, !tbaa !10
  %94 = xor i16 %93, %90
  %95 = lshr i16 %94, 8
  %96 = shl i16 %93, 8
  %97 = zext nneg i16 %95 to i64
  %98 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %97
  %99 = load i16, ptr %98, align 2, !tbaa !10
  %100 = xor i16 %99, %96
  %101 = trunc i32 %26 to i16
  %102 = xor i16 %101, %23
  %103 = shl i16 %23, 1
  %104 = xor i16 %103, 4129
  %105 = icmp slt i16 %102, 0
  %106 = select i1 %105, i16 %104, i16 %103
  %107 = shl i16 %101, 1
  %108 = xor i16 %106, %107
  %109 = shl i16 %106, 1
  %110 = xor i16 %109, 4129
  %111 = icmp slt i16 %108, 0
  %112 = select i1 %111, i16 %110, i16 %109
  %113 = shl i16 %101, 2
  %114 = xor i16 %112, %113
  %115 = shl i16 %112, 1
  %116 = xor i16 %115, 4129
  %117 = icmp slt i16 %114, 0
  %118 = select i1 %117, i16 %116, i16 %115
  %119 = shl i16 %101, 3
  %120 = xor i16 %118, %119
  %121 = shl i16 %118, 1
  %122 = xor i16 %121, 4129
  %123 = icmp slt i16 %120, 0
  %124 = select i1 %123, i16 %122, i16 %121
  %125 = shl i16 %101, 4
  %126 = xor i16 %124, %125
  %127 = shl i16 %124, 1
  %128 = xor i16 %127, 4129
  %129 = icmp slt i16 %126, 0
  %130 = select i1 %129, i16 %128, i16 %127
  %131 = shl i16 %101, 5
  %132 = xor i16 %130, %131
  %133 = shl i16 %130, 1
  %134 = xor i16 %133, 4129
  %135 = icmp slt i16 %132, 0
  %136 = select i1 %135, i16 %134, i16 %133
  %137 = shl i16 %101, 6
  %138 = xor i16 %136, %137
  %139 = shl i16 %136, 1
  %140 = xor i16 %139, 4129
  %141 = icmp slt i16 %138, 0
  %142 = select i1 %141, i16 %140, i16 %139
  %143 = shl i16 %101, 7
  %144 = xor i16 %142, %143
  %145 = shl i16 %142, 1
  %146 = xor i16 %145, 4129
  %147 = icmp slt i16 %144, 0
  %148 = select i1 %147, i16 %146, i16 %145
  %149 = shl i16 %101, 8
  %150 = xor i16 %148, %149
  %151 = shl i16 %148, 1
  %152 = xor i16 %151, 4129
  %153 = icmp slt i16 %150, 0
  %154 = select i1 %153, i16 %152, i16 %151
  %155 = shl i16 %101, 9
  %156 = xor i16 %154, %155
  %157 = shl i16 %154, 1
  %158 = xor i16 %157, 4129
  %159 = icmp slt i16 %156, 0
  %160 = select i1 %159, i16 %158, i16 %157
  %161 = shl i16 %101, 10
  %162 = xor i16 %160, %161
  %163 = shl i16 %160, 1
  %164 = xor i16 %163, 4129
  %165 = icmp slt i16 %162, 0
  %166 = select i1 %165, i16 %164, i16 %163
  %167 = shl i16 %101, 11
  %168 = xor i16 %166, %167
  %169 = shl i16 %166, 1
  %170 = xor i16 %169, 4129
  %171 = icmp slt i16 %168, 0
  %172 = select i1 %171, i16 %170, i16 %169
  %173 = shl i16 %101, 12
  %174 = xor i16 %172, %173
  %175 = shl i16 %172, 1
  %176 = xor i16 %175, 4129
  %177 = icmp slt i16 %174, 0
  %178 = select i1 %177, i16 %176, i16 %175
  %179 = shl i16 %101, 13
  %180 = xor i16 %178, %179
  %181 = shl i16 %178, 1
  %182 = xor i16 %181, 4129
  %183 = icmp slt i16 %180, 0
  %184 = select i1 %183, i16 %182, i16 %181
  %185 = shl i16 %101, 14
  %186 = xor i16 %184, %185
  %187 = shl i16 %184, 1
  %188 = xor i16 %187, 4129
  %189 = icmp slt i16 %186, 0
  %190 = select i1 %189, i16 %188, i16 %187
  %191 = shl i16 %101, 15
  %192 = xor i16 %190, %191
  %193 = shl i16 %190, 1
  %194 = xor i16 %193, 4129
  %195 = icmp slt i16 %192, 0
  %196 = select i1 %195, i16 %194, i16 %193
  %197 = shl i16 %196, 1
  %198 = xor i16 %197, 4129
  %199 = icmp slt i16 %196, 0
  %200 = select i1 %199, i16 %198, i16 %197
  %201 = shl i16 %200, 1
  %202 = xor i16 %201, 4129
  %203 = icmp slt i16 %200, 0
  %204 = select i1 %203, i16 %202, i16 %201
  %205 = shl i16 %204, 1
  %206 = xor i16 %205, 4129
  %207 = icmp slt i16 %204, 0
  %208 = select i1 %207, i16 %206, i16 %205
  %209 = shl i16 %208, 1
  %210 = xor i16 %209, 4129
  %211 = icmp slt i16 %208, 0
  %212 = select i1 %211, i16 %210, i16 %209
  %213 = shl i16 %212, 1
  %214 = xor i16 %213, 4129
  %215 = icmp slt i16 %212, 0
  %216 = select i1 %215, i16 %214, i16 %213
  %217 = shl i16 %216, 1
  %218 = xor i16 %217, 4129
  %219 = icmp slt i16 %216, 0
  %220 = select i1 %219, i16 %218, i16 %217
  %221 = shl i16 %220, 1
  %222 = xor i16 %221, 4129
  %223 = icmp slt i16 %220, 0
  %224 = select i1 %223, i16 %222, i16 %221
  %225 = shl i16 %224, 1
  %226 = xor i16 %225, 4129
  %227 = icmp slt i16 %224, 0
  %228 = select i1 %227, i16 %226, i16 %225
  %229 = shl i16 %228, 1
  %230 = xor i16 %229, 4129
  %231 = icmp slt i16 %228, 0
  %232 = select i1 %231, i16 %230, i16 %229
  %233 = shl i16 %232, 1
  %234 = xor i16 %233, 4129
  %235 = icmp slt i16 %232, 0
  %236 = select i1 %235, i16 %234, i16 %233
  %237 = shl i16 %236, 1
  %238 = xor i16 %237, 4129
  %239 = icmp slt i16 %236, 0
  %240 = select i1 %239, i16 %238, i16 %237
  %241 = shl i16 %240, 1
  %242 = xor i16 %241, 4129
  %243 = icmp slt i16 %240, 0
  %244 = select i1 %243, i16 %242, i16 %241
  %245 = shl i16 %244, 1
  %246 = xor i16 %245, 4129
  %247 = icmp slt i16 %244, 0
  %248 = select i1 %247, i16 %246, i16 %245
  %249 = shl i16 %248, 1
  %250 = xor i16 %249, 4129
  %251 = icmp slt i16 %248, 0
  %252 = select i1 %251, i16 %250, i16 %249
  %253 = shl i16 %252, 1
  %254 = xor i16 %253, 4129
  %255 = icmp slt i16 %252, 0
  %256 = select i1 %255, i16 %254, i16 %253
  %257 = shl i16 %256, 1
  %258 = xor i16 %257, 4129
  %259 = icmp slt i16 %256, 0
  %260 = select i1 %259, i16 %258, i16 %257
  %261 = icmp eq i16 %100, %260
  %262 = select i1 %261, i32 %20, i32 1
  %263 = add nuw nsw i64 %19, 1
  %264 = icmp eq i64 %263, 8
  br i1 %264, label %17, label %18, !llvm.loop !12
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"short", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
