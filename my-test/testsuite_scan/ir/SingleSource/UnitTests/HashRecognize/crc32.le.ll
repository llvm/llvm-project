; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc32.le.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc32.le.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@main.sample = internal unnamed_addr constant [8 x i32] [i32 0, i32 1, i32 11, i32 16, i32 129, i32 142, i32 196, i32 255], align 4
@CRCTable = internal unnamed_addr global [256 x i32] zeroinitializer, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @CRCTable, align 4
  %2 = xor i32 %1, 33800
  %3 = xor i32 %1, 16900
  %4 = xor i32 %1, 50700
  %5 = xor i32 %1, 8450
  %6 = xor i32 %1, 25350
  %7 = xor i32 %1, 42250
  %8 = xor i32 %1, 59150
  %9 = xor i32 %1, 4225
  %10 = xor i32 %1, 12675
  %11 = xor i32 %1, 21125
  %12 = xor i32 %1, 29575
  %13 = xor i32 %1, 38025
  %14 = xor i32 %1, 46475
  %15 = xor i32 %1, 54925
  %16 = xor i32 %1, 63375
  %17 = xor i32 %1, 35912
  %18 = xor i32 %1, 40137
  %19 = xor i32 %1, 44362
  %20 = xor i32 %1, 48587
  %21 = xor i32 %1, 52812
  %22 = xor i32 %1, 57037
  %23 = xor i32 %1, 61262
  %24 = xor i32 %1, 65487
  %25 = xor i32 %1, 2112
  %26 = xor i32 %1, 6337
  %27 = xor i32 %1, 10562
  %28 = xor i32 %1, 14787
  %29 = xor i32 %1, 19012
  %30 = xor i32 %1, 23237
  %31 = xor i32 %1, 27462
  %32 = xor i32 %1, 31687
  %33 = xor i32 %1, 17956
  %34 = xor i32 %1, 51820
  %35 = xor i32 %1, 22181
  %36 = xor i32 %1, 56045
  %37 = xor i32 %1, 26406
  %38 = xor i32 %1, 60270
  %39 = xor i32 %1, 30631
  %40 = xor i32 %1, 64495
  %41 = xor i32 %1, 1056
  %42 = xor i32 %1, 34920
  %43 = xor i32 %1, 5281
  %44 = xor i32 %1, 39145
  %45 = xor i32 %1, 9506
  %46 = xor i32 %1, 43370
  %47 = xor i32 %1, 13731
  %48 = xor i32 %1, 47595
  %49 = xor i32 %1, 49708
  %50 = xor i32 %1, 20068
  %51 = xor i32 %1, 53933
  %52 = xor i32 %1, 24293
  %53 = xor i32 %1, 58158
  %54 = xor i32 %1, 28518
  %55 = xor i32 %1, 62383
  %56 = xor i32 %1, 32743
  %57 = xor i32 %1, 32808
  %58 = xor i32 %1, 3168
  %59 = xor i32 %1, 37033
  %60 = xor i32 %1, 7393
  %61 = xor i32 %1, 41258
  %62 = xor i32 %1, 11618
  %63 = xor i32 %1, 45483
  %64 = xor i32 %1, 15843
  br label %66

65:                                               ; preds = %105
  ret i32 %357

66:                                               ; preds = %0, %105
  %67 = phi i64 [ 0, %0 ], [ %358, %105 ]
  %68 = phi i32 [ 0, %0 ], [ %357, %105 ]
  %69 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %67
  %70 = load i32, ptr %69, align 4, !tbaa !6
  %71 = sub nuw nsw i64 7, %67
  %72 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %71
  %73 = load i32, ptr %72, align 4, !tbaa !6
  %74 = load i32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 1020), align 4, !tbaa !6
  %75 = icmp eq i32 %74, 0
  br i1 %75, label %76, label %105

76:                                               ; preds = %66
  store i32 %2, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 512), align 4, !tbaa !6
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 256), align 4, !tbaa !6
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 768), align 4, !tbaa !6
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 4, !tbaa !6
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 384), align 4, !tbaa !6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 640), align 4, !tbaa !6
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 896), align 4, !tbaa !6
  store i32 %9, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 4, !tbaa !6
  store i32 %10, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 4, !tbaa !6
  store i32 %11, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 320), align 4, !tbaa !6
  store i32 %12, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 448), align 4, !tbaa !6
  store i32 %13, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 576), align 4, !tbaa !6
  store i32 %14, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 704), align 4, !tbaa !6
  store i32 %15, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 832), align 4, !tbaa !6
  store i32 %16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 960), align 4, !tbaa !6
  store i32 %17, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 4, !tbaa !6
  store i32 %18, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 4, !tbaa !6
  store i32 %19, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 4, !tbaa !6
  store i32 %20, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 4, !tbaa !6
  store i32 %21, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 288), align 4, !tbaa !6
  store i32 %22, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 352), align 4, !tbaa !6
  store i32 %23, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 416), align 4, !tbaa !6
  store i32 %24, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 480), align 4, !tbaa !6
  store i32 %25, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 544), align 4, !tbaa !6
  store i32 %26, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 608), align 4, !tbaa !6
  store i32 %27, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 672), align 4, !tbaa !6
  store i32 %28, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 736), align 4, !tbaa !6
  store i32 %29, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 800), align 4, !tbaa !6
  store i32 %30, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 864), align 4, !tbaa !6
  store i32 %31, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 928), align 4, !tbaa !6
  store i32 %32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 992), align 4, !tbaa !6
  store i32 %33, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 4, !tbaa !6
  store i32 %34, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 4, !tbaa !6
  store i32 %35, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 4, !tbaa !6
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 4, !tbaa !6
  store i32 %37, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 144), align 4, !tbaa !6
  store i32 %38, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 176), align 4, !tbaa !6
  store i32 %39, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 4, !tbaa !6
  store i32 %40, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 4, !tbaa !6
  store i32 %41, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 272), align 4, !tbaa !6
  store i32 %42, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 304), align 4, !tbaa !6
  store i32 %43, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 336), align 4, !tbaa !6
  store i32 %44, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 368), align 4, !tbaa !6
  store i32 %45, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 400), align 4, !tbaa !6
  store i32 %46, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 432), align 4, !tbaa !6
  store i32 %47, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 464), align 4, !tbaa !6
  store i32 %48, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 496), align 4, !tbaa !6
  store i32 %49, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 528), align 4, !tbaa !6
  store i32 %50, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 560), align 4, !tbaa !6
  store i32 %51, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 592), align 4, !tbaa !6
  store i32 %52, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 624), align 4, !tbaa !6
  store i32 %53, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 656), align 4, !tbaa !6
  store i32 %54, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 688), align 4, !tbaa !6
  store i32 %55, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 720), align 4, !tbaa !6
  store i32 %56, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 752), align 4, !tbaa !6
  store i32 %57, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 784), align 4, !tbaa !6
  store i32 %58, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 816), align 4, !tbaa !6
  store i32 %59, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 848), align 4, !tbaa !6
  store i32 %60, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 880), align 4, !tbaa !6
  store i32 %61, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 912), align 4, !tbaa !6
  store i32 %62, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 944), align 4, !tbaa !6
  store i32 %63, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 976), align 4, !tbaa !6
  store i32 %64, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 1008), align 4, !tbaa !6
  br label %77

77:                                               ; preds = %77, %76
  %78 = phi i64 [ 0, %76 ], [ %89, %77 ]
  %79 = shl i64 %78, 2
  %80 = or disjoint i64 %79, 4
  %81 = getelementptr inbounds nuw i32, ptr @CRCTable, i64 %79
  %82 = getelementptr inbounds nuw i32, ptr @CRCTable, i64 %80
  %83 = load i32, ptr %81, align 4, !tbaa !6
  %84 = load i32, ptr %82, align 4, !tbaa !6
  %85 = xor i32 %83, 8978
  %86 = xor i32 %84, 8978
  %87 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 8), i64 %79
  %88 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 8), i64 %80
  store i32 %85, ptr %87, align 4, !tbaa !6
  store i32 %86, ptr %88, align 4, !tbaa !6
  %89 = add nuw i64 %78, 2
  %90 = icmp eq i64 %89, 64
  br i1 %90, label %91, label %77, !llvm.loop !10

91:                                               ; preds = %77, %91
  %92 = phi i64 [ %103, %91 ], [ 0, %77 ]
  %93 = shl i64 %92, 1
  %94 = or disjoint i64 %93, 2
  %95 = getelementptr inbounds nuw i32, ptr @CRCTable, i64 %93
  %96 = getelementptr inbounds nuw i32, ptr @CRCTable, i64 %94
  %97 = load i32, ptr %95, align 4, !tbaa !6
  %98 = load i32, ptr %96, align 4, !tbaa !6
  %99 = xor i32 %97, 4489
  %100 = xor i32 %98, 4489
  %101 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 4), i64 %93
  %102 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 4), i64 %94
  store i32 %99, ptr %101, align 4, !tbaa !6
  store i32 %100, ptr %102, align 4, !tbaa !6
  %103 = add nuw i64 %92, 2
  %104 = icmp eq i64 %103, 128
  br i1 %104, label %105, label %91, !llvm.loop !14

105:                                              ; preds = %91, %66
  %106 = xor i32 %73, %70
  %107 = and i32 %106, 255
  %108 = zext nneg i32 %107 to i64
  %109 = getelementptr inbounds nuw i32, ptr @CRCTable, i64 %108
  %110 = load i32, ptr %109, align 4, !tbaa !6
  %111 = lshr i32 %106, 8
  %112 = xor i32 %110, %111
  %113 = lshr i32 %70, 16
  %114 = lshr i32 %110, 8
  %115 = xor i32 %114, %113
  %116 = and i32 %112, 255
  %117 = zext nneg i32 %116 to i64
  %118 = getelementptr inbounds nuw i32, ptr @CRCTable, i64 %117
  %119 = load i32, ptr %118, align 4, !tbaa !6
  %120 = xor i32 %115, %119
  %121 = lshr i32 %73, 16
  %122 = xor i32 %120, %121
  %123 = lshr i32 %120, 8
  %124 = and i32 %122, 255
  %125 = zext nneg i32 %124 to i64
  %126 = getelementptr inbounds nuw i32, ptr @CRCTable, i64 %125
  %127 = load i32, ptr %126, align 4, !tbaa !6
  %128 = xor i32 %123, %127
  %129 = lshr i32 %73, 24
  %130 = lshr i32 %128, 8
  %131 = and i32 %128, 255
  %132 = xor i32 %131, %129
  %133 = zext nneg i32 %132 to i64
  %134 = getelementptr inbounds nuw i32, ptr @CRCTable, i64 %133
  %135 = load i32, ptr %134, align 4, !tbaa !6
  %136 = xor i32 %130, %135
  %137 = lshr i32 %70, 1
  %138 = and i32 %106, 1
  %139 = icmp eq i32 %138, 0
  %140 = xor i32 %137, 33800
  %141 = select i1 %139, i32 %137, i32 %140
  %142 = lshr i32 %73, 1
  %143 = xor i32 %141, %142
  %144 = lshr i32 %141, 1
  %145 = and i32 %143, 1
  %146 = icmp eq i32 %145, 0
  %147 = xor i32 %144, 33800
  %148 = select i1 %146, i32 %144, i32 %147
  %149 = lshr i32 %73, 2
  %150 = xor i32 %148, %149
  %151 = lshr i32 %148, 1
  %152 = and i32 %150, 1
  %153 = icmp eq i32 %152, 0
  %154 = xor i32 %151, 33800
  %155 = select i1 %153, i32 %151, i32 %154
  %156 = lshr i32 %73, 3
  %157 = xor i32 %155, %156
  %158 = lshr i32 %155, 1
  %159 = and i32 %157, 1
  %160 = icmp eq i32 %159, 0
  %161 = xor i32 %158, 33800
  %162 = select i1 %160, i32 %158, i32 %161
  %163 = lshr i32 %73, 4
  %164 = xor i32 %162, %163
  %165 = lshr i32 %162, 1
  %166 = and i32 %164, 1
  %167 = icmp eq i32 %166, 0
  %168 = xor i32 %165, 33800
  %169 = select i1 %167, i32 %165, i32 %168
  %170 = lshr i32 %73, 5
  %171 = xor i32 %169, %170
  %172 = lshr i32 %169, 1
  %173 = and i32 %171, 1
  %174 = icmp eq i32 %173, 0
  %175 = xor i32 %172, 33800
  %176 = select i1 %174, i32 %172, i32 %175
  %177 = lshr i32 %73, 6
  %178 = xor i32 %176, %177
  %179 = lshr i32 %176, 1
  %180 = and i32 %178, 1
  %181 = icmp eq i32 %180, 0
  %182 = xor i32 %179, 33800
  %183 = select i1 %181, i32 %179, i32 %182
  %184 = lshr i32 %73, 7
  %185 = xor i32 %183, %184
  %186 = lshr i32 %183, 1
  %187 = and i32 %185, 1
  %188 = icmp eq i32 %187, 0
  %189 = xor i32 %186, 33800
  %190 = select i1 %188, i32 %186, i32 %189
  %191 = lshr i32 %73, 8
  %192 = xor i32 %190, %191
  %193 = lshr i32 %190, 1
  %194 = and i32 %192, 1
  %195 = icmp eq i32 %194, 0
  %196 = xor i32 %193, 33800
  %197 = select i1 %195, i32 %193, i32 %196
  %198 = lshr i32 %73, 9
  %199 = xor i32 %197, %198
  %200 = lshr i32 %197, 1
  %201 = and i32 %199, 1
  %202 = icmp eq i32 %201, 0
  %203 = xor i32 %200, 33800
  %204 = select i1 %202, i32 %200, i32 %203
  %205 = lshr i32 %73, 10
  %206 = xor i32 %204, %205
  %207 = lshr i32 %204, 1
  %208 = and i32 %206, 1
  %209 = icmp eq i32 %208, 0
  %210 = xor i32 %207, 33800
  %211 = select i1 %209, i32 %207, i32 %210
  %212 = lshr i32 %73, 11
  %213 = xor i32 %211, %212
  %214 = lshr i32 %211, 1
  %215 = and i32 %213, 1
  %216 = icmp eq i32 %215, 0
  %217 = xor i32 %214, 33800
  %218 = select i1 %216, i32 %214, i32 %217
  %219 = lshr i32 %73, 12
  %220 = xor i32 %218, %219
  %221 = lshr i32 %218, 1
  %222 = and i32 %220, 1
  %223 = icmp eq i32 %222, 0
  %224 = xor i32 %221, 33800
  %225 = select i1 %223, i32 %221, i32 %224
  %226 = lshr i32 %73, 13
  %227 = xor i32 %225, %226
  %228 = lshr i32 %225, 1
  %229 = and i32 %227, 1
  %230 = icmp eq i32 %229, 0
  %231 = xor i32 %228, 33800
  %232 = select i1 %230, i32 %228, i32 %231
  %233 = lshr i32 %73, 14
  %234 = xor i32 %232, %233
  %235 = lshr i32 %232, 1
  %236 = and i32 %234, 1
  %237 = icmp eq i32 %236, 0
  %238 = xor i32 %235, 33800
  %239 = select i1 %237, i32 %235, i32 %238
  %240 = lshr i32 %73, 15
  %241 = xor i32 %239, %240
  %242 = lshr i32 %239, 1
  %243 = and i32 %241, 1
  %244 = icmp eq i32 %243, 0
  %245 = xor i32 %242, 33800
  %246 = select i1 %244, i32 %242, i32 %245
  %247 = xor i32 %246, %121
  %248 = lshr i32 %246, 1
  %249 = and i32 %247, 1
  %250 = icmp eq i32 %249, 0
  %251 = xor i32 %248, 33800
  %252 = select i1 %250, i32 %248, i32 %251
  %253 = lshr i32 %73, 17
  %254 = xor i32 %252, %253
  %255 = lshr i32 %252, 1
  %256 = and i32 %254, 1
  %257 = icmp eq i32 %256, 0
  %258 = xor i32 %255, 33800
  %259 = select i1 %257, i32 %255, i32 %258
  %260 = lshr i32 %73, 18
  %261 = xor i32 %259, %260
  %262 = lshr i32 %259, 1
  %263 = and i32 %261, 1
  %264 = icmp eq i32 %263, 0
  %265 = xor i32 %262, 33800
  %266 = select i1 %264, i32 %262, i32 %265
  %267 = lshr i32 %73, 19
  %268 = xor i32 %266, %267
  %269 = lshr i32 %266, 1
  %270 = and i32 %268, 1
  %271 = icmp eq i32 %270, 0
  %272 = xor i32 %269, 33800
  %273 = select i1 %271, i32 %269, i32 %272
  %274 = lshr i32 %73, 20
  %275 = xor i32 %273, %274
  %276 = lshr i32 %273, 1
  %277 = and i32 %275, 1
  %278 = icmp eq i32 %277, 0
  %279 = xor i32 %276, 33800
  %280 = select i1 %278, i32 %276, i32 %279
  %281 = lshr i32 %73, 21
  %282 = xor i32 %280, %281
  %283 = lshr i32 %280, 1
  %284 = and i32 %282, 1
  %285 = icmp eq i32 %284, 0
  %286 = xor i32 %283, 33800
  %287 = select i1 %285, i32 %283, i32 %286
  %288 = lshr i32 %73, 22
  %289 = xor i32 %287, %288
  %290 = lshr i32 %287, 1
  %291 = and i32 %289, 1
  %292 = icmp eq i32 %291, 0
  %293 = xor i32 %290, 33800
  %294 = select i1 %292, i32 %290, i32 %293
  %295 = lshr i32 %73, 23
  %296 = xor i32 %294, %295
  %297 = lshr i32 %294, 1
  %298 = and i32 %296, 1
  %299 = icmp eq i32 %298, 0
  %300 = xor i32 %297, 33800
  %301 = select i1 %299, i32 %297, i32 %300
  %302 = xor i32 %301, %129
  %303 = lshr i32 %301, 1
  %304 = and i32 %302, 1
  %305 = icmp eq i32 %304, 0
  %306 = xor i32 %303, 33800
  %307 = select i1 %305, i32 %303, i32 %306
  %308 = lshr i32 %73, 25
  %309 = xor i32 %307, %308
  %310 = lshr i32 %307, 1
  %311 = and i32 %309, 1
  %312 = icmp eq i32 %311, 0
  %313 = xor i32 %310, 33800
  %314 = select i1 %312, i32 %310, i32 %313
  %315 = lshr i32 %73, 26
  %316 = xor i32 %314, %315
  %317 = lshr i32 %314, 1
  %318 = and i32 %316, 1
  %319 = icmp eq i32 %318, 0
  %320 = xor i32 %317, 33800
  %321 = select i1 %319, i32 %317, i32 %320
  %322 = lshr i32 %73, 27
  %323 = xor i32 %321, %322
  %324 = lshr i32 %321, 1
  %325 = and i32 %323, 1
  %326 = icmp eq i32 %325, 0
  %327 = xor i32 %324, 33800
  %328 = select i1 %326, i32 %324, i32 %327
  %329 = lshr i32 %73, 28
  %330 = xor i32 %328, %329
  %331 = lshr i32 %328, 1
  %332 = and i32 %330, 1
  %333 = icmp eq i32 %332, 0
  %334 = xor i32 %331, 33800
  %335 = select i1 %333, i32 %331, i32 %334
  %336 = lshr i32 %73, 29
  %337 = xor i32 %335, %336
  %338 = lshr i32 %335, 1
  %339 = and i32 %337, 1
  %340 = icmp eq i32 %339, 0
  %341 = xor i32 %338, 33800
  %342 = select i1 %340, i32 %338, i32 %341
  %343 = lshr i32 %73, 30
  %344 = xor i32 %342, %343
  %345 = lshr i32 %342, 1
  %346 = and i32 %344, 1
  %347 = icmp eq i32 %346, 0
  %348 = xor i32 %345, 33800
  %349 = select i1 %347, i32 %345, i32 %348
  %350 = lshr i32 %73, 31
  %351 = lshr i32 %349, 1
  %352 = and i32 %349, 1
  %353 = icmp eq i32 %350, %352
  %354 = xor i32 %351, 33800
  %355 = select i1 %353, i32 %351, i32 %354
  %356 = icmp eq i32 %136, %355
  %357 = select i1 %356, i32 %68, i32 1
  %358 = add nuw nsw i64 %67, 1
  %359 = icmp eq i64 %358, 8
  br i1 %359, label %65, label %66, !llvm.loop !15
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
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11, !12, !13}
!15 = distinct !{!15, !11}
