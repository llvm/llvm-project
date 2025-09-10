; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/va-arg-22.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/va-arg-22.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.A31 = type { [31 x i8] }
%struct.A32 = type { [32 x i8] }
%struct.A35 = type { [35 x i8] }
%struct.A72 = type { [72 x i8] }
%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@bar.lastn = internal unnamed_addr global i32 -1, align 4
@bar.lastc = internal unnamed_addr global i32 -1, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local void @bar(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = load i32, ptr @bar.lastn, align 4, !tbaa !6
  %4 = icmp eq i32 %3, %0
  %5 = load i32, ptr @bar.lastc, align 4, !tbaa !6
  br i1 %4, label %10, label %6

6:                                                ; preds = %2
  %7 = icmp eq i32 %5, %3
  br i1 %7, label %9, label %8

8:                                                ; preds = %6
  tail call void @abort() #7
  unreachable

9:                                                ; preds = %6
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 %0, ptr @bar.lastn, align 4, !tbaa !6
  br label %10

10:                                               ; preds = %9, %2
  %11 = phi i32 [ 0, %9 ], [ %5, %2 ]
  %12 = shl i32 %0, 3
  %13 = xor i32 %11, %12
  %14 = and i32 %13, 255
  %15 = icmp eq i32 %1, %14
  br i1 %15, label %17, label %16

16:                                               ; preds = %10
  tail call void @abort() #7
  unreachable

17:                                               ; preds = %10
  %18 = add nsw i32 %11, 1
  store i32 %18, ptr @bar.lastc, align 4, !tbaa !6
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(i32 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.A31, align 4
  %3 = alloca %struct.A32, align 4
  %4 = alloca %struct.A35, align 4
  %5 = alloca %struct.A72, align 4
  %6 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #8
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #8
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #8
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #8
  %7 = icmp eq i32 %0, 21
  br i1 %7, label %9, label %8

8:                                                ; preds = %1
  tail call void @abort() #7
  unreachable

9:                                                ; preds = %1
  call void @llvm.va_start.p0(ptr nonnull %6)
  %10 = getelementptr inbounds nuw i8, ptr %6, i64 24
  %11 = load i32, ptr %10, align 8
  %12 = icmp sgt i32 %11, -1
  br i1 %12, label %21, label %13

13:                                               ; preds = %9
  %14 = add nsw i32 %11, 8
  store i32 %14, ptr %10, align 8
  %15 = icmp samesign ult i32 %11, -7
  br i1 %15, label %16, label %21

16:                                               ; preds = %13
  %17 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %18 = load ptr, ptr %17, align 8
  %19 = sext i32 %11 to i64
  %20 = getelementptr inbounds i8, ptr %18, i64 %19
  br label %25

21:                                               ; preds = %13, %9
  %22 = phi i32 [ %14, %13 ], [ %11, %9 ]
  %23 = load ptr, ptr %6, align 8
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 8
  store ptr %24, ptr %6, align 8
  br label %25

25:                                               ; preds = %21, %16
  %26 = phi i32 [ %14, %16 ], [ %22, %21 ]
  %27 = phi ptr [ %20, %16 ], [ %23, %21 ]
  %28 = load i8, ptr %27, align 8, !tbaa !10
  %29 = load i32, ptr @bar.lastn, align 4
  %30 = load i32, ptr @bar.lastc, align 4
  %31 = zext i8 %28 to i32
  %32 = icmp eq i32 %29, 1
  br i1 %32, label %37, label %33

33:                                               ; preds = %25
  %34 = icmp eq i32 %30, %29
  br i1 %34, label %36, label %35

35:                                               ; preds = %33
  call void @abort() #7
  unreachable

36:                                               ; preds = %33
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 1, ptr @bar.lastn, align 4, !tbaa !6
  br label %37

37:                                               ; preds = %36, %25
  %38 = phi i32 [ 0, %36 ], [ %30, %25 ]
  %39 = and i32 %38, 255
  %40 = xor i32 %39, %31
  %41 = icmp eq i32 %40, 8
  br i1 %41, label %43, label %42

42:                                               ; preds = %37
  call void @abort() #7
  unreachable

43:                                               ; preds = %37
  %44 = add nsw i32 %38, 1
  store i32 %44, ptr @bar.lastc, align 4, !tbaa !6
  %45 = icmp sgt i32 %26, -1
  br i1 %45, label %54, label %46

46:                                               ; preds = %43
  %47 = add nsw i32 %26, 8
  store i32 %47, ptr %10, align 8
  %48 = icmp samesign ult i32 %26, -7
  br i1 %48, label %49, label %54

49:                                               ; preds = %46
  %50 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %51 = load ptr, ptr %50, align 8
  %52 = sext i32 %26 to i64
  %53 = getelementptr inbounds i8, ptr %51, i64 %52
  br label %58

54:                                               ; preds = %46, %43
  %55 = phi i32 [ %47, %46 ], [ %26, %43 ]
  %56 = load ptr, ptr %6, align 8
  %57 = getelementptr inbounds nuw i8, ptr %56, i64 8
  store ptr %57, ptr %6, align 8
  br label %58

58:                                               ; preds = %49, %54
  %59 = phi i32 [ %47, %49 ], [ %55, %54 ]
  %60 = phi ptr [ %53, %49 ], [ %56, %54 ]
  %61 = load i16, ptr %60, align 8, !tbaa !10
  %62 = icmp eq i32 %38, 0
  br i1 %62, label %64, label %63

63:                                               ; preds = %58
  call void @abort() #7
  unreachable

64:                                               ; preds = %58
  %65 = and i16 %61, 255
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 2, ptr @bar.lastn, align 4, !tbaa !6
  %66 = icmp eq i16 %65, 16
  br i1 %66, label %68, label %67

67:                                               ; preds = %68, %64
  call void @abort() #7
  unreachable

68:                                               ; preds = %64
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %69 = and i16 %61, -256
  %70 = icmp eq i16 %69, 4352
  br i1 %70, label %71, label %67

71:                                               ; preds = %68
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %72 = icmp sgt i32 %59, -1
  br i1 %72, label %81, label %73

73:                                               ; preds = %71
  %74 = add nsw i32 %59, 8
  store i32 %74, ptr %10, align 8
  %75 = icmp samesign ult i32 %59, -7
  br i1 %75, label %76, label %81

76:                                               ; preds = %73
  %77 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %78 = load ptr, ptr %77, align 8
  %79 = sext i32 %59 to i64
  %80 = getelementptr inbounds i8, ptr %78, i64 %79
  br label %85

81:                                               ; preds = %73, %71
  %82 = phi i32 [ %74, %73 ], [ %59, %71 ]
  %83 = load ptr, ptr %6, align 8
  %84 = getelementptr inbounds nuw i8, ptr %83, i64 8
  store ptr %84, ptr %6, align 8
  br label %85

85:                                               ; preds = %76, %81
  %86 = phi i32 [ %74, %76 ], [ %82, %81 ]
  %87 = phi ptr [ %80, %76 ], [ %83, %81 ]
  %88 = load i8, ptr %87, align 8
  %89 = getelementptr inbounds nuw i8, ptr %87, i64 1
  %90 = load i8, ptr %89, align 1
  %91 = getelementptr inbounds nuw i8, ptr %87, i64 2
  %92 = load i8, ptr %91, align 2, !tbaa !10
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 3, ptr @bar.lastn, align 4, !tbaa !6
  %93 = icmp eq i8 %88, 24
  br i1 %93, label %95, label %94

94:                                               ; preds = %97, %95, %85
  call void @abort() #7
  unreachable

95:                                               ; preds = %85
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %96 = icmp eq i8 %90, 25
  br i1 %96, label %97, label %94

97:                                               ; preds = %95
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %98 = icmp eq i8 %92, 26
  br i1 %98, label %99, label %94

99:                                               ; preds = %97
  store i32 3, ptr @bar.lastc, align 4, !tbaa !6
  %100 = icmp sgt i32 %86, -1
  br i1 %100, label %109, label %101

101:                                              ; preds = %99
  %102 = add nsw i32 %86, 8
  store i32 %102, ptr %10, align 8
  %103 = icmp samesign ult i32 %86, -7
  br i1 %103, label %104, label %109

104:                                              ; preds = %101
  %105 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %106 = load ptr, ptr %105, align 8
  %107 = sext i32 %86 to i64
  %108 = getelementptr inbounds i8, ptr %106, i64 %107
  br label %113

109:                                              ; preds = %101, %99
  %110 = phi i32 [ %102, %101 ], [ %86, %99 ]
  %111 = load ptr, ptr %6, align 8
  %112 = getelementptr inbounds nuw i8, ptr %111, i64 8
  store ptr %112, ptr %6, align 8
  br label %113

113:                                              ; preds = %104, %109
  %114 = phi i32 [ %102, %104 ], [ %110, %109 ]
  %115 = phi ptr [ %108, %104 ], [ %111, %109 ]
  %116 = load i32, ptr %115, align 8, !tbaa !10
  %117 = and i32 %116, 255
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 4, ptr @bar.lastn, align 4, !tbaa !6
  %118 = icmp eq i32 %117, 32
  br i1 %118, label %120, label %119

119:                                              ; preds = %126, %123, %120, %113
  call void @abort() #7
  unreachable

120:                                              ; preds = %113
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %121 = and i32 %116, 65280
  %122 = icmp eq i32 %121, 8448
  br i1 %122, label %123, label %119

123:                                              ; preds = %120
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %124 = and i32 %116, 16711680
  %125 = icmp eq i32 %124, 2228224
  br i1 %125, label %126, label %119

126:                                              ; preds = %123
  store i32 3, ptr @bar.lastc, align 4, !tbaa !6
  %127 = and i32 %116, -16777216
  %128 = icmp eq i32 %127, 587202560
  br i1 %128, label %129, label %119

129:                                              ; preds = %126
  store i32 4, ptr @bar.lastc, align 4, !tbaa !6
  %130 = icmp sgt i32 %114, -1
  br i1 %130, label %139, label %131

131:                                              ; preds = %129
  %132 = add nsw i32 %114, 8
  store i32 %132, ptr %10, align 8
  %133 = icmp samesign ult i32 %114, -7
  br i1 %133, label %134, label %139

134:                                              ; preds = %131
  %135 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %136 = load ptr, ptr %135, align 8
  %137 = sext i32 %114 to i64
  %138 = getelementptr inbounds i8, ptr %136, i64 %137
  br label %143

139:                                              ; preds = %131, %129
  %140 = phi i32 [ %132, %131 ], [ %114, %129 ]
  %141 = load ptr, ptr %6, align 8
  %142 = getelementptr inbounds nuw i8, ptr %141, i64 8
  store ptr %142, ptr %6, align 8
  br label %143

143:                                              ; preds = %134, %139
  %144 = phi i32 [ %132, %134 ], [ %140, %139 ]
  %145 = phi ptr [ %138, %134 ], [ %141, %139 ]
  %146 = load i8, ptr %145, align 8
  %147 = getelementptr inbounds nuw i8, ptr %145, i64 1
  %148 = load i8, ptr %147, align 1
  %149 = getelementptr inbounds nuw i8, ptr %145, i64 2
  %150 = load i8, ptr %149, align 2
  %151 = getelementptr inbounds nuw i8, ptr %145, i64 3
  %152 = load i8, ptr %151, align 1
  %153 = getelementptr inbounds nuw i8, ptr %145, i64 4
  %154 = load i8, ptr %153, align 4, !tbaa !10
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 5, ptr @bar.lastn, align 4, !tbaa !6
  %155 = icmp eq i8 %146, 40
  br i1 %155, label %157, label %156

156:                                              ; preds = %163, %161, %159, %157, %143
  call void @abort() #7
  unreachable

157:                                              ; preds = %143
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %158 = icmp eq i8 %148, 41
  br i1 %158, label %159, label %156

159:                                              ; preds = %157
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %160 = icmp eq i8 %150, 42
  br i1 %160, label %161, label %156

161:                                              ; preds = %159
  store i32 3, ptr @bar.lastc, align 4, !tbaa !6
  %162 = icmp eq i8 %152, 43
  br i1 %162, label %163, label %156

163:                                              ; preds = %161
  store i32 4, ptr @bar.lastc, align 4, !tbaa !6
  %164 = icmp eq i8 %154, 44
  br i1 %164, label %165, label %156

165:                                              ; preds = %163
  store i32 5, ptr @bar.lastc, align 4, !tbaa !6
  %166 = icmp sgt i32 %144, -1
  br i1 %166, label %175, label %167

167:                                              ; preds = %165
  %168 = add nsw i32 %144, 8
  store i32 %168, ptr %10, align 8
  %169 = icmp samesign ult i32 %144, -7
  br i1 %169, label %170, label %175

170:                                              ; preds = %167
  %171 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %172 = load ptr, ptr %171, align 8
  %173 = sext i32 %144 to i64
  %174 = getelementptr inbounds i8, ptr %172, i64 %173
  br label %179

175:                                              ; preds = %167, %165
  %176 = phi i32 [ %168, %167 ], [ %144, %165 ]
  %177 = load ptr, ptr %6, align 8
  %178 = getelementptr inbounds nuw i8, ptr %177, i64 8
  store ptr %178, ptr %6, align 8
  br label %179

179:                                              ; preds = %170, %175
  %180 = phi i32 [ %168, %170 ], [ %176, %175 ]
  %181 = phi ptr [ %174, %170 ], [ %177, %175 ]
  %182 = load i8, ptr %181, align 8
  %183 = getelementptr inbounds nuw i8, ptr %181, i64 1
  %184 = load i8, ptr %183, align 1
  %185 = getelementptr inbounds nuw i8, ptr %181, i64 2
  %186 = load i8, ptr %185, align 2
  %187 = getelementptr inbounds nuw i8, ptr %181, i64 3
  %188 = load i8, ptr %187, align 1
  %189 = getelementptr inbounds nuw i8, ptr %181, i64 4
  %190 = load i8, ptr %189, align 4
  %191 = getelementptr inbounds nuw i8, ptr %181, i64 5
  %192 = load i8, ptr %191, align 1, !tbaa !10
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 6, ptr @bar.lastn, align 4, !tbaa !6
  %193 = icmp eq i8 %182, 48
  br i1 %193, label %195, label %194

194:                                              ; preds = %203, %201, %199, %197, %195, %179
  call void @abort() #7
  unreachable

195:                                              ; preds = %179
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %196 = icmp eq i8 %184, 49
  br i1 %196, label %197, label %194

197:                                              ; preds = %195
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %198 = icmp eq i8 %186, 50
  br i1 %198, label %199, label %194

199:                                              ; preds = %197
  store i32 3, ptr @bar.lastc, align 4, !tbaa !6
  %200 = icmp eq i8 %188, 51
  br i1 %200, label %201, label %194

201:                                              ; preds = %199
  store i32 4, ptr @bar.lastc, align 4, !tbaa !6
  %202 = icmp eq i8 %190, 52
  br i1 %202, label %203, label %194

203:                                              ; preds = %201
  store i32 5, ptr @bar.lastc, align 4, !tbaa !6
  %204 = icmp eq i8 %192, 53
  br i1 %204, label %205, label %194

205:                                              ; preds = %203
  store i32 6, ptr @bar.lastc, align 4, !tbaa !6
  %206 = icmp sgt i32 %180, -1
  br i1 %206, label %215, label %207

207:                                              ; preds = %205
  %208 = add nsw i32 %180, 8
  store i32 %208, ptr %10, align 8
  %209 = icmp samesign ult i32 %180, -7
  br i1 %209, label %210, label %215

210:                                              ; preds = %207
  %211 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %212 = load ptr, ptr %211, align 8
  %213 = sext i32 %180 to i64
  %214 = getelementptr inbounds i8, ptr %212, i64 %213
  br label %219

215:                                              ; preds = %207, %205
  %216 = phi i32 [ %208, %207 ], [ %180, %205 ]
  %217 = load ptr, ptr %6, align 8
  %218 = getelementptr inbounds nuw i8, ptr %217, i64 8
  store ptr %218, ptr %6, align 8
  br label %219

219:                                              ; preds = %210, %215
  %220 = phi i32 [ %208, %210 ], [ %216, %215 ]
  %221 = phi ptr [ %214, %210 ], [ %217, %215 ]
  %222 = load i8, ptr %221, align 8
  %223 = getelementptr inbounds nuw i8, ptr %221, i64 1
  %224 = load i8, ptr %223, align 1
  %225 = getelementptr inbounds nuw i8, ptr %221, i64 2
  %226 = load i8, ptr %225, align 2
  %227 = getelementptr inbounds nuw i8, ptr %221, i64 3
  %228 = load i8, ptr %227, align 1
  %229 = getelementptr inbounds nuw i8, ptr %221, i64 4
  %230 = load i8, ptr %229, align 4
  %231 = getelementptr inbounds nuw i8, ptr %221, i64 5
  %232 = load i8, ptr %231, align 1
  %233 = getelementptr inbounds nuw i8, ptr %221, i64 6
  %234 = load i8, ptr %233, align 2, !tbaa !10
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 7, ptr @bar.lastn, align 4, !tbaa !6
  %235 = icmp eq i8 %222, 56
  br i1 %235, label %237, label %236

236:                                              ; preds = %247, %245, %243, %241, %239, %237, %219
  call void @abort() #7
  unreachable

237:                                              ; preds = %219
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %238 = icmp eq i8 %224, 57
  br i1 %238, label %239, label %236

239:                                              ; preds = %237
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %240 = icmp eq i8 %226, 58
  br i1 %240, label %241, label %236

241:                                              ; preds = %239
  store i32 3, ptr @bar.lastc, align 4, !tbaa !6
  %242 = icmp eq i8 %228, 59
  br i1 %242, label %243, label %236

243:                                              ; preds = %241
  store i32 4, ptr @bar.lastc, align 4, !tbaa !6
  %244 = icmp eq i8 %230, 60
  br i1 %244, label %245, label %236

245:                                              ; preds = %243
  store i32 5, ptr @bar.lastc, align 4, !tbaa !6
  %246 = icmp eq i8 %232, 61
  br i1 %246, label %247, label %236

247:                                              ; preds = %245
  store i32 6, ptr @bar.lastc, align 4, !tbaa !6
  %248 = icmp eq i8 %234, 62
  br i1 %248, label %249, label %236

249:                                              ; preds = %247
  %250 = icmp sgt i32 %220, -1
  br i1 %250, label %259, label %251

251:                                              ; preds = %249
  %252 = add nsw i32 %220, 8
  store i32 %252, ptr %10, align 8
  %253 = icmp samesign ult i32 %220, -7
  br i1 %253, label %254, label %259

254:                                              ; preds = %251
  %255 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %256 = load ptr, ptr %255, align 8
  %257 = sext i32 %220 to i64
  %258 = getelementptr inbounds i8, ptr %256, i64 %257
  br label %263

259:                                              ; preds = %251, %249
  %260 = phi i32 [ %252, %251 ], [ %220, %249 ]
  %261 = load ptr, ptr %6, align 8
  %262 = getelementptr inbounds nuw i8, ptr %261, i64 8
  store ptr %262, ptr %6, align 8
  br label %263

263:                                              ; preds = %254, %259
  %264 = phi i32 [ %252, %254 ], [ %260, %259 ]
  %265 = phi ptr [ %258, %254 ], [ %261, %259 ]
  %266 = load i64, ptr %265, align 8, !tbaa !10
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 8, ptr @bar.lastn, align 4, !tbaa !6
  %267 = and i64 %266, 255
  %268 = icmp eq i64 %267, 64
  br i1 %268, label %270, label %269

269:                                              ; preds = %288, %285, %282, %279, %276, %273, %270, %263
  call void @abort() #7
  unreachable

270:                                              ; preds = %263
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %271 = and i64 %266, 65280
  %272 = icmp eq i64 %271, 16640
  br i1 %272, label %273, label %269

273:                                              ; preds = %270
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %274 = and i64 %266, 16711680
  %275 = icmp eq i64 %274, 4325376
  br i1 %275, label %276, label %269

276:                                              ; preds = %273
  store i32 3, ptr @bar.lastc, align 4, !tbaa !6
  %277 = and i64 %266, 4278190080
  %278 = icmp eq i64 %277, 1124073472
  br i1 %278, label %279, label %269

279:                                              ; preds = %276
  store i32 4, ptr @bar.lastc, align 4, !tbaa !6
  %280 = and i64 %266, 1095216660480
  %281 = icmp eq i64 %280, 292057776128
  br i1 %281, label %282, label %269

282:                                              ; preds = %279
  store i32 5, ptr @bar.lastc, align 4, !tbaa !6
  %283 = and i64 %266, 280375465082880
  %284 = icmp eq i64 %283, 75866302316544
  br i1 %284, label %285, label %269

285:                                              ; preds = %282
  store i32 6, ptr @bar.lastc, align 4, !tbaa !6
  %286 = and i64 %266, 71776119061217280
  %287 = icmp eq i64 %286, 19703248369745920
  br i1 %287, label %288, label %269

288:                                              ; preds = %285
  store i32 7, ptr @bar.lastc, align 4, !tbaa !6
  %289 = and i64 %266, -72057594037927936
  %290 = icmp eq i64 %289, 5116089176692883456
  br i1 %290, label %291, label %269

291:                                              ; preds = %288
  store i32 8, ptr @bar.lastc, align 4, !tbaa !6
  %292 = icmp sgt i32 %264, -1
  br i1 %292, label %301, label %293

293:                                              ; preds = %291
  %294 = add nsw i32 %264, 16
  store i32 %294, ptr %10, align 8
  %295 = icmp samesign ult i32 %264, -15
  br i1 %295, label %296, label %301

296:                                              ; preds = %293
  %297 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %298 = load ptr, ptr %297, align 8
  %299 = sext i32 %264 to i64
  %300 = getelementptr inbounds i8, ptr %298, i64 %299
  br label %305

301:                                              ; preds = %293, %291
  %302 = phi i32 [ %294, %293 ], [ %264, %291 ]
  %303 = load ptr, ptr %6, align 8
  %304 = getelementptr inbounds nuw i8, ptr %303, i64 16
  store ptr %304, ptr %6, align 8
  br label %305

305:                                              ; preds = %296, %301
  %306 = phi i32 [ %294, %296 ], [ %302, %301 ]
  %307 = phi ptr [ %300, %296 ], [ %303, %301 ]
  %308 = load i8, ptr %307, align 8
  %309 = getelementptr inbounds nuw i8, ptr %307, i64 1
  %310 = load i8, ptr %309, align 1
  %311 = getelementptr inbounds nuw i8, ptr %307, i64 2
  %312 = load i8, ptr %311, align 2
  %313 = getelementptr inbounds nuw i8, ptr %307, i64 3
  %314 = load i8, ptr %313, align 1
  %315 = getelementptr inbounds nuw i8, ptr %307, i64 4
  %316 = load i8, ptr %315, align 4
  %317 = getelementptr inbounds nuw i8, ptr %307, i64 5
  %318 = load i8, ptr %317, align 1
  %319 = getelementptr inbounds nuw i8, ptr %307, i64 6
  %320 = load i8, ptr %319, align 2
  %321 = getelementptr inbounds nuw i8, ptr %307, i64 7
  %322 = load i8, ptr %321, align 1
  %323 = getelementptr inbounds nuw i8, ptr %307, i64 8
  %324 = load i8, ptr %323, align 8, !tbaa !10
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 9, ptr @bar.lastn, align 4, !tbaa !6
  %325 = icmp eq i8 %308, 72
  br i1 %325, label %327, label %326

326:                                              ; preds = %341, %339, %337, %335, %333, %331, %329, %327, %305
  call void @abort() #7
  unreachable

327:                                              ; preds = %305
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %328 = icmp eq i8 %310, 73
  br i1 %328, label %329, label %326

329:                                              ; preds = %327
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %330 = icmp eq i8 %312, 74
  br i1 %330, label %331, label %326

331:                                              ; preds = %329
  store i32 3, ptr @bar.lastc, align 4, !tbaa !6
  %332 = icmp eq i8 %314, 75
  br i1 %332, label %333, label %326

333:                                              ; preds = %331
  store i32 4, ptr @bar.lastc, align 4, !tbaa !6
  %334 = icmp eq i8 %316, 76
  br i1 %334, label %335, label %326

335:                                              ; preds = %333
  store i32 5, ptr @bar.lastc, align 4, !tbaa !6
  %336 = icmp eq i8 %318, 77
  br i1 %336, label %337, label %326

337:                                              ; preds = %335
  store i32 6, ptr @bar.lastc, align 4, !tbaa !6
  %338 = icmp eq i8 %320, 78
  br i1 %338, label %339, label %326

339:                                              ; preds = %337
  store i32 7, ptr @bar.lastc, align 4, !tbaa !6
  %340 = icmp eq i8 %322, 79
  br i1 %340, label %341, label %326

341:                                              ; preds = %339
  store i32 8, ptr @bar.lastc, align 4, !tbaa !6
  %342 = icmp eq i8 %324, 64
  br i1 %342, label %343, label %326

343:                                              ; preds = %341
  store i32 9, ptr @bar.lastc, align 4, !tbaa !6
  %344 = icmp sgt i32 %306, -1
  br i1 %344, label %353, label %345

345:                                              ; preds = %343
  %346 = add nsw i32 %306, 16
  store i32 %346, ptr %10, align 8
  %347 = icmp samesign ult i32 %306, -15
  br i1 %347, label %348, label %353

348:                                              ; preds = %345
  %349 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %350 = load ptr, ptr %349, align 8
  %351 = sext i32 %306 to i64
  %352 = getelementptr inbounds i8, ptr %350, i64 %351
  br label %356

353:                                              ; preds = %345, %343
  %354 = load ptr, ptr %6, align 8
  %355 = getelementptr inbounds nuw i8, ptr %354, i64 16
  store ptr %355, ptr %6, align 8
  br label %356

356:                                              ; preds = %348, %353
  %357 = phi ptr [ %352, %348 ], [ %354, %353 ]
  %358 = load i8, ptr %357, align 8
  %359 = getelementptr inbounds nuw i8, ptr %357, i64 1
  %360 = load i8, ptr %359, align 1
  %361 = getelementptr inbounds nuw i8, ptr %357, i64 2
  %362 = load i8, ptr %361, align 2
  %363 = getelementptr inbounds nuw i8, ptr %357, i64 3
  %364 = load i8, ptr %363, align 1
  %365 = getelementptr inbounds nuw i8, ptr %357, i64 4
  %366 = load i8, ptr %365, align 4
  %367 = getelementptr inbounds nuw i8, ptr %357, i64 5
  %368 = load i8, ptr %367, align 1
  %369 = getelementptr inbounds nuw i8, ptr %357, i64 6
  %370 = load i8, ptr %369, align 2
  %371 = getelementptr inbounds nuw i8, ptr %357, i64 7
  %372 = load i8, ptr %371, align 1
  %373 = getelementptr inbounds nuw i8, ptr %357, i64 8
  %374 = load i8, ptr %373, align 8
  %375 = getelementptr inbounds nuw i8, ptr %357, i64 9
  %376 = load i8, ptr %375, align 1, !tbaa !10
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 10, ptr @bar.lastn, align 4, !tbaa !6
  %377 = icmp eq i8 %358, 80
  br i1 %377, label %379, label %378

378:                                              ; preds = %395, %393, %391, %389, %387, %385, %383, %381, %379, %356
  call void @abort() #7
  unreachable

379:                                              ; preds = %356
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %380 = icmp eq i8 %360, 81
  br i1 %380, label %381, label %378

381:                                              ; preds = %379
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %382 = icmp eq i8 %362, 82
  br i1 %382, label %383, label %378

383:                                              ; preds = %381
  store i32 3, ptr @bar.lastc, align 4, !tbaa !6
  %384 = icmp eq i8 %364, 83
  br i1 %384, label %385, label %378

385:                                              ; preds = %383
  store i32 4, ptr @bar.lastc, align 4, !tbaa !6
  %386 = icmp eq i8 %366, 84
  br i1 %386, label %387, label %378

387:                                              ; preds = %385
  store i32 5, ptr @bar.lastc, align 4, !tbaa !6
  %388 = icmp eq i8 %368, 85
  br i1 %388, label %389, label %378

389:                                              ; preds = %387
  store i32 6, ptr @bar.lastc, align 4, !tbaa !6
  %390 = icmp eq i8 %370, 86
  br i1 %390, label %391, label %378

391:                                              ; preds = %389
  store i32 7, ptr @bar.lastc, align 4, !tbaa !6
  %392 = icmp eq i8 %372, 87
  br i1 %392, label %393, label %378

393:                                              ; preds = %391
  store i32 8, ptr @bar.lastc, align 4, !tbaa !6
  %394 = icmp eq i8 %374, 88
  br i1 %394, label %395, label %378

395:                                              ; preds = %393
  store i32 9, ptr @bar.lastc, align 4, !tbaa !6
  %396 = icmp eq i8 %376, 89
  br i1 %396, label %397, label %378

397:                                              ; preds = %395
  store i32 10, ptr @bar.lastc, align 4, !tbaa !6
  %398 = load i32, ptr %10, align 8
  %399 = icmp sgt i32 %398, -1
  br i1 %399, label %408, label %400

400:                                              ; preds = %397
  %401 = add nsw i32 %398, 16
  store i32 %401, ptr %10, align 8
  %402 = icmp samesign ult i32 %398, -15
  br i1 %402, label %403, label %408

403:                                              ; preds = %400
  %404 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %405 = load ptr, ptr %404, align 8
  %406 = sext i32 %398 to i64
  %407 = getelementptr inbounds i8, ptr %405, i64 %406
  br label %412

408:                                              ; preds = %400, %397
  %409 = phi i32 [ %401, %400 ], [ %398, %397 ]
  %410 = load ptr, ptr %6, align 8
  %411 = getelementptr inbounds nuw i8, ptr %410, i64 16
  store ptr %411, ptr %6, align 8
  br label %412

412:                                              ; preds = %408, %403
  %413 = phi i32 [ %401, %403 ], [ %409, %408 ]
  %414 = phi ptr [ %407, %403 ], [ %410, %408 ]
  %415 = load i8, ptr %414, align 8
  %416 = getelementptr inbounds nuw i8, ptr %414, i64 1
  %417 = load i8, ptr %416, align 1
  %418 = getelementptr inbounds nuw i8, ptr %414, i64 2
  %419 = load i8, ptr %418, align 2
  %420 = getelementptr inbounds nuw i8, ptr %414, i64 3
  %421 = load i8, ptr %420, align 1
  %422 = getelementptr inbounds nuw i8, ptr %414, i64 4
  %423 = load i8, ptr %422, align 4
  %424 = getelementptr inbounds nuw i8, ptr %414, i64 5
  %425 = load i8, ptr %424, align 1
  %426 = getelementptr inbounds nuw i8, ptr %414, i64 6
  %427 = load i8, ptr %426, align 2
  %428 = getelementptr inbounds nuw i8, ptr %414, i64 7
  %429 = load i8, ptr %428, align 1
  %430 = getelementptr inbounds nuw i8, ptr %414, i64 8
  %431 = load i8, ptr %430, align 8
  %432 = getelementptr inbounds nuw i8, ptr %414, i64 9
  %433 = load i8, ptr %432, align 1
  %434 = getelementptr inbounds nuw i8, ptr %414, i64 10
  %435 = load i8, ptr %434, align 2, !tbaa !10
  %436 = load i32, ptr @bar.lastn, align 4, !tbaa !6
  %437 = zext i8 %415 to i32
  %438 = icmp eq i32 %436, 11
  br i1 %438, label %443, label %439

439:                                              ; preds = %412
  %440 = icmp eq i32 %436, 10
  br i1 %440, label %442, label %441

441:                                              ; preds = %439
  call void @abort() #7
  unreachable

442:                                              ; preds = %439
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 11, ptr @bar.lastn, align 4, !tbaa !6
  br label %443

443:                                              ; preds = %442, %412
  %444 = phi i32 [ 0, %442 ], [ 10, %412 ]
  %445 = xor i32 %444, %437
  %446 = icmp eq i32 %445, 88
  br i1 %446, label %448, label %447

447:                                              ; preds = %493, %488, %483, %478, %473, %468, %463, %458, %453, %448, %443
  call void @abort() #7
  unreachable

448:                                              ; preds = %443
  %449 = or disjoint i32 %444, 1
  store i32 %449, ptr @bar.lastc, align 4, !tbaa !6
  %450 = zext i8 %417 to i32
  %451 = xor i32 %449, %450
  %452 = icmp eq i32 %451, 88
  br i1 %452, label %453, label %447

453:                                              ; preds = %448
  %454 = add nuw nsw i32 %444, 2
  store i32 %454, ptr @bar.lastc, align 4, !tbaa !6
  %455 = zext i8 %419 to i32
  %456 = xor i32 %454, %455
  %457 = icmp eq i32 %456, 88
  br i1 %457, label %458, label %447

458:                                              ; preds = %453
  %459 = add nuw nsw i32 %444, 3
  store i32 %459, ptr @bar.lastc, align 4, !tbaa !6
  %460 = zext i8 %421 to i32
  %461 = xor i32 %459, %460
  %462 = icmp eq i32 %461, 88
  br i1 %462, label %463, label %447

463:                                              ; preds = %458
  %464 = or disjoint i32 %444, 4
  store i32 %464, ptr @bar.lastc, align 4, !tbaa !6
  %465 = zext i8 %423 to i32
  %466 = xor i32 %464, %465
  %467 = icmp eq i32 %466, 88
  br i1 %467, label %468, label %447

468:                                              ; preds = %463
  %469 = or disjoint i32 %444, 5
  store i32 %469, ptr @bar.lastc, align 4, !tbaa !6
  %470 = zext i8 %425 to i32
  %471 = xor i32 %469, %470
  %472 = icmp eq i32 %471, 88
  br i1 %472, label %473, label %447

473:                                              ; preds = %468
  %474 = add nuw nsw i32 %444, 6
  store i32 %474, ptr @bar.lastc, align 4, !tbaa !6
  %475 = zext i8 %427 to i32
  %476 = xor i32 %474, %475
  %477 = icmp eq i32 %476, 88
  br i1 %477, label %478, label %447

478:                                              ; preds = %473
  %479 = add nuw nsw i32 %444, 7
  store i32 %479, ptr @bar.lastc, align 4, !tbaa !6
  %480 = zext i8 %429 to i32
  %481 = xor i32 %479, %480
  %482 = icmp eq i32 %481, 88
  br i1 %482, label %483, label %447

483:                                              ; preds = %478
  %484 = add nuw nsw i32 %444, 8
  store i32 %484, ptr @bar.lastc, align 4, !tbaa !6
  %485 = zext i8 %431 to i32
  %486 = xor i32 %484, %485
  %487 = icmp eq i32 %486, 88
  br i1 %487, label %488, label %447

488:                                              ; preds = %483
  %489 = add nuw nsw i32 %444, 9
  store i32 %489, ptr @bar.lastc, align 4, !tbaa !6
  %490 = zext i8 %433 to i32
  %491 = xor i32 %489, %490
  %492 = icmp eq i32 %491, 88
  br i1 %492, label %493, label %447

493:                                              ; preds = %488
  %494 = add nuw nsw i32 %444, 10
  store i32 %494, ptr @bar.lastc, align 4, !tbaa !6
  %495 = zext i8 %435 to i32
  %496 = xor i32 %494, %495
  %497 = icmp eq i32 %496, 88
  br i1 %497, label %498, label %447

498:                                              ; preds = %493
  %499 = add nuw nsw i32 %444, 11
  store i32 %499, ptr @bar.lastc, align 4, !tbaa !6
  %500 = icmp sgt i32 %413, -1
  br i1 %500, label %509, label %501

501:                                              ; preds = %498
  %502 = add nsw i32 %413, 16
  store i32 %502, ptr %10, align 8
  %503 = icmp samesign ult i32 %413, -15
  br i1 %503, label %504, label %509

504:                                              ; preds = %501
  %505 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %506 = load ptr, ptr %505, align 8
  %507 = sext i32 %413 to i64
  %508 = getelementptr inbounds i8, ptr %506, i64 %507
  br label %513

509:                                              ; preds = %501, %498
  %510 = phi i32 [ %502, %501 ], [ %413, %498 ]
  %511 = load ptr, ptr %6, align 8
  %512 = getelementptr inbounds nuw i8, ptr %511, i64 16
  store ptr %512, ptr %6, align 8
  br label %513

513:                                              ; preds = %504, %509
  %514 = phi i32 [ %502, %504 ], [ %510, %509 ]
  %515 = phi ptr [ %508, %504 ], [ %511, %509 ]
  %516 = getelementptr inbounds nuw i8, ptr %515, i64 1
  %517 = load i8, ptr %516, align 1
  %518 = getelementptr inbounds nuw i8, ptr %515, i64 2
  %519 = load i8, ptr %518, align 2
  %520 = getelementptr inbounds nuw i8, ptr %515, i64 3
  %521 = load i8, ptr %520, align 1
  %522 = getelementptr inbounds nuw i8, ptr %515, i64 4
  %523 = load i8, ptr %522, align 4
  %524 = getelementptr inbounds nuw i8, ptr %515, i64 5
  %525 = load i8, ptr %524, align 1
  %526 = getelementptr inbounds nuw i8, ptr %515, i64 6
  %527 = load i8, ptr %526, align 2
  %528 = getelementptr inbounds nuw i8, ptr %515, i64 7
  %529 = load i8, ptr %528, align 1
  %530 = getelementptr inbounds nuw i8, ptr %515, i64 8
  %531 = load i8, ptr %530, align 8
  %532 = getelementptr inbounds nuw i8, ptr %515, i64 9
  %533 = load i8, ptr %532, align 1
  %534 = getelementptr inbounds nuw i8, ptr %515, i64 10
  %535 = load i8, ptr %534, align 2
  %536 = getelementptr inbounds nuw i8, ptr %515, i64 11
  %537 = load i8, ptr %536, align 1, !tbaa !10
  br i1 %438, label %538, label %539

538:                                              ; preds = %513
  call void @abort() #7
  unreachable

539:                                              ; preds = %513
  %540 = load i8, ptr %515, align 8
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 12, ptr @bar.lastn, align 4, !tbaa !6
  %541 = icmp eq i8 %540, 96
  br i1 %541, label %543, label %542

542:                                              ; preds = %563, %561, %559, %557, %555, %553, %551, %549, %547, %545, %543, %539
  call void @abort() #7
  unreachable

543:                                              ; preds = %539
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %544 = icmp eq i8 %517, 97
  br i1 %544, label %545, label %542

545:                                              ; preds = %543
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %546 = icmp eq i8 %519, 98
  br i1 %546, label %547, label %542

547:                                              ; preds = %545
  store i32 3, ptr @bar.lastc, align 4, !tbaa !6
  %548 = icmp eq i8 %521, 99
  br i1 %548, label %549, label %542

549:                                              ; preds = %547
  store i32 4, ptr @bar.lastc, align 4, !tbaa !6
  %550 = icmp eq i8 %523, 100
  br i1 %550, label %551, label %542

551:                                              ; preds = %549
  store i32 5, ptr @bar.lastc, align 4, !tbaa !6
  %552 = icmp eq i8 %525, 101
  br i1 %552, label %553, label %542

553:                                              ; preds = %551
  store i32 6, ptr @bar.lastc, align 4, !tbaa !6
  %554 = icmp eq i8 %527, 102
  br i1 %554, label %555, label %542

555:                                              ; preds = %553
  store i32 7, ptr @bar.lastc, align 4, !tbaa !6
  %556 = icmp eq i8 %529, 103
  br i1 %556, label %557, label %542

557:                                              ; preds = %555
  store i32 8, ptr @bar.lastc, align 4, !tbaa !6
  %558 = icmp eq i8 %531, 104
  br i1 %558, label %559, label %542

559:                                              ; preds = %557
  store i32 9, ptr @bar.lastc, align 4, !tbaa !6
  %560 = icmp eq i8 %533, 105
  br i1 %560, label %561, label %542

561:                                              ; preds = %559
  store i32 10, ptr @bar.lastc, align 4, !tbaa !6
  %562 = icmp eq i8 %535, 106
  br i1 %562, label %563, label %542

563:                                              ; preds = %561
  store i32 11, ptr @bar.lastc, align 4, !tbaa !6
  %564 = icmp eq i8 %537, 107
  br i1 %564, label %565, label %542

565:                                              ; preds = %563
  store i32 12, ptr @bar.lastc, align 4, !tbaa !6
  %566 = icmp sgt i32 %514, -1
  br i1 %566, label %575, label %567

567:                                              ; preds = %565
  %568 = add nsw i32 %514, 16
  store i32 %568, ptr %10, align 8
  %569 = icmp samesign ult i32 %514, -15
  br i1 %569, label %570, label %575

570:                                              ; preds = %567
  %571 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %572 = load ptr, ptr %571, align 8
  %573 = sext i32 %514 to i64
  %574 = getelementptr inbounds i8, ptr %572, i64 %573
  br label %579

575:                                              ; preds = %567, %565
  %576 = phi i32 [ %568, %567 ], [ %514, %565 ]
  %577 = load ptr, ptr %6, align 8
  %578 = getelementptr inbounds nuw i8, ptr %577, i64 16
  store ptr %578, ptr %6, align 8
  br label %579

579:                                              ; preds = %570, %575
  %580 = phi i32 [ %568, %570 ], [ %576, %575 ]
  %581 = phi ptr [ %574, %570 ], [ %577, %575 ]
  %582 = load i8, ptr %581, align 8
  %583 = getelementptr inbounds nuw i8, ptr %581, i64 1
  %584 = load i8, ptr %583, align 1
  %585 = getelementptr inbounds nuw i8, ptr %581, i64 2
  %586 = load i8, ptr %585, align 2
  %587 = getelementptr inbounds nuw i8, ptr %581, i64 3
  %588 = load i8, ptr %587, align 1
  %589 = getelementptr inbounds nuw i8, ptr %581, i64 4
  %590 = load i8, ptr %589, align 4
  %591 = getelementptr inbounds nuw i8, ptr %581, i64 5
  %592 = load i8, ptr %591, align 1
  %593 = getelementptr inbounds nuw i8, ptr %581, i64 6
  %594 = load i8, ptr %593, align 2
  %595 = getelementptr inbounds nuw i8, ptr %581, i64 7
  %596 = load i8, ptr %595, align 1
  %597 = getelementptr inbounds nuw i8, ptr %581, i64 8
  %598 = load i8, ptr %597, align 8
  %599 = getelementptr inbounds nuw i8, ptr %581, i64 9
  %600 = load i8, ptr %599, align 1
  %601 = getelementptr inbounds nuw i8, ptr %581, i64 10
  %602 = load i8, ptr %601, align 2
  %603 = getelementptr inbounds nuw i8, ptr %581, i64 11
  %604 = load i8, ptr %603, align 1
  %605 = getelementptr inbounds nuw i8, ptr %581, i64 12
  %606 = load i8, ptr %605, align 4, !tbaa !10
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 13, ptr @bar.lastn, align 4, !tbaa !6
  %607 = icmp eq i8 %582, 104
  br i1 %607, label %609, label %608

608:                                              ; preds = %631, %629, %627, %625, %623, %621, %619, %617, %615, %613, %611, %609, %579
  call void @abort() #7
  unreachable

609:                                              ; preds = %579
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %610 = icmp eq i8 %584, 105
  br i1 %610, label %611, label %608

611:                                              ; preds = %609
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %612 = icmp eq i8 %586, 106
  br i1 %612, label %613, label %608

613:                                              ; preds = %611
  store i32 3, ptr @bar.lastc, align 4, !tbaa !6
  %614 = icmp eq i8 %588, 107
  br i1 %614, label %615, label %608

615:                                              ; preds = %613
  store i32 4, ptr @bar.lastc, align 4, !tbaa !6
  %616 = icmp eq i8 %590, 108
  br i1 %616, label %617, label %608

617:                                              ; preds = %615
  store i32 5, ptr @bar.lastc, align 4, !tbaa !6
  %618 = icmp eq i8 %592, 109
  br i1 %618, label %619, label %608

619:                                              ; preds = %617
  store i32 6, ptr @bar.lastc, align 4, !tbaa !6
  %620 = icmp eq i8 %594, 110
  br i1 %620, label %621, label %608

621:                                              ; preds = %619
  store i32 7, ptr @bar.lastc, align 4, !tbaa !6
  %622 = icmp eq i8 %596, 111
  br i1 %622, label %623, label %608

623:                                              ; preds = %621
  store i32 8, ptr @bar.lastc, align 4, !tbaa !6
  %624 = icmp eq i8 %598, 96
  br i1 %624, label %625, label %608

625:                                              ; preds = %623
  store i32 9, ptr @bar.lastc, align 4, !tbaa !6
  %626 = icmp eq i8 %600, 97
  br i1 %626, label %627, label %608

627:                                              ; preds = %625
  store i32 10, ptr @bar.lastc, align 4, !tbaa !6
  %628 = icmp eq i8 %602, 98
  br i1 %628, label %629, label %608

629:                                              ; preds = %627
  store i32 11, ptr @bar.lastc, align 4, !tbaa !6
  %630 = icmp eq i8 %604, 99
  br i1 %630, label %631, label %608

631:                                              ; preds = %629
  store i32 12, ptr @bar.lastc, align 4, !tbaa !6
  %632 = icmp eq i8 %606, 100
  br i1 %632, label %633, label %608

633:                                              ; preds = %631
  store i32 13, ptr @bar.lastc, align 4, !tbaa !6
  %634 = icmp sgt i32 %580, -1
  br i1 %634, label %643, label %635

635:                                              ; preds = %633
  %636 = add nsw i32 %580, 16
  store i32 %636, ptr %10, align 8
  %637 = icmp samesign ult i32 %580, -15
  br i1 %637, label %638, label %643

638:                                              ; preds = %635
  %639 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %640 = load ptr, ptr %639, align 8
  %641 = sext i32 %580 to i64
  %642 = getelementptr inbounds i8, ptr %640, i64 %641
  br label %647

643:                                              ; preds = %635, %633
  %644 = phi i32 [ %636, %635 ], [ %580, %633 ]
  %645 = load ptr, ptr %6, align 8
  %646 = getelementptr inbounds nuw i8, ptr %645, i64 16
  store ptr %646, ptr %6, align 8
  br label %647

647:                                              ; preds = %638, %643
  %648 = phi i32 [ %636, %638 ], [ %644, %643 ]
  %649 = phi ptr [ %642, %638 ], [ %645, %643 ]
  %650 = load i8, ptr %649, align 8
  %651 = getelementptr inbounds nuw i8, ptr %649, i64 1
  %652 = load i8, ptr %651, align 1
  %653 = getelementptr inbounds nuw i8, ptr %649, i64 2
  %654 = load i8, ptr %653, align 2
  %655 = getelementptr inbounds nuw i8, ptr %649, i64 3
  %656 = load i8, ptr %655, align 1
  %657 = getelementptr inbounds nuw i8, ptr %649, i64 4
  %658 = load i8, ptr %657, align 4
  %659 = getelementptr inbounds nuw i8, ptr %649, i64 5
  %660 = load i8, ptr %659, align 1
  %661 = getelementptr inbounds nuw i8, ptr %649, i64 6
  %662 = load i8, ptr %661, align 2
  %663 = getelementptr inbounds nuw i8, ptr %649, i64 7
  %664 = load i8, ptr %663, align 1
  %665 = getelementptr inbounds nuw i8, ptr %649, i64 8
  %666 = load i8, ptr %665, align 8
  %667 = getelementptr inbounds nuw i8, ptr %649, i64 9
  %668 = load i8, ptr %667, align 1
  %669 = getelementptr inbounds nuw i8, ptr %649, i64 10
  %670 = load i8, ptr %669, align 2
  %671 = getelementptr inbounds nuw i8, ptr %649, i64 11
  %672 = load i8, ptr %671, align 1
  %673 = getelementptr inbounds nuw i8, ptr %649, i64 12
  %674 = load i8, ptr %673, align 4
  %675 = getelementptr inbounds nuw i8, ptr %649, i64 13
  %676 = load i8, ptr %675, align 1, !tbaa !10
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 14, ptr @bar.lastn, align 4, !tbaa !6
  %677 = icmp eq i8 %650, 112
  br i1 %677, label %679, label %678

678:                                              ; preds = %703, %701, %699, %697, %695, %693, %691, %689, %687, %685, %683, %681, %679, %647
  call void @abort() #7
  unreachable

679:                                              ; preds = %647
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %680 = icmp eq i8 %652, 113
  br i1 %680, label %681, label %678

681:                                              ; preds = %679
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %682 = icmp eq i8 %654, 114
  br i1 %682, label %683, label %678

683:                                              ; preds = %681
  store i32 3, ptr @bar.lastc, align 4, !tbaa !6
  %684 = icmp eq i8 %656, 115
  br i1 %684, label %685, label %678

685:                                              ; preds = %683
  store i32 4, ptr @bar.lastc, align 4, !tbaa !6
  %686 = icmp eq i8 %658, 116
  br i1 %686, label %687, label %678

687:                                              ; preds = %685
  store i32 5, ptr @bar.lastc, align 4, !tbaa !6
  %688 = icmp eq i8 %660, 117
  br i1 %688, label %689, label %678

689:                                              ; preds = %687
  store i32 6, ptr @bar.lastc, align 4, !tbaa !6
  %690 = icmp eq i8 %662, 118
  br i1 %690, label %691, label %678

691:                                              ; preds = %689
  store i32 7, ptr @bar.lastc, align 4, !tbaa !6
  %692 = icmp eq i8 %664, 119
  br i1 %692, label %693, label %678

693:                                              ; preds = %691
  store i32 8, ptr @bar.lastc, align 4, !tbaa !6
  %694 = icmp eq i8 %666, 120
  br i1 %694, label %695, label %678

695:                                              ; preds = %693
  store i32 9, ptr @bar.lastc, align 4, !tbaa !6
  %696 = icmp eq i8 %668, 121
  br i1 %696, label %697, label %678

697:                                              ; preds = %695
  store i32 10, ptr @bar.lastc, align 4, !tbaa !6
  %698 = icmp eq i8 %670, 122
  br i1 %698, label %699, label %678

699:                                              ; preds = %697
  store i32 11, ptr @bar.lastc, align 4, !tbaa !6
  %700 = icmp eq i8 %672, 123
  br i1 %700, label %701, label %678

701:                                              ; preds = %699
  store i32 12, ptr @bar.lastc, align 4, !tbaa !6
  %702 = icmp eq i8 %674, 124
  br i1 %702, label %703, label %678

703:                                              ; preds = %701
  store i32 13, ptr @bar.lastc, align 4, !tbaa !6
  %704 = icmp eq i8 %676, 125
  br i1 %704, label %705, label %678

705:                                              ; preds = %703
  store i32 14, ptr @bar.lastc, align 4, !tbaa !6
  %706 = icmp sgt i32 %648, -1
  br i1 %706, label %715, label %707

707:                                              ; preds = %705
  %708 = add nsw i32 %648, 16
  store i32 %708, ptr %10, align 8
  %709 = icmp samesign ult i32 %648, -15
  br i1 %709, label %710, label %715

710:                                              ; preds = %707
  %711 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %712 = load ptr, ptr %711, align 8
  %713 = sext i32 %648 to i64
  %714 = getelementptr inbounds i8, ptr %712, i64 %713
  br label %718

715:                                              ; preds = %707, %705
  %716 = load ptr, ptr %6, align 8
  %717 = getelementptr inbounds nuw i8, ptr %716, i64 16
  store ptr %717, ptr %6, align 8
  br label %718

718:                                              ; preds = %710, %715
  %719 = phi ptr [ %714, %710 ], [ %716, %715 ]
  %720 = load i8, ptr %719, align 8
  %721 = getelementptr inbounds nuw i8, ptr %719, i64 1
  %722 = load i8, ptr %721, align 1
  %723 = getelementptr inbounds nuw i8, ptr %719, i64 2
  %724 = load i8, ptr %723, align 2
  %725 = getelementptr inbounds nuw i8, ptr %719, i64 3
  %726 = load i8, ptr %725, align 1
  %727 = getelementptr inbounds nuw i8, ptr %719, i64 4
  %728 = load i8, ptr %727, align 4
  %729 = getelementptr inbounds nuw i8, ptr %719, i64 5
  %730 = load i8, ptr %729, align 1
  %731 = getelementptr inbounds nuw i8, ptr %719, i64 6
  %732 = load i8, ptr %731, align 2
  %733 = getelementptr inbounds nuw i8, ptr %719, i64 7
  %734 = load i8, ptr %733, align 1
  %735 = getelementptr inbounds nuw i8, ptr %719, i64 8
  %736 = load i8, ptr %735, align 8
  %737 = getelementptr inbounds nuw i8, ptr %719, i64 9
  %738 = load i8, ptr %737, align 1
  %739 = getelementptr inbounds nuw i8, ptr %719, i64 10
  %740 = load i8, ptr %739, align 2
  %741 = getelementptr inbounds nuw i8, ptr %719, i64 11
  %742 = load i8, ptr %741, align 1
  %743 = getelementptr inbounds nuw i8, ptr %719, i64 12
  %744 = load i8, ptr %743, align 4
  %745 = getelementptr inbounds nuw i8, ptr %719, i64 13
  %746 = load i8, ptr %745, align 1
  %747 = getelementptr inbounds nuw i8, ptr %719, i64 14
  %748 = load i8, ptr %747, align 2, !tbaa !10
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 15, ptr @bar.lastn, align 4, !tbaa !6
  %749 = icmp eq i8 %720, 120
  br i1 %749, label %751, label %750

750:                                              ; preds = %777, %775, %773, %771, %769, %767, %765, %763, %761, %759, %757, %755, %753, %751, %718
  call void @abort() #7
  unreachable

751:                                              ; preds = %718
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  %752 = icmp eq i8 %722, 121
  br i1 %752, label %753, label %750

753:                                              ; preds = %751
  store i32 2, ptr @bar.lastc, align 4, !tbaa !6
  %754 = icmp eq i8 %724, 122
  br i1 %754, label %755, label %750

755:                                              ; preds = %753
  store i32 3, ptr @bar.lastc, align 4, !tbaa !6
  %756 = icmp eq i8 %726, 123
  br i1 %756, label %757, label %750

757:                                              ; preds = %755
  store i32 4, ptr @bar.lastc, align 4, !tbaa !6
  %758 = icmp eq i8 %728, 124
  br i1 %758, label %759, label %750

759:                                              ; preds = %757
  store i32 5, ptr @bar.lastc, align 4, !tbaa !6
  %760 = icmp eq i8 %730, 125
  br i1 %760, label %761, label %750

761:                                              ; preds = %759
  store i32 6, ptr @bar.lastc, align 4, !tbaa !6
  %762 = icmp eq i8 %732, 126
  br i1 %762, label %763, label %750

763:                                              ; preds = %761
  store i32 7, ptr @bar.lastc, align 4, !tbaa !6
  %764 = icmp eq i8 %734, 127
  br i1 %764, label %765, label %750

765:                                              ; preds = %763
  store i32 8, ptr @bar.lastc, align 4, !tbaa !6
  %766 = icmp eq i8 %736, 112
  br i1 %766, label %767, label %750

767:                                              ; preds = %765
  store i32 9, ptr @bar.lastc, align 4, !tbaa !6
  %768 = icmp eq i8 %738, 113
  br i1 %768, label %769, label %750

769:                                              ; preds = %767
  store i32 10, ptr @bar.lastc, align 4, !tbaa !6
  %770 = icmp eq i8 %740, 114
  br i1 %770, label %771, label %750

771:                                              ; preds = %769
  store i32 11, ptr @bar.lastc, align 4, !tbaa !6
  %772 = icmp eq i8 %742, 115
  br i1 %772, label %773, label %750

773:                                              ; preds = %771
  store i32 12, ptr @bar.lastc, align 4, !tbaa !6
  %774 = icmp eq i8 %744, 116
  br i1 %774, label %775, label %750

775:                                              ; preds = %773
  store i32 13, ptr @bar.lastc, align 4, !tbaa !6
  %776 = icmp eq i8 %746, 117
  br i1 %776, label %777, label %750

777:                                              ; preds = %775
  store i32 14, ptr @bar.lastc, align 4, !tbaa !6
  %778 = icmp eq i8 %748, 118
  br i1 %778, label %779, label %750

779:                                              ; preds = %777
  store i32 15, ptr @bar.lastc, align 4, !tbaa !6
  %780 = load i32, ptr %10, align 8
  %781 = icmp sgt i32 %780, -1
  br i1 %781, label %790, label %782

782:                                              ; preds = %779
  %783 = add nsw i32 %780, 16
  store i32 %783, ptr %10, align 8
  %784 = icmp samesign ult i32 %780, -15
  br i1 %784, label %785, label %790

785:                                              ; preds = %782
  %786 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %787 = load ptr, ptr %786, align 8
  %788 = sext i32 %780 to i64
  %789 = getelementptr inbounds i8, ptr %787, i64 %788
  br label %794

790:                                              ; preds = %782, %779
  %791 = phi i32 [ %783, %782 ], [ %780, %779 ]
  %792 = load ptr, ptr %6, align 8
  %793 = getelementptr inbounds nuw i8, ptr %792, i64 16
  store ptr %793, ptr %6, align 8
  br label %794

794:                                              ; preds = %790, %785
  %795 = phi i32 [ %783, %785 ], [ %791, %790 ]
  %796 = phi ptr [ %789, %785 ], [ %792, %790 ]
  %797 = load i8, ptr %796, align 8
  %798 = getelementptr inbounds nuw i8, ptr %796, i64 1
  %799 = load i8, ptr %798, align 1
  %800 = getelementptr inbounds nuw i8, ptr %796, i64 2
  %801 = load i8, ptr %800, align 2
  %802 = getelementptr inbounds nuw i8, ptr %796, i64 3
  %803 = load i8, ptr %802, align 1
  %804 = getelementptr inbounds nuw i8, ptr %796, i64 4
  %805 = load i8, ptr %804, align 4
  %806 = getelementptr inbounds nuw i8, ptr %796, i64 5
  %807 = load i8, ptr %806, align 1
  %808 = getelementptr inbounds nuw i8, ptr %796, i64 6
  %809 = load i8, ptr %808, align 2
  %810 = getelementptr inbounds nuw i8, ptr %796, i64 7
  %811 = load i8, ptr %810, align 1
  %812 = getelementptr inbounds nuw i8, ptr %796, i64 8
  %813 = load i8, ptr %812, align 8
  %814 = getelementptr inbounds nuw i8, ptr %796, i64 9
  %815 = load i8, ptr %814, align 1
  %816 = getelementptr inbounds nuw i8, ptr %796, i64 10
  %817 = load i8, ptr %816, align 2
  %818 = getelementptr inbounds nuw i8, ptr %796, i64 11
  %819 = load i8, ptr %818, align 1
  %820 = getelementptr inbounds nuw i8, ptr %796, i64 12
  %821 = load i8, ptr %820, align 4
  %822 = getelementptr inbounds nuw i8, ptr %796, i64 13
  %823 = load i8, ptr %822, align 1
  %824 = getelementptr inbounds nuw i8, ptr %796, i64 14
  %825 = load i8, ptr %824, align 2
  %826 = getelementptr inbounds nuw i8, ptr %796, i64 15
  %827 = load i8, ptr %826, align 1, !tbaa !10
  %828 = load i32, ptr @bar.lastn, align 4, !tbaa !6
  %829 = zext i8 %797 to i32
  switch i32 %828, label %830 [
    i32 16, label %832
    i32 15, label %831
  ]

830:                                              ; preds = %794
  call void @abort() #7
  unreachable

831:                                              ; preds = %794
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 16, ptr @bar.lastn, align 4, !tbaa !6
  br label %832

832:                                              ; preds = %794, %831
  %833 = phi i1 [ true, %831 ], [ false, %794 ]
  %834 = phi i32 [ 0, %831 ], [ 15, %794 ]
  %835 = xor i32 %834, %829
  %836 = icmp eq i32 %835, 128
  br i1 %836, label %838, label %837

837:                                              ; preds = %908, %903, %898, %893, %888, %883, %878, %873, %868, %863, %858, %853, %848, %843, %838, %832
  call void @abort() #7
  unreachable

838:                                              ; preds = %832
  %839 = add nuw nsw i32 %834, 1
  store i32 %839, ptr @bar.lastc, align 4, !tbaa !6
  %840 = zext i8 %799 to i32
  %841 = xor i32 %839, %840
  %842 = icmp eq i32 %841, 128
  br i1 %842, label %843, label %837

843:                                              ; preds = %838
  %844 = add nuw nsw i32 %834, 2
  store i32 %844, ptr @bar.lastc, align 4, !tbaa !6
  %845 = zext i8 %801 to i32
  %846 = xor i32 %844, %845
  %847 = icmp eq i32 %846, 128
  br i1 %847, label %848, label %837

848:                                              ; preds = %843
  %849 = add nuw nsw i32 %834, 3
  store i32 %849, ptr @bar.lastc, align 4, !tbaa !6
  %850 = zext i8 %803 to i32
  %851 = xor i32 %849, %850
  %852 = icmp eq i32 %851, 128
  br i1 %852, label %853, label %837

853:                                              ; preds = %848
  %854 = add nuw nsw i32 %834, 4
  store i32 %854, ptr @bar.lastc, align 4, !tbaa !6
  %855 = zext i8 %805 to i32
  %856 = xor i32 %854, %855
  %857 = icmp eq i32 %856, 128
  br i1 %857, label %858, label %837

858:                                              ; preds = %853
  %859 = add nuw nsw i32 %834, 5
  store i32 %859, ptr @bar.lastc, align 4, !tbaa !6
  %860 = zext i8 %807 to i32
  %861 = xor i32 %859, %860
  %862 = icmp eq i32 %861, 128
  br i1 %862, label %863, label %837

863:                                              ; preds = %858
  %864 = add nuw nsw i32 %834, 6
  store i32 %864, ptr @bar.lastc, align 4, !tbaa !6
  %865 = zext i8 %809 to i32
  %866 = xor i32 %864, %865
  %867 = icmp eq i32 %866, 128
  br i1 %867, label %868, label %837

868:                                              ; preds = %863
  %869 = add nuw nsw i32 %834, 7
  store i32 %869, ptr @bar.lastc, align 4, !tbaa !6
  %870 = zext i8 %811 to i32
  %871 = xor i32 %869, %870
  %872 = icmp eq i32 %871, 128
  br i1 %872, label %873, label %837

873:                                              ; preds = %868
  %874 = add nuw nsw i32 %834, 8
  store i32 %874, ptr @bar.lastc, align 4, !tbaa !6
  %875 = zext i8 %813 to i32
  %876 = xor i32 %874, %875
  %877 = icmp eq i32 %876, 128
  br i1 %877, label %878, label %837

878:                                              ; preds = %873
  %879 = add nuw nsw i32 %834, 9
  store i32 %879, ptr @bar.lastc, align 4, !tbaa !6
  %880 = zext i8 %815 to i32
  %881 = xor i32 %879, %880
  %882 = icmp eq i32 %881, 128
  br i1 %882, label %883, label %837

883:                                              ; preds = %878
  %884 = add nuw nsw i32 %834, 10
  store i32 %884, ptr @bar.lastc, align 4, !tbaa !6
  %885 = zext i8 %817 to i32
  %886 = xor i32 %884, %885
  %887 = icmp eq i32 %886, 128
  br i1 %887, label %888, label %837

888:                                              ; preds = %883
  %889 = add nuw nsw i32 %834, 11
  store i32 %889, ptr @bar.lastc, align 4, !tbaa !6
  %890 = zext i8 %819 to i32
  %891 = xor i32 %889, %890
  %892 = icmp eq i32 %891, 128
  br i1 %892, label %893, label %837

893:                                              ; preds = %888
  %894 = add nuw nsw i32 %834, 12
  store i32 %894, ptr @bar.lastc, align 4, !tbaa !6
  %895 = zext i8 %821 to i32
  %896 = xor i32 %894, %895
  %897 = icmp eq i32 %896, 128
  br i1 %897, label %898, label %837

898:                                              ; preds = %893
  %899 = add nuw nsw i32 %834, 13
  store i32 %899, ptr @bar.lastc, align 4, !tbaa !6
  %900 = zext i8 %823 to i32
  %901 = xor i32 %899, %900
  %902 = icmp eq i32 %901, 128
  br i1 %902, label %903, label %837

903:                                              ; preds = %898
  %904 = add nuw nsw i32 %834, 14
  store i32 %904, ptr @bar.lastc, align 4, !tbaa !6
  %905 = zext i8 %825 to i32
  %906 = xor i32 %904, %905
  %907 = icmp eq i32 %906, 128
  br i1 %907, label %908, label %837

908:                                              ; preds = %903
  %909 = add nuw nsw i32 %834, 15
  store i32 %909, ptr @bar.lastc, align 4, !tbaa !6
  %910 = zext i8 %827 to i32
  %911 = xor i32 %909, %910
  %912 = icmp eq i32 %911, 128
  br i1 %912, label %913, label %837

913:                                              ; preds = %908
  %914 = or disjoint i32 %834, 16
  store i32 %914, ptr @bar.lastc, align 4, !tbaa !6
  %915 = icmp sgt i32 %795, -1
  br i1 %915, label %924, label %916

916:                                              ; preds = %913
  %917 = add nsw i32 %795, 8
  store i32 %917, ptr %10, align 8
  %918 = icmp samesign ult i32 %795, -7
  br i1 %918, label %919, label %924

919:                                              ; preds = %916
  %920 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %921 = load ptr, ptr %920, align 8
  %922 = sext i32 %795 to i64
  %923 = getelementptr inbounds i8, ptr %921, i64 %922
  br label %928

924:                                              ; preds = %916, %913
  %925 = phi i32 [ %917, %916 ], [ %795, %913 ]
  %926 = load ptr, ptr %6, align 8
  %927 = getelementptr inbounds nuw i8, ptr %926, i64 8
  store ptr %927, ptr %6, align 8
  br label %928

928:                                              ; preds = %924, %919
  %929 = phi i32 [ %917, %919 ], [ %925, %924 ]
  %930 = phi ptr [ %923, %919 ], [ %926, %924 ]
  %931 = load ptr, ptr %930, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(31) %2, ptr noundef nonnull align 1 dereferenceable(31) %931, i64 31, i1 false), !tbaa.struct !11
  %932 = load i8, ptr %2, align 4, !tbaa !10
  br i1 %833, label %933, label %945

933:                                              ; preds = %928
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 31, ptr @bar.lastn, align 4, !tbaa !6
  %934 = icmp eq i8 %932, -8
  br i1 %934, label %935, label %946

935:                                              ; preds = %933
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  br label %936

936:                                              ; preds = %935, %947
  %937 = phi i64 [ 1, %935 ], [ %949, %947 ]
  %938 = phi i32 [ 1, %935 ], [ %948, %947 ]
  %939 = getelementptr inbounds nuw i8, ptr %2, i64 %937
  %940 = load i8, ptr %939, align 1, !tbaa !10
  %941 = zext i8 %940 to i32
  %942 = and i32 %938, 255
  %943 = xor i32 %942, %941
  %944 = icmp eq i32 %943, 248
  br i1 %944, label %947, label %946

945:                                              ; preds = %928
  call void @abort() #7
  unreachable

946:                                              ; preds = %936, %933
  call void @abort() #7
  unreachable

947:                                              ; preds = %936
  %948 = add nuw nsw i32 %938, 1
  store i32 %948, ptr @bar.lastc, align 4, !tbaa !6
  %949 = add nuw nsw i64 %937, 1
  %950 = icmp eq i64 %949, 31
  br i1 %950, label %951, label %936, !llvm.loop !12

951:                                              ; preds = %947
  %952 = icmp eq i32 %948, 31
  %953 = icmp sgt i32 %929, -1
  br i1 %953, label %962, label %954

954:                                              ; preds = %951
  %955 = add nsw i32 %929, 8
  store i32 %955, ptr %10, align 8
  %956 = icmp samesign ult i32 %929, -7
  br i1 %956, label %957, label %962

957:                                              ; preds = %954
  %958 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %959 = load ptr, ptr %958, align 8
  %960 = sext i32 %929 to i64
  %961 = getelementptr inbounds i8, ptr %959, i64 %960
  br label %966

962:                                              ; preds = %954, %951
  %963 = phi i32 [ %955, %954 ], [ %929, %951 ]
  %964 = load ptr, ptr %6, align 8
  %965 = getelementptr inbounds nuw i8, ptr %964, i64 8
  store ptr %965, ptr %6, align 8
  br label %966

966:                                              ; preds = %962, %957
  %967 = phi i32 [ %955, %957 ], [ %963, %962 ]
  %968 = phi ptr [ %961, %957 ], [ %964, %962 ]
  %969 = load ptr, ptr %968, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(32) %3, ptr noundef nonnull align 1 dereferenceable(32) %969, i64 32, i1 false), !tbaa.struct !15
  %970 = load i8, ptr %3, align 4, !tbaa !10
  br i1 %952, label %971, label %981

971:                                              ; preds = %966
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 32, ptr @bar.lastn, align 4, !tbaa !6
  %972 = icmp eq i8 %970, 0
  br i1 %972, label %973, label %982

973:                                              ; preds = %971
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  br label %974

974:                                              ; preds = %973, %983
  %975 = phi i64 [ 1, %973 ], [ %985, %983 ]
  %976 = phi i32 [ 1, %973 ], [ %984, %983 ]
  %977 = getelementptr inbounds nuw i8, ptr %3, i64 %975
  %978 = load i8, ptr %977, align 1, !tbaa !10
  %979 = trunc i32 %976 to i8
  %980 = icmp eq i8 %978, %979
  br i1 %980, label %983, label %982

981:                                              ; preds = %966
  call void @abort() #7
  unreachable

982:                                              ; preds = %974, %971
  call void @abort() #7
  unreachable

983:                                              ; preds = %974
  %984 = add nuw nsw i32 %976, 1
  store i32 %984, ptr @bar.lastc, align 4, !tbaa !6
  %985 = add nuw nsw i64 %975, 1
  %986 = icmp eq i64 %985, 32
  br i1 %986, label %987, label %974, !llvm.loop !16

987:                                              ; preds = %983
  %988 = icmp eq i32 %984, 32
  %989 = icmp sgt i32 %967, -1
  br i1 %989, label %998, label %990

990:                                              ; preds = %987
  %991 = add nsw i32 %967, 8
  store i32 %991, ptr %10, align 8
  %992 = icmp samesign ult i32 %967, -7
  br i1 %992, label %993, label %998

993:                                              ; preds = %990
  %994 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %995 = load ptr, ptr %994, align 8
  %996 = sext i32 %967 to i64
  %997 = getelementptr inbounds i8, ptr %995, i64 %996
  br label %1002

998:                                              ; preds = %990, %987
  %999 = phi i32 [ %991, %990 ], [ %967, %987 ]
  %1000 = load ptr, ptr %6, align 8
  %1001 = getelementptr inbounds nuw i8, ptr %1000, i64 8
  store ptr %1001, ptr %6, align 8
  br label %1002

1002:                                             ; preds = %998, %993
  %1003 = phi i32 [ %991, %993 ], [ %999, %998 ]
  %1004 = phi ptr [ %997, %993 ], [ %1000, %998 ]
  %1005 = load ptr, ptr %1004, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(35) %4, ptr noundef nonnull align 1 dereferenceable(35) %1005, i64 35, i1 false), !tbaa.struct !17
  %1006 = load i8, ptr %4, align 4, !tbaa !10
  br i1 %988, label %1007, label %1019

1007:                                             ; preds = %1002
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 35, ptr @bar.lastn, align 4, !tbaa !6
  %1008 = icmp eq i8 %1006, 24
  br i1 %1008, label %1009, label %1020

1009:                                             ; preds = %1007
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  br label %1010

1010:                                             ; preds = %1009, %1021
  %1011 = phi i64 [ 1, %1009 ], [ %1023, %1021 ]
  %1012 = phi i32 [ 1, %1009 ], [ %1022, %1021 ]
  %1013 = getelementptr inbounds nuw i8, ptr %4, i64 %1011
  %1014 = load i8, ptr %1013, align 1, !tbaa !10
  %1015 = zext i8 %1014 to i32
  %1016 = and i32 %1012, 255
  %1017 = xor i32 %1016, %1015
  %1018 = icmp eq i32 %1017, 24
  br i1 %1018, label %1021, label %1020

1019:                                             ; preds = %1002
  call void @abort() #7
  unreachable

1020:                                             ; preds = %1010, %1007
  call void @abort() #7
  unreachable

1021:                                             ; preds = %1010
  %1022 = add nuw nsw i32 %1012, 1
  store i32 %1022, ptr @bar.lastc, align 4, !tbaa !6
  %1023 = add nuw nsw i64 %1011, 1
  %1024 = icmp eq i64 %1023, 35
  br i1 %1024, label %1025, label %1010, !llvm.loop !18

1025:                                             ; preds = %1021
  %1026 = icmp eq i32 %1022, 35
  %1027 = icmp sgt i32 %1003, -1
  br i1 %1027, label %1036, label %1028

1028:                                             ; preds = %1025
  %1029 = add nsw i32 %1003, 8
  store i32 %1029, ptr %10, align 8
  %1030 = icmp samesign ult i32 %1003, -7
  br i1 %1030, label %1031, label %1036

1031:                                             ; preds = %1028
  %1032 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %1033 = load ptr, ptr %1032, align 8
  %1034 = sext i32 %1003 to i64
  %1035 = getelementptr inbounds i8, ptr %1033, i64 %1034
  br label %1039

1036:                                             ; preds = %1028, %1025
  %1037 = load ptr, ptr %6, align 8
  %1038 = getelementptr inbounds nuw i8, ptr %1037, i64 8
  store ptr %1038, ptr %6, align 8
  br label %1039

1039:                                             ; preds = %1036, %1031
  %1040 = phi ptr [ %1035, %1031 ], [ %1037, %1036 ]
  %1041 = load ptr, ptr %1040, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(72) %5, ptr noundef nonnull align 1 dereferenceable(72) %1041, i64 72, i1 false), !tbaa.struct !19
  %1042 = load i8, ptr %5, align 4, !tbaa !10
  br i1 %1026, label %1043, label %1055

1043:                                             ; preds = %1039
  store i32 0, ptr @bar.lastc, align 4, !tbaa !6
  store i32 72, ptr @bar.lastn, align 4, !tbaa !6
  %1044 = icmp eq i8 %1042, 64
  br i1 %1044, label %1045, label %1056

1045:                                             ; preds = %1043
  store i32 1, ptr @bar.lastc, align 4, !tbaa !6
  br label %1046

1046:                                             ; preds = %1045, %1057
  %1047 = phi i64 [ 1, %1045 ], [ %1059, %1057 ]
  %1048 = phi i32 [ 1, %1045 ], [ %1058, %1057 ]
  %1049 = getelementptr inbounds nuw i8, ptr %5, i64 %1047
  %1050 = load i8, ptr %1049, align 1, !tbaa !10
  %1051 = zext i8 %1050 to i32
  %1052 = and i32 %1048, 255
  %1053 = xor i32 %1052, %1051
  %1054 = icmp eq i32 %1053, 64
  br i1 %1054, label %1057, label %1056

1055:                                             ; preds = %1039
  call void @abort() #7
  unreachable

1056:                                             ; preds = %1046, %1043
  call void @abort() #7
  unreachable

1057:                                             ; preds = %1046
  %1058 = add nuw nsw i32 %1048, 1
  store i32 %1058, ptr @bar.lastc, align 4, !tbaa !6
  %1059 = add nuw nsw i64 %1047, 1
  %1060 = icmp eq i64 %1059, 72
  br i1 %1060, label %1061, label %1046, !llvm.loop !20

1061:                                             ; preds = %1057
  call void @llvm.va_end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  %1 = alloca %struct.A31, align 4
  %2 = alloca %struct.A32, align 4
  %3 = alloca %struct.A35, align 4
  %4 = alloca %struct.A72, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #8
  store i8 -8, ptr %1, align 4
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 1
  store i8 -7, ptr %5, align 1
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 2
  store i8 -6, ptr %6, align 2
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 3
  store i8 -5, ptr %7, align 1
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 4
  store i8 -4, ptr %8, align 4
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 5
  store i8 -3, ptr %9, align 1
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 6
  store i8 -2, ptr %10, align 2
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 7
  store i8 -1, ptr %11, align 1
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i8 -16, ptr %12, align 4
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 9
  store i8 -15, ptr %13, align 1
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 10
  store i8 -14, ptr %14, align 2
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 11
  store i8 -13, ptr %15, align 1
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 12
  store i8 -12, ptr %16, align 4
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 13
  store i8 -11, ptr %17, align 1
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 14
  store i8 -10, ptr %18, align 2
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 15
  store i8 -9, ptr %19, align 1
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i8 -24, ptr %20, align 4
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 17
  store i8 -23, ptr %21, align 1
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 18
  store i8 -22, ptr %22, align 2
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 19
  store i8 -21, ptr %23, align 1
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 20
  store i8 -20, ptr %24, align 4
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 21
  store i8 -19, ptr %25, align 1
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 22
  store i8 -18, ptr %26, align 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 23
  store i8 -17, ptr %27, align 1
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i8 -32, ptr %28, align 4
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 25
  store i8 -31, ptr %29, align 1
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 26
  store i8 -30, ptr %30, align 2
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 27
  store i8 -29, ptr %31, align 1
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 28
  store i8 -28, ptr %32, align 4
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 29
  store i8 -27, ptr %33, align 1
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 30
  store i8 -26, ptr %34, align 2, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  store i8 0, ptr %2, align 4
  %35 = getelementptr inbounds nuw i8, ptr %2, i64 1
  store i8 1, ptr %35, align 1
  %36 = getelementptr inbounds nuw i8, ptr %2, i64 2
  store i8 2, ptr %36, align 2
  %37 = getelementptr inbounds nuw i8, ptr %2, i64 3
  store i8 3, ptr %37, align 1
  %38 = getelementptr inbounds nuw i8, ptr %2, i64 4
  store i8 4, ptr %38, align 4
  %39 = getelementptr inbounds nuw i8, ptr %2, i64 5
  store i8 5, ptr %39, align 1
  %40 = getelementptr inbounds nuw i8, ptr %2, i64 6
  store i8 6, ptr %40, align 2
  %41 = getelementptr inbounds nuw i8, ptr %2, i64 7
  store i8 7, ptr %41, align 1
  %42 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i8 8, ptr %42, align 4
  %43 = getelementptr inbounds nuw i8, ptr %2, i64 9
  store i8 9, ptr %43, align 1
  %44 = getelementptr inbounds nuw i8, ptr %2, i64 10
  store i8 10, ptr %44, align 2
  %45 = getelementptr inbounds nuw i8, ptr %2, i64 11
  store i8 11, ptr %45, align 1
  %46 = getelementptr inbounds nuw i8, ptr %2, i64 12
  store i8 12, ptr %46, align 4
  %47 = getelementptr inbounds nuw i8, ptr %2, i64 13
  store i8 13, ptr %47, align 1
  %48 = getelementptr inbounds nuw i8, ptr %2, i64 14
  store i8 14, ptr %48, align 2
  %49 = getelementptr inbounds nuw i8, ptr %2, i64 15
  store i8 15, ptr %49, align 1
  %50 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store i8 16, ptr %50, align 4
  %51 = getelementptr inbounds nuw i8, ptr %2, i64 17
  store i8 17, ptr %51, align 1
  %52 = getelementptr inbounds nuw i8, ptr %2, i64 18
  store i8 18, ptr %52, align 2
  %53 = getelementptr inbounds nuw i8, ptr %2, i64 19
  store i8 19, ptr %53, align 1
  %54 = getelementptr inbounds nuw i8, ptr %2, i64 20
  store i8 20, ptr %54, align 4
  %55 = getelementptr inbounds nuw i8, ptr %2, i64 21
  store i8 21, ptr %55, align 1
  %56 = getelementptr inbounds nuw i8, ptr %2, i64 22
  store i8 22, ptr %56, align 2
  %57 = getelementptr inbounds nuw i8, ptr %2, i64 23
  store i8 23, ptr %57, align 1
  %58 = getelementptr inbounds nuw i8, ptr %2, i64 24
  store i8 24, ptr %58, align 4
  %59 = getelementptr inbounds nuw i8, ptr %2, i64 25
  store i8 25, ptr %59, align 1
  %60 = getelementptr inbounds nuw i8, ptr %2, i64 26
  store i8 26, ptr %60, align 2
  %61 = getelementptr inbounds nuw i8, ptr %2, i64 27
  store i8 27, ptr %61, align 1
  %62 = getelementptr inbounds nuw i8, ptr %2, i64 28
  store i8 28, ptr %62, align 4
  %63 = getelementptr inbounds nuw i8, ptr %2, i64 29
  store i8 29, ptr %63, align 1
  %64 = getelementptr inbounds nuw i8, ptr %2, i64 30
  store i8 30, ptr %64, align 2
  %65 = getelementptr inbounds nuw i8, ptr %2, i64 31
  store i8 31, ptr %65, align 1, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #8
  store i8 24, ptr %3, align 4
  %66 = getelementptr inbounds nuw i8, ptr %3, i64 1
  store i8 25, ptr %66, align 1
  %67 = getelementptr inbounds nuw i8, ptr %3, i64 2
  store i8 26, ptr %67, align 2
  %68 = getelementptr inbounds nuw i8, ptr %3, i64 3
  store i8 27, ptr %68, align 1
  %69 = getelementptr inbounds nuw i8, ptr %3, i64 4
  store i8 28, ptr %69, align 4
  %70 = getelementptr inbounds nuw i8, ptr %3, i64 5
  store i8 29, ptr %70, align 1
  %71 = getelementptr inbounds nuw i8, ptr %3, i64 6
  store i8 30, ptr %71, align 2
  %72 = getelementptr inbounds nuw i8, ptr %3, i64 7
  store i8 31, ptr %72, align 1
  %73 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i8 16, ptr %73, align 4
  %74 = getelementptr inbounds nuw i8, ptr %3, i64 9
  store i8 17, ptr %74, align 1
  %75 = getelementptr inbounds nuw i8, ptr %3, i64 10
  store i8 18, ptr %75, align 2
  %76 = getelementptr inbounds nuw i8, ptr %3, i64 11
  store i8 19, ptr %76, align 1
  %77 = getelementptr inbounds nuw i8, ptr %3, i64 12
  store i8 20, ptr %77, align 4
  %78 = getelementptr inbounds nuw i8, ptr %3, i64 13
  store i8 21, ptr %78, align 1
  %79 = getelementptr inbounds nuw i8, ptr %3, i64 14
  store i8 22, ptr %79, align 2
  %80 = getelementptr inbounds nuw i8, ptr %3, i64 15
  store i8 23, ptr %80, align 1
  %81 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store i8 8, ptr %81, align 4
  %82 = getelementptr inbounds nuw i8, ptr %3, i64 17
  store i8 9, ptr %82, align 1
  %83 = getelementptr inbounds nuw i8, ptr %3, i64 18
  store i8 10, ptr %83, align 2
  %84 = getelementptr inbounds nuw i8, ptr %3, i64 19
  store i8 11, ptr %84, align 1
  %85 = getelementptr inbounds nuw i8, ptr %3, i64 20
  store i8 12, ptr %85, align 4
  %86 = getelementptr inbounds nuw i8, ptr %3, i64 21
  store i8 13, ptr %86, align 1
  %87 = getelementptr inbounds nuw i8, ptr %3, i64 22
  store i8 14, ptr %87, align 2
  %88 = getelementptr inbounds nuw i8, ptr %3, i64 23
  store i8 15, ptr %88, align 1
  %89 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store i8 0, ptr %89, align 4
  %90 = getelementptr inbounds nuw i8, ptr %3, i64 25
  store i8 1, ptr %90, align 1
  %91 = getelementptr inbounds nuw i8, ptr %3, i64 26
  store i8 2, ptr %91, align 2
  %92 = getelementptr inbounds nuw i8, ptr %3, i64 27
  store i8 3, ptr %92, align 1
  %93 = getelementptr inbounds nuw i8, ptr %3, i64 28
  store i8 4, ptr %93, align 4
  %94 = getelementptr inbounds nuw i8, ptr %3, i64 29
  store i8 5, ptr %94, align 1
  %95 = getelementptr inbounds nuw i8, ptr %3, i64 30
  store i8 6, ptr %95, align 2
  %96 = getelementptr inbounds nuw i8, ptr %3, i64 31
  store i8 7, ptr %96, align 1
  %97 = getelementptr inbounds nuw i8, ptr %3, i64 32
  store i8 56, ptr %97, align 4
  %98 = getelementptr inbounds nuw i8, ptr %3, i64 33
  store i8 57, ptr %98, align 1
  %99 = getelementptr inbounds nuw i8, ptr %3, i64 34
  store i8 58, ptr %99, align 2, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #8
  store i8 64, ptr %4, align 4
  %100 = getelementptr inbounds nuw i8, ptr %4, i64 1
  store i8 65, ptr %100, align 1
  %101 = getelementptr inbounds nuw i8, ptr %4, i64 2
  store i8 66, ptr %101, align 2
  %102 = getelementptr inbounds nuw i8, ptr %4, i64 3
  store i8 67, ptr %102, align 1
  %103 = getelementptr inbounds nuw i8, ptr %4, i64 4
  store i8 68, ptr %103, align 4
  %104 = getelementptr inbounds nuw i8, ptr %4, i64 5
  store i8 69, ptr %104, align 1
  %105 = getelementptr inbounds nuw i8, ptr %4, i64 6
  store i8 70, ptr %105, align 2
  %106 = getelementptr inbounds nuw i8, ptr %4, i64 7
  store i8 71, ptr %106, align 1
  %107 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i8 72, ptr %107, align 4
  %108 = getelementptr inbounds nuw i8, ptr %4, i64 9
  store i8 73, ptr %108, align 1
  %109 = getelementptr inbounds nuw i8, ptr %4, i64 10
  store i8 74, ptr %109, align 2
  %110 = getelementptr inbounds nuw i8, ptr %4, i64 11
  store i8 75, ptr %110, align 1
  %111 = getelementptr inbounds nuw i8, ptr %4, i64 12
  store i8 76, ptr %111, align 4
  %112 = getelementptr inbounds nuw i8, ptr %4, i64 13
  store i8 77, ptr %112, align 1
  %113 = getelementptr inbounds nuw i8, ptr %4, i64 14
  store i8 78, ptr %113, align 2
  %114 = getelementptr inbounds nuw i8, ptr %4, i64 15
  store i8 79, ptr %114, align 1
  %115 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i8 80, ptr %115, align 4
  %116 = getelementptr inbounds nuw i8, ptr %4, i64 17
  store i8 81, ptr %116, align 1
  %117 = getelementptr inbounds nuw i8, ptr %4, i64 18
  store i8 82, ptr %117, align 2
  %118 = getelementptr inbounds nuw i8, ptr %4, i64 19
  store i8 83, ptr %118, align 1
  %119 = getelementptr inbounds nuw i8, ptr %4, i64 20
  store i8 84, ptr %119, align 4
  %120 = getelementptr inbounds nuw i8, ptr %4, i64 21
  store i8 85, ptr %120, align 1
  %121 = getelementptr inbounds nuw i8, ptr %4, i64 22
  store i8 86, ptr %121, align 2
  %122 = getelementptr inbounds nuw i8, ptr %4, i64 23
  store i8 87, ptr %122, align 1
  %123 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store i8 88, ptr %123, align 4
  %124 = getelementptr inbounds nuw i8, ptr %4, i64 25
  store i8 89, ptr %124, align 1
  %125 = getelementptr inbounds nuw i8, ptr %4, i64 26
  store i8 90, ptr %125, align 2
  %126 = getelementptr inbounds nuw i8, ptr %4, i64 27
  store i8 91, ptr %126, align 1
  %127 = getelementptr inbounds nuw i8, ptr %4, i64 28
  store i8 92, ptr %127, align 4
  %128 = getelementptr inbounds nuw i8, ptr %4, i64 29
  store i8 93, ptr %128, align 1
  %129 = getelementptr inbounds nuw i8, ptr %4, i64 30
  store i8 94, ptr %129, align 2
  %130 = getelementptr inbounds nuw i8, ptr %4, i64 31
  store i8 95, ptr %130, align 1
  %131 = getelementptr inbounds nuw i8, ptr %4, i64 32
  store i8 96, ptr %131, align 4
  %132 = getelementptr inbounds nuw i8, ptr %4, i64 33
  store i8 97, ptr %132, align 1
  %133 = getelementptr inbounds nuw i8, ptr %4, i64 34
  store i8 98, ptr %133, align 2
  %134 = getelementptr inbounds nuw i8, ptr %4, i64 35
  store i8 99, ptr %134, align 1
  %135 = getelementptr inbounds nuw i8, ptr %4, i64 36
  store i8 100, ptr %135, align 4
  %136 = getelementptr inbounds nuw i8, ptr %4, i64 37
  store i8 101, ptr %136, align 1
  %137 = getelementptr inbounds nuw i8, ptr %4, i64 38
  store i8 102, ptr %137, align 2
  %138 = getelementptr inbounds nuw i8, ptr %4, i64 39
  store i8 103, ptr %138, align 1
  %139 = getelementptr inbounds nuw i8, ptr %4, i64 40
  store i8 104, ptr %139, align 4
  %140 = getelementptr inbounds nuw i8, ptr %4, i64 41
  store i8 105, ptr %140, align 1
  %141 = getelementptr inbounds nuw i8, ptr %4, i64 42
  store i8 106, ptr %141, align 2
  %142 = getelementptr inbounds nuw i8, ptr %4, i64 43
  store i8 107, ptr %142, align 1
  %143 = getelementptr inbounds nuw i8, ptr %4, i64 44
  store i8 108, ptr %143, align 4
  %144 = getelementptr inbounds nuw i8, ptr %4, i64 45
  store i8 109, ptr %144, align 1
  %145 = getelementptr inbounds nuw i8, ptr %4, i64 46
  store i8 110, ptr %145, align 2
  %146 = getelementptr inbounds nuw i8, ptr %4, i64 47
  store i8 111, ptr %146, align 1
  %147 = getelementptr inbounds nuw i8, ptr %4, i64 48
  store i8 112, ptr %147, align 4
  %148 = getelementptr inbounds nuw i8, ptr %4, i64 49
  store i8 113, ptr %148, align 1
  %149 = getelementptr inbounds nuw i8, ptr %4, i64 50
  store i8 114, ptr %149, align 2
  %150 = getelementptr inbounds nuw i8, ptr %4, i64 51
  store i8 115, ptr %150, align 1
  %151 = getelementptr inbounds nuw i8, ptr %4, i64 52
  store i8 116, ptr %151, align 4
  %152 = getelementptr inbounds nuw i8, ptr %4, i64 53
  store i8 117, ptr %152, align 1
  %153 = getelementptr inbounds nuw i8, ptr %4, i64 54
  store i8 118, ptr %153, align 2
  %154 = getelementptr inbounds nuw i8, ptr %4, i64 55
  store i8 119, ptr %154, align 1
  %155 = getelementptr inbounds nuw i8, ptr %4, i64 56
  store i8 120, ptr %155, align 4
  %156 = getelementptr inbounds nuw i8, ptr %4, i64 57
  store i8 121, ptr %156, align 1
  %157 = getelementptr inbounds nuw i8, ptr %4, i64 58
  store i8 122, ptr %157, align 2
  %158 = getelementptr inbounds nuw i8, ptr %4, i64 59
  store i8 123, ptr %158, align 1
  %159 = getelementptr inbounds nuw i8, ptr %4, i64 60
  store i8 124, ptr %159, align 4
  %160 = getelementptr inbounds nuw i8, ptr %4, i64 61
  store i8 125, ptr %160, align 1
  %161 = getelementptr inbounds nuw i8, ptr %4, i64 62
  store i8 126, ptr %161, align 2
  %162 = getelementptr inbounds nuw i8, ptr %4, i64 63
  store i8 127, ptr %162, align 1
  %163 = getelementptr inbounds nuw i8, ptr %4, i64 64
  store i8 0, ptr %163, align 4
  %164 = getelementptr inbounds nuw i8, ptr %4, i64 65
  store i8 1, ptr %164, align 1
  %165 = getelementptr inbounds nuw i8, ptr %4, i64 66
  store i8 2, ptr %165, align 2
  %166 = getelementptr inbounds nuw i8, ptr %4, i64 67
  store i8 3, ptr %166, align 1
  %167 = getelementptr inbounds nuw i8, ptr %4, i64 68
  store i8 4, ptr %167, align 4
  %168 = getelementptr inbounds nuw i8, ptr %4, i64 69
  store i8 5, ptr %168, align 1
  %169 = getelementptr inbounds nuw i8, ptr %4, i64 70
  store i8 6, ptr %169, align 2
  %170 = getelementptr inbounds nuw i8, ptr %4, i64 71
  store i8 7, ptr %170, align 1, !tbaa !10
  call void (i32, ...) @foo(i32 noundef 21, i64 8, i64 4368, i64 1710360, i64 589439264, i64 189702744360, i64 58498313498928, i64 17518777457064248, i64 5135868584551137600, [2 x i64] [i64 5714589967255750984, i64 64], [2 x i64] [i64 6293311349960364368, i64 22872], [2 x i64] [i64 6872032732664977752, i64 5394768], [2 x i64] [i64 7450754115369591136, i64 1802135912], [2 x i64] [i64 8029475498074204520, i64 431164121440], [2 x i64] [i64 8608196880778817904, i64 137973601040760], [2 x i64] [i64 9186918263483431288, i64 33343190265393520], [2 x i64] [i64 -8681104427521506944, i64 -8102383044816893560], ptr dead_on_return noundef nonnull %1, ptr dead_on_return noundef nonnull %2, ptr dead_on_return noundef nonnull %3, ptr dead_on_return noundef nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #8
  call void @exit(i32 noundef 0) #7
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #6

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { noreturn nounwind }
attributes #8 = { nounwind }

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
!11 = !{i64 0, i64 31, !10}
!12 = distinct !{!12, !13, !14}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!"llvm.loop.peeled.count", i32 1}
!15 = !{i64 0, i64 32, !10}
!16 = distinct !{!16, !13, !14}
!17 = !{i64 0, i64 35, !10}
!18 = distinct !{!18, !13, !14}
!19 = !{i64 0, i64 72, !10}
!20 = distinct !{!20, !13, !14}
