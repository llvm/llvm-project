; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/uint64_to_float.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/uint64_to_float.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [28 x i8] c"Error detected @ 0x%016llx\0A\00", align 1
@.str.1 = private unnamed_addr constant [31 x i8] c"\09Expected result: %a (0x%08x)\0A\00", align 1
@.str.2 = private unnamed_addr constant [31 x i8] c"\09Observed result: %a (0x%08x)\0A\00", align 1
@.str.3 = private constant [11 x i8] c"to nearest\00", align 1
@.str.4 = private constant [5 x i8] c"down\00", align 1
@.str.5 = private constant [3 x i8] c"up\00", align 1
@.str.6 = private constant [13 x i8] c"towards zero\00", align 1
@__const.main.modeNames.rel = private unnamed_addr constant [4 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @.str.3 to i64), i64 ptrtoint (ptr @__const.main.modeNames.rel to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @.str.4 to i64), i64 ptrtoint (ptr @__const.main.modeNames.rel to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @.str.5 to i64), i64 ptrtoint (ptr @__const.main.modeNames.rel to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @.str.6 to i64), i64 ptrtoint (ptr @__const.main.modeNames.rel to i64)) to i32)], align 4
@.str.7 = private unnamed_addr constant [55 x i8] c"Testing uint64_t --> float conversions in round %s...\0A\00", align 1
@str = private unnamed_addr constant [18 x i8] c"Finished Testing.\00", align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local float @floatundisf(i64 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i64 %0, 0
  br i1 %2, label %45, label %3

3:                                                ; preds = %1
  %4 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %0, i1 true)
  %5 = trunc nuw nsw i64 %4 to i32
  %6 = sub nuw nsw i32 64, %5
  %7 = xor i32 %5, 63
  %8 = icmp ugt i64 %0, 16777215
  br i1 %8, label %9, label %32

9:                                                ; preds = %3
  switch i32 %5, label %12 [
    i32 39, label %10
    i32 38, label %21
  ]

10:                                               ; preds = %9
  %11 = shl i64 %0, 1
  br label %21

12:                                               ; preds = %9
  %13 = sub nsw i64 38, %4
  %14 = and i64 %13, 4294967295
  %15 = lshr i64 %0, %14
  %16 = lshr i64 274877906943, %4
  %17 = and i64 %16, %0
  %18 = icmp ne i64 %17, 0
  %19 = zext i1 %18 to i64
  %20 = or i64 %15, %19
  br label %21

21:                                               ; preds = %12, %9, %10
  %22 = phi i64 [ %20, %12 ], [ %11, %10 ], [ %0, %9 ]
  %23 = lshr i64 %22, 2
  %24 = and i64 %23, 1
  %25 = or i64 %24, %22
  %26 = add i64 %25, 1
  %27 = and i64 %26, 67108864
  %28 = icmp eq i64 %27, 0
  %29 = select i1 %28, i64 2, i64 3
  %30 = lshr i64 %26, %29
  %31 = select i1 %28, i32 %7, i32 %6
  br label %36

32:                                               ; preds = %3
  %33 = add nuw nsw i64 %4, 4294967256
  %34 = and i64 %33, 4294967295
  %35 = shl i64 %0, %34
  br label %36

36:                                               ; preds = %21, %32
  %37 = phi i64 [ %35, %32 ], [ %30, %21 ]
  %38 = phi i32 [ %7, %32 ], [ %31, %21 ]
  %39 = shl nuw nsw i32 %38, 23
  %40 = add nuw nsw i32 %39, 1065353216
  %41 = trunc i64 %37 to i32
  %42 = and i32 %41, 8388607
  %43 = or disjoint i32 %40, %42
  %44 = bitcast i32 %43 to float
  br label %45

45:                                               ; preds = %1, %36
  %46 = phi float [ %44, %36 ], [ 0.000000e+00, %1 ]
  ret float %46
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.ctlz.i64(i64, i1 immarg) #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @test(i64 noundef %0) local_unnamed_addr #2 {
  %2 = icmp eq i64 %0, 0
  br i1 %2, label %45, label %3

3:                                                ; preds = %1
  %4 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %0, i1 true)
  %5 = trunc nuw nsw i64 %4 to i32
  %6 = sub nuw nsw i32 64, %5
  %7 = xor i32 %5, 63
  %8 = icmp ugt i64 %0, 16777215
  br i1 %8, label %9, label %32

9:                                                ; preds = %3
  switch i32 %5, label %12 [
    i32 39, label %10
    i32 38, label %21
  ]

10:                                               ; preds = %9
  %11 = shl i64 %0, 1
  br label %21

12:                                               ; preds = %9
  %13 = sub nsw i64 38, %4
  %14 = and i64 %13, 4294967295
  %15 = lshr i64 %0, %14
  %16 = lshr i64 274877906943, %4
  %17 = and i64 %16, %0
  %18 = icmp ne i64 %17, 0
  %19 = zext i1 %18 to i64
  %20 = or i64 %15, %19
  br label %21

21:                                               ; preds = %12, %10, %9
  %22 = phi i64 [ %20, %12 ], [ %11, %10 ], [ %0, %9 ]
  %23 = lshr i64 %22, 2
  %24 = and i64 %23, 1
  %25 = or i64 %24, %22
  %26 = add i64 %25, 1
  %27 = and i64 %26, 67108864
  %28 = icmp eq i64 %27, 0
  %29 = select i1 %28, i64 2, i64 3
  %30 = lshr i64 %26, %29
  %31 = select i1 %28, i32 %7, i32 %6
  br label %36

32:                                               ; preds = %3
  %33 = add nuw nsw i64 %4, 4294967256
  %34 = and i64 %33, 4294967295
  %35 = shl i64 %0, %34
  br label %36

36:                                               ; preds = %32, %21
  %37 = phi i64 [ %35, %32 ], [ %30, %21 ]
  %38 = phi i32 [ %7, %32 ], [ %31, %21 ]
  %39 = shl nuw nsw i32 %38, 23
  %40 = add nuw nsw i32 %39, 1065353216
  %41 = trunc i64 %37 to i32
  %42 = and i32 %41, 8388607
  %43 = or disjoint i32 %40, %42
  %44 = bitcast i32 %43 to float
  br label %45

45:                                               ; preds = %1, %36
  %46 = phi float [ %44, %36 ], [ 0.000000e+00, %1 ]
  %47 = bitcast float %46 to i32
  %48 = uitofp i64 %0 to float
  %49 = bitcast float %48 to i32
  %50 = icmp eq i32 %47, %49
  br i1 %50, label %57, label %51

51:                                               ; preds = %45
  %52 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %0)
  %53 = fpext float %46 to double
  %54 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %53, i32 noundef %47)
  %55 = fpext float %48 to double
  %56 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %55, i32 noundef %49)
  br label %57

57:                                               ; preds = %51, %45
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #4 {
  br label %3

3:                                                ; preds = %2, %1704
  %4 = phi i64 [ 0, %2 ], [ %1705, %1704 ]
  %5 = tail call i32 @fesetround(i32 noundef 0) #8
  %6 = shl i64 %4, 2
  %7 = call ptr @llvm.load.relative.i64(ptr @__const.main.modeNames.rel, i64 %6)
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, ptr noundef %7)
  br label %9

9:                                                ; preds = %3, %1701
  %10 = phi i64 [ 0, %3 ], [ %1702, %1701 ]
  %11 = shl nuw i64 1, %10
  %12 = trunc i64 %10 to i32
  %13 = sub i32 63, %12
  %14 = add i32 %12, 1
  %15 = xor i32 %13, 63
  %16 = icmp samesign ugt i64 %10, 23
  br i1 %16, label %17, label %37

17:                                               ; preds = %9
  switch i32 %12, label %20 [
    i32 24, label %18
    i32 25, label %24
  ]

18:                                               ; preds = %17
  %19 = shl i64 2, %10
  br label %24

20:                                               ; preds = %17
  %21 = add nuw nsw i64 %10, 4294967271
  %22 = and i64 %21, 4294967295
  %23 = lshr i64 %11, %22
  br label %24

24:                                               ; preds = %20, %18, %17
  %25 = phi i64 [ %23, %20 ], [ %19, %18 ], [ %11, %17 ]
  %26 = lshr i64 %25, 2
  %27 = and i64 %26, 1
  %28 = or i64 %27, %25
  %29 = add i64 %28, 1
  %30 = and i64 %29, 67108864
  %31 = icmp eq i64 %30, 0
  %32 = select i1 %31, i64 2, i64 3
  %33 = lshr i64 %29, %32
  %34 = select i1 %31, i32 %15, i32 %14
  %35 = trunc i64 %33 to i32
  %36 = and i32 %35, 8388607
  br label %37

37:                                               ; preds = %9, %24
  %38 = phi i32 [ %36, %24 ], [ 0, %9 ]
  %39 = phi i32 [ %34, %24 ], [ %15, %9 ]
  %40 = shl nuw nsw i32 %39, 23
  %41 = add nuw nsw i32 %40, 1065353216
  %42 = or disjoint i32 %41, %38
  %43 = uitofp i64 %11 to float
  %44 = bitcast float %43 to i32
  %45 = icmp eq i32 %42, %44
  br i1 %45, label %53, label %46

46:                                               ; preds = %37
  %47 = bitcast i32 %42 to float
  %48 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %11)
  %49 = fpext float %47 to double
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %49, i32 noundef %42)
  %51 = fpext float %43 to double
  %52 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %51, i32 noundef %44)
  br label %53

53:                                               ; preds = %37, %46
  %54 = shl nsw i64 -1, %10
  %55 = lshr i64 %54, 38
  %56 = and i64 %54, 274877906943
  %57 = icmp ne i64 %56, 0
  %58 = zext i1 %57 to i64
  %59 = lshr i64 %54, 40
  %60 = and i64 %59, 1
  %61 = or i64 %60, %55
  %62 = or i64 %61, %58
  %63 = add nuw nsw i64 %62, 1
  %64 = icmp eq i64 %62, 67108863
  %65 = select i1 %64, i64 3, i64 2
  %66 = lshr i64 %63, %65
  %67 = select i1 %64, i32 1602224128, i32 1593835520
  %68 = trunc nuw nsw i64 %66 to i32
  %69 = and i32 %68, 8388607
  %70 = or disjoint i32 %69, %67
  %71 = uitofp i64 %54 to float
  %72 = bitcast float %71 to i32
  %73 = icmp eq i32 %70, %72
  br i1 %73, label %81, label %74

74:                                               ; preds = %53
  %75 = bitcast i32 %70 to float
  %76 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %54)
  %77 = fpext float %75 to double
  %78 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %77, i32 noundef %70)
  %79 = fpext float %71 to double
  %80 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %79, i32 noundef %72)
  br label %81

81:                                               ; preds = %53, %74
  %82 = icmp eq i64 %10, 0
  br i1 %82, label %1701, label %83

83:                                               ; preds = %81, %1698
  %84 = phi i64 [ %1699, %1698 ], [ 0, %81 ]
  %85 = shl nuw i64 1, %84
  %86 = add i64 %85, %11
  %87 = icmp eq i64 %86, 0
  br i1 %87, label %130, label %88

88:                                               ; preds = %83
  %89 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %86, i1 true)
  %90 = trunc nuw nsw i64 %89 to i32
  %91 = sub nuw nsw i32 64, %90
  %92 = xor i32 %90, 63
  %93 = icmp ugt i64 %86, 16777215
  br i1 %93, label %94, label %117

94:                                               ; preds = %88
  switch i32 %90, label %97 [
    i32 39, label %95
    i32 38, label %106
  ]

95:                                               ; preds = %94
  %96 = shl i64 %86, 1
  br label %106

97:                                               ; preds = %94
  %98 = sub nsw i64 38, %89
  %99 = and i64 %98, 4294967295
  %100 = lshr i64 %86, %99
  %101 = lshr i64 274877906943, %89
  %102 = and i64 %101, %86
  %103 = icmp ne i64 %102, 0
  %104 = zext i1 %103 to i64
  %105 = or i64 %100, %104
  br label %106

106:                                              ; preds = %97, %95, %94
  %107 = phi i64 [ %105, %97 ], [ %96, %95 ], [ %86, %94 ]
  %108 = lshr i64 %107, 2
  %109 = and i64 %108, 1
  %110 = or i64 %109, %107
  %111 = add i64 %110, 1
  %112 = and i64 %111, 67108864
  %113 = icmp eq i64 %112, 0
  %114 = select i1 %113, i64 2, i64 3
  %115 = lshr i64 %111, %114
  %116 = select i1 %113, i32 %92, i32 %91
  br label %121

117:                                              ; preds = %88
  %118 = add nuw nsw i64 %89, 4294967256
  %119 = and i64 %118, 4294967295
  %120 = shl i64 %86, %119
  br label %121

121:                                              ; preds = %117, %106
  %122 = phi i64 [ %120, %117 ], [ %115, %106 ]
  %123 = phi i32 [ %92, %117 ], [ %116, %106 ]
  %124 = shl nuw nsw i32 %123, 23
  %125 = add nuw nsw i32 %124, 1065353216
  %126 = trunc i64 %122 to i32
  %127 = and i32 %126, 8388607
  %128 = or disjoint i32 %125, %127
  %129 = bitcast i32 %128 to float
  br label %130

130:                                              ; preds = %121, %83
  %131 = phi float [ %129, %121 ], [ 0.000000e+00, %83 ]
  %132 = bitcast float %131 to i32
  %133 = uitofp i64 %86 to float
  %134 = bitcast float %133 to i32
  %135 = icmp eq i32 %132, %134
  br i1 %135, label %142, label %136

136:                                              ; preds = %130
  %137 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %86)
  %138 = fpext float %131 to double
  %139 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %138, i32 noundef %132)
  %140 = fpext float %133 to double
  %141 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %140, i32 noundef %134)
  br label %142

142:                                              ; preds = %130, %136
  %143 = shl nsw i64 -1, %84
  %144 = add i64 %143, %11
  %145 = icmp eq i64 %144, 0
  br i1 %145, label %188, label %146

146:                                              ; preds = %142
  %147 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %144, i1 true)
  %148 = trunc nuw nsw i64 %147 to i32
  %149 = sub nuw nsw i32 64, %148
  %150 = xor i32 %148, 63
  %151 = icmp ugt i64 %144, 16777215
  br i1 %151, label %152, label %175

152:                                              ; preds = %146
  switch i32 %148, label %155 [
    i32 39, label %153
    i32 38, label %164
  ]

153:                                              ; preds = %152
  %154 = shl i64 %144, 1
  br label %164

155:                                              ; preds = %152
  %156 = sub nsw i64 38, %147
  %157 = and i64 %156, 4294967295
  %158 = lshr i64 %144, %157
  %159 = lshr i64 274877906943, %147
  %160 = and i64 %159, %144
  %161 = icmp ne i64 %160, 0
  %162 = zext i1 %161 to i64
  %163 = or i64 %158, %162
  br label %164

164:                                              ; preds = %155, %153, %152
  %165 = phi i64 [ %163, %155 ], [ %154, %153 ], [ %144, %152 ]
  %166 = lshr i64 %165, 2
  %167 = and i64 %166, 1
  %168 = or i64 %167, %165
  %169 = add i64 %168, 1
  %170 = and i64 %169, 67108864
  %171 = icmp eq i64 %170, 0
  %172 = select i1 %171, i64 2, i64 3
  %173 = lshr i64 %169, %172
  %174 = select i1 %171, i32 %150, i32 %149
  br label %179

175:                                              ; preds = %146
  %176 = add nuw nsw i64 %147, 4294967256
  %177 = and i64 %176, 4294967295
  %178 = shl i64 %144, %177
  br label %179

179:                                              ; preds = %175, %164
  %180 = phi i64 [ %178, %175 ], [ %173, %164 ]
  %181 = phi i32 [ %150, %175 ], [ %174, %164 ]
  %182 = shl nuw nsw i32 %181, 23
  %183 = add nuw nsw i32 %182, 1065353216
  %184 = trunc i64 %180 to i32
  %185 = and i32 %184, 8388607
  %186 = or disjoint i32 %183, %185
  %187 = bitcast i32 %186 to float
  br label %188

188:                                              ; preds = %179, %142
  %189 = phi float [ %187, %179 ], [ 0.000000e+00, %142 ]
  %190 = bitcast float %189 to i32
  %191 = uitofp i64 %144 to float
  %192 = bitcast float %191 to i32
  %193 = icmp eq i32 %190, %192
  br i1 %193, label %200, label %194

194:                                              ; preds = %188
  %195 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %144)
  %196 = fpext float %189 to double
  %197 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %196, i32 noundef %190)
  %198 = fpext float %191 to double
  %199 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %198, i32 noundef %192)
  br label %200

200:                                              ; preds = %188, %194
  %201 = add i64 %85, %54
  %202 = icmp eq i64 %201, 0
  br i1 %202, label %245, label %203

203:                                              ; preds = %200
  %204 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %201, i1 true)
  %205 = trunc nuw nsw i64 %204 to i32
  %206 = sub nuw nsw i32 64, %205
  %207 = xor i32 %205, 63
  %208 = icmp ugt i64 %201, 16777215
  br i1 %208, label %209, label %232

209:                                              ; preds = %203
  switch i32 %205, label %212 [
    i32 39, label %210
    i32 38, label %221
  ]

210:                                              ; preds = %209
  %211 = shl i64 %201, 1
  br label %221

212:                                              ; preds = %209
  %213 = sub nsw i64 38, %204
  %214 = and i64 %213, 4294967295
  %215 = lshr i64 %201, %214
  %216 = lshr i64 274877906943, %204
  %217 = and i64 %216, %201
  %218 = icmp ne i64 %217, 0
  %219 = zext i1 %218 to i64
  %220 = or i64 %215, %219
  br label %221

221:                                              ; preds = %212, %210, %209
  %222 = phi i64 [ %220, %212 ], [ %211, %210 ], [ %201, %209 ]
  %223 = lshr i64 %222, 2
  %224 = and i64 %223, 1
  %225 = or i64 %224, %222
  %226 = add i64 %225, 1
  %227 = and i64 %226, 67108864
  %228 = icmp eq i64 %227, 0
  %229 = select i1 %228, i64 2, i64 3
  %230 = lshr i64 %226, %229
  %231 = select i1 %228, i32 %207, i32 %206
  br label %236

232:                                              ; preds = %203
  %233 = add nuw nsw i64 %204, 4294967256
  %234 = and i64 %233, 4294967295
  %235 = shl i64 %201, %234
  br label %236

236:                                              ; preds = %232, %221
  %237 = phi i64 [ %235, %232 ], [ %230, %221 ]
  %238 = phi i32 [ %207, %232 ], [ %231, %221 ]
  %239 = shl nuw nsw i32 %238, 23
  %240 = add nuw nsw i32 %239, 1065353216
  %241 = trunc i64 %237 to i32
  %242 = and i32 %241, 8388607
  %243 = or disjoint i32 %240, %242
  %244 = bitcast i32 %243 to float
  br label %245

245:                                              ; preds = %236, %200
  %246 = phi float [ %244, %236 ], [ 0.000000e+00, %200 ]
  %247 = bitcast float %246 to i32
  %248 = uitofp i64 %201 to float
  %249 = bitcast float %248 to i32
  %250 = icmp eq i32 %247, %249
  br i1 %250, label %257, label %251

251:                                              ; preds = %245
  %252 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %201)
  %253 = fpext float %246 to double
  %254 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %253, i32 noundef %247)
  %255 = fpext float %248 to double
  %256 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %255, i32 noundef %249)
  br label %257

257:                                              ; preds = %245, %251
  %258 = add i64 %143, %54
  %259 = icmp eq i64 %258, 0
  br i1 %259, label %302, label %260

260:                                              ; preds = %257
  %261 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %258, i1 true)
  %262 = trunc nuw nsw i64 %261 to i32
  %263 = sub nuw nsw i32 64, %262
  %264 = xor i32 %262, 63
  %265 = icmp ugt i64 %258, 16777215
  br i1 %265, label %266, label %289

266:                                              ; preds = %260
  switch i32 %262, label %269 [
    i32 39, label %267
    i32 38, label %278
  ]

267:                                              ; preds = %266
  %268 = shl i64 %258, 1
  br label %278

269:                                              ; preds = %266
  %270 = sub nsw i64 38, %261
  %271 = and i64 %270, 4294967295
  %272 = lshr i64 %258, %271
  %273 = lshr i64 274877906943, %261
  %274 = and i64 %273, %258
  %275 = icmp ne i64 %274, 0
  %276 = zext i1 %275 to i64
  %277 = or i64 %272, %276
  br label %278

278:                                              ; preds = %269, %267, %266
  %279 = phi i64 [ %277, %269 ], [ %268, %267 ], [ %258, %266 ]
  %280 = lshr i64 %279, 2
  %281 = and i64 %280, 1
  %282 = or i64 %281, %279
  %283 = add i64 %282, 1
  %284 = and i64 %283, 67108864
  %285 = icmp eq i64 %284, 0
  %286 = select i1 %285, i64 2, i64 3
  %287 = lshr i64 %283, %286
  %288 = select i1 %285, i32 %264, i32 %263
  br label %293

289:                                              ; preds = %260
  %290 = add nuw nsw i64 %261, 4294967256
  %291 = and i64 %290, 4294967295
  %292 = shl i64 %258, %291
  br label %293

293:                                              ; preds = %289, %278
  %294 = phi i64 [ %292, %289 ], [ %287, %278 ]
  %295 = phi i32 [ %264, %289 ], [ %288, %278 ]
  %296 = shl nuw nsw i32 %295, 23
  %297 = add nuw nsw i32 %296, 1065353216
  %298 = trunc i64 %294 to i32
  %299 = and i32 %298, 8388607
  %300 = or disjoint i32 %297, %299
  %301 = bitcast i32 %300 to float
  br label %302

302:                                              ; preds = %293, %257
  %303 = phi float [ %301, %293 ], [ 0.000000e+00, %257 ]
  %304 = bitcast float %303 to i32
  %305 = uitofp i64 %258 to float
  %306 = bitcast float %305 to i32
  %307 = icmp eq i32 %304, %306
  br i1 %307, label %314, label %308

308:                                              ; preds = %302
  %309 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %258)
  %310 = fpext float %303 to double
  %311 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %310, i32 noundef %304)
  %312 = fpext float %305 to double
  %313 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %312, i32 noundef %306)
  br label %314

314:                                              ; preds = %302, %308
  %315 = icmp eq i64 %84, 0
  br i1 %315, label %1698, label %316

316:                                              ; preds = %314, %1695
  %317 = phi i64 [ %1696, %1695 ], [ 0, %314 ]
  %318 = shl nuw i64 1, %317
  %319 = add i64 %318, %86
  %320 = icmp eq i64 %319, 0
  br i1 %320, label %363, label %321

321:                                              ; preds = %316
  %322 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %319, i1 true)
  %323 = trunc nuw nsw i64 %322 to i32
  %324 = sub nuw nsw i32 64, %323
  %325 = xor i32 %323, 63
  %326 = icmp ugt i64 %319, 16777215
  br i1 %326, label %327, label %350

327:                                              ; preds = %321
  switch i32 %323, label %330 [
    i32 39, label %328
    i32 38, label %339
  ]

328:                                              ; preds = %327
  %329 = shl i64 %319, 1
  br label %339

330:                                              ; preds = %327
  %331 = sub nsw i64 38, %322
  %332 = and i64 %331, 4294967295
  %333 = lshr i64 %319, %332
  %334 = lshr i64 274877906943, %322
  %335 = and i64 %334, %319
  %336 = icmp ne i64 %335, 0
  %337 = zext i1 %336 to i64
  %338 = or i64 %333, %337
  br label %339

339:                                              ; preds = %330, %328, %327
  %340 = phi i64 [ %338, %330 ], [ %329, %328 ], [ %319, %327 ]
  %341 = lshr i64 %340, 2
  %342 = and i64 %341, 1
  %343 = or i64 %342, %340
  %344 = add i64 %343, 1
  %345 = and i64 %344, 67108864
  %346 = icmp eq i64 %345, 0
  %347 = select i1 %346, i64 2, i64 3
  %348 = lshr i64 %344, %347
  %349 = select i1 %346, i32 %325, i32 %324
  br label %354

350:                                              ; preds = %321
  %351 = add nuw nsw i64 %322, 4294967256
  %352 = and i64 %351, 4294967295
  %353 = shl i64 %319, %352
  br label %354

354:                                              ; preds = %350, %339
  %355 = phi i64 [ %353, %350 ], [ %348, %339 ]
  %356 = phi i32 [ %325, %350 ], [ %349, %339 ]
  %357 = shl nuw nsw i32 %356, 23
  %358 = add nuw nsw i32 %357, 1065353216
  %359 = trunc i64 %355 to i32
  %360 = and i32 %359, 8388607
  %361 = or disjoint i32 %358, %360
  %362 = bitcast i32 %361 to float
  br label %363

363:                                              ; preds = %354, %316
  %364 = phi float [ %362, %354 ], [ 0.000000e+00, %316 ]
  %365 = bitcast float %364 to i32
  %366 = uitofp i64 %319 to float
  %367 = bitcast float %366 to i32
  %368 = icmp eq i32 %365, %367
  br i1 %368, label %375, label %369

369:                                              ; preds = %363
  %370 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %319)
  %371 = fpext float %364 to double
  %372 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %371, i32 noundef %365)
  %373 = fpext float %366 to double
  %374 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %373, i32 noundef %367)
  br label %375

375:                                              ; preds = %363, %369
  %376 = shl nsw i64 -1, %317
  %377 = add i64 %376, %86
  %378 = icmp eq i64 %377, 0
  br i1 %378, label %421, label %379

379:                                              ; preds = %375
  %380 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %377, i1 true)
  %381 = trunc nuw nsw i64 %380 to i32
  %382 = sub nuw nsw i32 64, %381
  %383 = xor i32 %381, 63
  %384 = icmp ugt i64 %377, 16777215
  br i1 %384, label %385, label %408

385:                                              ; preds = %379
  switch i32 %381, label %388 [
    i32 39, label %386
    i32 38, label %397
  ]

386:                                              ; preds = %385
  %387 = shl i64 %377, 1
  br label %397

388:                                              ; preds = %385
  %389 = sub nsw i64 38, %380
  %390 = and i64 %389, 4294967295
  %391 = lshr i64 %377, %390
  %392 = lshr i64 274877906943, %380
  %393 = and i64 %392, %377
  %394 = icmp ne i64 %393, 0
  %395 = zext i1 %394 to i64
  %396 = or i64 %391, %395
  br label %397

397:                                              ; preds = %388, %386, %385
  %398 = phi i64 [ %396, %388 ], [ %387, %386 ], [ %377, %385 ]
  %399 = lshr i64 %398, 2
  %400 = and i64 %399, 1
  %401 = or i64 %400, %398
  %402 = add i64 %401, 1
  %403 = and i64 %402, 67108864
  %404 = icmp eq i64 %403, 0
  %405 = select i1 %404, i64 2, i64 3
  %406 = lshr i64 %402, %405
  %407 = select i1 %404, i32 %383, i32 %382
  br label %412

408:                                              ; preds = %379
  %409 = add nuw nsw i64 %380, 4294967256
  %410 = and i64 %409, 4294967295
  %411 = shl i64 %377, %410
  br label %412

412:                                              ; preds = %408, %397
  %413 = phi i64 [ %411, %408 ], [ %406, %397 ]
  %414 = phi i32 [ %383, %408 ], [ %407, %397 ]
  %415 = shl nuw nsw i32 %414, 23
  %416 = add nuw nsw i32 %415, 1065353216
  %417 = trunc i64 %413 to i32
  %418 = and i32 %417, 8388607
  %419 = or disjoint i32 %416, %418
  %420 = bitcast i32 %419 to float
  br label %421

421:                                              ; preds = %412, %375
  %422 = phi float [ %420, %412 ], [ 0.000000e+00, %375 ]
  %423 = bitcast float %422 to i32
  %424 = uitofp i64 %377 to float
  %425 = bitcast float %424 to i32
  %426 = icmp eq i32 %423, %425
  br i1 %426, label %433, label %427

427:                                              ; preds = %421
  %428 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %377)
  %429 = fpext float %422 to double
  %430 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %429, i32 noundef %423)
  %431 = fpext float %424 to double
  %432 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %431, i32 noundef %425)
  br label %433

433:                                              ; preds = %421, %427
  %434 = add i64 %318, %144
  %435 = icmp eq i64 %434, 0
  br i1 %435, label %478, label %436

436:                                              ; preds = %433
  %437 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %434, i1 true)
  %438 = trunc nuw nsw i64 %437 to i32
  %439 = sub nuw nsw i32 64, %438
  %440 = xor i32 %438, 63
  %441 = icmp ugt i64 %434, 16777215
  br i1 %441, label %442, label %465

442:                                              ; preds = %436
  switch i32 %438, label %445 [
    i32 39, label %443
    i32 38, label %454
  ]

443:                                              ; preds = %442
  %444 = shl i64 %434, 1
  br label %454

445:                                              ; preds = %442
  %446 = sub nsw i64 38, %437
  %447 = and i64 %446, 4294967295
  %448 = lshr i64 %434, %447
  %449 = lshr i64 274877906943, %437
  %450 = and i64 %449, %434
  %451 = icmp ne i64 %450, 0
  %452 = zext i1 %451 to i64
  %453 = or i64 %448, %452
  br label %454

454:                                              ; preds = %445, %443, %442
  %455 = phi i64 [ %453, %445 ], [ %444, %443 ], [ %434, %442 ]
  %456 = lshr i64 %455, 2
  %457 = and i64 %456, 1
  %458 = or i64 %457, %455
  %459 = add i64 %458, 1
  %460 = and i64 %459, 67108864
  %461 = icmp eq i64 %460, 0
  %462 = select i1 %461, i64 2, i64 3
  %463 = lshr i64 %459, %462
  %464 = select i1 %461, i32 %440, i32 %439
  br label %469

465:                                              ; preds = %436
  %466 = add nuw nsw i64 %437, 4294967256
  %467 = and i64 %466, 4294967295
  %468 = shl i64 %434, %467
  br label %469

469:                                              ; preds = %465, %454
  %470 = phi i64 [ %468, %465 ], [ %463, %454 ]
  %471 = phi i32 [ %440, %465 ], [ %464, %454 ]
  %472 = shl nuw nsw i32 %471, 23
  %473 = add nuw nsw i32 %472, 1065353216
  %474 = trunc i64 %470 to i32
  %475 = and i32 %474, 8388607
  %476 = or disjoint i32 %473, %475
  %477 = bitcast i32 %476 to float
  br label %478

478:                                              ; preds = %469, %433
  %479 = phi float [ %477, %469 ], [ 0.000000e+00, %433 ]
  %480 = bitcast float %479 to i32
  %481 = uitofp i64 %434 to float
  %482 = bitcast float %481 to i32
  %483 = icmp eq i32 %480, %482
  br i1 %483, label %490, label %484

484:                                              ; preds = %478
  %485 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %434)
  %486 = fpext float %479 to double
  %487 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %486, i32 noundef %480)
  %488 = fpext float %481 to double
  %489 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %488, i32 noundef %482)
  br label %490

490:                                              ; preds = %478, %484
  %491 = add i64 %376, %144
  %492 = icmp eq i64 %491, 0
  br i1 %492, label %535, label %493

493:                                              ; preds = %490
  %494 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %491, i1 true)
  %495 = trunc nuw nsw i64 %494 to i32
  %496 = sub nuw nsw i32 64, %495
  %497 = xor i32 %495, 63
  %498 = icmp ugt i64 %491, 16777215
  br i1 %498, label %499, label %522

499:                                              ; preds = %493
  switch i32 %495, label %502 [
    i32 39, label %500
    i32 38, label %511
  ]

500:                                              ; preds = %499
  %501 = shl i64 %491, 1
  br label %511

502:                                              ; preds = %499
  %503 = sub nsw i64 38, %494
  %504 = and i64 %503, 4294967295
  %505 = lshr i64 %491, %504
  %506 = lshr i64 274877906943, %494
  %507 = and i64 %506, %491
  %508 = icmp ne i64 %507, 0
  %509 = zext i1 %508 to i64
  %510 = or i64 %505, %509
  br label %511

511:                                              ; preds = %502, %500, %499
  %512 = phi i64 [ %510, %502 ], [ %501, %500 ], [ %491, %499 ]
  %513 = lshr i64 %512, 2
  %514 = and i64 %513, 1
  %515 = or i64 %514, %512
  %516 = add i64 %515, 1
  %517 = and i64 %516, 67108864
  %518 = icmp eq i64 %517, 0
  %519 = select i1 %518, i64 2, i64 3
  %520 = lshr i64 %516, %519
  %521 = select i1 %518, i32 %497, i32 %496
  br label %526

522:                                              ; preds = %493
  %523 = add nuw nsw i64 %494, 4294967256
  %524 = and i64 %523, 4294967295
  %525 = shl i64 %491, %524
  br label %526

526:                                              ; preds = %522, %511
  %527 = phi i64 [ %525, %522 ], [ %520, %511 ]
  %528 = phi i32 [ %497, %522 ], [ %521, %511 ]
  %529 = shl nuw nsw i32 %528, 23
  %530 = add nuw nsw i32 %529, 1065353216
  %531 = trunc i64 %527 to i32
  %532 = and i32 %531, 8388607
  %533 = or disjoint i32 %530, %532
  %534 = bitcast i32 %533 to float
  br label %535

535:                                              ; preds = %526, %490
  %536 = phi float [ %534, %526 ], [ 0.000000e+00, %490 ]
  %537 = bitcast float %536 to i32
  %538 = uitofp i64 %491 to float
  %539 = bitcast float %538 to i32
  %540 = icmp eq i32 %537, %539
  br i1 %540, label %547, label %541

541:                                              ; preds = %535
  %542 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %491)
  %543 = fpext float %536 to double
  %544 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %543, i32 noundef %537)
  %545 = fpext float %538 to double
  %546 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %545, i32 noundef %539)
  br label %547

547:                                              ; preds = %535, %541
  %548 = add i64 %318, %201
  %549 = icmp eq i64 %548, 0
  br i1 %549, label %592, label %550

550:                                              ; preds = %547
  %551 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %548, i1 true)
  %552 = trunc nuw nsw i64 %551 to i32
  %553 = sub nuw nsw i32 64, %552
  %554 = xor i32 %552, 63
  %555 = icmp ugt i64 %548, 16777215
  br i1 %555, label %556, label %579

556:                                              ; preds = %550
  switch i32 %552, label %559 [
    i32 39, label %557
    i32 38, label %568
  ]

557:                                              ; preds = %556
  %558 = shl i64 %548, 1
  br label %568

559:                                              ; preds = %556
  %560 = sub nsw i64 38, %551
  %561 = and i64 %560, 4294967295
  %562 = lshr i64 %548, %561
  %563 = lshr i64 274877906943, %551
  %564 = and i64 %563, %548
  %565 = icmp ne i64 %564, 0
  %566 = zext i1 %565 to i64
  %567 = or i64 %562, %566
  br label %568

568:                                              ; preds = %559, %557, %556
  %569 = phi i64 [ %567, %559 ], [ %558, %557 ], [ %548, %556 ]
  %570 = lshr i64 %569, 2
  %571 = and i64 %570, 1
  %572 = or i64 %571, %569
  %573 = add i64 %572, 1
  %574 = and i64 %573, 67108864
  %575 = icmp eq i64 %574, 0
  %576 = select i1 %575, i64 2, i64 3
  %577 = lshr i64 %573, %576
  %578 = select i1 %575, i32 %554, i32 %553
  br label %583

579:                                              ; preds = %550
  %580 = add nuw nsw i64 %551, 4294967256
  %581 = and i64 %580, 4294967295
  %582 = shl i64 %548, %581
  br label %583

583:                                              ; preds = %579, %568
  %584 = phi i64 [ %582, %579 ], [ %577, %568 ]
  %585 = phi i32 [ %554, %579 ], [ %578, %568 ]
  %586 = shl nuw nsw i32 %585, 23
  %587 = add nuw nsw i32 %586, 1065353216
  %588 = trunc i64 %584 to i32
  %589 = and i32 %588, 8388607
  %590 = or disjoint i32 %587, %589
  %591 = bitcast i32 %590 to float
  br label %592

592:                                              ; preds = %583, %547
  %593 = phi float [ %591, %583 ], [ 0.000000e+00, %547 ]
  %594 = bitcast float %593 to i32
  %595 = uitofp i64 %548 to float
  %596 = bitcast float %595 to i32
  %597 = icmp eq i32 %594, %596
  br i1 %597, label %604, label %598

598:                                              ; preds = %592
  %599 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %548)
  %600 = fpext float %593 to double
  %601 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %600, i32 noundef %594)
  %602 = fpext float %595 to double
  %603 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %602, i32 noundef %596)
  br label %604

604:                                              ; preds = %592, %598
  %605 = add i64 %376, %201
  %606 = icmp eq i64 %605, 0
  br i1 %606, label %649, label %607

607:                                              ; preds = %604
  %608 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %605, i1 true)
  %609 = trunc nuw nsw i64 %608 to i32
  %610 = sub nuw nsw i32 64, %609
  %611 = xor i32 %609, 63
  %612 = icmp ugt i64 %605, 16777215
  br i1 %612, label %613, label %636

613:                                              ; preds = %607
  switch i32 %609, label %616 [
    i32 39, label %614
    i32 38, label %625
  ]

614:                                              ; preds = %613
  %615 = shl i64 %605, 1
  br label %625

616:                                              ; preds = %613
  %617 = sub nsw i64 38, %608
  %618 = and i64 %617, 4294967295
  %619 = lshr i64 %605, %618
  %620 = lshr i64 274877906943, %608
  %621 = and i64 %620, %605
  %622 = icmp ne i64 %621, 0
  %623 = zext i1 %622 to i64
  %624 = or i64 %619, %623
  br label %625

625:                                              ; preds = %616, %614, %613
  %626 = phi i64 [ %624, %616 ], [ %615, %614 ], [ %605, %613 ]
  %627 = lshr i64 %626, 2
  %628 = and i64 %627, 1
  %629 = or i64 %628, %626
  %630 = add i64 %629, 1
  %631 = and i64 %630, 67108864
  %632 = icmp eq i64 %631, 0
  %633 = select i1 %632, i64 2, i64 3
  %634 = lshr i64 %630, %633
  %635 = select i1 %632, i32 %611, i32 %610
  br label %640

636:                                              ; preds = %607
  %637 = add nuw nsw i64 %608, 4294967256
  %638 = and i64 %637, 4294967295
  %639 = shl i64 %605, %638
  br label %640

640:                                              ; preds = %636, %625
  %641 = phi i64 [ %639, %636 ], [ %634, %625 ]
  %642 = phi i32 [ %611, %636 ], [ %635, %625 ]
  %643 = shl nuw nsw i32 %642, 23
  %644 = add nuw nsw i32 %643, 1065353216
  %645 = trunc i64 %641 to i32
  %646 = and i32 %645, 8388607
  %647 = or disjoint i32 %644, %646
  %648 = bitcast i32 %647 to float
  br label %649

649:                                              ; preds = %640, %604
  %650 = phi float [ %648, %640 ], [ 0.000000e+00, %604 ]
  %651 = bitcast float %650 to i32
  %652 = uitofp i64 %605 to float
  %653 = bitcast float %652 to i32
  %654 = icmp eq i32 %651, %653
  br i1 %654, label %661, label %655

655:                                              ; preds = %649
  %656 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %605)
  %657 = fpext float %650 to double
  %658 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %657, i32 noundef %651)
  %659 = fpext float %652 to double
  %660 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %659, i32 noundef %653)
  br label %661

661:                                              ; preds = %649, %655
  %662 = add i64 %318, %258
  %663 = icmp eq i64 %662, 0
  br i1 %663, label %706, label %664

664:                                              ; preds = %661
  %665 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %662, i1 true)
  %666 = trunc nuw nsw i64 %665 to i32
  %667 = sub nuw nsw i32 64, %666
  %668 = xor i32 %666, 63
  %669 = icmp ugt i64 %662, 16777215
  br i1 %669, label %670, label %693

670:                                              ; preds = %664
  switch i32 %666, label %673 [
    i32 39, label %671
    i32 38, label %682
  ]

671:                                              ; preds = %670
  %672 = shl i64 %662, 1
  br label %682

673:                                              ; preds = %670
  %674 = sub nsw i64 38, %665
  %675 = and i64 %674, 4294967295
  %676 = lshr i64 %662, %675
  %677 = lshr i64 274877906943, %665
  %678 = and i64 %677, %662
  %679 = icmp ne i64 %678, 0
  %680 = zext i1 %679 to i64
  %681 = or i64 %676, %680
  br label %682

682:                                              ; preds = %673, %671, %670
  %683 = phi i64 [ %681, %673 ], [ %672, %671 ], [ %662, %670 ]
  %684 = lshr i64 %683, 2
  %685 = and i64 %684, 1
  %686 = or i64 %685, %683
  %687 = add i64 %686, 1
  %688 = and i64 %687, 67108864
  %689 = icmp eq i64 %688, 0
  %690 = select i1 %689, i64 2, i64 3
  %691 = lshr i64 %687, %690
  %692 = select i1 %689, i32 %668, i32 %667
  br label %697

693:                                              ; preds = %664
  %694 = add nuw nsw i64 %665, 4294967256
  %695 = and i64 %694, 4294967295
  %696 = shl i64 %662, %695
  br label %697

697:                                              ; preds = %693, %682
  %698 = phi i64 [ %696, %693 ], [ %691, %682 ]
  %699 = phi i32 [ %668, %693 ], [ %692, %682 ]
  %700 = shl nuw nsw i32 %699, 23
  %701 = add nuw nsw i32 %700, 1065353216
  %702 = trunc i64 %698 to i32
  %703 = and i32 %702, 8388607
  %704 = or disjoint i32 %701, %703
  %705 = bitcast i32 %704 to float
  br label %706

706:                                              ; preds = %697, %661
  %707 = phi float [ %705, %697 ], [ 0.000000e+00, %661 ]
  %708 = bitcast float %707 to i32
  %709 = uitofp i64 %662 to float
  %710 = bitcast float %709 to i32
  %711 = icmp eq i32 %708, %710
  br i1 %711, label %718, label %712

712:                                              ; preds = %706
  %713 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %662)
  %714 = fpext float %707 to double
  %715 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %714, i32 noundef %708)
  %716 = fpext float %709 to double
  %717 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %716, i32 noundef %710)
  br label %718

718:                                              ; preds = %706, %712
  %719 = add i64 %376, %258
  %720 = icmp eq i64 %719, 0
  br i1 %720, label %763, label %721

721:                                              ; preds = %718
  %722 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %719, i1 true)
  %723 = trunc nuw nsw i64 %722 to i32
  %724 = sub nuw nsw i32 64, %723
  %725 = xor i32 %723, 63
  %726 = icmp ugt i64 %719, 16777215
  br i1 %726, label %727, label %750

727:                                              ; preds = %721
  switch i32 %723, label %730 [
    i32 39, label %728
    i32 38, label %739
  ]

728:                                              ; preds = %727
  %729 = shl i64 %719, 1
  br label %739

730:                                              ; preds = %727
  %731 = sub nsw i64 38, %722
  %732 = and i64 %731, 4294967295
  %733 = lshr i64 %719, %732
  %734 = lshr i64 274877906943, %722
  %735 = and i64 %734, %719
  %736 = icmp ne i64 %735, 0
  %737 = zext i1 %736 to i64
  %738 = or i64 %733, %737
  br label %739

739:                                              ; preds = %730, %728, %727
  %740 = phi i64 [ %738, %730 ], [ %729, %728 ], [ %719, %727 ]
  %741 = lshr i64 %740, 2
  %742 = and i64 %741, 1
  %743 = or i64 %742, %740
  %744 = add i64 %743, 1
  %745 = and i64 %744, 67108864
  %746 = icmp eq i64 %745, 0
  %747 = select i1 %746, i64 2, i64 3
  %748 = lshr i64 %744, %747
  %749 = select i1 %746, i32 %725, i32 %724
  br label %754

750:                                              ; preds = %721
  %751 = add nuw nsw i64 %722, 4294967256
  %752 = and i64 %751, 4294967295
  %753 = shl i64 %719, %752
  br label %754

754:                                              ; preds = %750, %739
  %755 = phi i64 [ %753, %750 ], [ %748, %739 ]
  %756 = phi i32 [ %725, %750 ], [ %749, %739 ]
  %757 = shl nuw nsw i32 %756, 23
  %758 = add nuw nsw i32 %757, 1065353216
  %759 = trunc i64 %755 to i32
  %760 = and i32 %759, 8388607
  %761 = or disjoint i32 %758, %760
  %762 = bitcast i32 %761 to float
  br label %763

763:                                              ; preds = %754, %718
  %764 = phi float [ %762, %754 ], [ 0.000000e+00, %718 ]
  %765 = bitcast float %764 to i32
  %766 = uitofp i64 %719 to float
  %767 = bitcast float %766 to i32
  %768 = icmp eq i32 %765, %767
  br i1 %768, label %775, label %769

769:                                              ; preds = %763
  %770 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %719)
  %771 = fpext float %764 to double
  %772 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %771, i32 noundef %765)
  %773 = fpext float %766 to double
  %774 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %773, i32 noundef %767)
  br label %775

775:                                              ; preds = %763, %769
  %776 = icmp eq i64 %317, 0
  br i1 %776, label %1695, label %777

777:                                              ; preds = %775, %1692
  %778 = phi i64 [ %1693, %1692 ], [ 0, %775 ]
  %779 = shl nuw i64 1, %778
  %780 = add i64 %779, %319
  %781 = icmp eq i64 %780, 0
  br i1 %781, label %824, label %782

782:                                              ; preds = %777
  %783 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %780, i1 true)
  %784 = trunc nuw nsw i64 %783 to i32
  %785 = sub nuw nsw i32 64, %784
  %786 = xor i32 %784, 63
  %787 = icmp ugt i64 %780, 16777215
  br i1 %787, label %788, label %811

788:                                              ; preds = %782
  switch i32 %784, label %791 [
    i32 39, label %789
    i32 38, label %800
  ]

789:                                              ; preds = %788
  %790 = shl i64 %780, 1
  br label %800

791:                                              ; preds = %788
  %792 = sub nsw i64 38, %783
  %793 = and i64 %792, 4294967295
  %794 = lshr i64 %780, %793
  %795 = lshr i64 274877906943, %783
  %796 = and i64 %795, %780
  %797 = icmp ne i64 %796, 0
  %798 = zext i1 %797 to i64
  %799 = or i64 %794, %798
  br label %800

800:                                              ; preds = %791, %789, %788
  %801 = phi i64 [ %799, %791 ], [ %790, %789 ], [ %780, %788 ]
  %802 = lshr i64 %801, 2
  %803 = and i64 %802, 1
  %804 = or i64 %803, %801
  %805 = add i64 %804, 1
  %806 = and i64 %805, 67108864
  %807 = icmp eq i64 %806, 0
  %808 = select i1 %807, i64 2, i64 3
  %809 = lshr i64 %805, %808
  %810 = select i1 %807, i32 %786, i32 %785
  br label %815

811:                                              ; preds = %782
  %812 = add nuw nsw i64 %783, 4294967256
  %813 = and i64 %812, 4294967295
  %814 = shl i64 %780, %813
  br label %815

815:                                              ; preds = %811, %800
  %816 = phi i64 [ %814, %811 ], [ %809, %800 ]
  %817 = phi i32 [ %786, %811 ], [ %810, %800 ]
  %818 = shl nuw nsw i32 %817, 23
  %819 = add nuw nsw i32 %818, 1065353216
  %820 = trunc i64 %816 to i32
  %821 = and i32 %820, 8388607
  %822 = or disjoint i32 %819, %821
  %823 = bitcast i32 %822 to float
  br label %824

824:                                              ; preds = %815, %777
  %825 = phi float [ %823, %815 ], [ 0.000000e+00, %777 ]
  %826 = bitcast float %825 to i32
  %827 = uitofp i64 %780 to float
  %828 = bitcast float %827 to i32
  %829 = icmp eq i32 %826, %828
  br i1 %829, label %836, label %830

830:                                              ; preds = %824
  %831 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %780)
  %832 = fpext float %825 to double
  %833 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %832, i32 noundef %826)
  %834 = fpext float %827 to double
  %835 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %834, i32 noundef %828)
  br label %836

836:                                              ; preds = %824, %830
  %837 = shl nsw i64 -1, %778
  %838 = add i64 %837, %319
  %839 = icmp eq i64 %838, 0
  br i1 %839, label %882, label %840

840:                                              ; preds = %836
  %841 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %838, i1 true)
  %842 = trunc nuw nsw i64 %841 to i32
  %843 = sub nuw nsw i32 64, %842
  %844 = xor i32 %842, 63
  %845 = icmp ugt i64 %838, 16777215
  br i1 %845, label %846, label %869

846:                                              ; preds = %840
  switch i32 %842, label %849 [
    i32 39, label %847
    i32 38, label %858
  ]

847:                                              ; preds = %846
  %848 = shl i64 %838, 1
  br label %858

849:                                              ; preds = %846
  %850 = sub nsw i64 38, %841
  %851 = and i64 %850, 4294967295
  %852 = lshr i64 %838, %851
  %853 = lshr i64 274877906943, %841
  %854 = and i64 %853, %838
  %855 = icmp ne i64 %854, 0
  %856 = zext i1 %855 to i64
  %857 = or i64 %852, %856
  br label %858

858:                                              ; preds = %849, %847, %846
  %859 = phi i64 [ %857, %849 ], [ %848, %847 ], [ %838, %846 ]
  %860 = lshr i64 %859, 2
  %861 = and i64 %860, 1
  %862 = or i64 %861, %859
  %863 = add i64 %862, 1
  %864 = and i64 %863, 67108864
  %865 = icmp eq i64 %864, 0
  %866 = select i1 %865, i64 2, i64 3
  %867 = lshr i64 %863, %866
  %868 = select i1 %865, i32 %844, i32 %843
  br label %873

869:                                              ; preds = %840
  %870 = add nuw nsw i64 %841, 4294967256
  %871 = and i64 %870, 4294967295
  %872 = shl i64 %838, %871
  br label %873

873:                                              ; preds = %869, %858
  %874 = phi i64 [ %872, %869 ], [ %867, %858 ]
  %875 = phi i32 [ %844, %869 ], [ %868, %858 ]
  %876 = shl nuw nsw i32 %875, 23
  %877 = add nuw nsw i32 %876, 1065353216
  %878 = trunc i64 %874 to i32
  %879 = and i32 %878, 8388607
  %880 = or disjoint i32 %877, %879
  %881 = bitcast i32 %880 to float
  br label %882

882:                                              ; preds = %873, %836
  %883 = phi float [ %881, %873 ], [ 0.000000e+00, %836 ]
  %884 = bitcast float %883 to i32
  %885 = uitofp i64 %838 to float
  %886 = bitcast float %885 to i32
  %887 = icmp eq i32 %884, %886
  br i1 %887, label %894, label %888

888:                                              ; preds = %882
  %889 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %838)
  %890 = fpext float %883 to double
  %891 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %890, i32 noundef %884)
  %892 = fpext float %885 to double
  %893 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %892, i32 noundef %886)
  br label %894

894:                                              ; preds = %882, %888
  %895 = add i64 %779, %377
  %896 = icmp eq i64 %895, 0
  br i1 %896, label %939, label %897

897:                                              ; preds = %894
  %898 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %895, i1 true)
  %899 = trunc nuw nsw i64 %898 to i32
  %900 = sub nuw nsw i32 64, %899
  %901 = xor i32 %899, 63
  %902 = icmp ugt i64 %895, 16777215
  br i1 %902, label %903, label %926

903:                                              ; preds = %897
  switch i32 %899, label %906 [
    i32 39, label %904
    i32 38, label %915
  ]

904:                                              ; preds = %903
  %905 = shl i64 %895, 1
  br label %915

906:                                              ; preds = %903
  %907 = sub nsw i64 38, %898
  %908 = and i64 %907, 4294967295
  %909 = lshr i64 %895, %908
  %910 = lshr i64 274877906943, %898
  %911 = and i64 %910, %895
  %912 = icmp ne i64 %911, 0
  %913 = zext i1 %912 to i64
  %914 = or i64 %909, %913
  br label %915

915:                                              ; preds = %906, %904, %903
  %916 = phi i64 [ %914, %906 ], [ %905, %904 ], [ %895, %903 ]
  %917 = lshr i64 %916, 2
  %918 = and i64 %917, 1
  %919 = or i64 %918, %916
  %920 = add i64 %919, 1
  %921 = and i64 %920, 67108864
  %922 = icmp eq i64 %921, 0
  %923 = select i1 %922, i64 2, i64 3
  %924 = lshr i64 %920, %923
  %925 = select i1 %922, i32 %901, i32 %900
  br label %930

926:                                              ; preds = %897
  %927 = add nuw nsw i64 %898, 4294967256
  %928 = and i64 %927, 4294967295
  %929 = shl i64 %895, %928
  br label %930

930:                                              ; preds = %926, %915
  %931 = phi i64 [ %929, %926 ], [ %924, %915 ]
  %932 = phi i32 [ %901, %926 ], [ %925, %915 ]
  %933 = shl nuw nsw i32 %932, 23
  %934 = add nuw nsw i32 %933, 1065353216
  %935 = trunc i64 %931 to i32
  %936 = and i32 %935, 8388607
  %937 = or disjoint i32 %934, %936
  %938 = bitcast i32 %937 to float
  br label %939

939:                                              ; preds = %930, %894
  %940 = phi float [ %938, %930 ], [ 0.000000e+00, %894 ]
  %941 = bitcast float %940 to i32
  %942 = uitofp i64 %895 to float
  %943 = bitcast float %942 to i32
  %944 = icmp eq i32 %941, %943
  br i1 %944, label %951, label %945

945:                                              ; preds = %939
  %946 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %895)
  %947 = fpext float %940 to double
  %948 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %947, i32 noundef %941)
  %949 = fpext float %942 to double
  %950 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %949, i32 noundef %943)
  br label %951

951:                                              ; preds = %939, %945
  %952 = add i64 %837, %377
  %953 = icmp eq i64 %952, 0
  br i1 %953, label %996, label %954

954:                                              ; preds = %951
  %955 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %952, i1 true)
  %956 = trunc nuw nsw i64 %955 to i32
  %957 = sub nuw nsw i32 64, %956
  %958 = xor i32 %956, 63
  %959 = icmp ugt i64 %952, 16777215
  br i1 %959, label %960, label %983

960:                                              ; preds = %954
  switch i32 %956, label %963 [
    i32 39, label %961
    i32 38, label %972
  ]

961:                                              ; preds = %960
  %962 = shl i64 %952, 1
  br label %972

963:                                              ; preds = %960
  %964 = sub nsw i64 38, %955
  %965 = and i64 %964, 4294967295
  %966 = lshr i64 %952, %965
  %967 = lshr i64 274877906943, %955
  %968 = and i64 %967, %952
  %969 = icmp ne i64 %968, 0
  %970 = zext i1 %969 to i64
  %971 = or i64 %966, %970
  br label %972

972:                                              ; preds = %963, %961, %960
  %973 = phi i64 [ %971, %963 ], [ %962, %961 ], [ %952, %960 ]
  %974 = lshr i64 %973, 2
  %975 = and i64 %974, 1
  %976 = or i64 %975, %973
  %977 = add i64 %976, 1
  %978 = and i64 %977, 67108864
  %979 = icmp eq i64 %978, 0
  %980 = select i1 %979, i64 2, i64 3
  %981 = lshr i64 %977, %980
  %982 = select i1 %979, i32 %958, i32 %957
  br label %987

983:                                              ; preds = %954
  %984 = add nuw nsw i64 %955, 4294967256
  %985 = and i64 %984, 4294967295
  %986 = shl i64 %952, %985
  br label %987

987:                                              ; preds = %983, %972
  %988 = phi i64 [ %986, %983 ], [ %981, %972 ]
  %989 = phi i32 [ %958, %983 ], [ %982, %972 ]
  %990 = shl nuw nsw i32 %989, 23
  %991 = add nuw nsw i32 %990, 1065353216
  %992 = trunc i64 %988 to i32
  %993 = and i32 %992, 8388607
  %994 = or disjoint i32 %991, %993
  %995 = bitcast i32 %994 to float
  br label %996

996:                                              ; preds = %987, %951
  %997 = phi float [ %995, %987 ], [ 0.000000e+00, %951 ]
  %998 = bitcast float %997 to i32
  %999 = uitofp i64 %952 to float
  %1000 = bitcast float %999 to i32
  %1001 = icmp eq i32 %998, %1000
  br i1 %1001, label %1008, label %1002

1002:                                             ; preds = %996
  %1003 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %952)
  %1004 = fpext float %997 to double
  %1005 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1004, i32 noundef %998)
  %1006 = fpext float %999 to double
  %1007 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1006, i32 noundef %1000)
  br label %1008

1008:                                             ; preds = %996, %1002
  %1009 = add i64 %779, %434
  %1010 = icmp eq i64 %1009, 0
  br i1 %1010, label %1053, label %1011

1011:                                             ; preds = %1008
  %1012 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %1009, i1 true)
  %1013 = trunc nuw nsw i64 %1012 to i32
  %1014 = sub nuw nsw i32 64, %1013
  %1015 = xor i32 %1013, 63
  %1016 = icmp ugt i64 %1009, 16777215
  br i1 %1016, label %1017, label %1040

1017:                                             ; preds = %1011
  switch i32 %1013, label %1020 [
    i32 39, label %1018
    i32 38, label %1029
  ]

1018:                                             ; preds = %1017
  %1019 = shl i64 %1009, 1
  br label %1029

1020:                                             ; preds = %1017
  %1021 = sub nsw i64 38, %1012
  %1022 = and i64 %1021, 4294967295
  %1023 = lshr i64 %1009, %1022
  %1024 = lshr i64 274877906943, %1012
  %1025 = and i64 %1024, %1009
  %1026 = icmp ne i64 %1025, 0
  %1027 = zext i1 %1026 to i64
  %1028 = or i64 %1023, %1027
  br label %1029

1029:                                             ; preds = %1020, %1018, %1017
  %1030 = phi i64 [ %1028, %1020 ], [ %1019, %1018 ], [ %1009, %1017 ]
  %1031 = lshr i64 %1030, 2
  %1032 = and i64 %1031, 1
  %1033 = or i64 %1032, %1030
  %1034 = add i64 %1033, 1
  %1035 = and i64 %1034, 67108864
  %1036 = icmp eq i64 %1035, 0
  %1037 = select i1 %1036, i64 2, i64 3
  %1038 = lshr i64 %1034, %1037
  %1039 = select i1 %1036, i32 %1015, i32 %1014
  br label %1044

1040:                                             ; preds = %1011
  %1041 = add nuw nsw i64 %1012, 4294967256
  %1042 = and i64 %1041, 4294967295
  %1043 = shl i64 %1009, %1042
  br label %1044

1044:                                             ; preds = %1040, %1029
  %1045 = phi i64 [ %1043, %1040 ], [ %1038, %1029 ]
  %1046 = phi i32 [ %1015, %1040 ], [ %1039, %1029 ]
  %1047 = shl nuw nsw i32 %1046, 23
  %1048 = add nuw nsw i32 %1047, 1065353216
  %1049 = trunc i64 %1045 to i32
  %1050 = and i32 %1049, 8388607
  %1051 = or disjoint i32 %1048, %1050
  %1052 = bitcast i32 %1051 to float
  br label %1053

1053:                                             ; preds = %1044, %1008
  %1054 = phi float [ %1052, %1044 ], [ 0.000000e+00, %1008 ]
  %1055 = bitcast float %1054 to i32
  %1056 = uitofp i64 %1009 to float
  %1057 = bitcast float %1056 to i32
  %1058 = icmp eq i32 %1055, %1057
  br i1 %1058, label %1065, label %1059

1059:                                             ; preds = %1053
  %1060 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1009)
  %1061 = fpext float %1054 to double
  %1062 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1061, i32 noundef %1055)
  %1063 = fpext float %1056 to double
  %1064 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1063, i32 noundef %1057)
  br label %1065

1065:                                             ; preds = %1053, %1059
  %1066 = add i64 %837, %434
  %1067 = icmp eq i64 %1066, 0
  br i1 %1067, label %1110, label %1068

1068:                                             ; preds = %1065
  %1069 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %1066, i1 true)
  %1070 = trunc nuw nsw i64 %1069 to i32
  %1071 = sub nuw nsw i32 64, %1070
  %1072 = xor i32 %1070, 63
  %1073 = icmp ugt i64 %1066, 16777215
  br i1 %1073, label %1074, label %1097

1074:                                             ; preds = %1068
  switch i32 %1070, label %1077 [
    i32 39, label %1075
    i32 38, label %1086
  ]

1075:                                             ; preds = %1074
  %1076 = shl i64 %1066, 1
  br label %1086

1077:                                             ; preds = %1074
  %1078 = sub nsw i64 38, %1069
  %1079 = and i64 %1078, 4294967295
  %1080 = lshr i64 %1066, %1079
  %1081 = lshr i64 274877906943, %1069
  %1082 = and i64 %1081, %1066
  %1083 = icmp ne i64 %1082, 0
  %1084 = zext i1 %1083 to i64
  %1085 = or i64 %1080, %1084
  br label %1086

1086:                                             ; preds = %1077, %1075, %1074
  %1087 = phi i64 [ %1085, %1077 ], [ %1076, %1075 ], [ %1066, %1074 ]
  %1088 = lshr i64 %1087, 2
  %1089 = and i64 %1088, 1
  %1090 = or i64 %1089, %1087
  %1091 = add i64 %1090, 1
  %1092 = and i64 %1091, 67108864
  %1093 = icmp eq i64 %1092, 0
  %1094 = select i1 %1093, i64 2, i64 3
  %1095 = lshr i64 %1091, %1094
  %1096 = select i1 %1093, i32 %1072, i32 %1071
  br label %1101

1097:                                             ; preds = %1068
  %1098 = add nuw nsw i64 %1069, 4294967256
  %1099 = and i64 %1098, 4294967295
  %1100 = shl i64 %1066, %1099
  br label %1101

1101:                                             ; preds = %1097, %1086
  %1102 = phi i64 [ %1100, %1097 ], [ %1095, %1086 ]
  %1103 = phi i32 [ %1072, %1097 ], [ %1096, %1086 ]
  %1104 = shl nuw nsw i32 %1103, 23
  %1105 = add nuw nsw i32 %1104, 1065353216
  %1106 = trunc i64 %1102 to i32
  %1107 = and i32 %1106, 8388607
  %1108 = or disjoint i32 %1105, %1107
  %1109 = bitcast i32 %1108 to float
  br label %1110

1110:                                             ; preds = %1101, %1065
  %1111 = phi float [ %1109, %1101 ], [ 0.000000e+00, %1065 ]
  %1112 = bitcast float %1111 to i32
  %1113 = uitofp i64 %1066 to float
  %1114 = bitcast float %1113 to i32
  %1115 = icmp eq i32 %1112, %1114
  br i1 %1115, label %1122, label %1116

1116:                                             ; preds = %1110
  %1117 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1066)
  %1118 = fpext float %1111 to double
  %1119 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1118, i32 noundef %1112)
  %1120 = fpext float %1113 to double
  %1121 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1120, i32 noundef %1114)
  br label %1122

1122:                                             ; preds = %1110, %1116
  %1123 = add i64 %779, %491
  %1124 = icmp eq i64 %1123, 0
  br i1 %1124, label %1167, label %1125

1125:                                             ; preds = %1122
  %1126 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %1123, i1 true)
  %1127 = trunc nuw nsw i64 %1126 to i32
  %1128 = sub nuw nsw i32 64, %1127
  %1129 = xor i32 %1127, 63
  %1130 = icmp ugt i64 %1123, 16777215
  br i1 %1130, label %1131, label %1154

1131:                                             ; preds = %1125
  switch i32 %1127, label %1134 [
    i32 39, label %1132
    i32 38, label %1143
  ]

1132:                                             ; preds = %1131
  %1133 = shl i64 %1123, 1
  br label %1143

1134:                                             ; preds = %1131
  %1135 = sub nsw i64 38, %1126
  %1136 = and i64 %1135, 4294967295
  %1137 = lshr i64 %1123, %1136
  %1138 = lshr i64 274877906943, %1126
  %1139 = and i64 %1138, %1123
  %1140 = icmp ne i64 %1139, 0
  %1141 = zext i1 %1140 to i64
  %1142 = or i64 %1137, %1141
  br label %1143

1143:                                             ; preds = %1134, %1132, %1131
  %1144 = phi i64 [ %1142, %1134 ], [ %1133, %1132 ], [ %1123, %1131 ]
  %1145 = lshr i64 %1144, 2
  %1146 = and i64 %1145, 1
  %1147 = or i64 %1146, %1144
  %1148 = add i64 %1147, 1
  %1149 = and i64 %1148, 67108864
  %1150 = icmp eq i64 %1149, 0
  %1151 = select i1 %1150, i64 2, i64 3
  %1152 = lshr i64 %1148, %1151
  %1153 = select i1 %1150, i32 %1129, i32 %1128
  br label %1158

1154:                                             ; preds = %1125
  %1155 = add nuw nsw i64 %1126, 4294967256
  %1156 = and i64 %1155, 4294967295
  %1157 = shl i64 %1123, %1156
  br label %1158

1158:                                             ; preds = %1154, %1143
  %1159 = phi i64 [ %1157, %1154 ], [ %1152, %1143 ]
  %1160 = phi i32 [ %1129, %1154 ], [ %1153, %1143 ]
  %1161 = shl nuw nsw i32 %1160, 23
  %1162 = add nuw nsw i32 %1161, 1065353216
  %1163 = trunc i64 %1159 to i32
  %1164 = and i32 %1163, 8388607
  %1165 = or disjoint i32 %1162, %1164
  %1166 = bitcast i32 %1165 to float
  br label %1167

1167:                                             ; preds = %1158, %1122
  %1168 = phi float [ %1166, %1158 ], [ 0.000000e+00, %1122 ]
  %1169 = bitcast float %1168 to i32
  %1170 = uitofp i64 %1123 to float
  %1171 = bitcast float %1170 to i32
  %1172 = icmp eq i32 %1169, %1171
  br i1 %1172, label %1179, label %1173

1173:                                             ; preds = %1167
  %1174 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1123)
  %1175 = fpext float %1168 to double
  %1176 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1175, i32 noundef %1169)
  %1177 = fpext float %1170 to double
  %1178 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1177, i32 noundef %1171)
  br label %1179

1179:                                             ; preds = %1167, %1173
  %1180 = add i64 %837, %491
  %1181 = icmp eq i64 %1180, 0
  br i1 %1181, label %1224, label %1182

1182:                                             ; preds = %1179
  %1183 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %1180, i1 true)
  %1184 = trunc nuw nsw i64 %1183 to i32
  %1185 = sub nuw nsw i32 64, %1184
  %1186 = xor i32 %1184, 63
  %1187 = icmp ugt i64 %1180, 16777215
  br i1 %1187, label %1188, label %1211

1188:                                             ; preds = %1182
  switch i32 %1184, label %1191 [
    i32 39, label %1189
    i32 38, label %1200
  ]

1189:                                             ; preds = %1188
  %1190 = shl i64 %1180, 1
  br label %1200

1191:                                             ; preds = %1188
  %1192 = sub nsw i64 38, %1183
  %1193 = and i64 %1192, 4294967295
  %1194 = lshr i64 %1180, %1193
  %1195 = lshr i64 274877906943, %1183
  %1196 = and i64 %1195, %1180
  %1197 = icmp ne i64 %1196, 0
  %1198 = zext i1 %1197 to i64
  %1199 = or i64 %1194, %1198
  br label %1200

1200:                                             ; preds = %1191, %1189, %1188
  %1201 = phi i64 [ %1199, %1191 ], [ %1190, %1189 ], [ %1180, %1188 ]
  %1202 = lshr i64 %1201, 2
  %1203 = and i64 %1202, 1
  %1204 = or i64 %1203, %1201
  %1205 = add i64 %1204, 1
  %1206 = and i64 %1205, 67108864
  %1207 = icmp eq i64 %1206, 0
  %1208 = select i1 %1207, i64 2, i64 3
  %1209 = lshr i64 %1205, %1208
  %1210 = select i1 %1207, i32 %1186, i32 %1185
  br label %1215

1211:                                             ; preds = %1182
  %1212 = add nuw nsw i64 %1183, 4294967256
  %1213 = and i64 %1212, 4294967295
  %1214 = shl i64 %1180, %1213
  br label %1215

1215:                                             ; preds = %1211, %1200
  %1216 = phi i64 [ %1214, %1211 ], [ %1209, %1200 ]
  %1217 = phi i32 [ %1186, %1211 ], [ %1210, %1200 ]
  %1218 = shl nuw nsw i32 %1217, 23
  %1219 = add nuw nsw i32 %1218, 1065353216
  %1220 = trunc i64 %1216 to i32
  %1221 = and i32 %1220, 8388607
  %1222 = or disjoint i32 %1219, %1221
  %1223 = bitcast i32 %1222 to float
  br label %1224

1224:                                             ; preds = %1215, %1179
  %1225 = phi float [ %1223, %1215 ], [ 0.000000e+00, %1179 ]
  %1226 = bitcast float %1225 to i32
  %1227 = uitofp i64 %1180 to float
  %1228 = bitcast float %1227 to i32
  %1229 = icmp eq i32 %1226, %1228
  br i1 %1229, label %1236, label %1230

1230:                                             ; preds = %1224
  %1231 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1180)
  %1232 = fpext float %1225 to double
  %1233 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1232, i32 noundef %1226)
  %1234 = fpext float %1227 to double
  %1235 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1234, i32 noundef %1228)
  br label %1236

1236:                                             ; preds = %1224, %1230
  %1237 = add i64 %779, %548
  %1238 = icmp eq i64 %1237, 0
  br i1 %1238, label %1281, label %1239

1239:                                             ; preds = %1236
  %1240 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %1237, i1 true)
  %1241 = trunc nuw nsw i64 %1240 to i32
  %1242 = sub nuw nsw i32 64, %1241
  %1243 = xor i32 %1241, 63
  %1244 = icmp ugt i64 %1237, 16777215
  br i1 %1244, label %1245, label %1268

1245:                                             ; preds = %1239
  switch i32 %1241, label %1248 [
    i32 39, label %1246
    i32 38, label %1257
  ]

1246:                                             ; preds = %1245
  %1247 = shl i64 %1237, 1
  br label %1257

1248:                                             ; preds = %1245
  %1249 = sub nsw i64 38, %1240
  %1250 = and i64 %1249, 4294967295
  %1251 = lshr i64 %1237, %1250
  %1252 = lshr i64 274877906943, %1240
  %1253 = and i64 %1252, %1237
  %1254 = icmp ne i64 %1253, 0
  %1255 = zext i1 %1254 to i64
  %1256 = or i64 %1251, %1255
  br label %1257

1257:                                             ; preds = %1248, %1246, %1245
  %1258 = phi i64 [ %1256, %1248 ], [ %1247, %1246 ], [ %1237, %1245 ]
  %1259 = lshr i64 %1258, 2
  %1260 = and i64 %1259, 1
  %1261 = or i64 %1260, %1258
  %1262 = add i64 %1261, 1
  %1263 = and i64 %1262, 67108864
  %1264 = icmp eq i64 %1263, 0
  %1265 = select i1 %1264, i64 2, i64 3
  %1266 = lshr i64 %1262, %1265
  %1267 = select i1 %1264, i32 %1243, i32 %1242
  br label %1272

1268:                                             ; preds = %1239
  %1269 = add nuw nsw i64 %1240, 4294967256
  %1270 = and i64 %1269, 4294967295
  %1271 = shl i64 %1237, %1270
  br label %1272

1272:                                             ; preds = %1268, %1257
  %1273 = phi i64 [ %1271, %1268 ], [ %1266, %1257 ]
  %1274 = phi i32 [ %1243, %1268 ], [ %1267, %1257 ]
  %1275 = shl nuw nsw i32 %1274, 23
  %1276 = add nuw nsw i32 %1275, 1065353216
  %1277 = trunc i64 %1273 to i32
  %1278 = and i32 %1277, 8388607
  %1279 = or disjoint i32 %1276, %1278
  %1280 = bitcast i32 %1279 to float
  br label %1281

1281:                                             ; preds = %1272, %1236
  %1282 = phi float [ %1280, %1272 ], [ 0.000000e+00, %1236 ]
  %1283 = bitcast float %1282 to i32
  %1284 = uitofp i64 %1237 to float
  %1285 = bitcast float %1284 to i32
  %1286 = icmp eq i32 %1283, %1285
  br i1 %1286, label %1293, label %1287

1287:                                             ; preds = %1281
  %1288 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1237)
  %1289 = fpext float %1282 to double
  %1290 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1289, i32 noundef %1283)
  %1291 = fpext float %1284 to double
  %1292 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1291, i32 noundef %1285)
  br label %1293

1293:                                             ; preds = %1281, %1287
  %1294 = add i64 %837, %548
  %1295 = icmp eq i64 %1294, 0
  br i1 %1295, label %1338, label %1296

1296:                                             ; preds = %1293
  %1297 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %1294, i1 true)
  %1298 = trunc nuw nsw i64 %1297 to i32
  %1299 = sub nuw nsw i32 64, %1298
  %1300 = xor i32 %1298, 63
  %1301 = icmp ugt i64 %1294, 16777215
  br i1 %1301, label %1302, label %1325

1302:                                             ; preds = %1296
  switch i32 %1298, label %1305 [
    i32 39, label %1303
    i32 38, label %1314
  ]

1303:                                             ; preds = %1302
  %1304 = shl i64 %1294, 1
  br label %1314

1305:                                             ; preds = %1302
  %1306 = sub nsw i64 38, %1297
  %1307 = and i64 %1306, 4294967295
  %1308 = lshr i64 %1294, %1307
  %1309 = lshr i64 274877906943, %1297
  %1310 = and i64 %1309, %1294
  %1311 = icmp ne i64 %1310, 0
  %1312 = zext i1 %1311 to i64
  %1313 = or i64 %1308, %1312
  br label %1314

1314:                                             ; preds = %1305, %1303, %1302
  %1315 = phi i64 [ %1313, %1305 ], [ %1304, %1303 ], [ %1294, %1302 ]
  %1316 = lshr i64 %1315, 2
  %1317 = and i64 %1316, 1
  %1318 = or i64 %1317, %1315
  %1319 = add i64 %1318, 1
  %1320 = and i64 %1319, 67108864
  %1321 = icmp eq i64 %1320, 0
  %1322 = select i1 %1321, i64 2, i64 3
  %1323 = lshr i64 %1319, %1322
  %1324 = select i1 %1321, i32 %1300, i32 %1299
  br label %1329

1325:                                             ; preds = %1296
  %1326 = add nuw nsw i64 %1297, 4294967256
  %1327 = and i64 %1326, 4294967295
  %1328 = shl i64 %1294, %1327
  br label %1329

1329:                                             ; preds = %1325, %1314
  %1330 = phi i64 [ %1328, %1325 ], [ %1323, %1314 ]
  %1331 = phi i32 [ %1300, %1325 ], [ %1324, %1314 ]
  %1332 = shl nuw nsw i32 %1331, 23
  %1333 = add nuw nsw i32 %1332, 1065353216
  %1334 = trunc i64 %1330 to i32
  %1335 = and i32 %1334, 8388607
  %1336 = or disjoint i32 %1333, %1335
  %1337 = bitcast i32 %1336 to float
  br label %1338

1338:                                             ; preds = %1329, %1293
  %1339 = phi float [ %1337, %1329 ], [ 0.000000e+00, %1293 ]
  %1340 = bitcast float %1339 to i32
  %1341 = uitofp i64 %1294 to float
  %1342 = bitcast float %1341 to i32
  %1343 = icmp eq i32 %1340, %1342
  br i1 %1343, label %1350, label %1344

1344:                                             ; preds = %1338
  %1345 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1294)
  %1346 = fpext float %1339 to double
  %1347 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1346, i32 noundef %1340)
  %1348 = fpext float %1341 to double
  %1349 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1348, i32 noundef %1342)
  br label %1350

1350:                                             ; preds = %1338, %1344
  %1351 = add i64 %779, %605
  %1352 = icmp eq i64 %1351, 0
  br i1 %1352, label %1395, label %1353

1353:                                             ; preds = %1350
  %1354 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %1351, i1 true)
  %1355 = trunc nuw nsw i64 %1354 to i32
  %1356 = sub nuw nsw i32 64, %1355
  %1357 = xor i32 %1355, 63
  %1358 = icmp ugt i64 %1351, 16777215
  br i1 %1358, label %1359, label %1382

1359:                                             ; preds = %1353
  switch i32 %1355, label %1362 [
    i32 39, label %1360
    i32 38, label %1371
  ]

1360:                                             ; preds = %1359
  %1361 = shl i64 %1351, 1
  br label %1371

1362:                                             ; preds = %1359
  %1363 = sub nsw i64 38, %1354
  %1364 = and i64 %1363, 4294967295
  %1365 = lshr i64 %1351, %1364
  %1366 = lshr i64 274877906943, %1354
  %1367 = and i64 %1366, %1351
  %1368 = icmp ne i64 %1367, 0
  %1369 = zext i1 %1368 to i64
  %1370 = or i64 %1365, %1369
  br label %1371

1371:                                             ; preds = %1362, %1360, %1359
  %1372 = phi i64 [ %1370, %1362 ], [ %1361, %1360 ], [ %1351, %1359 ]
  %1373 = lshr i64 %1372, 2
  %1374 = and i64 %1373, 1
  %1375 = or i64 %1374, %1372
  %1376 = add i64 %1375, 1
  %1377 = and i64 %1376, 67108864
  %1378 = icmp eq i64 %1377, 0
  %1379 = select i1 %1378, i64 2, i64 3
  %1380 = lshr i64 %1376, %1379
  %1381 = select i1 %1378, i32 %1357, i32 %1356
  br label %1386

1382:                                             ; preds = %1353
  %1383 = add nuw nsw i64 %1354, 4294967256
  %1384 = and i64 %1383, 4294967295
  %1385 = shl i64 %1351, %1384
  br label %1386

1386:                                             ; preds = %1382, %1371
  %1387 = phi i64 [ %1385, %1382 ], [ %1380, %1371 ]
  %1388 = phi i32 [ %1357, %1382 ], [ %1381, %1371 ]
  %1389 = shl nuw nsw i32 %1388, 23
  %1390 = add nuw nsw i32 %1389, 1065353216
  %1391 = trunc i64 %1387 to i32
  %1392 = and i32 %1391, 8388607
  %1393 = or disjoint i32 %1390, %1392
  %1394 = bitcast i32 %1393 to float
  br label %1395

1395:                                             ; preds = %1386, %1350
  %1396 = phi float [ %1394, %1386 ], [ 0.000000e+00, %1350 ]
  %1397 = bitcast float %1396 to i32
  %1398 = uitofp i64 %1351 to float
  %1399 = bitcast float %1398 to i32
  %1400 = icmp eq i32 %1397, %1399
  br i1 %1400, label %1407, label %1401

1401:                                             ; preds = %1395
  %1402 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1351)
  %1403 = fpext float %1396 to double
  %1404 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1403, i32 noundef %1397)
  %1405 = fpext float %1398 to double
  %1406 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1405, i32 noundef %1399)
  br label %1407

1407:                                             ; preds = %1395, %1401
  %1408 = add i64 %837, %605
  %1409 = icmp eq i64 %1408, 0
  br i1 %1409, label %1452, label %1410

1410:                                             ; preds = %1407
  %1411 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %1408, i1 true)
  %1412 = trunc nuw nsw i64 %1411 to i32
  %1413 = sub nuw nsw i32 64, %1412
  %1414 = xor i32 %1412, 63
  %1415 = icmp ugt i64 %1408, 16777215
  br i1 %1415, label %1416, label %1439

1416:                                             ; preds = %1410
  switch i32 %1412, label %1419 [
    i32 39, label %1417
    i32 38, label %1428
  ]

1417:                                             ; preds = %1416
  %1418 = shl i64 %1408, 1
  br label %1428

1419:                                             ; preds = %1416
  %1420 = sub nsw i64 38, %1411
  %1421 = and i64 %1420, 4294967295
  %1422 = lshr i64 %1408, %1421
  %1423 = lshr i64 274877906943, %1411
  %1424 = and i64 %1423, %1408
  %1425 = icmp ne i64 %1424, 0
  %1426 = zext i1 %1425 to i64
  %1427 = or i64 %1422, %1426
  br label %1428

1428:                                             ; preds = %1419, %1417, %1416
  %1429 = phi i64 [ %1427, %1419 ], [ %1418, %1417 ], [ %1408, %1416 ]
  %1430 = lshr i64 %1429, 2
  %1431 = and i64 %1430, 1
  %1432 = or i64 %1431, %1429
  %1433 = add i64 %1432, 1
  %1434 = and i64 %1433, 67108864
  %1435 = icmp eq i64 %1434, 0
  %1436 = select i1 %1435, i64 2, i64 3
  %1437 = lshr i64 %1433, %1436
  %1438 = select i1 %1435, i32 %1414, i32 %1413
  br label %1443

1439:                                             ; preds = %1410
  %1440 = add nuw nsw i64 %1411, 4294967256
  %1441 = and i64 %1440, 4294967295
  %1442 = shl i64 %1408, %1441
  br label %1443

1443:                                             ; preds = %1439, %1428
  %1444 = phi i64 [ %1442, %1439 ], [ %1437, %1428 ]
  %1445 = phi i32 [ %1414, %1439 ], [ %1438, %1428 ]
  %1446 = shl nuw nsw i32 %1445, 23
  %1447 = add nuw nsw i32 %1446, 1065353216
  %1448 = trunc i64 %1444 to i32
  %1449 = and i32 %1448, 8388607
  %1450 = or disjoint i32 %1447, %1449
  %1451 = bitcast i32 %1450 to float
  br label %1452

1452:                                             ; preds = %1443, %1407
  %1453 = phi float [ %1451, %1443 ], [ 0.000000e+00, %1407 ]
  %1454 = bitcast float %1453 to i32
  %1455 = uitofp i64 %1408 to float
  %1456 = bitcast float %1455 to i32
  %1457 = icmp eq i32 %1454, %1456
  br i1 %1457, label %1464, label %1458

1458:                                             ; preds = %1452
  %1459 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1408)
  %1460 = fpext float %1453 to double
  %1461 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1460, i32 noundef %1454)
  %1462 = fpext float %1455 to double
  %1463 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1462, i32 noundef %1456)
  br label %1464

1464:                                             ; preds = %1452, %1458
  %1465 = add i64 %779, %662
  %1466 = icmp eq i64 %1465, 0
  br i1 %1466, label %1509, label %1467

1467:                                             ; preds = %1464
  %1468 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %1465, i1 true)
  %1469 = trunc nuw nsw i64 %1468 to i32
  %1470 = sub nuw nsw i32 64, %1469
  %1471 = xor i32 %1469, 63
  %1472 = icmp ugt i64 %1465, 16777215
  br i1 %1472, label %1473, label %1496

1473:                                             ; preds = %1467
  switch i32 %1469, label %1476 [
    i32 39, label %1474
    i32 38, label %1485
  ]

1474:                                             ; preds = %1473
  %1475 = shl i64 %1465, 1
  br label %1485

1476:                                             ; preds = %1473
  %1477 = sub nsw i64 38, %1468
  %1478 = and i64 %1477, 4294967295
  %1479 = lshr i64 %1465, %1478
  %1480 = lshr i64 274877906943, %1468
  %1481 = and i64 %1480, %1465
  %1482 = icmp ne i64 %1481, 0
  %1483 = zext i1 %1482 to i64
  %1484 = or i64 %1479, %1483
  br label %1485

1485:                                             ; preds = %1476, %1474, %1473
  %1486 = phi i64 [ %1484, %1476 ], [ %1475, %1474 ], [ %1465, %1473 ]
  %1487 = lshr i64 %1486, 2
  %1488 = and i64 %1487, 1
  %1489 = or i64 %1488, %1486
  %1490 = add i64 %1489, 1
  %1491 = and i64 %1490, 67108864
  %1492 = icmp eq i64 %1491, 0
  %1493 = select i1 %1492, i64 2, i64 3
  %1494 = lshr i64 %1490, %1493
  %1495 = select i1 %1492, i32 %1471, i32 %1470
  br label %1500

1496:                                             ; preds = %1467
  %1497 = add nuw nsw i64 %1468, 4294967256
  %1498 = and i64 %1497, 4294967295
  %1499 = shl i64 %1465, %1498
  br label %1500

1500:                                             ; preds = %1496, %1485
  %1501 = phi i64 [ %1499, %1496 ], [ %1494, %1485 ]
  %1502 = phi i32 [ %1471, %1496 ], [ %1495, %1485 ]
  %1503 = shl nuw nsw i32 %1502, 23
  %1504 = add nuw nsw i32 %1503, 1065353216
  %1505 = trunc i64 %1501 to i32
  %1506 = and i32 %1505, 8388607
  %1507 = or disjoint i32 %1504, %1506
  %1508 = bitcast i32 %1507 to float
  br label %1509

1509:                                             ; preds = %1500, %1464
  %1510 = phi float [ %1508, %1500 ], [ 0.000000e+00, %1464 ]
  %1511 = bitcast float %1510 to i32
  %1512 = uitofp i64 %1465 to float
  %1513 = bitcast float %1512 to i32
  %1514 = icmp eq i32 %1511, %1513
  br i1 %1514, label %1521, label %1515

1515:                                             ; preds = %1509
  %1516 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1465)
  %1517 = fpext float %1510 to double
  %1518 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1517, i32 noundef %1511)
  %1519 = fpext float %1512 to double
  %1520 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1519, i32 noundef %1513)
  br label %1521

1521:                                             ; preds = %1509, %1515
  %1522 = add i64 %837, %662
  %1523 = icmp eq i64 %1522, 0
  br i1 %1523, label %1566, label %1524

1524:                                             ; preds = %1521
  %1525 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %1522, i1 true)
  %1526 = trunc nuw nsw i64 %1525 to i32
  %1527 = sub nuw nsw i32 64, %1526
  %1528 = xor i32 %1526, 63
  %1529 = icmp ugt i64 %1522, 16777215
  br i1 %1529, label %1530, label %1553

1530:                                             ; preds = %1524
  switch i32 %1526, label %1533 [
    i32 39, label %1531
    i32 38, label %1542
  ]

1531:                                             ; preds = %1530
  %1532 = shl i64 %1522, 1
  br label %1542

1533:                                             ; preds = %1530
  %1534 = sub nsw i64 38, %1525
  %1535 = and i64 %1534, 4294967295
  %1536 = lshr i64 %1522, %1535
  %1537 = lshr i64 274877906943, %1525
  %1538 = and i64 %1537, %1522
  %1539 = icmp ne i64 %1538, 0
  %1540 = zext i1 %1539 to i64
  %1541 = or i64 %1536, %1540
  br label %1542

1542:                                             ; preds = %1533, %1531, %1530
  %1543 = phi i64 [ %1541, %1533 ], [ %1532, %1531 ], [ %1522, %1530 ]
  %1544 = lshr i64 %1543, 2
  %1545 = and i64 %1544, 1
  %1546 = or i64 %1545, %1543
  %1547 = add i64 %1546, 1
  %1548 = and i64 %1547, 67108864
  %1549 = icmp eq i64 %1548, 0
  %1550 = select i1 %1549, i64 2, i64 3
  %1551 = lshr i64 %1547, %1550
  %1552 = select i1 %1549, i32 %1528, i32 %1527
  br label %1557

1553:                                             ; preds = %1524
  %1554 = add nuw nsw i64 %1525, 4294967256
  %1555 = and i64 %1554, 4294967295
  %1556 = shl i64 %1522, %1555
  br label %1557

1557:                                             ; preds = %1553, %1542
  %1558 = phi i64 [ %1556, %1553 ], [ %1551, %1542 ]
  %1559 = phi i32 [ %1528, %1553 ], [ %1552, %1542 ]
  %1560 = shl nuw nsw i32 %1559, 23
  %1561 = add nuw nsw i32 %1560, 1065353216
  %1562 = trunc i64 %1558 to i32
  %1563 = and i32 %1562, 8388607
  %1564 = or disjoint i32 %1561, %1563
  %1565 = bitcast i32 %1564 to float
  br label %1566

1566:                                             ; preds = %1557, %1521
  %1567 = phi float [ %1565, %1557 ], [ 0.000000e+00, %1521 ]
  %1568 = bitcast float %1567 to i32
  %1569 = uitofp i64 %1522 to float
  %1570 = bitcast float %1569 to i32
  %1571 = icmp eq i32 %1568, %1570
  br i1 %1571, label %1578, label %1572

1572:                                             ; preds = %1566
  %1573 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1522)
  %1574 = fpext float %1567 to double
  %1575 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1574, i32 noundef %1568)
  %1576 = fpext float %1569 to double
  %1577 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1576, i32 noundef %1570)
  br label %1578

1578:                                             ; preds = %1566, %1572
  %1579 = add i64 %779, %719
  %1580 = icmp eq i64 %1579, 0
  br i1 %1580, label %1623, label %1581

1581:                                             ; preds = %1578
  %1582 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %1579, i1 true)
  %1583 = trunc nuw nsw i64 %1582 to i32
  %1584 = sub nuw nsw i32 64, %1583
  %1585 = xor i32 %1583, 63
  %1586 = icmp ugt i64 %1579, 16777215
  br i1 %1586, label %1587, label %1610

1587:                                             ; preds = %1581
  switch i32 %1583, label %1590 [
    i32 39, label %1588
    i32 38, label %1599
  ]

1588:                                             ; preds = %1587
  %1589 = shl i64 %1579, 1
  br label %1599

1590:                                             ; preds = %1587
  %1591 = sub nsw i64 38, %1582
  %1592 = and i64 %1591, 4294967295
  %1593 = lshr i64 %1579, %1592
  %1594 = lshr i64 274877906943, %1582
  %1595 = and i64 %1594, %1579
  %1596 = icmp ne i64 %1595, 0
  %1597 = zext i1 %1596 to i64
  %1598 = or i64 %1593, %1597
  br label %1599

1599:                                             ; preds = %1590, %1588, %1587
  %1600 = phi i64 [ %1598, %1590 ], [ %1589, %1588 ], [ %1579, %1587 ]
  %1601 = lshr i64 %1600, 2
  %1602 = and i64 %1601, 1
  %1603 = or i64 %1602, %1600
  %1604 = add i64 %1603, 1
  %1605 = and i64 %1604, 67108864
  %1606 = icmp eq i64 %1605, 0
  %1607 = select i1 %1606, i64 2, i64 3
  %1608 = lshr i64 %1604, %1607
  %1609 = select i1 %1606, i32 %1585, i32 %1584
  br label %1614

1610:                                             ; preds = %1581
  %1611 = add nuw nsw i64 %1582, 4294967256
  %1612 = and i64 %1611, 4294967295
  %1613 = shl i64 %1579, %1612
  br label %1614

1614:                                             ; preds = %1610, %1599
  %1615 = phi i64 [ %1613, %1610 ], [ %1608, %1599 ]
  %1616 = phi i32 [ %1585, %1610 ], [ %1609, %1599 ]
  %1617 = shl nuw nsw i32 %1616, 23
  %1618 = add nuw nsw i32 %1617, 1065353216
  %1619 = trunc i64 %1615 to i32
  %1620 = and i32 %1619, 8388607
  %1621 = or disjoint i32 %1618, %1620
  %1622 = bitcast i32 %1621 to float
  br label %1623

1623:                                             ; preds = %1614, %1578
  %1624 = phi float [ %1622, %1614 ], [ 0.000000e+00, %1578 ]
  %1625 = bitcast float %1624 to i32
  %1626 = uitofp i64 %1579 to float
  %1627 = bitcast float %1626 to i32
  %1628 = icmp eq i32 %1625, %1627
  br i1 %1628, label %1635, label %1629

1629:                                             ; preds = %1623
  %1630 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1579)
  %1631 = fpext float %1624 to double
  %1632 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1631, i32 noundef %1625)
  %1633 = fpext float %1626 to double
  %1634 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1633, i32 noundef %1627)
  br label %1635

1635:                                             ; preds = %1623, %1629
  %1636 = add i64 %837, %719
  %1637 = icmp eq i64 %1636, 0
  br i1 %1637, label %1680, label %1638

1638:                                             ; preds = %1635
  %1639 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %1636, i1 true)
  %1640 = trunc nuw nsw i64 %1639 to i32
  %1641 = sub nuw nsw i32 64, %1640
  %1642 = xor i32 %1640, 63
  %1643 = icmp ugt i64 %1636, 16777215
  br i1 %1643, label %1644, label %1667

1644:                                             ; preds = %1638
  switch i32 %1640, label %1647 [
    i32 39, label %1645
    i32 38, label %1656
  ]

1645:                                             ; preds = %1644
  %1646 = shl i64 %1636, 1
  br label %1656

1647:                                             ; preds = %1644
  %1648 = sub nsw i64 38, %1639
  %1649 = and i64 %1648, 4294967295
  %1650 = lshr i64 %1636, %1649
  %1651 = lshr i64 274877906943, %1639
  %1652 = and i64 %1651, %1636
  %1653 = icmp ne i64 %1652, 0
  %1654 = zext i1 %1653 to i64
  %1655 = or i64 %1650, %1654
  br label %1656

1656:                                             ; preds = %1647, %1645, %1644
  %1657 = phi i64 [ %1655, %1647 ], [ %1646, %1645 ], [ %1636, %1644 ]
  %1658 = lshr i64 %1657, 2
  %1659 = and i64 %1658, 1
  %1660 = or i64 %1659, %1657
  %1661 = add i64 %1660, 1
  %1662 = and i64 %1661, 67108864
  %1663 = icmp eq i64 %1662, 0
  %1664 = select i1 %1663, i64 2, i64 3
  %1665 = lshr i64 %1661, %1664
  %1666 = select i1 %1663, i32 %1642, i32 %1641
  br label %1671

1667:                                             ; preds = %1638
  %1668 = add nuw nsw i64 %1639, 4294967256
  %1669 = and i64 %1668, 4294967295
  %1670 = shl i64 %1636, %1669
  br label %1671

1671:                                             ; preds = %1667, %1656
  %1672 = phi i64 [ %1670, %1667 ], [ %1665, %1656 ]
  %1673 = phi i32 [ %1642, %1667 ], [ %1666, %1656 ]
  %1674 = shl nuw nsw i32 %1673, 23
  %1675 = add nuw nsw i32 %1674, 1065353216
  %1676 = trunc i64 %1672 to i32
  %1677 = and i32 %1676, 8388607
  %1678 = or disjoint i32 %1675, %1677
  %1679 = bitcast i32 %1678 to float
  br label %1680

1680:                                             ; preds = %1671, %1635
  %1681 = phi float [ %1679, %1671 ], [ 0.000000e+00, %1635 ]
  %1682 = bitcast float %1681 to i32
  %1683 = uitofp i64 %1636 to float
  %1684 = bitcast float %1683 to i32
  %1685 = icmp eq i32 %1682, %1684
  br i1 %1685, label %1692, label %1686

1686:                                             ; preds = %1680
  %1687 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1636)
  %1688 = fpext float %1681 to double
  %1689 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %1688, i32 noundef %1682)
  %1690 = fpext float %1683 to double
  %1691 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %1690, i32 noundef %1684)
  br label %1692

1692:                                             ; preds = %1680, %1686
  %1693 = add nuw nsw i64 %778, 1
  %1694 = icmp eq i64 %1693, %317
  br i1 %1694, label %1695, label %777, !llvm.loop !6

1695:                                             ; preds = %1692, %775
  %1696 = add nuw nsw i64 %317, 1
  %1697 = icmp eq i64 %1696, %84
  br i1 %1697, label %1698, label %316, !llvm.loop !8

1698:                                             ; preds = %1695, %314
  %1699 = add nuw nsw i64 %84, 1
  %1700 = icmp eq i64 %1699, %10
  br i1 %1700, label %1701, label %83, !llvm.loop !9

1701:                                             ; preds = %1698, %81
  %1702 = add nuw nsw i64 %10, 1
  %1703 = icmp eq i64 %1702, 64
  br i1 %1703, label %1704, label %9, !llvm.loop !10

1704:                                             ; preds = %1701
  %1705 = add nuw nsw i64 %4, 1
  %1706 = icmp eq i64 %1705, 4
  br i1 %1706, label %1707, label %3, !llvm.loop !11

1707:                                             ; preds = %1704
  %1708 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @fesetround(i32 noundef) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare ptr @llvm.load.relative.i64(ptr, i64) #7

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nounwind }
attributes #7 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #8 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
