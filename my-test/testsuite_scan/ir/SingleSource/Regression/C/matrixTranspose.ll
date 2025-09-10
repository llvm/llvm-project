; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/matrixTranspose.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/matrixTranspose.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@A = dso_local local_unnamed_addr global [2048 x float] zeroinitializer, align 4
@.str = private unnamed_addr constant [23 x i8] c"Checksum before = %lf\0A\00", align 1
@.str.1 = private unnamed_addr constant [23 x i8] c"Checksum  after = %lf\0A\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @complex_transpose(ptr noundef captures(none) %0, ptr noundef captures(none) %1, i32 noundef %2, i32 noundef %3, i32 noundef %4) local_unnamed_addr #0 {
  %6 = icmp sgt i32 %2, 1
  br i1 %6, label %7, label %73

7:                                                ; preds = %5
  %8 = sext i32 %4 to i64
  %9 = sext i32 %3 to i64
  %10 = zext nneg i32 %2 to i64
  %11 = icmp ne i32 %4, 1
  %12 = icmp ne i32 %3, 1
  %13 = or i1 %11, %12
  br label %14

14:                                               ; preds = %7, %69
  %15 = phi i64 [ 0, %7 ], [ %72, %69 ]
  %16 = phi i64 [ 1, %7 ], [ %70, %69 ]
  %17 = shl nuw nsw i64 %15, 2
  %18 = add nuw i64 %17, 4
  %19 = getelementptr i8, ptr %0, i64 %18
  %20 = shl nuw nsw i64 %15, 3
  %21 = add nuw i64 %20, 8
  %22 = getelementptr i8, ptr %0, i64 %21
  %23 = getelementptr i8, ptr %1, i64 %18
  %24 = getelementptr i8, ptr %1, i64 %21
  %25 = mul nsw i64 %16, %9
  %26 = mul nsw i64 %16, %8
  %27 = icmp samesign ult i64 %16, 4
  %28 = or i1 %27, %13
  br i1 %28, label %51, label %29

29:                                               ; preds = %14
  %30 = icmp ult ptr %19, %24
  %31 = icmp ult ptr %23, %22
  %32 = and i1 %30, %31
  br i1 %32, label %51, label %33

33:                                               ; preds = %29
  %34 = and i64 %16, 9223372036854775804
  br label %35

35:                                               ; preds = %35, %33
  %36 = phi i64 [ 0, %33 ], [ %47, %35 ]
  %37 = add nsw i64 %36, %25
  %38 = getelementptr inbounds float, ptr %0, i64 %37
  %39 = load <4 x float>, ptr %38, align 4, !tbaa !6, !alias.scope !10, !noalias !13
  %40 = getelementptr inbounds float, ptr %1, i64 %37
  %41 = load <4 x float>, ptr %40, align 4, !tbaa !6, !alias.scope !13
  %42 = add nsw i64 %36, %26
  %43 = getelementptr inbounds float, ptr %0, i64 %42
  %44 = load <4 x float>, ptr %43, align 4, !tbaa !6, !alias.scope !10, !noalias !13
  %45 = getelementptr inbounds float, ptr %1, i64 %42
  %46 = load <4 x float>, ptr %45, align 4, !tbaa !6, !alias.scope !13
  store <4 x float> %39, ptr %43, align 4, !tbaa !6, !alias.scope !10, !noalias !13
  store <4 x float> %41, ptr %45, align 4, !tbaa !6, !alias.scope !13
  store <4 x float> %44, ptr %38, align 4, !tbaa !6, !alias.scope !10, !noalias !13
  store <4 x float> %46, ptr %40, align 4, !tbaa !6, !alias.scope !13
  %47 = add nuw i64 %36, 4
  %48 = icmp eq i64 %47, %34
  br i1 %48, label %49, label %35, !llvm.loop !15

49:                                               ; preds = %35
  %50 = icmp eq i64 %16, %34
  br i1 %50, label %69, label %51

51:                                               ; preds = %14, %29, %49
  %52 = phi i64 [ 0, %29 ], [ 0, %14 ], [ %34, %49 ]
  br label %53

53:                                               ; preds = %51, %53
  %54 = phi i64 [ %67, %53 ], [ %52, %51 ]
  %55 = mul nsw i64 %54, %8
  %56 = add nsw i64 %55, %25
  %57 = getelementptr inbounds float, ptr %0, i64 %56
  %58 = load float, ptr %57, align 4, !tbaa !6
  %59 = getelementptr inbounds float, ptr %1, i64 %56
  %60 = load float, ptr %59, align 4, !tbaa !6
  %61 = mul nsw i64 %54, %9
  %62 = add nsw i64 %61, %26
  %63 = getelementptr inbounds float, ptr %0, i64 %62
  %64 = load float, ptr %63, align 4, !tbaa !6
  %65 = getelementptr inbounds float, ptr %1, i64 %62
  %66 = load float, ptr %65, align 4, !tbaa !6
  store float %58, ptr %63, align 4, !tbaa !6
  store float %60, ptr %65, align 4, !tbaa !6
  store float %64, ptr %57, align 4, !tbaa !6
  store float %66, ptr %59, align 4, !tbaa !6
  %67 = add nuw nsw i64 %54, 1
  %68 = icmp eq i64 %67, %16
  br i1 %68, label %69, label %53, !llvm.loop !19

69:                                               ; preds = %53, %49
  %70 = add nuw nsw i64 %16, 1
  %71 = icmp eq i64 %70, %10
  %72 = add i64 %15, 1
  br i1 %71, label %73, label %14, !llvm.loop !20

73:                                               ; preds = %69, %5
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #1 {
  br label %3

3:                                                ; preds = %3, %2
  %4 = phi i64 [ 0, %2 ], [ %14, %3 ]
  %5 = phi float [ 0.000000e+00, %2 ], [ %13, %3 ]
  %6 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %2 ], [ %15, %3 ]
  %7 = add <4 x i32> %6, splat (i32 4)
  %8 = uitofp nneg <4 x i32> %6 to <4 x float>
  %9 = uitofp nneg <4 x i32> %7 to <4 x float>
  %10 = getelementptr inbounds nuw float, ptr @A, i64 %4
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 16
  store <4 x float> %8, ptr %10, align 4, !tbaa !6
  store <4 x float> %9, ptr %11, align 4, !tbaa !6
  %12 = tail call float @llvm.vector.reduce.fadd.v4f32(float %5, <4 x float> %8)
  %13 = tail call float @llvm.vector.reduce.fadd.v4f32(float %12, <4 x float> %9)
  %14 = add nuw i64 %4, 8
  %15 = add <4 x i32> %6, splat (i32 8)
  %16 = icmp eq i64 %14, 2048
  br i1 %16, label %17, label %3, !llvm.loop !21

17:                                               ; preds = %3
  %18 = fpext float %13 to double
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %18)
  br label %20

20:                                               ; preds = %17, %36
  %21 = phi i64 [ %37, %36 ], [ 1, %17 ]
  %22 = shl nuw nsw i64 %21, 3
  %23 = getelementptr inbounds i8, ptr @A, i64 %22
  %24 = shl nsw i64 %21, 8
  %25 = getelementptr inbounds i8, ptr @A, i64 %24
  br label %26

26:                                               ; preds = %26, %20
  %27 = phi i64 [ 0, %20 ], [ %34, %26 ]
  %28 = shl nsw i64 %27, 8
  %29 = getelementptr inbounds i8, ptr %23, i64 %28
  %30 = shl nuw nsw i64 %27, 3
  %31 = getelementptr inbounds i8, ptr %25, i64 %30
  %32 = load <2 x float>, ptr %31, align 4, !tbaa !6
  %33 = load <2 x float>, ptr %29, align 4, !tbaa !6
  store <2 x float> %33, ptr %31, align 4, !tbaa !6
  store <2 x float> %32, ptr %29, align 4, !tbaa !6
  %34 = add nuw nsw i64 %27, 1
  %35 = icmp eq i64 %34, %21
  br i1 %35, label %36, label %26, !llvm.loop !22

36:                                               ; preds = %26
  %37 = add nuw nsw i64 %21, 1
  %38 = icmp eq i64 %37, 32
  br i1 %38, label %39, label %20, !llvm.loop !20

39:                                               ; preds = %36, %55
  %40 = phi i64 [ %56, %55 ], [ 1, %36 ]
  %41 = shl nuw nsw i64 %40, 3
  %42 = getelementptr inbounds i8, ptr @A, i64 %41
  %43 = shl nsw i64 %40, 8
  %44 = getelementptr inbounds i8, ptr @A, i64 %43
  br label %45

45:                                               ; preds = %45, %39
  %46 = phi i64 [ 0, %39 ], [ %53, %45 ]
  %47 = shl nsw i64 %46, 8
  %48 = getelementptr inbounds i8, ptr %42, i64 %47
  %49 = shl nuw nsw i64 %46, 3
  %50 = getelementptr inbounds i8, ptr %44, i64 %49
  %51 = load <2 x float>, ptr %50, align 4, !tbaa !6
  %52 = load <2 x float>, ptr %48, align 4, !tbaa !6
  store <2 x float> %52, ptr %50, align 4, !tbaa !6
  store <2 x float> %51, ptr %48, align 4, !tbaa !6
  %53 = add nuw nsw i64 %46, 1
  %54 = icmp eq i64 %53, %40
  br i1 %54, label %55, label %45, !llvm.loop !22

55:                                               ; preds = %45
  %56 = add nuw nsw i64 %40, 1
  %57 = icmp eq i64 %56, 32
  br i1 %57, label %58, label %39, !llvm.loop !20

58:                                               ; preds = %55, %74
  %59 = phi i64 [ %75, %74 ], [ 1, %55 ]
  %60 = shl nuw nsw i64 %59, 3
  %61 = getelementptr inbounds i8, ptr @A, i64 %60
  %62 = shl nsw i64 %59, 8
  %63 = getelementptr inbounds i8, ptr @A, i64 %62
  br label %64

64:                                               ; preds = %64, %58
  %65 = phi i64 [ 0, %58 ], [ %72, %64 ]
  %66 = shl nsw i64 %65, 8
  %67 = getelementptr inbounds i8, ptr %61, i64 %66
  %68 = shl nuw nsw i64 %65, 3
  %69 = getelementptr inbounds i8, ptr %63, i64 %68
  %70 = load <2 x float>, ptr %69, align 4, !tbaa !6
  %71 = load <2 x float>, ptr %67, align 4, !tbaa !6
  store <2 x float> %71, ptr %69, align 4, !tbaa !6
  store <2 x float> %70, ptr %67, align 4, !tbaa !6
  %72 = add nuw nsw i64 %65, 1
  %73 = icmp eq i64 %72, %59
  br i1 %73, label %74, label %64, !llvm.loop !22

74:                                               ; preds = %64
  %75 = add nuw nsw i64 %59, 1
  %76 = icmp eq i64 %75, 32
  br i1 %76, label %77, label %58, !llvm.loop !20

77:                                               ; preds = %74, %93
  %78 = phi i64 [ %94, %93 ], [ 1, %74 ]
  %79 = shl nuw nsw i64 %78, 3
  %80 = getelementptr inbounds i8, ptr @A, i64 %79
  %81 = shl nsw i64 %78, 8
  %82 = getelementptr inbounds i8, ptr @A, i64 %81
  br label %83

83:                                               ; preds = %83, %77
  %84 = phi i64 [ 0, %77 ], [ %91, %83 ]
  %85 = shl nsw i64 %84, 8
  %86 = getelementptr inbounds i8, ptr %80, i64 %85
  %87 = shl nuw nsw i64 %84, 3
  %88 = getelementptr inbounds i8, ptr %82, i64 %87
  %89 = load <2 x float>, ptr %88, align 4, !tbaa !6
  %90 = load <2 x float>, ptr %86, align 4, !tbaa !6
  store <2 x float> %90, ptr %88, align 4, !tbaa !6
  store <2 x float> %89, ptr %86, align 4, !tbaa !6
  %91 = add nuw nsw i64 %84, 1
  %92 = icmp eq i64 %91, %78
  br i1 %92, label %93, label %83, !llvm.loop !22

93:                                               ; preds = %83
  %94 = add nuw nsw i64 %78, 1
  %95 = icmp eq i64 %94, 32
  br i1 %95, label %96, label %77, !llvm.loop !20

96:                                               ; preds = %93, %112
  %97 = phi i64 [ %113, %112 ], [ 1, %93 ]
  %98 = shl nuw nsw i64 %97, 3
  %99 = getelementptr inbounds i8, ptr @A, i64 %98
  %100 = shl nsw i64 %97, 8
  %101 = getelementptr inbounds i8, ptr @A, i64 %100
  br label %102

102:                                              ; preds = %102, %96
  %103 = phi i64 [ 0, %96 ], [ %110, %102 ]
  %104 = shl nsw i64 %103, 8
  %105 = getelementptr inbounds i8, ptr %99, i64 %104
  %106 = shl nuw nsw i64 %103, 3
  %107 = getelementptr inbounds i8, ptr %101, i64 %106
  %108 = load <2 x float>, ptr %107, align 4, !tbaa !6
  %109 = load <2 x float>, ptr %105, align 4, !tbaa !6
  store <2 x float> %109, ptr %107, align 4, !tbaa !6
  store <2 x float> %108, ptr %105, align 4, !tbaa !6
  %110 = add nuw nsw i64 %103, 1
  %111 = icmp eq i64 %110, %97
  br i1 %111, label %112, label %102, !llvm.loop !22

112:                                              ; preds = %102
  %113 = add nuw nsw i64 %97, 1
  %114 = icmp eq i64 %113, 32
  br i1 %114, label %115, label %96, !llvm.loop !20

115:                                              ; preds = %112, %131
  %116 = phi i64 [ %132, %131 ], [ 1, %112 ]
  %117 = shl nuw nsw i64 %116, 3
  %118 = getelementptr inbounds i8, ptr @A, i64 %117
  %119 = shl nsw i64 %116, 8
  %120 = getelementptr inbounds i8, ptr @A, i64 %119
  br label %121

121:                                              ; preds = %121, %115
  %122 = phi i64 [ 0, %115 ], [ %129, %121 ]
  %123 = shl nsw i64 %122, 8
  %124 = getelementptr inbounds i8, ptr %118, i64 %123
  %125 = shl nuw nsw i64 %122, 3
  %126 = getelementptr inbounds i8, ptr %120, i64 %125
  %127 = load <2 x float>, ptr %126, align 4, !tbaa !6
  %128 = load <2 x float>, ptr %124, align 4, !tbaa !6
  store <2 x float> %128, ptr %126, align 4, !tbaa !6
  store <2 x float> %127, ptr %124, align 4, !tbaa !6
  %129 = add nuw nsw i64 %122, 1
  %130 = icmp eq i64 %129, %116
  br i1 %130, label %131, label %121, !llvm.loop !22

131:                                              ; preds = %121
  %132 = add nuw nsw i64 %116, 1
  %133 = icmp eq i64 %132, 32
  br i1 %133, label %134, label %115, !llvm.loop !20

134:                                              ; preds = %131, %150
  %135 = phi i64 [ %151, %150 ], [ 1, %131 ]
  %136 = shl nuw nsw i64 %135, 3
  %137 = getelementptr inbounds i8, ptr @A, i64 %136
  %138 = shl nsw i64 %135, 8
  %139 = getelementptr inbounds i8, ptr @A, i64 %138
  br label %140

140:                                              ; preds = %140, %134
  %141 = phi i64 [ 0, %134 ], [ %148, %140 ]
  %142 = shl nsw i64 %141, 8
  %143 = getelementptr inbounds i8, ptr %137, i64 %142
  %144 = shl nuw nsw i64 %141, 3
  %145 = getelementptr inbounds i8, ptr %139, i64 %144
  %146 = load <2 x float>, ptr %145, align 4, !tbaa !6
  %147 = load <2 x float>, ptr %143, align 4, !tbaa !6
  store <2 x float> %147, ptr %145, align 4, !tbaa !6
  store <2 x float> %146, ptr %143, align 4, !tbaa !6
  %148 = add nuw nsw i64 %141, 1
  %149 = icmp eq i64 %148, %135
  br i1 %149, label %150, label %140, !llvm.loop !22

150:                                              ; preds = %140
  %151 = add nuw nsw i64 %135, 1
  %152 = icmp eq i64 %151, 32
  br i1 %152, label %153, label %134, !llvm.loop !20

153:                                              ; preds = %150, %169
  %154 = phi i64 [ %170, %169 ], [ 1, %150 ]
  %155 = shl nuw nsw i64 %154, 3
  %156 = getelementptr inbounds i8, ptr @A, i64 %155
  %157 = shl nsw i64 %154, 8
  %158 = getelementptr inbounds i8, ptr @A, i64 %157
  br label %159

159:                                              ; preds = %159, %153
  %160 = phi i64 [ 0, %153 ], [ %167, %159 ]
  %161 = shl nsw i64 %160, 8
  %162 = getelementptr inbounds i8, ptr %156, i64 %161
  %163 = shl nuw nsw i64 %160, 3
  %164 = getelementptr inbounds i8, ptr %158, i64 %163
  %165 = load <2 x float>, ptr %164, align 4, !tbaa !6
  %166 = load <2 x float>, ptr %162, align 4, !tbaa !6
  store <2 x float> %166, ptr %164, align 4, !tbaa !6
  store <2 x float> %165, ptr %162, align 4, !tbaa !6
  %167 = add nuw nsw i64 %160, 1
  %168 = icmp eq i64 %167, %154
  br i1 %168, label %169, label %159, !llvm.loop !22

169:                                              ; preds = %159
  %170 = add nuw nsw i64 %154, 1
  %171 = icmp eq i64 %170, 32
  br i1 %171, label %172, label %153, !llvm.loop !20

172:                                              ; preds = %169, %188
  %173 = phi i64 [ %189, %188 ], [ 1, %169 ]
  %174 = shl nuw nsw i64 %173, 3
  %175 = getelementptr inbounds i8, ptr @A, i64 %174
  %176 = shl nsw i64 %173, 8
  %177 = getelementptr inbounds i8, ptr @A, i64 %176
  br label %178

178:                                              ; preds = %178, %172
  %179 = phi i64 [ 0, %172 ], [ %186, %178 ]
  %180 = shl nsw i64 %179, 8
  %181 = getelementptr inbounds i8, ptr %175, i64 %180
  %182 = shl nuw nsw i64 %179, 3
  %183 = getelementptr inbounds i8, ptr %177, i64 %182
  %184 = load <2 x float>, ptr %183, align 4, !tbaa !6
  %185 = load <2 x float>, ptr %181, align 4, !tbaa !6
  store <2 x float> %185, ptr %183, align 4, !tbaa !6
  store <2 x float> %184, ptr %181, align 4, !tbaa !6
  %186 = add nuw nsw i64 %179, 1
  %187 = icmp eq i64 %186, %173
  br i1 %187, label %188, label %178, !llvm.loop !22

188:                                              ; preds = %178
  %189 = add nuw nsw i64 %173, 1
  %190 = icmp eq i64 %189, 32
  br i1 %190, label %191, label %172, !llvm.loop !20

191:                                              ; preds = %188, %207
  %192 = phi i64 [ %208, %207 ], [ 1, %188 ]
  %193 = shl nuw nsw i64 %192, 3
  %194 = getelementptr inbounds i8, ptr @A, i64 %193
  %195 = shl nsw i64 %192, 8
  %196 = getelementptr inbounds i8, ptr @A, i64 %195
  br label %197

197:                                              ; preds = %197, %191
  %198 = phi i64 [ 0, %191 ], [ %205, %197 ]
  %199 = shl nsw i64 %198, 8
  %200 = getelementptr inbounds i8, ptr %194, i64 %199
  %201 = shl nuw nsw i64 %198, 3
  %202 = getelementptr inbounds i8, ptr %196, i64 %201
  %203 = load <2 x float>, ptr %202, align 4, !tbaa !6
  %204 = load <2 x float>, ptr %200, align 4, !tbaa !6
  store <2 x float> %204, ptr %202, align 4, !tbaa !6
  store <2 x float> %203, ptr %200, align 4, !tbaa !6
  %205 = add nuw nsw i64 %198, 1
  %206 = icmp eq i64 %205, %192
  br i1 %206, label %207, label %197, !llvm.loop !22

207:                                              ; preds = %197
  %208 = add nuw nsw i64 %192, 1
  %209 = icmp eq i64 %208, 32
  br i1 %209, label %210, label %191, !llvm.loop !20

210:                                              ; preds = %207, %210
  %211 = phi i64 [ %219, %210 ], [ 0, %207 ]
  %212 = phi float [ %218, %210 ], [ 0.000000e+00, %207 ]
  %213 = getelementptr inbounds nuw float, ptr @A, i64 %211
  %214 = getelementptr inbounds nuw i8, ptr %213, i64 16
  %215 = load <4 x float>, ptr %213, align 4, !tbaa !6
  %216 = load <4 x float>, ptr %214, align 4, !tbaa !6
  %217 = tail call float @llvm.vector.reduce.fadd.v4f32(float %212, <4 x float> %215)
  %218 = tail call float @llvm.vector.reduce.fadd.v4f32(float %217, <4 x float> %216)
  %219 = add nuw i64 %211, 8
  %220 = icmp eq i64 %219, 2048
  br i1 %220, label %221, label %210, !llvm.loop !23

221:                                              ; preds = %210
  %222 = fpext float %218 to double
  %223 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %222)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>) #3

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"float", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11}
!11 = distinct !{!11, !12}
!12 = distinct !{!12, !"LVerDomain"}
!13 = !{!14}
!14 = distinct !{!14, !12}
!15 = distinct !{!15, !16, !17, !18}
!16 = !{!"llvm.loop.mustprogress"}
!17 = !{!"llvm.loop.isvectorized", i32 1}
!18 = !{!"llvm.loop.unroll.runtime.disable"}
!19 = distinct !{!19, !16, !17}
!20 = distinct !{!20, !16}
!21 = distinct !{!21, !16, !17, !18}
!22 = distinct !{!22, !16}
!23 = distinct !{!23, !16, !17, !18}
