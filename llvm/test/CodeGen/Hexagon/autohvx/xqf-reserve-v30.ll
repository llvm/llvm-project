; RUN: llc -O3 -mtriple=hexagon -mcpu=hexagonv79 \
; RUN:   -mattr=+hvxv79,+hvx-length128b -verify-machineinstrs %s -o /dev/null

; V30 is used by the post-RA QFloat handling pass to hold a vector of zeroes.
; Verify that register allocation reserves it when QFloat instructions are
; present.

define i32 @f(i32 %.idx69, ptr %gep83, ptr %0, <32 x float> %1, ptr %2,
              ptr %3, <32 x float> %4, ptr %5) {
.preheader81:
  br label %6

6:                                                ; preds = %6, %.preheader81
  %7 = phi [8 x <32 x float>] [ zeroinitializer, %.preheader81 ], [ %89, %6 ]
  %8 = phi i32 [ 0, %.preheader81 ], [ %.idx69, %6 ]
  %9 = load <32 x float>, ptr null, align 4
  %.idx68 = shl i32 %8, 8
  %gep = getelementptr i8, ptr null, i32 %.idx68
  %10 = load <32 x float>, ptr %gep, align 4
  %11 = extractvalue [8 x <32 x float>] %7, 4
  %12 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> zeroinitializer, <32 x float> %10, <32 x float> %11)
  %13 = load float, ptr null, align 4
  %14 = load float, ptr null, align 4
  %15 = insertelement <32 x float> zeroinitializer, float %14, i64 0
  %16 = shufflevector <32 x float> %15, <32 x float> zeroinitializer, <32 x i32> zeroinitializer
  %17 = extractvalue [8 x <32 x float>] %7, 6
  %18 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %16, <32 x float> %1, <32 x float> %17)
  %19 = load float, ptr inttoptr (i32 4 to ptr), align 4
  %20 = insertelement <32 x float> zeroinitializer, float %19, i64 0
  %21 = extractvalue [8 x <32 x float>] %7, 0
  %22 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> zeroinitializer, <32 x float> %10, <32 x float> %21)
  %23 = load float, ptr inttoptr (i32 1024 to ptr), align 4
  %24 = insertelement <32 x float> zeroinitializer, float %23, i64 0
  %25 = extractvalue [8 x <32 x float>] %7, 1
  %26 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> zeroinitializer, <32 x float> zeroinitializer, <32 x float> %25)
  %27 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %24, <32 x float> zeroinitializer, <32 x float> %26)
  %28 = load float, ptr inttoptr (i32 4096 to ptr), align 4
  %29 = insertelement <32 x float> zeroinitializer, float %28, i64 0
  %30 = load float, ptr inttoptr (i32 5120 to ptr), align 4
  %31 = insertelement <32 x float> zeroinitializer, float %30, i64 0
  %32 = insertelement <32 x float> zeroinitializer, float %13, i64 0
  %33 = extractvalue [8 x <32 x float>] %7, 5
  %34 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %32, <32 x float> %4, <32 x float> %33)
  %35 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %31, <32 x float> zeroinitializer, <32 x float> %34)
  %36 = load float, ptr inttoptr (i32 6144 to ptr), align 4
  %37 = insertelement <32 x float> zeroinitializer, float %36, i64 0
  %38 = load float, ptr %3, align 4
  %39 = insertelement <32 x float> zeroinitializer, float %38, i64 0
  %40 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %20, <32 x float> zeroinitializer, <32 x float> %22)
  %41 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> zeroinitializer, <32 x float> zeroinitializer, <32 x float> %40)
  %42 = load float, ptr %0, align 4
  %43 = load float, ptr null, align 4
  %44 = load float, ptr %gep83, align 4
  %45 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %29, <32 x float> zeroinitializer, <32 x float> %12)
  %46 = or disjoint i32 %8, 1
  %.idx72 = shl i32 %46, 8
  %gep89 = getelementptr i8, ptr null, i32 %.idx72
  %47 = load <32 x float>, ptr %gep89, align 4
  %48 = load <32 x float>, ptr null, align 4
  %49 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> zeroinitializer, <32 x float> zeroinitializer, <32 x float> %35)
  %50 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> zeroinitializer, <32 x float> %48, <32 x float> %49)
  %51 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %4, <32 x float> %47, <32 x float> %50)
  %52 = load float, ptr %2, align 4
  %53 = insertelement <32 x float> zeroinitializer, float %52, i64 0
  %54 = insertelement <32 x float> zeroinitializer, float %42, i64 0
  %55 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %37, <32 x float> %9, <32 x float> %18)
  %56 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %54, <32 x float> zeroinitializer, <32 x float> %55)
  %57 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %53, <32 x float> zeroinitializer, <32 x float> %56)
  %58 = insertelement <32 x float> zeroinitializer, float %43, i64 0
  %59 = shufflevector <32 x float> %58, <32 x float> zeroinitializer, <32 x i32> zeroinitializer
  %60 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %39, <32 x float> zeroinitializer, <32 x float> %41)
  %61 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %59, <32 x float> splat (float 1.000000e+00), <32 x float> %60)
  %62 = insertelement <32 x float> zeroinitializer, float %44, i64 0
  %63 = shufflevector <32 x float> %62, <32 x float> zeroinitializer, <32 x i32> zeroinitializer
  %64 = load float, ptr null, align 4
  %65 = insertelement <32 x float> zeroinitializer, float %64, i64 0
  %66 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %63, <32 x float> %1, <32 x float> %27)
  %67 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %65, <32 x float> zeroinitializer, <32 x float> %66)
  %68 = or disjoint i32 %8, 5
  %.idx73 = shl i32 %68, 8
  %gep91 = getelementptr i8, ptr null, i32 %.idx73
  %69 = load <32 x float>, ptr %gep91, align 4
  %70 = load float, ptr %gep83, align 4
  %71 = insertelement <32 x float> zeroinitializer, float %70, i64 0
  %72 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> zeroinitializer, <32 x float> zeroinitializer, <32 x float> %45)
  %73 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %1, <32 x float> %48, <32 x float> %72)
  %74 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %71, <32 x float> zeroinitializer, <32 x float> %73)
  %75 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> zeroinitializer, <32 x float> %69, <32 x float> %74)
  %76 = load float, ptr %5, align 4
  %77 = insertelement <32 x float> zeroinitializer, float %76, i64 0
  %78 = shufflevector <32 x float> %77, <32 x float> zeroinitializer, <32 x i32> zeroinitializer
  %79 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> zeroinitializer, <32 x float> zeroinitializer, <32 x float> %61)
  %80 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %78, <32 x float> %1, <32 x float> %79)
  %81 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> zeroinitializer, <32 x float> zeroinitializer, <32 x float> %75)
  %.idx74 = shl i32 %8, 8
  %gep93 = getelementptr i8, ptr null, i32 %.idx74
  %82 = load <32 x float>, ptr %gep93, align 4
  %83 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> zeroinitializer, <32 x float> %82, <32 x float> %51)
  %84 = call <32 x float> @llvm.fmuladd.v32f32(<32 x float> zeroinitializer, <32 x float> zeroinitializer, <32 x float> %67)
  %85 = insertvalue [8 x <32 x float>] zeroinitializer, <32 x float> %80, 0
  %86 = insertvalue [8 x <32 x float>] %85, <32 x float> %84, 1
  %87 = insertvalue [8 x <32 x float>] %86, <32 x float> %81, 4
  %88 = insertvalue [8 x <32 x float>] %87, <32 x float> %83, 5
  %89 = insertvalue [8 x <32 x float>] %88, <32 x float> %57, 6
  br label %6
}

declare <32 x float> @llvm.fmuladd.v32f32(<32 x float>, <32 x float>,
                                          <32 x float>)
