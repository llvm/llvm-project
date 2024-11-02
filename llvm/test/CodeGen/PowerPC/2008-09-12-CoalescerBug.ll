; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s

	%struct.CGLDI = type { ptr, i32, i32, i32, i32, i32, ptr, i32, ptr, ptr, %struct.vv_t }
	%struct.cgli = type { i32, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, ptr, ptr, ptr, ptr, float, float, float, float, i32, ptr, float, ptr, [16 x i32] }
	%struct.CGLSI = type { ptr, i32, ptr, ptr, i32, i32, ptr, ptr, %struct.vv_t, %struct.vv_t, ptr }
	%struct._cgro = type opaque
	%struct.xx_t = type { [3 x %struct.vv_t], [2 x %struct.vv_t], [2 x [3 x ptr]] }
	%struct.vv_t = type { <16 x i8> }
@llvm.used = appending global [1 x ptr] [ ptr @lb ], section "llvm.metadata"		; <ptr> [#uses=0]

; CHECK: lb
; CHECK: blr
define void @lb(ptr %src, i32 %n, ptr %dst) nounwind {
entry:
	%0 = load i32, ptr null, align 4		; <i32> [#uses=1]
	%1 = icmp sgt i32 %0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb.nph4945, label %return

bb.nph4945:		; preds = %entry
	%2 = getelementptr [2 x i64], ptr null, i32 0, i32 1		; <ptr> [#uses=6]
	%3 = getelementptr [2 x i64], ptr null, i32 0, i32 1		; <ptr> [#uses=3]
	br label %bb2326

bb2217:		; preds = %bb2326
	%4 = or i64 0, 0		; <i64> [#uses=2]
	%5 = fptosi float 0.000000e+00 to i32		; <i32> [#uses=1]
	%6 = fptosi float 0.000000e+00 to i32		; <i32> [#uses=1]
	%7 = getelementptr float, ptr null, i32 2		; <ptr> [#uses=1]
	%8 = load float, ptr %7, align 4		; <float> [#uses=1]
	%9 = getelementptr float, ptr null, i32 3		; <ptr> [#uses=1]
	%10 = load float, ptr %9, align 4		; <float> [#uses=1]
	%11 = fmul float %8, 6.553500e+04		; <float> [#uses=1]
	%12 = fadd float %11, 5.000000e-01		; <float> [#uses=1]
	%13 = fmul float %10, 6.553500e+04		; <float> [#uses=1]
	%14 = fadd float %13, 5.000000e-01		; <float> [#uses=3]
	%15 = fcmp olt float %12, 0.000000e+00		; <i1> [#uses=0]
	%16 = fcmp olt float %14, 0.000000e+00		; <i1> [#uses=1]
	br i1 %16, label %bb2265, label %bb2262

bb2262:		; preds = %bb2217
	%17 = fcmp ogt float %14, 6.553500e+04		; <i1> [#uses=1]
	br i1 %17, label %bb2264, label %bb2265

bb2264:		; preds = %bb2262
	br label %bb2265

bb2265:		; preds = %bb2264, %bb2262, %bb2217
	%f3596.0 = phi float [ 6.553500e+04, %bb2264 ], [ 0.000000e+00, %bb2217 ], [ %14, %bb2262 ]		; <float> [#uses=1]
	%18 = fptosi float 0.000000e+00 to i32		; <i32> [#uses=1]
	%19 = fptosi float %f3596.0 to i32		; <i32> [#uses=1]
	%20 = zext i32 %5 to i64		; <i64> [#uses=1]
	%21 = shl i64 %20, 48		; <i64> [#uses=1]
	%22 = zext i32 %6 to i64		; <i64> [#uses=1]
	%23 = shl i64 %22, 32		; <i64> [#uses=1]
	%24 = sext i32 %18 to i64		; <i64> [#uses=1]
	%25 = shl i64 %24, 16		; <i64> [#uses=1]
	%26 = sext i32 %19 to i64		; <i64> [#uses=1]
	%27 = or i64 %23, %21		; <i64> [#uses=1]
	%28 = or i64 %27, %25		; <i64> [#uses=1]
	%29 = or i64 %28, %26		; <i64> [#uses=2]
	%30 = shl i64 %4, 48		; <i64> [#uses=1]
	%31 = shl i64 %29, 32		; <i64> [#uses=1]
	%32 = and i64 %31, 281470681743360		; <i64> [#uses=1]
	store i64 %4, ptr null, align 16
	store i64 %29, ptr %2, align 8
	%33 = getelementptr i8, ptr null, i32 0		; <ptr> [#uses=1]
	%34 = load float, ptr %33, align 4		; <float> [#uses=1]
	%35 = getelementptr float, ptr %33, i32 1		; <ptr> [#uses=1]
	%36 = load float, ptr %35, align 4		; <float> [#uses=1]
	%37 = fmul float %34, 6.553500e+04		; <float> [#uses=1]
	%38 = fadd float %37, 5.000000e-01		; <float> [#uses=1]
	%39 = fmul float %36, 6.553500e+04		; <float> [#uses=1]
	%40 = fadd float %39, 5.000000e-01		; <float> [#uses=3]
	%41 = fcmp olt float %38, 0.000000e+00		; <i1> [#uses=0]
	%42 = fcmp olt float %40, 0.000000e+00		; <i1> [#uses=1]
	br i1 %42, label %bb2277, label %bb2274

bb2274:		; preds = %bb2265
	%43 = fcmp ogt float %40, 6.553500e+04		; <i1> [#uses=0]
	br label %bb2277

bb2277:		; preds = %bb2274, %bb2265
	%f1582.0 = phi float [ 0.000000e+00, %bb2265 ], [ %40, %bb2274 ]		; <float> [#uses=1]
	%44 = fptosi float 0.000000e+00 to i32		; <i32> [#uses=1]
	%45 = fptosi float %f1582.0 to i32		; <i32> [#uses=1]
	%46 = getelementptr float, ptr %33, i32 2		; <ptr> [#uses=1]
	%47 = load float, ptr %46, align 4		; <float> [#uses=1]
	%48 = getelementptr float, ptr %33, i32 3		; <ptr> [#uses=1]
	%49 = load float, ptr %48, align 4		; <float> [#uses=1]
	%50 = fmul float %47, 6.553500e+04		; <float> [#uses=1]
	%51 = fadd float %50, 5.000000e-01		; <float> [#uses=1]
	%52 = fmul float %49, 6.553500e+04		; <float> [#uses=1]
	%53 = fadd float %52, 5.000000e-01		; <float> [#uses=1]
	%54 = fcmp olt float %51, 0.000000e+00		; <i1> [#uses=0]
	%55 = fcmp olt float %53, 0.000000e+00		; <i1> [#uses=0]
	%56 = fptosi float 0.000000e+00 to i32		; <i32> [#uses=1]
	%57 = fptosi float 0.000000e+00 to i32		; <i32> [#uses=1]
	%58 = zext i32 %44 to i64		; <i64> [#uses=1]
	%59 = shl i64 %58, 48		; <i64> [#uses=1]
	%60 = zext i32 %45 to i64		; <i64> [#uses=1]
	%61 = shl i64 %60, 32		; <i64> [#uses=1]
	%62 = sext i32 %56 to i64		; <i64> [#uses=1]
	%63 = shl i64 %62, 16		; <i64> [#uses=1]
	%64 = sext i32 %57 to i64		; <i64> [#uses=1]
	%65 = or i64 %61, %59		; <i64> [#uses=1]
	%66 = or i64 %65, %63		; <i64> [#uses=1]
	%67 = or i64 %66, %64		; <i64> [#uses=2]
	%68 = getelementptr i8, ptr null, i32 0		; <ptr> [#uses=1]
	%69 = load float, ptr %68, align 4		; <float> [#uses=1]
	%70 = getelementptr float, ptr %68, i32 1		; <ptr> [#uses=1]
	%71 = load float, ptr %70, align 4		; <float> [#uses=1]
	%72 = fmul float %69, 6.553500e+04		; <float> [#uses=1]
	%73 = fadd float %72, 5.000000e-01		; <float> [#uses=3]
	%74 = fmul float %71, 6.553500e+04		; <float> [#uses=1]
	%75 = fadd float %74, 5.000000e-01		; <float> [#uses=1]
	%76 = fcmp olt float %73, 0.000000e+00		; <i1> [#uses=1]
	br i1 %76, label %bb2295, label %bb2292

bb2292:		; preds = %bb2277
	%77 = fcmp ogt float %73, 6.553500e+04		; <i1> [#uses=1]
	br i1 %77, label %bb2294, label %bb2295

bb2294:		; preds = %bb2292
	br label %bb2295

bb2295:		; preds = %bb2294, %bb2292, %bb2277
	%f0569.0 = phi float [ 6.553500e+04, %bb2294 ], [ 0.000000e+00, %bb2277 ], [ %73, %bb2292 ]		; <float> [#uses=1]
	%78 = fcmp olt float %75, 0.000000e+00		; <i1> [#uses=0]
	%79 = fptosi float %f0569.0 to i32		; <i32> [#uses=1]
	%80 = fptosi float 0.000000e+00 to i32		; <i32> [#uses=1]
	%81 = getelementptr float, ptr %68, i32 2		; <ptr> [#uses=1]
	%82 = load float, ptr %81, align 4		; <float> [#uses=1]
	%83 = getelementptr float, ptr %68, i32 3		; <ptr> [#uses=1]
	%84 = load float, ptr %83, align 4		; <float> [#uses=1]
	%85 = fmul float %82, 6.553500e+04		; <float> [#uses=1]
	%86 = fadd float %85, 5.000000e-01		; <float> [#uses=1]
	%87 = fmul float %84, 6.553500e+04		; <float> [#uses=1]
	%88 = fadd float %87, 5.000000e-01		; <float> [#uses=1]
	%89 = fcmp olt float %86, 0.000000e+00		; <i1> [#uses=0]
	%90 = fcmp olt float %88, 0.000000e+00		; <i1> [#uses=0]
	%91 = fptosi float 0.000000e+00 to i32		; <i32> [#uses=1]
	%92 = fptosi float 0.000000e+00 to i32		; <i32> [#uses=1]
	%93 = zext i32 %79 to i64		; <i64> [#uses=1]
	%94 = shl i64 %93, 48		; <i64> [#uses=1]
	%95 = zext i32 %80 to i64		; <i64> [#uses=1]
	%96 = shl i64 %95, 32		; <i64> [#uses=1]
	%97 = sext i32 %91 to i64		; <i64> [#uses=1]
	%98 = shl i64 %97, 16		; <i64> [#uses=1]
	%99 = sext i32 %92 to i64		; <i64> [#uses=1]
	%100 = or i64 %96, %94		; <i64> [#uses=1]
	%101 = or i64 %100, %98		; <i64> [#uses=1]
	%102 = or i64 %101, %99		; <i64> [#uses=2]
	%103 = shl i64 %67, 16		; <i64> [#uses=1]
	%104 = and i64 %103, 4294901760		; <i64> [#uses=1]
	%105 = and i64 %102, 65535		; <i64> [#uses=1]
	%106 = or i64 %32, %30		; <i64> [#uses=1]
	%107 = or i64 %106, %104		; <i64> [#uses=1]
	%108 = or i64 %107, %105		; <i64> [#uses=1]
	store i64 %67, ptr null, align 16
	store i64 %102, ptr %3, align 8
	%109 = icmp eq i64 %108, 0		; <i1> [#uses=1]
	br i1 %109, label %bb2325, label %bb2315

bb2315:		; preds = %bb2295
	%110 = icmp eq ptr %155, null		; <i1> [#uses=1]
	br i1 %110, label %bb2318, label %bb2317

bb2317:		; preds = %bb2315
	%111 = load i64, ptr null, align 16		; <i64> [#uses=1]
	%112 = call i32 (...) @_u16a_cm( i64 %111, ptr %155, double 0.000000e+00, double 1.047551e+06 ) nounwind		; <i32> [#uses=1]
	%113 = sext i32 %112 to i64		; <i64> [#uses=1]
	store i64 %113, ptr null, align 16
	%114 = load i64, ptr %2, align 8		; <i64> [#uses=1]
	%115 = call i32 (...) @_u16a_cm( i64 %114, ptr %155, double 0.000000e+00, double 1.047551e+06 ) nounwind		; <i32> [#uses=1]
	%116 = sext i32 %115 to i64		; <i64> [#uses=1]
	store i64 %116, ptr %2, align 8
	%117 = load i64, ptr null, align 16		; <i64> [#uses=1]
	%118 = call i32 (...) @_u16a_cm( i64 %117, ptr %155, double 0.000000e+00, double 1.047551e+06 ) nounwind		; <i32> [#uses=1]
	%119 = sext i32 %118 to i64		; <i64> [#uses=1]
	store i64 %119, ptr null, align 16
	%120 = load i64, ptr %3, align 8		; <i64> [#uses=1]
	%121 = call i32 (...) @_u16a_cm( i64 %120, ptr %155, double 0.000000e+00, double 1.047551e+06 ) nounwind		; <i32> [#uses=0]
	unreachable

bb2318:		; preds = %bb2315
	%122 = getelementptr %struct.CGLSI, ptr %src, i32 %indvar5021, i32 8		; <ptr> [#uses=1]
	%123 = load i64, ptr %122, align 8		; <i64> [#uses=1]
	%124 = trunc i64 %123 to i32		; <i32> [#uses=4]
	%125 = load i64, ptr null, align 16		; <i64> [#uses=1]
	%126 = call i32 (...) @_u16_ff( i64 %125, i32 %124 ) nounwind		; <i32> [#uses=1]
	%127 = sext i32 %126 to i64		; <i64> [#uses=1]
	store i64 %127, ptr null, align 16
	%128 = load i64, ptr %2, align 8		; <i64> [#uses=1]
	%129 = call i32 (...) @_u16_ff( i64 %128, i32 %124 ) nounwind		; <i32> [#uses=1]
	%130 = sext i32 %129 to i64		; <i64> [#uses=1]
	store i64 %130, ptr %2, align 8
	%131 = load i64, ptr null, align 16		; <i64> [#uses=1]
	%132 = call i32 (...) @_u16_ff( i64 %131, i32 %124 ) nounwind		; <i32> [#uses=1]
	%133 = sext i32 %132 to i64		; <i64> [#uses=1]
	store i64 %133, ptr null, align 16
	%134 = load i64, ptr %3, align 8		; <i64> [#uses=1]
	%135 = call i32 (...) @_u16_ff( i64 %134, i32 %124 ) nounwind		; <i32> [#uses=0]
	unreachable

bb2319:		; preds = %bb2326
	%136 = getelementptr %struct.CGLSI, ptr %src, i32 %indvar5021, i32 2		; <ptr> [#uses=1]
	%137 = load ptr, ptr %136, align 4		; <ptr> [#uses=4]
	%138 = getelementptr i8, ptr %137, i32 0		; <ptr> [#uses=1]
	%139 = call i32 (...) @_u16_sf32( double 0.000000e+00, double 6.553500e+04, double 5.000000e-01, ptr %138 ) nounwind		; <i32> [#uses=1]
	%140 = sext i32 %139 to i64		; <i64> [#uses=2]
	%141 = getelementptr i8, ptr %137, i32 0		; <ptr> [#uses=1]
	%142 = call i32 (...) @_u16_sf32( double 0.000000e+00, double 6.553500e+04, double 5.000000e-01, ptr %141 ) nounwind		; <i32> [#uses=1]
	%143 = sext i32 %142 to i64		; <i64> [#uses=2]
	%144 = shl i64 %140, 48		; <i64> [#uses=0]
	%145 = shl i64 %143, 32		; <i64> [#uses=1]
	%146 = and i64 %145, 281470681743360		; <i64> [#uses=0]
	store i64 %140, ptr null, align 16
	store i64 %143, ptr %2, align 8
	%147 = getelementptr i8, ptr %137, i32 0		; <ptr> [#uses=1]
	%148 = call i32 (...) @_u16_sf32( double 0.000000e+00, double 6.553500e+04, double 5.000000e-01, ptr %147 ) nounwind		; <i32> [#uses=1]
	%149 = sext i32 %148 to i64		; <i64> [#uses=0]
	%150 = getelementptr i8, ptr %137, i32 0		; <ptr> [#uses=1]
	%151 = call i32 (...) @_u16_sf32( double 0.000000e+00, double 6.553500e+04, double 5.000000e-01, ptr %150 ) nounwind		; <i32> [#uses=0]
	unreachable

bb2325:		; preds = %bb2326, %bb2295
	%indvar.next5145 = add i32 %indvar5021, 1		; <i32> [#uses=1]
	br label %bb2326

bb2326:		; preds = %bb2325, %bb.nph4945
	%indvar5021 = phi i32 [ 0, %bb.nph4945 ], [ %indvar.next5145, %bb2325 ]		; <i32> [#uses=6]
	%152 = icmp slt i32 %indvar5021, %n		; <i1> [#uses=0]
	%153 = getelementptr %struct.CGLSI, ptr %src, i32 %indvar5021, i32 10		; <ptr> [#uses=1]
	%154 = load ptr, ptr %153, align 4		; <ptr> [#uses=5]
	%155 = getelementptr %struct.CGLSI, ptr %src, i32 %indvar5021, i32 1		; <ptr> [#uses=1]
	%156 = load i32, ptr %155, align 4		; <i32> [#uses=1]
	%157 = and i32 %156, 255		; <i32> [#uses=1]
	switch i32 %157, label %bb2325 [
		 i32 59, label %bb2217
		 i32 60, label %bb2319
	]

return:		; preds = %entry
	ret void
}

declare i32 @_u16_ff(...)

declare i32 @_u16a_cm(...)

declare i32 @_u16_sf32(...)
