; RUN: opt -S -inline -inline-threshold=275 < %s | FileCheck %s
; PR13095

; The performance of the c-ray benchmark largely depends on the inlining of a
; specific call to @ray_sphere. This test case is designed to verify that it's
; inlined at -O3.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.sphere = type { %struct.vec3, double, %struct.material, %struct.sphere* }
%struct.vec3 = type { double, double, double }
%struct.material = type { %struct.vec3, double, double }
%struct.ray = type { %struct.vec3, %struct.vec3 }
%struct.spoint = type { %struct.vec3, %struct.vec3, %struct.vec3, double }

define i32 @caller(%struct.sphere* %i) {
  %shadow_ray = alloca %struct.ray, align 8
  call void @fix(%struct.ray* %shadow_ray)

  %call = call i32 @ray_sphere(%struct.sphere* %i, %struct.ray* byval align 8 %shadow_ray, %struct.spoint* null)
  ret i32 %call

; CHECK: @caller
; CHECK-NOT: call i32 @ray_sphere
; CHECK: ret i32
}

declare void @fix(%struct.ray*)

define i32 @ray_sphere(%struct.sphere* nocapture %sph, %struct.ray* nocapture byval align 8 %ray, %struct.spoint* %sp) nounwind uwtable ssp {
  %1 = getelementptr inbounds %struct.ray* %ray, i64 0, i32 1, i32 0
  %2 = load double* %1, align 8
  %3 = fmul double %2, %2
  %4 = getelementptr inbounds %struct.ray* %ray, i64 0, i32 1, i32 1
  %5 = load double* %4, align 8
  %6 = fmul double %5, %5
  %7 = fadd double %3, %6
  %8 = getelementptr inbounds %struct.ray* %ray, i64 0, i32 1, i32 2
  %9 = load double* %8, align 8
  %10 = fmul double %9, %9
  %11 = fadd double %7, %10
  %12 = fmul double %2, 2.000000e+00
  %13 = getelementptr inbounds %struct.ray* %ray, i64 0, i32 0, i32 0
  %14 = load double* %13, align 8
  %15 = getelementptr inbounds %struct.sphere* %sph, i64 0, i32 0, i32 0
  %16 = load double* %15, align 8
  %17 = fsub double %14, %16
  %18 = fmul double %12, %17
  %19 = fmul double %5, 2.000000e+00
  %20 = getelementptr inbounds %struct.ray* %ray, i64 0, i32 0, i32 1
  %21 = load double* %20, align 8
  %22 = getelementptr inbounds %struct.sphere* %sph, i64 0, i32 0, i32 1
  %23 = load double* %22, align 8
  %24 = fsub double %21, %23
  %25 = fmul double %19, %24
  %26 = fadd double %18, %25
  %27 = fmul double %9, 2.000000e+00
  %28 = getelementptr inbounds %struct.ray* %ray, i64 0, i32 0, i32 2
  %29 = load double* %28, align 8
  %30 = getelementptr inbounds %struct.sphere* %sph, i64 0, i32 0, i32 2
  %31 = load double* %30, align 8
  %32 = fsub double %29, %31
  %33 = fmul double %27, %32
  %34 = fadd double %26, %33
  %35 = fmul double %16, %16
  %36 = fmul double %23, %23
  %37 = fadd double %35, %36
  %38 = fmul double %31, %31
  %39 = fadd double %37, %38
  %40 = fmul double %14, %14
  %41 = fadd double %40, %39
  %42 = fmul double %21, %21
  %43 = fadd double %42, %41
  %44 = fmul double %29, %29
  %45 = fadd double %44, %43
  %46 = fsub double -0.000000e+00, %16
  %47 = fmul double %14, %46
  %48 = fmul double %21, %23
  %49 = fsub double %47, %48
  %50 = fmul double %29, %31
  %51 = fsub double %49, %50
  %52 = fmul double %51, 2.000000e+00
  %53 = fadd double %52, %45
  %54 = getelementptr inbounds %struct.sphere* %sph, i64 0, i32 1
  %55 = load double* %54, align 8
  %56 = fmul double %55, %55
  %57 = fsub double %53, %56
  %58 = fmul double %34, %34
  %59 = fmul double %11, 4.000000e+00
  %60 = fmul double %59, %57
  %61 = fsub double %58, %60
  %62 = fcmp olt double %61, 0.000000e+00
  br i1 %62, label %130, label %63

; <label>:63                                      ; preds = %0
  %64 = tail call double @sqrt(double %61) nounwind readnone
  %65 = fsub double -0.000000e+00, %34
  %66 = fsub double %64, %34
  %67 = fmul double %11, 2.000000e+00
  %68 = fdiv double %66, %67
  %69 = fsub double %65, %64
  %70 = fdiv double %69, %67
  %71 = fcmp olt double %68, 1.000000e-06
  %72 = fcmp olt double %70, 1.000000e-06
  %or.cond = and i1 %71, %72
  br i1 %or.cond, label %130, label %73

; <label>:73                                      ; preds = %63
  %74 = fcmp ogt double %68, 1.000000e+00
  %75 = fcmp ogt double %70, 1.000000e+00
  %or.cond1 = and i1 %74, %75
  br i1 %or.cond1, label %130, label %76

; <label>:76                                      ; preds = %73
  %77 = icmp eq %struct.spoint* %sp, null
  br i1 %77, label %130, label %78

; <label>:78                                      ; preds = %76
  %t1.0 = select i1 %71, double %70, double %68
  %t2.0 = select i1 %72, double %t1.0, double %70
  %79 = fcmp olt double %t1.0, %t2.0
  %80 = select i1 %79, double %t1.0, double %t2.0
  %81 = getelementptr inbounds %struct.spoint* %sp, i64 0, i32 3
  store double %80, double* %81, align 8
  %82 = fmul double %80, %2
  %83 = fadd double %14, %82
  %84 = getelementptr inbounds %struct.spoint* %sp, i64 0, i32 0, i32 0
  store double %83, double* %84, align 8
  %85 = fmul double %5, %80
  %86 = fadd double %21, %85
  %87 = getelementptr inbounds %struct.spoint* %sp, i64 0, i32 0, i32 1
  store double %86, double* %87, align 8
  %88 = fmul double %9, %80
  %89 = fadd double %29, %88
  %90 = getelementptr inbounds %struct.spoint* %sp, i64 0, i32 0, i32 2
  store double %89, double* %90, align 8
  %91 = load double* %15, align 8
  %92 = fsub double %83, %91
  %93 = load double* %54, align 8
  %94 = fdiv double %92, %93
  %95 = getelementptr inbounds %struct.spoint* %sp, i64 0, i32 1, i32 0
  store double %94, double* %95, align 8
  %96 = load double* %22, align 8
  %97 = fsub double %86, %96
  %98 = load double* %54, align 8
  %99 = fdiv double %97, %98
  %100 = getelementptr inbounds %struct.spoint* %sp, i64 0, i32 1, i32 1
  store double %99, double* %100, align 8
  %101 = load double* %30, align 8
  %102 = fsub double %89, %101
  %103 = load double* %54, align 8
  %104 = fdiv double %102, %103
  %105 = getelementptr inbounds %struct.spoint* %sp, i64 0, i32 1, i32 2
  store double %104, double* %105, align 8
  %106 = fmul double %2, %94
  %107 = fmul double %5, %99
  %108 = fadd double %106, %107
  %109 = fmul double %9, %104
  %110 = fadd double %108, %109
  %111 = fmul double %110, 2.000000e+00
  %112 = fmul double %94, %111
  %113 = fsub double %112, %2
  %114 = fsub double -0.000000e+00, %113
  %115 = fmul double %99, %111
  %116 = fsub double %115, %5
  %117 = fsub double -0.000000e+00, %116
  %118 = fmul double %104, %111
  %119 = fsub double %118, %9
  %120 = fsub double -0.000000e+00, %119
  %.06 = getelementptr inbounds %struct.spoint* %sp, i64 0, i32 2, i32 0
  %.18 = getelementptr inbounds %struct.spoint* %sp, i64 0, i32 2, i32 1
  %.210 = getelementptr inbounds %struct.spoint* %sp, i64 0, i32 2, i32 2
  %121 = fmul double %113, %113
  %122 = fmul double %116, %116
  %123 = fadd double %121, %122
  %124 = fmul double %119, %119
  %125 = fadd double %123, %124
  %126 = tail call double @sqrt(double %125) nounwind readnone
  %127 = fdiv double %114, %126
  store double %127, double* %.06, align 8
  %128 = fdiv double %117, %126
  store double %128, double* %.18, align 8
  %129 = fdiv double %120, %126
  store double %129, double* %.210, align 8
  br label %130

; <label>:130                                     ; preds = %78, %76, %73, %63, %0
  %.0 = phi i32 [ 0, %0 ], [ 0, %73 ], [ 0, %63 ], [ 1, %76 ], [ 1, %78 ]
  ret i32 %.0
}

declare double @sqrt(double) nounwind readnone
