; NOTE: There must be no spill reload inside the loop starting with LBB0_1:
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 < %s | FileCheck %s
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9-p9:192:256:256:32"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @foo(ptr %.sroa.1.0.copyload, ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15, ptr %16, ptr %17, ptr %18, ptr %19, ptr %20, ptr %21, ptr %22, ptr %23, ptr %24, ptr %25, ptr %26, ptr %27, ptr %28, ptr %29, ptr %30, ptr %31, ptr %32, ptr %33, double %34, double %35, double %36, float %37, float %38, float %39, float %40, ptr %41) {
; CHECK-LABEL: foo:
; CHECK-LABEL: .LBB0_1
; CHECK-NOT: buffer_load_dword {{v[0-9]+}}

.lr.ph:
  %.pre = load double, ptr null, align 8
  br label %42

42:                                               ; preds = %42, %.lr.ph
  %.0.i4402 = phi i32 [ 1, %.lr.ph ], [ 0, %42 ]
  %43 = zext i32 %.0.i4402 to i64
  %44 = load double, ptr %2, align 8
  %45 = load double, ptr %4, align 8
  %46 = load double, ptr %7, align 8
  %47 = load double, ptr %13, align 8
  %48 = load double, ptr %15, align 8
  %49 = load double, ptr %17, align 8
  %50 = load double, ptr %19, align 8
  %51 = load double, ptr %18, align 8
  %52 = load double, ptr %27, align 8
  %53 = load double, ptr %23, align 8
  %54 = load double, ptr %31, align 8
  %55 = load double, ptr %33, align 8
  %56 = load double, ptr %25, align 8
  %57 = load double, ptr %16, align 8
  %58 = fpext float %40 to double
  %59 = fmul double %52, %58
  %60 = fadd double %59, %51
  %61 = fsub double %60, %48
  %62 = fmul double 0.000000e+00, %36
  %63 = fsub double %61, %62
  %64 = fadd double %49, %63
  %65 = fptrunc double %64 to float
  %66 = fsub double 0.000000e+00, %34
  %67 = fpext float %39 to double
  %68 = fmul double %53, %67
  %69 = fsub double %66, %68
  %70 = fadd double %50, %69
  %71 = fptrunc double %70 to float
  store float 0.000000e+00, ptr %30, align 4
  store float 0.000000e+00, ptr %26, align 4
  %72 = getelementptr float, ptr %41, i64 %43
  store float %38, ptr %72, align 4
  store float %65, ptr %29, align 4
  store float %71, ptr %14, align 4
  store float %39, ptr %3, align 4
  store float %39, ptr %11, align 4
  %73 = fsub double %46, %44
  %74 = fptrunc double %73 to float
  %75 = fsub double %47, %45
  %76 = fptrunc double %75 to float
  %77 = fadd float %74, %76
  %78 = fpext float %37 to double
  %79 = fmul contract double %56, 0.000000e+00
  %80 = fsub contract double %34, %79
  %81 = fpext float %77 to double
  %82 = fmul double %.pre, %81
  %83 = fsub double %80, %82
  %84 = fpext float %38 to double
  %85 = fmul double %57, %84
  %86 = fsub double %83, %85
  %87 = fptrunc double %86 to float
  %88 = fmul double %34, 0.000000e+00
  %89 = fmul double %54, %78
  %90 = fadd double %89, %88
  %91 = fsub double %90, %55
  %92 = fmul double 0.000000e+00, %35
  %93 = fsub double %91, %92
  %94 = fmul double %34, %34
  %95 = fadd double %93, %94
  %96 = fptrunc double %95 to float
  store float %87, ptr %1, align 4
  store float %37, ptr %21, align 4
  store float %96, ptr %0, align 4
  store float 0.000000e+00, ptr %9, align 4
  store float 0.000000e+00, ptr %32, align 4
  store float 0.000000e+00, ptr %20, align 4
  store float 0.000000e+00, ptr %22, align 4
  store float 0.000000e+00, ptr %5, align 4
  store float 0.000000e+00, ptr %28, align 4
  store float 0.000000e+00, ptr %12, align 4
  store float 0.000000e+00, ptr %6, align 4
  store float 0.000000e+00, ptr %8, align 4
  store float 0.000000e+00, ptr %.sroa.1.0.copyload, align 4
  store float %37, ptr %10, align 4
  store float 0.000000e+00, ptr %24, align 4
  br label %42
}
