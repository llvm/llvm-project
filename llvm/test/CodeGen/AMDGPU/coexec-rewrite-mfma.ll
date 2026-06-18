; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -amdgpu-sched-strategy=coexec --enable-post-misched=0 %s -o - 2>&1 | FileCheck %s

; The coexec scheduler strategy pushes RewriteMFMAForm into its pipeline so MFMA
; chains can be rewritten to the AGPR form when ArchVGPR pressure is excessive.
;
; This IR is reduced from a Triton-generated GEMM kernel with multiple (28 to be
; exact) MFMAs in a phi-carried loop with high VGPR pressure, every MFMA in the
; loop is rewritten to the AGPR-destination form (v_mfma_f32_16x16x32_f16 a*).

; FIXME: To enable on gfx950?
; CHECK: warning: {{.*}}'amdgpu-sched-strategy'='coexec' is only supported for gfx1250
; CHECK-LABEL: v5_local_prefetch:
; CHECK-COUNT-28: v_mfma_f32_16x16x32_f16 a
; CHECK-NOT: v_mfma_f32_16x16x32_f16 v

target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @v5_local_prefetch() #0 {
.lr.ph:
  %0 = load <8 x half>, ptr addrspace(3) null, align 16
  %1 = shufflevector <8 x half> %0, <8 x half> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %2 = shufflevector <8 x half> %0, <8 x half> zeroinitializer, <2 x i32> <i32 2, i32 3>
  br label %3

3:                                                ; preds = %3, %.lr.ph
  %4 = phi float [ 0.000000e+00, %.lr.ph ], [ %53, %3 ]
  %5 = phi float [ 0.000000e+00, %.lr.ph ], [ %54, %3 ]
  %6 = phi float [ 0.000000e+00, %.lr.ph ], [ %61, %3 ]
  %7 = phi float [ 0.000000e+00, %.lr.ph ], [ %62, %3 ]
  %8 = phi float [ 0.000000e+00, %.lr.ph ], [ %63, %3 ]
  %9 = phi float [ 0.000000e+00, %.lr.ph ], [ %64, %3 ]
  %10 = phi float [ 0.000000e+00, %.lr.ph ], [ %71, %3 ]
  %11 = phi float [ 0.000000e+00, %.lr.ph ], [ %72, %3 ]
  %12 = phi float [ 0.000000e+00, %.lr.ph ], [ %73, %3 ]
  %13 = phi float [ 0.000000e+00, %.lr.ph ], [ %74, %3 ]
  %14 = phi float [ 0.000000e+00, %.lr.ph ], [ %80, %3 ]
  %15 = phi float [ 0.000000e+00, %.lr.ph ], [ %81, %3 ]
  %16 = phi float [ 0.000000e+00, %.lr.ph ], [ %82, %3 ]
  %17 = phi float [ 0.000000e+00, %.lr.ph ], [ %87, %3 ]
  %18 = phi float [ 0.000000e+00, %.lr.ph ], [ %88, %3 ]
  %19 = phi float [ 0.000000e+00, %.lr.ph ], [ %93, %3 ]
  %20 = phi float [ 0.000000e+00, %.lr.ph ], [ %94, %3 ]
  %21 = phi float [ 0.000000e+00, %.lr.ph ], [ %100, %3 ]
  %22 = phi float [ 0.000000e+00, %.lr.ph ], [ %101, %3 ]
  %23 = phi float [ 0.000000e+00, %.lr.ph ], [ %24, %3 ]
  %24 = phi float [ 0.000000e+00, %.lr.ph ], [ %106, %3 ]
  %25 = phi float [ 0.000000e+00, %.lr.ph ], [ %113, %3 ]
  %26 = phi float [ 0.000000e+00, %.lr.ph ], [ %114, %3 ]
  %27 = phi float [ 0.000000e+00, %.lr.ph ], [ %115, %3 ]
  %28 = phi float [ 0.000000e+00, %.lr.ph ], [ %116, %3 ]
  %29 = phi float [ 0.000000e+00, %.lr.ph ], [ %123, %3 ]
  %30 = phi float [ 0.000000e+00, %.lr.ph ], [ %124, %3 ]
  %31 = phi float [ 0.000000e+00, %.lr.ph ], [ %125, %3 ]
  %32 = phi float [ 0.000000e+00, %.lr.ph ], [ %126, %3 ]
  %33 = phi float [ 0.000000e+00, %.lr.ph ], [ %133, %3 ]
  %34 = phi float [ 0.000000e+00, %.lr.ph ], [ %134, %3 ]
  %35 = phi float [ 0.000000e+00, %.lr.ph ], [ %135, %3 ]
  %36 = phi float [ 0.000000e+00, %.lr.ph ], [ %136, %3 ]
  %37 = phi float [ 0.000000e+00, %.lr.ph ], [ %141, %3 ]
  %38 = phi float [ 0.000000e+00, %.lr.ph ], [ %142, %3 ]
  %39 = phi float [ 0.000000e+00, %.lr.ph ], [ %147, %3 ]
  %40 = phi float [ 0.000000e+00, %.lr.ph ], [ %148, %3 ]
  %41 = phi float [ 0.000000e+00, %.lr.ph ], [ %153, %3 ]
  %42 = phi float [ 0.000000e+00, %.lr.ph ], [ %41, %3 ]
  %43 = phi <2 x half> [ %1, %.lr.ph ], [ zeroinitializer, %3 ]
  %44 = phi <2 x half> [ %2, %.lr.ph ], [ zeroinitializer, %3 ]
  %45 = phi <8 x half> [ zeroinitializer, %.lr.ph ], [ <half 0.000000e+00, half 0.000000e+00, half poison, half poison, half poison, half poison, half poison, half poison>, %3 ]
  %46 = shufflevector <2 x half> %43, <2 x half> %44, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %47 = shufflevector <8 x half> %46, <8 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %48 = shufflevector <8 x half> %46, <8 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %49 = insertelement <4 x float> zeroinitializer, float %4, i64 2
  %50 = insertelement <4 x float> %49, float %5, i64 3
  %51 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %47, <8 x half> zeroinitializer, <4 x float> %50, i32 0, i32 0, i32 0)
  %52 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %51, i32 0, i32 0, i32 0)
  %53 = extractelement <4 x float> %52, i64 2
  %54 = extractelement <4 x float> %52, i64 3
  %55 = insertelement <4 x float> zeroinitializer, float %6, i64 0
  %56 = insertelement <4 x float> %55, float %7, i64 1
  %57 = insertelement <4 x float> %56, float %8, i64 2
  %58 = insertelement <4 x float> %57, float %9, i64 3
  %59 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %58, i32 0, i32 0, i32 0)
  %60 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %59, i32 0, i32 0, i32 0)
  %61 = extractelement <4 x float> %60, i64 0
  %62 = extractelement <4 x float> %60, i64 1
  %63 = extractelement <4 x float> %60, i64 2
  %64 = extractelement <4 x float> %60, i64 3
  %65 = insertelement <4 x float> zeroinitializer, float %10, i64 0
  %66 = insertelement <4 x float> %65, float %11, i64 1
  %67 = insertelement <4 x float> %66, float %12, i64 2
  %68 = insertelement <4 x float> %67, float %13, i64 3
  %69 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %68, i32 0, i32 0, i32 0)
  %70 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %69, i32 0, i32 0, i32 0)
  %71 = extractelement <4 x float> %70, i64 0
  %72 = extractelement <4 x float> %70, i64 1
  %73 = extractelement <4 x float> %70, i64 2
  %74 = extractelement <4 x float> %70, i64 3
  %75 = insertelement <4 x float> zeroinitializer, float %14, i64 0
  %76 = insertelement <4 x float> %75, float %15, i64 1
  %77 = insertelement <4 x float> %76, float %16, i64 2
  %78 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %77, i32 0, i32 0, i32 0)
  %79 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %78, i32 0, i32 0, i32 0)
  %80 = extractelement <4 x float> %79, i64 0
  %81 = extractelement <4 x float> %79, i64 1
  %82 = extractelement <4 x float> %79, i64 2
  %83 = insertelement <4 x float> zeroinitializer, float %17, i64 2
  %84 = insertelement <4 x float> %83, float %18, i64 3
  %85 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %84, i32 0, i32 0, i32 0)
  %86 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %85, i32 0, i32 0, i32 0)
  %87 = extractelement <4 x float> %86, i64 2
  %88 = extractelement <4 x float> %86, i64 3
  %89 = insertelement <4 x float> zeroinitializer, float %19, i64 0
  %90 = insertelement <4 x float> %89, float %20, i64 1
  %91 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %90, i32 0, i32 0, i32 0)
  %92 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %91, i32 0, i32 0, i32 0)
  %93 = extractelement <4 x float> %92, i64 0
  %94 = extractelement <4 x float> %92, i64 1
  %95 = shufflevector <8 x half> zeroinitializer, <8 x half> %45, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %96 = insertelement <4 x float> zeroinitializer, float %21, i64 0
  %97 = insertelement <4 x float> %96, float %22, i64 1
  %98 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %95, <8 x half> zeroinitializer, <4 x float> %97, i32 0, i32 0, i32 0)
  %99 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %98, i32 0, i32 0, i32 0)
  %100 = extractelement <4 x float> %99, i64 0
  %101 = extractelement <4 x float> %99, i64 1
  %102 = insertelement <4 x float> poison, float %23, i64 0
  %103 = insertelement <4 x float> %102, float %24, i64 0
  %104 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %103, i32 0, i32 0, i32 0)
  %105 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %104, i32 0, i32 0, i32 0)
  %106 = extractelement <4 x float> %105, i64 0
  %107 = insertelement <4 x float> zeroinitializer, float %25, i64 0
  %108 = insertelement <4 x float> %107, float %26, i64 1
  %109 = insertelement <4 x float> %108, float %27, i64 2
  %110 = insertelement <4 x float> %109, float %28, i64 3
  %111 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %48, <8 x half> zeroinitializer, <4 x float> %110, i32 0, i32 0, i32 0)
  %112 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %111, i32 0, i32 0, i32 0)
  %113 = extractelement <4 x float> %112, i64 0
  %114 = extractelement <4 x float> %112, i64 1
  %115 = extractelement <4 x float> %112, i64 2
  %116 = extractelement <4 x float> %112, i64 3
  %117 = insertelement <4 x float> zeroinitializer, float %29, i64 0
  %118 = insertelement <4 x float> %117, float %30, i64 1
  %119 = insertelement <4 x float> %118, float %31, i64 2
  %120 = insertelement <4 x float> %119, float %32, i64 3
  %121 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %120, i32 0, i32 0, i32 0)
  %122 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %121, i32 0, i32 0, i32 0)
  %123 = extractelement <4 x float> %122, i64 0
  %124 = extractelement <4 x float> %122, i64 1
  %125 = extractelement <4 x float> %122, i64 2
  %126 = extractelement <4 x float> %122, i64 3
  %127 = insertelement <4 x float> zeroinitializer, float %33, i64 0
  %128 = insertelement <4 x float> %127, float %34, i64 1
  %129 = insertelement <4 x float> %128, float %35, i64 2
  %130 = insertelement <4 x float> %129, float %36, i64 3
  %131 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %130, i32 0, i32 0, i32 0)
  %132 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %131, i32 0, i32 0, i32 0)
  %133 = extractelement <4 x float> %132, i64 0
  %134 = extractelement <4 x float> %132, i64 1
  %135 = extractelement <4 x float> %132, i64 2
  %136 = extractelement <4 x float> %132, i64 3
  %137 = insertelement <4 x float> zeroinitializer, float %37, i64 0
  %138 = insertelement <4 x float> %137, float %38, i64 1
  %139 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %138, i32 0, i32 0, i32 0)
  %140 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %139, i32 0, i32 0, i32 0)
  %141 = extractelement <4 x float> %140, i64 0
  %142 = extractelement <4 x float> %140, i64 1
  %143 = insertelement <4 x float> zeroinitializer, float %39, i64 0
  %144 = insertelement <4 x float> %143, float %40, i64 1
  %145 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %144, i32 0, i32 0, i32 0)
  %146 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %145, i32 0, i32 0, i32 0)
  %147 = extractelement <4 x float> %146, i64 0
  %148 = extractelement <4 x float> %146, i64 1
  %149 = insertelement <4 x float> <float 0.000000e+00, float 1.000000e+00, float 0.000000e+00, float 0.000000e+00>, float %41, i64 2
  %150 = insertelement <4 x float> %149, float %42, i64 3
  %151 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %150, i32 0, i32 0, i32 0)
  %152 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %151, i32 0, i32 0, i32 0)
  %153 = extractelement <4 x float> %152, i64 0
  br label %3
}

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }
