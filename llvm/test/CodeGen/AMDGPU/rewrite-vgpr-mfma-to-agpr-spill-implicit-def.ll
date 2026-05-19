; REQUIRES: asserts
; RUN: llc -O3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
; RUN:   -stop-after=amdgpu-rewrite-agpr-copy-mfma \
; RUN:   -debug-only=amdgpu-rewrite-agpr-copy-mfma -filetype=null %s 2>&1 \
; RUN:   | FileCheck %s

; Regression test from https://github.com/llvm/llvm-project/issues/196671

; It is legal for a spill reload to not have a dominating spill store.
; When the AGPR rewrite pass unspills such a slot into a vreg, it must insert
; IMPLICIT_DEF so the vreg has defs on all paths.

; CHECK: Inserted IMPLICIT_DEF for %{{[0-9]+}} in %bb.{{[0-9]+}}

define amdgpu_kernel void @rewrite_vgpr_mfma_to_agpr_spill_implicit_def(i1 %0, <16 x float> %.sroa.366.2) #0 {
.lr.ph.i:
  br label %1

1:                                                ; preds = %51, %.lr.ph.i
  %.sroa.01121.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %80, %51 ]
  %.sroa.54.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %82, %51 ]
  %.sroa.106.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %84, %51 ]
  %.sroa.1581182.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %85, %51 ]
  %.sroa.210.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %86, %51 ]
  %.sroa.262.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %87, %51 ]
  %.sroa.314.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %88, %51 ]
  %.sroa.366.21 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %72, %51 ]
  %.sroa.418.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %89, %51 ]
  %.sroa.470.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %74, %51 ]
  %.sroa.522.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %75, %51 ]
  %.sroa.574.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %90, %51 ]
  %.sroa.626.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %76, %51 ]
  %.sroa.678.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %91, %51 ]
  %.sroa.730.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %92, %51 ]
  %.sroa.782.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %79, %51 ]
  %2 = phi i64 [ 0, %.lr.ph.i ], [ 1, %51 ]
  br i1 %0, label %3, label %4

3:                                                ; preds = %1
  store <4 x i32> zeroinitializer, ptr addrspace(5) null, align 16
  br label %51

4:                                                ; preds = %1
  %5 = fmul <16 x float> %.sroa.01121.2, zeroinitializer
  %6 = fmul <16 x float> %.sroa.54.2, zeroinitializer
  %7 = fmul <16 x float> %.sroa.106.2, zeroinitializer
  %8 = fmul <16 x float> %.sroa.1581182.2, zeroinitializer
  %9 = fmul <16 x float> %.sroa.210.2, zeroinitializer
  %10 = fmul <16 x float> %.sroa.262.2, zeroinitializer
  %11 = fmul <16 x float> %.sroa.314.2, zeroinitializer
  %12 = fmul <16 x float> %.sroa.366.21, zeroinitializer
  %13 = fmul <16 x float> %.sroa.418.2, zeroinitializer
  %14 = fmul <16 x float> %.sroa.470.2, zeroinitializer
  %15 = fmul <16 x float> %.sroa.522.2, zeroinitializer
  %16 = fmul <16 x float> %.sroa.574.2, zeroinitializer
  %17 = fmul <16 x float> %.sroa.626.2, zeroinitializer
  %18 = fmul <16 x float> %.sroa.678.2, zeroinitializer
  %19 = fmul <16 x float> %.sroa.730.2, zeroinitializer
  %20 = fmul <16 x float> %.sroa.782.2, zeroinitializer
  %21 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %5, i32 0, i32 0, i32 0)
  %22 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %6, i32 0, i32 0, i32 0)
  %23 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %7, i32 0, i32 0, i32 0)
  %24 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %8, i32 0, i32 0, i32 0)
  %25 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %9, i32 0, i32 0, i32 0)
  %26 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %10, i32 0, i32 0, i32 0)
  %27 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %11, i32 0, i32 0, i32 0)
  %28 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %12, i32 0, i32 0, i32 0)
  %29 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %13, i32 0, i32 0, i32 0)
  %30 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %14, i32 0, i32 0, i32 0)
  %31 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %15, i32 0, i32 0, i32 0)
  %32 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %17, i32 0, i32 0, i32 0)
  %33 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %18, i32 0, i32 0, i32 0)
  %34 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %19, i32 0, i32 0, i32 0)
  %35 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %20, i32 0, i32 0, i32 0)
  %36 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %21, i32 0, i32 0, i32 0)
  %37 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %22, i32 0, i32 0, i32 0)
  %38 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %23, i32 0, i32 0, i32 0)
  %39 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %24, i32 0, i32 0, i32 0)
  %40 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %25, i32 0, i32 0, i32 0)
  %41 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %26, i32 0, i32 0, i32 0)
  %42 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %27, i32 0, i32 0, i32 0)
  %43 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %28, i32 0, i32 0, i32 0)
  %44 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %29, i32 0, i32 0, i32 0)
  %45 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %30, i32 0, i32 0, i32 0)
  %46 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %31, i32 0, i32 0, i32 0)
  %47 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %32, i32 0, i32 0, i32 0)
  %48 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %33, i32 0, i32 0, i32 0)
  %49 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %34, i32 0, i32 0, i32 0)
  %50 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %35, i32 0, i32 0, i32 0)
  br label %51

51:                                               ; preds = %4, %3
  %.sroa.01121.3 = phi <16 x float> [ %.sroa.01121.2, %3 ], [ %36, %4 ]
  %.sroa.54.3 = phi <16 x float> [ %.sroa.54.2, %3 ], [ %37, %4 ]
  %.sroa.106.3 = phi <16 x float> [ %.sroa.106.2, %3 ], [ %38, %4 ]
  %.sroa.1581182.3 = phi <16 x float> [ %.sroa.1581182.2, %3 ], [ %39, %4 ]
  %.sroa.210.3 = phi <16 x float> [ %.sroa.210.2, %3 ], [ %40, %4 ]
  %.sroa.262.3 = phi <16 x float> [ %.sroa.262.2, %3 ], [ %41, %4 ]
  %.sroa.314.3 = phi <16 x float> [ %.sroa.314.2, %3 ], [ %42, %4 ]
  %.sroa.366.3 = phi <16 x float> [ %.sroa.366.2, %3 ], [ %43, %4 ]
  %.sroa.418.3 = phi <16 x float> [ %.sroa.418.2, %3 ], [ %44, %4 ]
  %.sroa.470.3 = phi <16 x float> [ %.sroa.470.2, %3 ], [ %45, %4 ]
  %.sroa.522.3 = phi <16 x float> [ %.sroa.522.2, %3 ], [ %46, %4 ]
  %.sroa.574.3 = phi <16 x float> [ %.sroa.574.2, %3 ], [ %16, %4 ]
  %.sroa.626.3 = phi <16 x float> [ zeroinitializer, %3 ], [ %47, %4 ]
  %.sroa.678.3 = phi <16 x float> [ %.sroa.678.2, %3 ], [ %48, %4 ]
  %.sroa.730.3 = phi <16 x float> [ %.sroa.730.2, %3 ], [ %49, %4 ]
  %.sroa.782.3 = phi <16 x float> [ %.sroa.782.2, %3 ], [ %50, %4 ]
  %52 = fmul <16 x float> %.sroa.01121.3, zeroinitializer
  %53 = fmul <16 x float> %.sroa.54.3, zeroinitializer
  %54 = fmul <16 x float> %.sroa.106.3, zeroinitializer
  %55 = fmul <16 x float> %.sroa.1581182.3, zeroinitializer
  %56 = fmul <16 x float> %.sroa.210.3, zeroinitializer
  %57 = fmul <16 x float> %.sroa.262.3, zeroinitializer
  %58 = fmul <16 x float> %.sroa.314.3, zeroinitializer
  %59 = fmul <16 x float> %.sroa.366.3, zeroinitializer
  %60 = fmul <16 x float> %.sroa.418.3, zeroinitializer
  %61 = fmul <16 x float> %.sroa.470.3, zeroinitializer
  %62 = fmul <16 x float> %.sroa.522.3, zeroinitializer
  %63 = fmul <16 x float> %.sroa.574.3, zeroinitializer
  %64 = fmul <16 x float> %.sroa.626.3, zeroinitializer
  %65 = fmul <16 x float> %.sroa.678.3, zeroinitializer
  %66 = fmul <16 x float> %.sroa.730.3, zeroinitializer
  %67 = fmul <16 x float> %.sroa.782.3, zeroinitializer
  %68 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %55, i32 0, i32 0, i32 0)
  %69 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %56, i32 0, i32 0, i32 0)
  %70 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %57, i32 0, i32 0, i32 0)
  %71 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %58, i32 0, i32 0, i32 0)
  %72 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %59, i32 0, i32 0, i32 0)
  %73 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %60, i32 0, i32 0, i32 0)
  %74 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %61, i32 0, i32 0, i32 0)
  %75 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %62, i32 0, i32 0, i32 0)
  %76 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %64, i32 0, i32 0, i32 0)
  %77 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %65, i32 0, i32 0, i32 0)
  %78 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %66, i32 0, i32 0, i32 0)
  %79 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %67, i32 0, i32 0, i32 0)
  %80 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %52, i32 0, i32 0, i32 0)
  %81 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %53, i32 0, i32 0, i32 0)
  %82 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %81, i32 0, i32 0, i32 0)
  %83 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %54, i32 0, i32 0, i32 0)
  %84 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %83, i32 0, i32 0, i32 0)
  %85 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %68, i32 0, i32 0, i32 0)
  %86 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %69, i32 0, i32 0, i32 0)
  %87 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %70, i32 0, i32 0, i32 0)
  %88 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %71, i32 0, i32 0, i32 0)
  %89 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %73, i32 0, i32 0, i32 0)
  %90 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %.sroa.574.2, i32 0, i32 0, i32 0)
  %91 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %77, i32 0, i32 0, i32 0)
  %92 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %78, i32 0, i32 0, i32 0)
  %exitcond.not.i = icmp eq i64 %2, 0
  br i1 %exitcond.not.i, label %._crit_edge.i.loopexit, label %1

._crit_edge.i.loopexit:                           ; preds = %51
  ret void
}

declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat>, <8 x bfloat>, <16 x float>, i32 immarg, i32 immarg, i32 immarg)

attributes #0 = { "amdgpu-flat-work-group-size"="1, 256" "target-cpu"="gfx950" }
