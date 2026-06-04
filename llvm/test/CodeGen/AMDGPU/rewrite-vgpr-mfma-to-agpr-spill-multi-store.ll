; FIXME: the following line is added to cleanup bots, will be removed in weeks.
; RUN: rm -f %S/rewrite-vgpr-mfma-to-agpr-spill-multi-store.s
; REQUIRES: asserts
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -O3 \
; RUN:   -amdgpu-use-amdgpu-trackers=1 -verify-machineinstrs \
; RUN:   -stop-after=amdgpu-rewrite-agpr-copy-mfma \
; RUN:   -debug-only=amdgpu-rewrite-agpr-copy-mfma -filetype=null %s 2>&1 | FileCheck %s

; This test verifies that multiple connected live range components are not
; created by the VGPR-to-AGPR MFMA rewrite pass. If multiple components exist,
; the verifier would error out. Check that the unspilled interval was split
; into separate components.

; CHECK: Split unspilled interval into {{[0-9]+}} components

define amdgpu_kernel void @multi_store_spill_slot() #0 {
entry:
  br label %do.body

do.body:
  %c_block_tile.sroa.37.0 = phi float [ 0.000000e+00, %entry ], [ %c_block_tile.sroa.70.0, %do.body ]
  %c_block_tile.sroa.70.0 = phi float [ 0.000000e+00, %entry ], [ %ext0, %do.body ]
  %c_block_tile.sroa.961.0 = phi float [ 0.000000e+00, %entry ], [ %ext1, %do.body ]
  %c_block_tile.sroa.994.0 = phi float [ 0.000000e+00, %entry ], [ %ext2, %do.body ]
  %c_block_tile.sroa.1060.0 = phi float [ 0.000000e+00, %entry ], [ %ext3, %do.body ]
  %c_block_tile.sroa.1093.0 = phi float [ 0.000000e+00, %entry ], [ %c_block_tile.sroa.1060.0, %do.body ]
  %c_block_tile.sroa.1588.0 = phi float [ 0.000000e+00, %entry ], [ %ext4, %do.body ]
  %c_block_tile.sroa.1687.0 = phi float [ 0.000000e+00, %entry ], [ %ext5, %do.body ]
  %c_block_tile.sroa.2578.0 = phi float [ 0.000000e+00, %entry ], [ %ext6, %do.body ]
  %c_block_tile.sroa.2611.0 = phi float [ 0.000000e+00, %entry ], [ %ext7, %do.body ]
  %c_block_tile.sroa.2644.0 = phi float [ 0.000000e+00, %entry ], [ %ext8, %do.body ]
  %c_block_tile.sroa.2677.0 = phi float [ 0.000000e+00, %entry ], [ %ext9, %do.body ]
  %c_block_tile.sroa.3568.0 = phi float [ 0.000000e+00, %entry ], [ %ext10, %do.body ]
  %c_block_tile.sroa.3634.0 = phi float [ 0.000000e+00, %entry ], [ %ext11, %do.body ]
  %c_block_tile.sroa.3898.0 = phi float [ 0.000000e+00, %entry ], [ %ext12, %do.body ]
  %c_block_tile.sroa.3931.0 = phi float [ 0.000000e+00, %entry ], [ %ext13, %do.body ]
  %c_block_tile.sroa.4624.0 = phi float [ 0.000000e+00, %entry ], [ %ext14, %do.body ]
  %c_block_tile.sroa.4657.0 = phi float [ 0.000000e+00, %entry ], [ %ext15, %do.body ]
  %c_block_tile.sroa.5185.0 = phi float [ 0.000000e+00, %entry ], [ %ext16, %do.body ]
  %c_block_tile.sroa.5218.0 = phi float [ 0.000000e+00, %entry ], [ %ext17, %do.body ]
  %c_block_tile.sroa.5746.0 = phi float [ 0.000000e+00, %entry ], [ %ext18, %do.body ]
  %c_block_tile.sroa.5779.0 = phi float [ 0.000000e+00, %entry ], [ %c_block_tile.sroa.5746.0, %do.body ]
  %c_block_tile.sroa.5812.0 = phi float [ 0.000000e+00, %entry ], [ %ext19, %do.body ]
  %c_block_tile.sroa.6373.0 = phi float [ 0.000000e+00, %entry ], [ %c_block_tile.sroa.6406.0, %do.body ]
  %c_block_tile.sroa.6406.0 = phi float [ 0.000000e+00, %entry ], [ %ext20, %do.body ]
  %c_block_tile.sroa.7297.0 = phi float [ 0.000000e+00, %entry ], [ %ext21, %do.body ]
  %c_block_tile.sroa.7363.0 = phi float [ 0.000000e+00, %entry ], [ %ext22, %do.body ]
  %c_block_tile.sroa.7396.0 = phi float [ 0.000000e+00, %entry ], [ %ext23, %do.body ]
  %c_block_tile.sroa.7429.0 = phi float [ 0.000000e+00, %entry ], [ %ext24, %do.body ]
  %v0 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.37.0, i64 0
  %v1 = insertelement <16 x float> %v0, float %c_block_tile.sroa.70.0, i64 0
  %0 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v1, i32 0, i32 0, i32 0)
  %v2 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.961.0, i64 13
  %v3 = insertelement <16 x float> %v2, float %c_block_tile.sroa.994.0, i64 14
  %v4 = insertelement <16 x float> %v3, float 0x7FF8000000000000, i64 0
  %1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v4, i32 0, i32 0, i32 0)
  %v5 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.1588.0, i64 0
  %v6 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.1687.0, i64 3
  %v7 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.2578.0, i64 0
  %v8 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.2611.0, i64 0
  %v9 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.2677.0, i64 0
  %v10 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.3568.0, i64 0
  %v11 = insertelement <16 x float> splat (float 1.000000e+00), float %c_block_tile.sroa.3634.0, i64 0
  %2 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v11, i32 0, i32 0, i32 0)
  %v12 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.3898.0, i64 1
  %v13 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.4624.0, i64 0
  %v14 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.5185.0, i64 0
  %v15 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.5218.0, i64 0
  %v16 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.5746.0, i64 14
  %v17 = insertelement <16 x float> %v16, float %c_block_tile.sroa.5779.0, i64 15
  %3 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v17, i32 0, i32 0, i32 0)
  %v18 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.5812.0, i64 0
  %v19 = insertelement <16 x float> %v18, float %c_block_tile.sroa.5812.0, i64 0
  %v20 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.6373.0, i64 1
  %v21 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.7297.0, i64 0
  %v22 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.7363.0, i64 15
  %v23 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.7396.0, i64 0
  %v24 = insertelement <16 x float> %v23, float %c_block_tile.sroa.7429.0, i64 1
  %4 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v15, i32 0, i32 0, i32 0)
  %5 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %4, i32 0, i32 0, i32 0)
  %6 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v22, i32 0, i32 0, i32 0)
  %7 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %6, i32 0, i32 0, i32 0)
  %8 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v6, i32 0, i32 0, i32 0)
  %9 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %0, i32 0, i32 0, i32 0)
  %10 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %2, i32 0, i32 0, i32 0)
  %v25 = insertelement <16 x float> %v12, float %c_block_tile.sroa.3931.0, i64 0
  %11 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v25, i32 0, i32 0, i32 0)
  %v26 = insertelement <16 x float> %v20, float %c_block_tile.sroa.6406.0, i64 0
  %12 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v26, i32 0, i32 0, i32 0)
  %13 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v24, i32 0, i32 0, i32 0)
  %14 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %9, i32 0, i32 0, i32 0)
  %15 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %11, i32 0, i32 0, i32 0)
  %16 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %3, i32 0, i32 0, i32 0)
  %17 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %1, i32 0, i32 0, i32 0)
  %18 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %17, i32 0, i32 0, i32 0)
  %v27 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.1060.0, i64 0
  %v28 = insertelement <16 x float> %v27, float %c_block_tile.sroa.1093.0, i64 1
  %19 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v28, i32 0, i32 0, i32 0)
  %20 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %8, i32 0, i32 0, i32 0)
  %21 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %20, i32 0, i32 0, i32 0)
  %22 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %16, i32 0, i32 0, i32 0)
  %23 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %12, i32 0, i32 0, i32 0)
  %ext0 = extractelement <16 x float> %14, i64 0
  %24 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %18, i32 0, i32 0, i32 0)
  %ext1 = extractelement <16 x float> %24, i64 0
  %ext2 = extractelement <16 x float> %17, i64 0
  %25 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %19, i32 0, i32 0, i32 0)
  %ext3 = extractelement <16 x float> %25, i64 0
  %ext4 = extractelement <16 x float> %8, i64 0
  %ext5 = extractelement <16 x float> %21, i64 0
  %26 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v8, i32 0, i32 0, i32 0)
  %ext6 = extractelement <16 x float> %26, i64 0
  %27 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %26, i32 0, i32 0, i32 0)
  %ext7 = extractelement <16 x float> %27, i64 0
  %v29 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.2644.0, i64 0
  %28 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v29, i32 0, i32 0, i32 0)
  %29 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %28, i32 0, i32 0, i32 0)
  %ext8 = extractelement <16 x float> %29, i64 0
  %ext9 = extractelement <16 x float> %28, i64 0
  %ext10 = extractelement <16 x float> %2, i64 0
  %30 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %10, i32 0, i32 0, i32 0)
  %ext11 = extractelement <16 x float> %30, i64 0
  %ext12 = extractelement <16 x float> %11, i64 0
  %31 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %15, i32 0, i32 0, i32 0)
  %32 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %31, i32 0, i32 0, i32 0)
  %33 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> splat (half 0xH3C00), <16 x float> %32, i32 0, i32 0, i32 0)
  %ext13 = extractelement <16 x float> %33, i64 7
  %v30 = insertelement <16 x float> zeroinitializer, float %c_block_tile.sroa.4657.0, i64 0
  %34 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v30, i32 0, i32 0, i32 0)
  %ext14 = extractelement <16 x float> %34, i64 0
  %35 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %34, i32 0, i32 0, i32 0)
  %ext15 = extractelement <16 x float> %35, i64 0
  %ext16 = extractelement <16 x float> %4, i64 0
  %ext17 = extractelement <16 x float> %5, i64 0
  %ext18 = extractelement <16 x float> %22, i64 0
  %36 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %v19, i32 0, i32 0, i32 0)
  %ext19 = extractelement <16 x float> %36, i64 0
  %37 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %23, i32 0, i32 0, i32 0)
  %ext20 = extractelement <16 x float> %37, i64 0
  %ext21 = extractelement <16 x float> %6, i64 0
  %ext22 = extractelement <16 x float> %7, i64 0
  %ext23 = extractelement <16 x float> %13, i64 0
  %38 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> %13, i32 0, i32 0, i32 0)
  %ext24 = extractelement <16 x float> %38, i64 0
  br label %do.body
}

declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half>, <4 x half>, <16 x float>, i32 immarg, i32 immarg, i32 immarg)

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" "amdgpu-lds-size"="32768" }
