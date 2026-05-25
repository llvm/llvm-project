; REQUIRES: asserts
; RUN: llc -O3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
; RUN:   -stop-after=amdgpu-rewrite-agpr-copy-mfma \
; RUN:   -debug-only=amdgpu-rewrite-agpr-copy-mfma -filetype=null %s 2>&1 \
; RUN:   | FileCheck %s

; Regression test from https://github.com/llvm/llvm-project/issues/196671

; It is legal for a spill reload to not have a dominating spill store. When
; the AGPR rewrite pass unspills such a slot into a vreg, it must insert an
; IMPLICIT_DEF so the vreg has a def on all paths to such reloads. For the
; unspill to occur, the physical register must be interference-free for the
; extended live range produced by the IMPLICIT_DEF. In this case, no physical
; register is found that satisfies the interference property, so the unspill
; does not occur.

; CHECK: Selected IMPLICIT_DEF block for SS#{{[0-9]+}}: %bb.{{[0-9]+}} ({{[0-9]+}} problematic reload block(s))
; CHECK: IMPLICIT_DEF extension interferes

define amdgpu_kernel void @rewrite_vgpr_mfma_to_agpr_spill_joint_dom(i1 %arg, <16 x float> %.sroa.366.2) #0 {
.lr.ph.i:
  br label %bb

bb:                                               ; preds = %bb49, %.lr.ph.i
  %.sroa.01121.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i78, %bb49 ]
  %.sroa.54.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i80, %bb49 ]
  %.sroa.106.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i82, %bb49 ]
  %.sroa.1581182.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i83, %bb49 ]
  %.sroa.210.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i84, %bb49 ]
  %.sroa.262.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i85, %bb49 ]
  %.sroa.314.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i86, %bb49 ]
  %.sroa.366.21 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i70, %bb49 ]
  %.sroa.418.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i87, %bb49 ]
  %.sroa.470.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i72, %bb49 ]
  %.sroa.522.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i73, %bb49 ]
  %.sroa.574.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i88, %bb49 ]
  %.sroa.626.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i74, %bb49 ]
  %.sroa.678.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i89, %bb49 ]
  %.sroa.730.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i90, %bb49 ]
  %.sroa.782.2 = phi <16 x float> [ zeroinitializer, %.lr.ph.i ], [ %i77, %bb49 ]
  %i = phi i64 [ 0, %.lr.ph.i ], [ 1, %bb49 ]
  br i1 %arg, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  store <4 x i32> zeroinitializer, ptr addrspace(5) null, align 16
  br label %bb49

bb2:                                              ; preds = %bb
  %i3 = fmul <16 x float> %.sroa.01121.2, zeroinitializer
  %i4 = fmul <16 x float> %.sroa.54.2, zeroinitializer
  %i5 = fmul <16 x float> %.sroa.106.2, zeroinitializer
  %i6 = fmul <16 x float> %.sroa.1581182.2, zeroinitializer
  %i7 = fmul <16 x float> %.sroa.210.2, zeroinitializer
  %i8 = fmul <16 x float> %.sroa.262.2, zeroinitializer
  %i9 = fmul <16 x float> %.sroa.314.2, zeroinitializer
  %i10 = fmul <16 x float> %.sroa.366.21, zeroinitializer
  %i11 = fmul <16 x float> %.sroa.418.2, zeroinitializer
  %i12 = fmul <16 x float> %.sroa.470.2, zeroinitializer
  %i13 = fmul <16 x float> %.sroa.522.2, zeroinitializer
  %i14 = fmul <16 x float> %.sroa.574.2, zeroinitializer
  %i15 = fmul <16 x float> %.sroa.626.2, zeroinitializer
  %i16 = fmul <16 x float> %.sroa.678.2, zeroinitializer
  %i17 = fmul <16 x float> %.sroa.730.2, zeroinitializer
  %i18 = fmul <16 x float> %.sroa.782.2, zeroinitializer
  %i19 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i3, i32 0, i32 0, i32 0)
  %i20 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i4, i32 0, i32 0, i32 0)
  %i21 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i5, i32 0, i32 0, i32 0)
  %i22 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i6, i32 0, i32 0, i32 0)
  %i23 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i7, i32 0, i32 0, i32 0)
  %i24 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i8, i32 0, i32 0, i32 0)
  %i25 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i9, i32 0, i32 0, i32 0)
  %i26 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i10, i32 0, i32 0, i32 0)
  %i27 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i11, i32 0, i32 0, i32 0)
  %i28 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i12, i32 0, i32 0, i32 0)
  %i29 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i13, i32 0, i32 0, i32 0)
  %i30 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i15, i32 0, i32 0, i32 0)
  %i31 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i16, i32 0, i32 0, i32 0)
  %i32 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i17, i32 0, i32 0, i32 0)
  %i33 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i18, i32 0, i32 0, i32 0)
  %i34 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i19, i32 0, i32 0, i32 0)
  %i35 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i20, i32 0, i32 0, i32 0)
  %i36 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i21, i32 0, i32 0, i32 0)
  %i37 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i22, i32 0, i32 0, i32 0)
  %i38 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i23, i32 0, i32 0, i32 0)
  %i39 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i24, i32 0, i32 0, i32 0)
  %i40 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i25, i32 0, i32 0, i32 0)
  %i41 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i26, i32 0, i32 0, i32 0)
  %i42 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i27, i32 0, i32 0, i32 0)
  %i43 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i28, i32 0, i32 0, i32 0)
  %i44 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i29, i32 0, i32 0, i32 0)
  %i45 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i30, i32 0, i32 0, i32 0)
  %i46 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i31, i32 0, i32 0, i32 0)
  %i47 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i32, i32 0, i32 0, i32 0)
  %i48 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i33, i32 0, i32 0, i32 0)
  br label %bb49

bb49:                                             ; preds = %bb2, %bb1
  %.sroa.01121.3 = phi <16 x float> [ %.sroa.01121.2, %bb1 ], [ %i34, %bb2 ]
  %.sroa.54.3 = phi <16 x float> [ %.sroa.54.2, %bb1 ], [ %i35, %bb2 ]
  %.sroa.106.3 = phi <16 x float> [ %.sroa.106.2, %bb1 ], [ %i36, %bb2 ]
  %.sroa.1581182.3 = phi <16 x float> [ %.sroa.1581182.2, %bb1 ], [ %i37, %bb2 ]
  %.sroa.210.3 = phi <16 x float> [ %.sroa.210.2, %bb1 ], [ %i38, %bb2 ]
  %.sroa.262.3 = phi <16 x float> [ %.sroa.262.2, %bb1 ], [ %i39, %bb2 ]
  %.sroa.314.3 = phi <16 x float> [ %.sroa.314.2, %bb1 ], [ %i40, %bb2 ]
  %.sroa.366.3 = phi <16 x float> [ %.sroa.366.2, %bb1 ], [ %i41, %bb2 ]
  %.sroa.418.3 = phi <16 x float> [ %.sroa.418.2, %bb1 ], [ %i42, %bb2 ]
  %.sroa.470.3 = phi <16 x float> [ %.sroa.470.2, %bb1 ], [ %i43, %bb2 ]
  %.sroa.522.3 = phi <16 x float> [ %.sroa.522.2, %bb1 ], [ %i44, %bb2 ]
  %.sroa.574.3 = phi <16 x float> [ %.sroa.574.2, %bb1 ], [ %i14, %bb2 ]
  %.sroa.626.3 = phi <16 x float> [ zeroinitializer, %bb1 ], [ %i45, %bb2 ]
  %.sroa.678.3 = phi <16 x float> [ %.sroa.678.2, %bb1 ], [ %i46, %bb2 ]
  %.sroa.730.3 = phi <16 x float> [ %.sroa.730.2, %bb1 ], [ %i47, %bb2 ]
  %.sroa.782.3 = phi <16 x float> [ %.sroa.782.2, %bb1 ], [ %i48, %bb2 ]
  %i50 = fmul <16 x float> %.sroa.01121.3, zeroinitializer
  %i51 = fmul <16 x float> %.sroa.54.3, zeroinitializer
  %i52 = fmul <16 x float> %.sroa.106.3, zeroinitializer
  %i53 = fmul <16 x float> %.sroa.1581182.3, zeroinitializer
  %i54 = fmul <16 x float> %.sroa.210.3, zeroinitializer
  %i55 = fmul <16 x float> %.sroa.262.3, zeroinitializer
  %i56 = fmul <16 x float> %.sroa.314.3, zeroinitializer
  %i57 = fmul <16 x float> %.sroa.366.3, zeroinitializer
  %i58 = fmul <16 x float> %.sroa.418.3, zeroinitializer
  %i59 = fmul <16 x float> %.sroa.470.3, zeroinitializer
  %i60 = fmul <16 x float> %.sroa.522.3, zeroinitializer
  %i61 = fmul <16 x float> %.sroa.574.3, zeroinitializer
  %i62 = fmul <16 x float> %.sroa.626.3, zeroinitializer
  %i63 = fmul <16 x float> %.sroa.678.3, zeroinitializer
  %i64 = fmul <16 x float> %.sroa.730.3, zeroinitializer
  %i65 = fmul <16 x float> %.sroa.782.3, zeroinitializer
  %i66 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i53, i32 0, i32 0, i32 0)
  %i67 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i54, i32 0, i32 0, i32 0)
  %i68 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i55, i32 0, i32 0, i32 0)
  %i69 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i56, i32 0, i32 0, i32 0)
  %i70 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i57, i32 0, i32 0, i32 0)
  %i71 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i58, i32 0, i32 0, i32 0)
  %i72 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i59, i32 0, i32 0, i32 0)
  %i73 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i60, i32 0, i32 0, i32 0)
  %i74 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i62, i32 0, i32 0, i32 0)
  %i75 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i63, i32 0, i32 0, i32 0)
  %i76 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i64, i32 0, i32 0, i32 0)
  %i77 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i65, i32 0, i32 0, i32 0)
  %i78 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i50, i32 0, i32 0, i32 0)
  %i79 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i51, i32 0, i32 0, i32 0)
  %i80 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i79, i32 0, i32 0, i32 0)
  %i81 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i52, i32 0, i32 0, i32 0)
  %i82 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i81, i32 0, i32 0, i32 0)
  %i83 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i66, i32 0, i32 0, i32 0)
  %i84 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i67, i32 0, i32 0, i32 0)
  %i85 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i68, i32 0, i32 0, i32 0)
  %i86 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i69, i32 0, i32 0, i32 0)
  %i87 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i71, i32 0, i32 0, i32 0)
  %i88 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %.sroa.574.2, i32 0, i32 0, i32 0)
  %i89 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i75, i32 0, i32 0, i32 0)
  %i90 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %i76, i32 0, i32 0, i32 0)
  %exitcond.not.i = icmp eq i64 %i, 0
  br i1 %exitcond.not.i, label %._crit_edge.i.loopexit, label %bb

._crit_edge.i.loopexit:                           ; preds = %bb49
  ret void
}

declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat>, <8 x bfloat>, <16 x float>, i32 immarg, i32 immarg, i32 immarg)

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }
