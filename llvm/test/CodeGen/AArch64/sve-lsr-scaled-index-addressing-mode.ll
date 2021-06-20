; RUN: opt -S -loop-reduce < %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s --check-prefix=ASM
; Note: To update this test, please run utils/update_test_checks.py and utils/update_llc_test_checks.py separately on opt/llc run line.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linux-gnu"

; These tests check that the IR coming out of LSR does not cast input/output pointer from i16* to i8* type.
; And scaled-index addressing mode is leveraged in the generated assembly, i.e. ld1h { z1.h }, p0/z, [x0, x8, lsl #1].

define void @ld_st_nxv8i16(i16* %in, i16* %out) {
; IR-LABEL: @ld_st_nxv8i16(
; IR-NEXT:  entry:
; IR-NEXT:    br label [[LOOP_PH:%.*]]
; IR:       loop.ph:
; IR-NEXT:    [[P_VEC_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i16> undef, i16 3, i32 0
; IR-NEXT:    [[P_VEC_SPLAT:%.*]] = shufflevector <vscale x 8 x i16> [[P_VEC_SPLATINSERT]], <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
; IR-NEXT:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
; IR-NEXT:    [[SCALED_VF:%.*]] = shl i64 [[VSCALE]], 3
; IR-NEXT:    br label [[LOOP:%.*]]
; IR:       loop:
; IR-NEXT:    [[INDVAR:%.*]] = phi i64 [ 0, [[LOOP_PH]] ], [ [[INDVAR_NEXT:%.*]], [[LOOP]] ]
; IR-NEXT:    [[SCEVGEP2:%.*]] = getelementptr i16, i16* [[IN:%.*]], i64 [[INDVAR]]
; IR-NEXT:    [[SCEVGEP23:%.*]] = bitcast i16* [[SCEVGEP2]] to <vscale x 8 x i16>*
; IR-NEXT:    [[SCEVGEP:%.*]] = getelementptr i16, i16* [[OUT:%.*]], i64 [[INDVAR]]
; IR-NEXT:    [[SCEVGEP1:%.*]] = bitcast i16* [[SCEVGEP]] to <vscale x 8 x i16>*
; IR-NEXT:    [[VAL:%.*]] = load <vscale x 8 x i16>, <vscale x 8 x i16>* [[SCEVGEP23]], align 16
; IR-NEXT:    [[ADDP_VEC:%.*]] = add <vscale x 8 x i16> [[VAL]], [[P_VEC_SPLAT]]
; IR-NEXT:    store <vscale x 8 x i16> [[ADDP_VEC]], <vscale x 8 x i16>* [[SCEVGEP1]], align 16
; IR-NEXT:    [[INDVAR_NEXT]] = add nsw i64 [[INDVAR]], [[SCALED_VF]]
; IR-NEXT:    [[EXIT_COND:%.*]] = icmp eq i64 [[INDVAR_NEXT]], 1024
; IR-NEXT:    br i1 [[EXIT_COND]], label [[LOOP_EXIT:%.*]], label [[LOOP]]
; IR:       loop.exit:
; IR-NEXT:    br label [[EXIT:%.*]]
; IR:       exit:
; IR-NEXT:    ret void
;
; ASM-LABEL: ld_st_nxv8i16:
; ASM:       // %bb.0: // %entry
; ASM-NEXT:    mov x8, xzr
; ASM-NEXT:    mov z0.h, #3 // =0x3
; ASM-NEXT:    cnth x9
; ASM-NEXT:    ptrue p0.h
; ASM-NEXT:  .LBB0_1: // %loop
; ASM-NEXT:    // =>This Inner Loop Header: Depth=1
; ASM-NEXT:    ld1h { z1.h }, p0/z, [x0, x8, lsl #1]
; ASM-NEXT:    add z1.h, z1.h, z0.h
; ASM-NEXT:    st1h { z1.h }, p0, [x1, x8, lsl #1]
; ASM-NEXT:    add x8, x8, x9
; ASM-NEXT:    cmp x8, #1024 // =1024
; ASM-NEXT:    b.ne .LBB0_1
; ASM-NEXT:  // %bb.2: // %exit
; ASM-NEXT:    ret
entry:
  br label %loop.ph

loop.ph:
  %p_vec.splatinsert = insertelement <vscale x 8 x i16> undef, i16 3, i32 0
  %p_vec.splat = shufflevector <vscale x 8 x i16> %p_vec.splatinsert, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %vscale = call i64 @llvm.vscale.i64()
  %scaled_vf = shl i64 %vscale, 3
  br label %loop

loop:                                             ; preds = %loop, %loop.ph
  %indvar = phi i64 [ 0, %loop.ph ], [ %indvar.next, %loop ]
  %ptr.in = getelementptr inbounds i16, i16* %in, i64 %indvar
  %ptr.out = getelementptr inbounds i16, i16* %out, i64 %indvar
  %in.ptrcast = bitcast i16* %ptr.in to <vscale x 8 x i16>*
  %out.ptrcast = bitcast i16* %ptr.out to <vscale x 8 x i16>*
  %val = load <vscale x 8 x i16>, <vscale x 8 x i16>* %in.ptrcast, align 16
  %addp_vec = add <vscale x 8 x i16> %val, %p_vec.splat
  store <vscale x 8 x i16> %addp_vec, <vscale x 8 x i16>* %out.ptrcast, align 16
  %indvar.next = add nsw i64 %indvar, %scaled_vf
  %exit.cond = icmp eq i64 %indvar.next, 1024
  br i1 %exit.cond, label %loop.exit, label %loop

loop.exit:                                        ; preds = %loop
  br label %exit

exit:
  ret void
}

define void @masked_ld_st_nxv8i16(i16* %in, i16* %out, i64 %n) {
; IR-LABEL: @masked_ld_st_nxv8i16(
; IR-NEXT:  entry:
; IR-NEXT:    br label [[LOOP_PH:%.*]]
; IR:       loop.ph:
; IR-NEXT:    [[P_VEC_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i16> undef, i16 3, i32 0
; IR-NEXT:    [[P_VEC_SPLAT:%.*]] = shufflevector <vscale x 8 x i16> [[P_VEC_SPLATINSERT]], <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
; IR-NEXT:    [[PTRUE_VEC_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i1> undef, i1 true, i32 0
; IR-NEXT:    [[PTRUE_VEC_SPLAT:%.*]] = shufflevector <vscale x 8 x i1> [[PTRUE_VEC_SPLATINSERT]], <vscale x 8 x i1> undef, <vscale x 8 x i32> zeroinitializer
; IR-NEXT:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
; IR-NEXT:    [[SCALED_VF:%.*]] = shl i64 [[VSCALE]], 3
; IR-NEXT:    br label [[LOOP:%.*]]
; IR:       loop:
; IR-NEXT:    [[INDVAR:%.*]] = phi i64 [ 0, [[LOOP_PH]] ], [ [[INDVAR_NEXT:%.*]], [[LOOP]] ]
; IR-NEXT:    [[SCEVGEP2:%.*]] = getelementptr i16, i16* [[IN:%.*]], i64 [[INDVAR]]
; IR-NEXT:    [[SCEVGEP23:%.*]] = bitcast i16* [[SCEVGEP2]] to <vscale x 8 x i16>*
; IR-NEXT:    [[SCEVGEP:%.*]] = getelementptr i16, i16* [[OUT:%.*]], i64 [[INDVAR]]
; IR-NEXT:    [[SCEVGEP1:%.*]] = bitcast i16* [[SCEVGEP]] to <vscale x 8 x i16>*
; IR-NEXT:    [[VAL:%.*]] = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16.p0nxv8i16(<vscale x 8 x i16>* [[SCEVGEP23]], i32 4, <vscale x 8 x i1> [[PTRUE_VEC_SPLAT]], <vscale x 8 x i16> undef)
; IR-NEXT:    [[ADDP_VEC:%.*]] = add <vscale x 8 x i16> [[VAL]], [[P_VEC_SPLAT]]
; IR-NEXT:    call void @llvm.masked.store.nxv8i16.p0nxv8i16(<vscale x 8 x i16> [[ADDP_VEC]], <vscale x 8 x i16>* [[SCEVGEP1]], i32 4, <vscale x 8 x i1> [[PTRUE_VEC_SPLAT]])
; IR-NEXT:    [[INDVAR_NEXT]] = add nsw i64 [[INDVAR]], [[SCALED_VF]]
; IR-NEXT:    [[EXIT_COND:%.*]] = icmp eq i64 [[N:%.*]], [[INDVAR_NEXT]]
; IR-NEXT:    br i1 [[EXIT_COND]], label [[LOOP_EXIT:%.*]], label [[LOOP]]
; IR:       loop.exit:
; IR-NEXT:    br label [[EXIT:%.*]]
; IR:       exit:
; IR-NEXT:    ret void
;
; ASM-LABEL: masked_ld_st_nxv8i16:
; ASM:       // %bb.0: // %entry
; ASM-NEXT:    mov x8, xzr
; ASM-NEXT:    mov z0.h, #3 // =0x3
; ASM-NEXT:    ptrue p0.h
; ASM-NEXT:    cnth x9
; ASM-NEXT:  .LBB1_1: // %loop
; ASM-NEXT:    // =>This Inner Loop Header: Depth=1
; ASM-NEXT:    ld1h { z1.h }, p0/z, [x0, x8, lsl #1]
; ASM-NEXT:    add z1.h, z1.h, z0.h
; ASM-NEXT:    st1h { z1.h }, p0, [x1, x8, lsl #1]
; ASM-NEXT:    add x8, x8, x9
; ASM-NEXT:    cmp x2, x8
; ASM-NEXT:    b.ne .LBB1_1
; ASM-NEXT:  // %bb.2: // %exit
; ASM-NEXT:    ret
entry:
  br label %loop.ph

loop.ph:
  %p_vec.splatinsert = insertelement <vscale x 8 x i16> undef, i16 3, i32 0
  %p_vec.splat = shufflevector <vscale x 8 x i16> %p_vec.splatinsert, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %ptrue_vec.splatinsert = insertelement <vscale x 8 x i1> undef, i1 true, i32 0
  %ptrue_vec.splat = shufflevector <vscale x 8 x i1> %ptrue_vec.splatinsert, <vscale x 8 x i1> undef, <vscale x 8 x i32> zeroinitializer
  %vscale = call i64 @llvm.vscale.i64()
  %scaled_vf = shl i64 %vscale, 3
  br label %loop

loop:                                             ; preds = %loop, %loop.ph
  %indvar = phi i64 [ 0, %loop.ph ], [ %indvar.next, %loop ]
  %ptr.in = getelementptr inbounds i16, i16* %in, i64 %indvar
  %ptr.out = getelementptr inbounds i16, i16* %out, i64 %indvar
  %in.ptrcast = bitcast i16* %ptr.in to <vscale x 8 x i16>*
  %out.ptrcast = bitcast i16* %ptr.out to <vscale x 8 x i16>*
  %val = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16.p0nxv8i16(<vscale x 8 x i16>* %in.ptrcast, i32 4, <vscale x 8 x i1> %ptrue_vec.splat, <vscale x 8 x i16> undef)
  %addp_vec = add <vscale x 8 x i16> %val, %p_vec.splat
  call void @llvm.masked.store.nxv8i16.p0nxv8i16(<vscale x 8 x i16> %addp_vec, <vscale x 8 x i16>* %out.ptrcast, i32 4, <vscale x 8 x i1> %ptrue_vec.splat)
  %indvar.next = add nsw i64 %indvar, %scaled_vf
  %exit.cond = icmp eq i64 %indvar.next, %n
  br i1 %exit.cond, label %loop.exit, label %loop

loop.exit:                                        ; preds = %loop
  br label %exit

exit:
  ret void
}

declare i64 @llvm.vscale.i64()

declare <vscale x 8 x i16> @llvm.masked.load.nxv8i16.p0nxv8i16(<vscale x 8 x i16>*, i32 immarg, <vscale x 8 x i1>, <vscale x 8 x i16>)

declare void @llvm.masked.store.nxv8i16.p0nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>*, i32 immarg, <vscale x 8 x i1>)
