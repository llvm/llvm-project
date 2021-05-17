; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck %s

; Test that unused lanes in the s_xor result are masked out with v_cndmask.

; CHECK-LABEL: combine_add_zext_xor:
; CHECK: s_xor_b32 [[RESULT:s[0-9]+]]
; CHECK: v_cndmask_b32_e64 [[ARG:v[0-9]+]], 0, 1, [[RESULT]]
; CHECK: v_add_nc_u32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[ARG]]
define i32 @combine_add_zext_xor() {
.entry:
  br label %.a

.a:                                               ; preds = %bb9, %.entry
  %.2 = phi i32 [ 0, %.entry ], [ %i11, %bb9 ]
  br i1 undef, label %bb9, label %bb

bb:                                               ; preds = %.a
  %.i3 = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> undef, i32 %.2, i32 64, i32 1)
  %i5 = icmp eq i32 %.i3, 0
  br label %bb9

bb9:                                              ; preds = %bb, %.a
  %.2.0.in.in = phi i1 [ %i5, %bb ], [ undef, %.a ]
  %.2.0.in = xor i1 %.2.0.in.in, true
  %.2.0 = zext i1 %.2.0.in to i32
  %i11 = add i32 %.2, %.2.0
  %i12 = icmp sgt i32 %.2, -1050
  br i1 %i12, label %.a, label %.exit

.exit:                                            ; preds = %bb9
  ret i32 %.2.0
}

; Test that unused lanes in the s_xor result are masked out with v_cndmask.

; CHECK-LABEL: combine_sub_zext_xor:
; CHECK: s_xor_b32 [[RESULT:s[0-9]+]]
; CHECK: v_cndmask_b32_e64 [[ARG:v[0-9]+]], 0, 1, [[RESULT]]
; CHECK: v_sub_nc_u32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[ARG]]

define i32 @combine_sub_zext_xor() {
.entry:
  br label %.a

.a:                                               ; preds = %bb9, %.entry
  %.2 = phi i32 [ 0, %.entry ], [ %i11, %bb9 ]
  br i1 undef, label %bb9, label %bb

bb:                                               ; preds = %.a
  %.i3 = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> undef, i32 %.2, i32 64, i32 1)
  %i5 = icmp eq i32 %.i3, 0
  br label %bb9

bb9:                                              ; preds = %bb, %.a
  %.2.0.in.in = phi i1 [ %i5, %bb ], [ undef, %.a ]
  %.2.0.in = xor i1 %.2.0.in.in, true
  %.2.0 = zext i1 %.2.0.in to i32
  %i11 = sub i32 %.2, %.2.0
  %i12 = icmp sgt i32 %.2, -1050
  br i1 %i12, label %.a, label %.exit

.exit:                                            ; preds = %bb9
  ret i32 %.2.0
}

; Test that unused lanes in the s_or result are masked out with v_cndmask.

; CHECK-LABEL: combine_add_zext_or:
; CHECK: s_or_b32 [[RESULT:s[0-9]+]]
; CHECK: v_cndmask_b32_e64 [[ARG:v[0-9]+]], 0, 1, [[RESULT]]
; CHECK: v_add_nc_u32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[ARG]]

define i32 @combine_add_zext_or() {
.entry:
  br label %.a

.a:                                               ; preds = %bb9, %.entry
  %.2 = phi i32 [ 0, %.entry ], [ %i11, %bb9 ]
  br i1 undef, label %bb9, label %bb

bb:                                               ; preds = %.a
  %.i3 = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> undef, i32 %.2, i32 64, i32 1)
  %i5 = icmp eq i32 %.i3, 0
  br label %bb9

bb9:                                              ; preds = %bb, %.a
  %.2.0.in.in = phi i1 [ %i5, %bb ], [ undef, %.a ]
  %t = icmp sgt i32 %.2, -1050
  %.2.0.in = or i1 %.2.0.in.in, %t
  %.2.0 = zext i1 %.2.0.in to i32
  %i11 = add i32 %.2, %.2.0
  %i12 = icmp sgt i32 %.2, -1050
  br i1 %i12, label %.a, label %.exit

.exit:                                            ; preds = %bb9
  ret i32 %.2.0
}

; Test that unused lanes in the s_or result are masked out with v_cndmask.

; CHECK-LABEL: combine_sub_zext_or:
; CHECK: s_or_b32 [[RESULT:s[0-9]+]]
; CHECK: v_cndmask_b32_e64 [[ARG:v[0-9]+]], 0, 1, [[RESULT]]
; CHECK: v_sub_nc_u32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[ARG]]

define i32 @combine_sub_zext_or() {
.entry:
  br label %.a

.a:                                               ; preds = %bb9, %.entry
  %.2 = phi i32 [ 0, %.entry ], [ %i11, %bb9 ]
  br i1 undef, label %bb9, label %bb

bb:                                               ; preds = %.a
  %.i3 = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> undef, i32 %.2, i32 64, i32 1)
  %i5 = icmp eq i32 %.i3, 0
  br label %bb9

bb9:                                              ; preds = %bb, %.a
  %.2.0.in.in = phi i1 [ %i5, %bb ], [ undef, %.a ]
  %t = icmp sgt i32 %.2, -1050
  %.2.0.in = or i1 %.2.0.in.in, %t
  %.2.0 = zext i1 %.2.0.in to i32
  %i11 = sub i32 %.2, %.2.0
  %i12 = icmp sgt i32 %.2, -1050
  br i1 %i12, label %.a, label %.exit

.exit:                                            ; preds = %bb9
  ret i32 %.2.0
}

; Test that unused lanes in the s_and result are masked out with v_cndmask.

; CHECK-LABEL: combine_add_zext_and:
; CHECK: s_and_b32 [[RESULT:s[0-9]+]]
; CHECK: v_cndmask_b32_e64 [[ARG:v[0-9]+]], 0, 1, [[RESULT]]
; CHECK: v_add_nc_u32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[ARG]]

define i32 @combine_add_zext_and() {
.entry:
  br label %.a

.a:                                               ; preds = %bb9, %.entry
  %.2 = phi i32 [ 0, %.entry ], [ %i11, %bb9 ]
  br i1 undef, label %bb9, label %bb

bb:                                               ; preds = %.a
  %.i3 = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> undef, i32 %.2, i32 64, i32 1)
  %i5 = icmp eq i32 %.i3, 0
  br label %bb9

bb9:                                              ; preds = %bb, %.a
  %.2.0.in.in = phi i1 [ %i5, %bb ], [ undef, %.a ]
  %t = icmp sgt i32 %.2, -1050
  %.2.0.in = and i1 %.2.0.in.in, %t
  %.2.0 = zext i1 %.2.0.in to i32
  %i11 = add i32 %.2, %.2.0
  %i12 = icmp sgt i32 %.2, -1050
  br i1 %i12, label %.a, label %.exit

.exit:                                            ; preds = %bb9
  ret i32 %.2.0
}

; Test that unused lanes in the s_and result are masked out with v_cndmask.

; CHECK-LABEL: combine_sub_zext_and:
; CHECK: s_and_b32 [[RESULT:s[0-9]+]]
; CHECK: v_cndmask_b32_e64 [[ARG:v[0-9]+]], 0, 1, [[RESULT]]
; CHECK: v_sub_nc_u32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[ARG]]

define i32 @combine_sub_zext_and() {
.entry:
  br label %.a

.a:                                               ; preds = %bb9, %.entry
  %.2 = phi i32 [ 0, %.entry ], [ %i11, %bb9 ]
  br i1 undef, label %bb9, label %bb

bb:                                               ; preds = %.a
  %.i3 = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> undef, i32 %.2, i32 64, i32 1)
  %i5 = icmp eq i32 %.i3, 0
  br label %bb9

bb9:                                              ; preds = %bb, %.a
  %.2.0.in.in = phi i1 [ %i5, %bb ], [ undef, %.a ]
  %t = icmp sgt i32 %.2, -1050
  %.2.0.in = and i1 %.2.0.in.in, %t
  %.2.0 = zext i1 %.2.0.in to i32
  %i11 = sub i32 %.2, %.2.0
  %i12 = icmp sgt i32 %.2, -1050
  br i1 %i12, label %.a, label %.exit

.exit:                                            ; preds = %bb9
  ret i32 %.2.0
}


; Function Attrs: nounwind readonly willreturn
declare i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32>, i32, i32, i32 immarg) #0

attributes #0 = { nounwind readonly willreturn }

