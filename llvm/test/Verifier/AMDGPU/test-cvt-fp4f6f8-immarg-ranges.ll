; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

; --------------------------------------------------------------------
; llvm.amdgcn.cvt.sr.fp8.f16 - byte_sel out of range
; --------------------------------------------------------------------

; CHECK: immarg value 4 out of range [0, 4)
; CHECK-NEXT: %result = call i32 @llvm.amdgcn.cvt.sr.fp8.f16(half %src, i32 %seed, i32 %old, i32 4)
define i32 @test_cvt_sr_fp8_f16_byte_sel_out_of_range(half %src, i32 %seed, i32 %old) {
  %result = call i32 @llvm.amdgcn.cvt.sr.fp8.f16(half %src, i32 %seed, i32 %old, i32 4)
  ret i32 %result
}

; CHECK: immarg value 10 out of range [0, 4)
; CHECK-NEXT: %result = call i32 @llvm.amdgcn.cvt.sr.fp8.f16(half %src, i32 %seed, i32 %old, i32 10)
define i32 @test_cvt_sr_fp8_f16_byte_sel_way_out_of_range(half %src, i32 %seed, i32 %old) {
  %result = call i32 @llvm.amdgcn.cvt.sr.fp8.f16(half %src, i32 %seed, i32 %old, i32 10)
  ret i32 %result
}

; --------------------------------------------------------------------
; llvm.amdgcn.cvt.sr.bf8.f16 - byte_sel out of range
; --------------------------------------------------------------------

; CHECK: immarg value 4 out of range [0, 4)
; CHECK-NEXT: %result = call i32 @llvm.amdgcn.cvt.sr.bf8.f16(half %src, i32 %seed, i32 %old, i32 4)
define i32 @test_cvt_sr_bf8_f16_byte_sel_out_of_range(half %src, i32 %seed, i32 %old) {
  %result = call i32 @llvm.amdgcn.cvt.sr.bf8.f16(half %src, i32 %seed, i32 %old, i32 4)
  ret i32 %result
}

; --------------------------------------------------------------------
; llvm.amdgcn.cvt.scale.pk8.f16.fp8 - scale_sel out of range
; --------------------------------------------------------------------

; CHECK: immarg value 16 out of range [0, 16)
; CHECK-NEXT: %result = call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp8(<2 x i32> %src, i32 0, i32 16)
define <8 x half> @test_cvt_scale_pk8_f16_fp8_scale_sel_out_of_range(<2 x i32> %src) {
  %result = call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp8(<2 x i32> %src, i32 0, i32 16)
  ret <8 x half> %result
}

; CHECK: immarg value 100 out of range [0, 16)
; CHECK-NEXT: %result = call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp8(<2 x i32> %src, i32 0, i32 100)
define <8 x half> @test_cvt_scale_pk8_f16_fp8_scale_sel_way_out_of_range(<2 x i32> %src) {
  %result = call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp8(<2 x i32> %src, i32 0, i32 100)
  ret <8 x half> %result
}

; --------------------------------------------------------------------
; llvm.amdgcn.cvt.scalef32.f32.fp8 - src_sel out of range
; --------------------------------------------------------------------

; CHECK: immarg value 4 out of range [0, 4)
; CHECK-NEXT: %result = call float @llvm.amdgcn.cvt.scalef32.f32.fp8(i32 %src, float %scale, i32 4)
define float @test_cvt_scalef32_f32_fp8_src_sel_out_of_range(i32 %src, float %scale) {
  %result = call float @llvm.amdgcn.cvt.scalef32.f32.fp8(i32 %src, float %scale, i32 4)
  ret float %result
}

; CHECK: immarg value 7 out of range [0, 4)
; CHECK-NEXT: %result = call float @llvm.amdgcn.cvt.scalef32.f32.fp8(i32 %src, float %scale, i32 7)
define float @test_cvt_scalef32_f32_fp8_src_sel_way_out_of_range(i32 %src, float %scale) {
  %result = call float @llvm.amdgcn.cvt.scalef32.f32.fp8(i32 %src, float %scale, i32 7)
  ret float %result
}

; --------------------------------------------------------------------
; llvm.amdgcn.cvt.scalef32.f16.fp8 - src_sel_index out of range
; --------------------------------------------------------------------

; CHECK: immarg value 4 out of range [0, 4)
; CHECK-NEXT: %result = call <2 x half> @llvm.amdgcn.cvt.scalef32.f16.fp8(<2 x half> %old, i32 %src, float %scale, i32 4, i1 false)
define <2 x half> @test_cvt_scalef32_f16_fp8_src_sel_index_out_of_range(<2 x half> %old, i32 %src, float %scale) {
  %result = call <2 x half> @llvm.amdgcn.cvt.scalef32.f16.fp8(<2 x half> %old, i32 %src, float %scale, i32 4, i1 false)
  ret <2 x half> %result
}

; CHECK: immarg value 15 out of range [0, 4)
; CHECK-NEXT: %result = call <2 x half> @llvm.amdgcn.cvt.scalef32.f16.fp8(<2 x half> %old, i32 %src, float %scale, i32 15, i1 true)
define <2 x half> @test_cvt_scalef32_f16_fp8_src_sel_index_way_out_of_range(<2 x half> %old, i32 %src, float %scale) {
  %result = call <2 x half> @llvm.amdgcn.cvt.scalef32.f16.fp8(<2 x half> %old, i32 %src, float %scale, i32 15, i1 true)
  ret <2 x half> %result
}

; --------------------------------------------------------------------
; llvm.amdgcn.cvt.scalef32.pk.fp4.f32 - dst_sel_index out of range
; --------------------------------------------------------------------

; CHECK: immarg value 4 out of range [0, 4)
; CHECK-NEXT: %result = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %old, float %src0, float %src1, float %scale, i32 4)
define i32 @test_cvt_scalef32_pk_fp4_f32_dst_sel_index_out_of_range(i32 %old, float %src0, float %src1, float %scale) {
  %result = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %old, float %src0, float %src1, float %scale, i32 4)
  ret i32 %result
}

; CHECK: immarg value 8 out of range [0, 4)
; CHECK-NEXT: %result = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %old, float %src0, float %src1, float %scale, i32 8)
define i32 @test_cvt_scalef32_pk_fp4_f32_dst_sel_index_way_out_of_range(i32 %old, float %src0, float %src1, float %scale) {
  %result = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %old, float %src0, float %src1, float %scale, i32 8)
  ret i32 %result
}

; --------------------------------------------------------------------
; llvm.amdgcn.cvt.scalef32.pk.fp4.f16 - dest_sel_index out of range
; --------------------------------------------------------------------

; CHECK: immarg value 4 out of range [0, 4)
; CHECK-NEXT: %result = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f16(i32 %old, <2 x half> %src, float %scale, i32 4)
define i32 @test_cvt_scalef32_pk_fp4_f16_dest_sel_index_out_of_range(i32 %old, <2 x half> %src, float %scale) {
  %result = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f16(i32 %old, <2 x half> %src, float %scale, i32 4)
  ret i32 %result
}

; CHECK: immarg value 12 out of range [0, 4)
; CHECK-NEXT: %result = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f16(i32 %old, <2 x half> %src, float %scale, i32 12)
define i32 @test_cvt_scalef32_pk_fp4_f16_dest_sel_index_way_out_of_range(i32 %old, <2 x half> %src, float %scale) {
  %result = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f16(i32 %old, <2 x half> %src, float %scale, i32 12)
  ret i32 %result
}

; --------------------------------------------------------------------
; llvm.amdgcn.cvt.scalef32.sr.pk.fp4.f16 - dst_sel_index out of range
; --------------------------------------------------------------------

; CHECK: immarg value 4 out of range [0, 4)
; CHECK-NEXT: %result = call i32 @llvm.amdgcn.cvt.scalef32.sr.pk.fp4.f16(i32 %old, <2 x half> %src, i32 %seed, float %scale, i32 4)
define i32 @test_cvt_scalef32_sr_pk_fp4_f16_dst_sel_index_out_of_range(i32 %old, <2 x half> %src, i32 %seed, float %scale) {
  %result = call i32 @llvm.amdgcn.cvt.scalef32.sr.pk.fp4.f16(i32 %old, <2 x half> %src, i32 %seed, float %scale, i32 4)
  ret i32 %result
}

; CHECK: immarg value 9 out of range [0, 4)
; CHECK-NEXT: %result = call i32 @llvm.amdgcn.cvt.scalef32.sr.pk.fp4.f16(i32 %old, <2 x half> %src, i32 %seed, float %scale, i32 9)
define i32 @test_cvt_scalef32_sr_pk_fp4_f16_dst_sel_index_way_out_of_range(i32 %old, <2 x half> %src, i32 %seed, float %scale) {
  %result = call i32 @llvm.amdgcn.cvt.scalef32.sr.pk.fp4.f16(i32 %old, <2 x half> %src, i32 %seed, float %scale, i32 9)
  ret i32 %result
}

declare i32 @llvm.amdgcn.cvt.sr.fp8.f16(half, i32, i32, i32)
declare i32 @llvm.amdgcn.cvt.sr.bf8.f16(half, i32, i32, i32)
declare <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp8(<2 x i32>, i32, i32)
declare float @llvm.amdgcn.cvt.scalef32.f32.fp8(i32, float, i32)
declare <2 x half> @llvm.amdgcn.cvt.scalef32.f16.fp8(<2 x half>, i32, float, i32, i1)
declare i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32, float, float, float, i32)
declare i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f16(i32, <2 x half>, float, i32)
declare i32 @llvm.amdgcn.cvt.scalef32.sr.pk.fp4.f16(i32, <2 x half>, i32, float, i32)
