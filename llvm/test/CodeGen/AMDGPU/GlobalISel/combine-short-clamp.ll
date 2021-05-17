; RUN: llc -global-isel -mcpu=tahiti -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs < %s | FileCheck --check-prefixes=GFX678,GFX6789 %s
; RUN: llc -global-isel -mcpu=gfx900 -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs < %s | FileCheck --check-prefixes=GFX9,GFX6789 %s
; RUN: llc -global-isel -mcpu=gfx1010 -march=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefix=GFX10 %s

declare i64 @llvm.smax.i64(i64, i64)
declare i64 @llvm.smin.i64(i64, i64)

; GFX10-LABEL: {{^}}v_clamp_i64_i16
; GFX678: v_cvt_pk_i16_i32_e32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX9: v_cvt_pk_i16_i32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX6789: v_mov_b32_e32 [[B]], 0xffff8000
; GFX6789: v_mov_b32_e32 [[C:v[0-9]+]], 0x7fff
; GFX6789: v_med3_i32 [[A]], [[B]], [[A]], [[C]]
; GFX10: v_cvt_pk_i16_i32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX10: v_mov_b32_e32 [[B]], 0x7fff
; GFX10: v_med3_i32 [[A]], 0xffff8000, [[A]], [[B]]
define i16 @v_clamp_i64_i16(i64 %in) #0 {
entry:
  %max = call i64 @llvm.smax.i64(i64 %in, i64 -32768)
  %min = call i64 @llvm.smin.i64(i64 %max, i64 32767)
  %result = trunc i64 %min to i16
  ret i16 %result
}

; GFX10-LABEL: {{^}}v_clamp_i64_i16_reverse
; GFX678: v_cvt_pk_i16_i32_e32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX9: v_cvt_pk_i16_i32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX6789: v_mov_b32_e32 [[B]], 0xffff8000
; GFX6789: v_mov_b32_e32 [[C:v[0-9]+]], 0x7fff
; GFX6789: v_med3_i32 [[A]], [[B]], [[A]], [[C]]
; GFX10: v_cvt_pk_i16_i32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX10: v_mov_b32_e32 [[B]], 0x7fff
; GFX10: v_med3_i32 [[A]], 0xffff8000, [[A]], [[B]] 
define i16 @v_clamp_i64_i16_reverse(i64 %in) #0 {
entry:
  %min = call i64 @llvm.smin.i64(i64 %in, i64 32767)
  %max = call i64 @llvm.smax.i64(i64 %min, i64 -32768)
  %result = trunc i64 %max to i16
  ret i16 %result
}

; GFX10-LABEL: {{^}}v_clamp_i64_i16_invalid_lower
; GFX6789: v_mov_b32_e32 [[B:v[0-9]+]], 0x8001
; GFX6789: v_cndmask_b32_e32 [[A:v[0-9]+]], [[B]], [[A]], vcc
; GFX6789: v_cndmask_b32_e32 [[C:v[0-9]+]], 0, [[C]], vcc

; GFX10: v_cndmask_b32_e32 [[A:v[0-9]+]], 0x8001, [[A]], vcc_lo
; GFX10: v_cndmask_b32_e32 [[B:v[0-9]+]], 0, [[B]], vcc_lo
define i16 @v_clamp_i64_i16_invalid_lower(i64 %in) #0 {
entry:
  %min = call i64 @llvm.smin.i64(i64 %in, i64 32769)
  %max = call i64 @llvm.smax.i64(i64 %min, i64 -32768)
  %result = trunc i64 %max to i16
  ret i16 %result
}

; GFX10-LABEL: {{^}}v_clamp_i64_i16_invalid_lower_and_higher
; GFX6789: v_mov_b32_e32 [[B:v[0-9]+]], 0x8000
; GFX6789: v_cndmask_b32_e32 [[A:v[0-9]+]], [[B]], [[A]], vcc
; GFX10: v_cndmask_b32_e32 [[A:v[0-9]+]], 0x8000, [[A]], vcc_lo
define i16 @v_clamp_i64_i16_invalid_lower_and_higher(i64 %in) #0 {
entry:
  %max = call i64 @llvm.smax.i64(i64 %in, i64 -32769)
  %min = call i64 @llvm.smin.i64(i64 %max, i64 32768)
  %result = trunc i64 %min to i16
  ret i16 %result
}

; GFX10-LABEL: {{^}}v_clamp_i64_i16_lower_than_short
; GFX678: v_cvt_pk_i16_i32_e32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX9: v_cvt_pk_i16_i32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX6789: v_mov_b32_e32 [[B]], 0xffffff01
; GFX6789: v_mov_b32_e32 [[C:v[0-9]+]], 0x100
; GFX6789: v_med3_i32 [[A]], [[B]], [[A]], [[C]]
; GFX10: v_cvt_pk_i16_i32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX10: v_mov_b32_e32 [[B]], 0x100
; GFX10: v_med3_i32 [[A]], 0xffffff01, [[A]], [[B]]
define i16 @v_clamp_i64_i16_lower_than_short(i64 %in) #0 {
entry:
  %min = call i64 @llvm.smin.i64(i64 %in, i64 256)
  %max = call i64 @llvm.smax.i64(i64 %min, i64 -255)
  %result = trunc i64 %max to i16
  ret i16 %result
}

; GFX10-LABEL: {{^}}v_clamp_i64_i16_lower_than_short_reverse
; GFX678: v_cvt_pk_i16_i32_e32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX9: v_cvt_pk_i16_i32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX6789: v_mov_b32_e32 [[B]], 0xffffff01
; GFX6789: v_mov_b32_e32 [[C:v[0-9]+]], 0x100
; GFX6789: v_med3_i32 [[A]], [[B]], [[A]], [[C]]
; GFX10: v_cvt_pk_i16_i32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX10: v_mov_b32_e32 [[B]], 0x100
; GFX10: v_med3_i32 [[A]], 0xffffff01, [[A]], [[B]]
define i16 @v_clamp_i64_i16_lower_than_short_reverse(i64 %in) #0 {
entry:
  %max = call i64 @llvm.smax.i64(i64 %in, i64 -255)
  %min = call i64 @llvm.smin.i64(i64 %max, i64 256)
  %result = trunc i64 %min to i16
  ret i16 %result
}

; GFX10-LABEL: {{^}}v_clamp_i64_i16_zero
; GFX6789: v_mov_b32_e32 v0, 0
; GFX10: v_mov_b32_e32 v0, 0
define i16 @v_clamp_i64_i16_zero(i64 %in) #0 {
entry:
  %max = call i64 @llvm.smax.i64(i64 %in, i64 0)
  %min = call i64 @llvm.smin.i64(i64 %max, i64 0)
  %result = trunc i64 %min to i16
  ret i16 %result
}
