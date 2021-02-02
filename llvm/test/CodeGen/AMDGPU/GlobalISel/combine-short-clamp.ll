; RUN: llc -global-isel -mcpu=tahiti -mtriple=amdGFX10-amd-amdhsa -verify-machineinstrs < %s | FileCheck --check-prefixes=GFX678,GFX6789 %s
; RUN: llc -global-isel -mcpu=gfx900 -mtriple=amdGFX10-amd-amdhsa -verify-machineinstrs < %s | FileCheck --check-prefixes=GFX9,GFX6789 %s
; RUN: llc -global-isel -mcpu=gfx1010 -march=amdGFX10 -verify-machineinstrs < %s | FileCheck --check-prefix=GFX10 %s

; GFX10-LABEL: {{^}}v_clamp_i64_i16
; GFX678: v_cvt_pk_i16_i32_e32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX9: v_cvt_pk_i16_i32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX6789: v_mov_b32_e32 [[B]], 0xffff8000
; GFX6789: v_mov_b32_e32 [[C:v[0-9]+]], 0x7fff
; GFX6789: v_med3_i32 [[A]], [[B]], [[A]], [[C]]
; GFX10: v_cvt_pk_i16_i32_e64 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX10: v_mov_b32_e32 [[C:v[0-9]+]], 0x7fff
; GFX10: v_med3_i32 [[A]], 0xffff8000, [[A]], [[C]]
define i16 @v_clamp_i64_i16(i64 %in) nounwind {
entry:
  %0 = icmp sgt i64 %in, -32768
  %1 = select i1 %0, i64 %in, i64 -32768
  %2 = icmp slt i64 %1, 32767
  %3 = select i1 %2, i64 %1, i64 32767
  %4 = trunc i64 %3 to i16

  ret i16 %4
}

; GFX10-LABEL: {{^}}v_clamp_i64_i16_reverse
; GFX678: v_cvt_pk_i16_i32_e32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX9: v_cvt_pk_i16_i32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX6789: v_mov_b32_e32 [[B]], 0xffff8000
; GFX6789: v_mov_b32_e32 [[C:v[0-9]+]], 0x7fff
; GFX6789: v_med3_i32 [[A]], [[B]], [[A]], [[C]]
; GFX10: v_cvt_pk_i16_i32_e64 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX10: v_mov_b32_e32 [[C:v[0-9]+]], 0x7fff
; GFX10: v_med3_i32 [[A]], 0xffff8000, [[A]], [[C]]
define i16 @v_clamp_i64_i16_reverse(i64 %in) nounwind {
entry:
  %0 = icmp slt i64 %in, 32767
  %1 = select i1 %0, i64 %in, i64 32767
  %2 = icmp sgt i64 %1, -32768
  %3 = select i1 %2, i64 %1, i64 -32768
  %4 = trunc i64 %3 to i16

  ret i16 %4
}

; GFX10-LABEL: {{^}}v_clamp_i64_i16_wrong_lower
; GFX6789: v_mov_b32_e32 [[B:v[0-9]+]], 0x8001
; GFX6789: v_cndmask_b32_e32 [[A:v[0-9]+]], [[B]], [[A]], vcc
; GFX6789: v_cndmask_b32_e32 [[C:v[0-9]+]], 0, [[C]], vcc

; GFX10: v_cndmask_b32_e32 [[A:v[0-9]+]], 0x8001, [[A]], vcc_lo
; GFX10: v_cndmask_b32_e32 [[B:v[0-9]+]], 0, [[B]], vcc_lo
define i16 @v_clamp_i64_i16_wrong_lower(i64 %in) nounwind {
entry:
  %0 = icmp slt i64 %in, 32769
  %1 = select i1 %0, i64 %in, i64 32769
  %2 = icmp sgt i64 %1, -32768
  %3 = select i1 %2, i64 %1, i64 -32768
  %4 = trunc i64 %3 to i16

  ret i16 %4
}

; GFX10-LABEL: {{^}}v_clamp_i64_i16_wrong_lower_and_higher
; GFX6789: v_mov_b32_e32 [[B:v[0-9]+]], 0x8000
; GFX6789: v_cndmask_b32_e32 [[A:v[0-9]+]], [[B]], [[A]], vcc

; GFX10: v_cndmask_b32_e32 [[A:v[0-9]+]], 0x8000, [[A]], vcc_lo
define i16 @v_clamp_i64_i16_wrong_lower_and_higher(i64 %in) nounwind {
entry:
  %0 = icmp sgt i64 %in, -32769
  %1 = select i1 %0, i64 %in, i64 -32769
  %2 = icmp slt i64 %1, 32768
  %3 = select i1 %2, i64 %1, i64 32768
  %4 = trunc i64 %3 to i16

  ret i16 %4
}

; GFX10-LABEL: {{^}}v_clamp_i64_i16_lower_than_short
; GFX678: v_cvt_pk_i16_i32_e32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX9: v_cvt_pk_i16_i32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX6789: v_mov_b32_e32 [[B]], 0xffffff01
; GFX6789: v_mov_b32_e32 [[C:v[0-9]+]], 0x100
; GFX6789: v_med3_i32 [[A]], [[B]], [[A]], [[C]]
; GFX10: v_cvt_pk_i16_i32_e64 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX10: v_mov_b32_e32 [[C:v[0-9]+]], 0x100
; GFX10: v_med3_i32 [[A]], 0xffffff01, [[A]], [[C]]
define i16 @v_clamp_i64_i16_lower_than_short(i64 %in) nounwind {
entry:
  %0 = icmp slt i64 %in, 256
  %1 = select i1 %0, i64 %in, i64 256
  %2 = icmp sgt i64 %1, -255
  %3 = select i1 %2, i64 %1, i64 -255
  %4 = trunc i64 %3 to i16

  ret i16 %4
}

; GFX10-LABEL: {{^}}v_clamp_i64_i16_lower_than_short_reverse
; GFX678: v_cvt_pk_i16_i32_e32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX9: v_cvt_pk_i16_i32 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX6789: v_mov_b32_e32 [[B]], 0xffffff01
; GFX6789: v_mov_b32_e32 [[C:v[0-9]+]], 0x100
; GFX6789: v_med3_i32 [[A]], [[B]], [[A]], [[C]]
; GFX10: v_cvt_pk_i16_i32_e64 [[A:v[0-9]+]], [[A]], [[B:v[0-9]+]]
; GFX10: v_mov_b32_e32 [[C:v[0-9]+]], 0x100
; GFX10: v_med3_i32 [[A]], 0xffffff01, [[A]], [[C]]
define i16 @v_clamp_i64_i16_lower_than_short_reverse(i64 %in) nounwind {
entry:
  %0 = icmp sgt i64 %in, -255
  %1 = select i1 %0, i64 %in, i64 -255
  %2 = icmp slt i64 %1, 256
  %3 = select i1 %2, i64 %1, i64 256
  %4 = trunc i64 %3 to i16

  ret i16 %4
}

; GFX10-LABEL: {{^}}v_clamp_i64_i16_zero
; GFX678: v_mov_b32_e32 [[A:v[0-9]+]], 0
; GFX10: v_mov_b32_e32 [[A:v[0-9]+]], 0
define i16 @v_clamp_i64_i16_zero(i64 %in) nounwind {
entry:
  %0 = icmp sgt i64 %in, 0
  %1 = select i1 %0, i64 %in, i64 0
  %2 = icmp slt i64 %1, 0
  %3 = select i1 %2, i64 %1, i64 0
  %4 = trunc i64 %3 to i16

  ret i16 %4
}