; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Test that NVPTX promotes globals with non-standard integer widths
; (anything other than i1/i8/i16/i32/i64) to the next supported PTX
; integer storage type (.u8/.u16/.u32/.u64). See issue #154337.

target triple = "nvptx64-nvidia-cuda"

; Sub-byte: lower and upper edges of the .u8 bucket.
; CHECK: .global .align 1 .u8 g_i2 = 1;
@g_i2 = addrspace(1) constant i2 1, align 1
; CHECK: .global .align 1 .u8 g_i7 = 5;
@g_i7 = addrspace(1) constant i7 5, align 1

; Mid-range non-supported widths.
; CHECK: .global .align 2 .u16 g_i12 = 42;
@g_i12 = addrspace(1) constant i12 42, align 2
; CHECK: .global .align 4 .u32 g_i24 = 100;
@g_i24 = addrspace(1) constant i24 100, align 4

; Lower edge of the .u64 bucket: width > 32, value also > 2^32 so it must
; actually use .u64 storage (a smaller bucket would truncate). i34 is used
; instead of i33 because i33's signed range cannot represent a > 2^32 value
; without setting the sign bit, which would trip APInt's default signed
; printer (separate issue: printScalarConstant should zero-extend after
; type promotion).
; CHECK: .global .align 8 .u64 g_i34 = 8589934591;
@g_i34 = addrspace(1) constant i34 8589934591, align 8
