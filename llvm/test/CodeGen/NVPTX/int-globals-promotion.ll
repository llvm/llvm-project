; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Test that NVPTX promotes globals with non-standard integer widths
; (anything other than i1/i8/i16/i32/i64) to the next supported PTX
; integer storage type (.u8/.u16/.u32/.u64), and that the emitted
; initializer literal zero-extends to that storage width so the cubin
; bit pattern matches the IR ConstantInt. See issue #154337.

target triple = "nvptx64-nvidia-cuda"

; ----------------------------------------------------------------------------
; i1: pin down the latent fix. Previously emitted `.u8 g_i1 = -1` (cubin byte
; 0xFF). The byte pattern now matches the IR's canonical zero-extension of
; `i1 true` (cubin byte 0x01). Both are truthy, so this is functionally
; transparent for boolean uses; the fix matters for byte-level inspection.
; CHECK: .global .align 1 .u8 g_i1 = 1;
@g_i1 = addrspace(1) constant i1 true, align 1

; ----------------------------------------------------------------------------
; Sub-byte: lower and upper edges of the .u8 bucket.
; CHECK: .global .align 1 .u8 g_i2 = 1;
@g_i2 = addrspace(1) constant i2 1, align 1
; CHECK: .global .align 1 .u8 g_i7 = 5;
@g_i7 = addrspace(1) constant i7 5, align 1

; ----------------------------------------------------------------------------
; Sign-bit case in the .u16 bucket: i12 -1 = 0xFFF must zero-extend to 4095,
; not emit as -1 (which ptxas would silently reinterpret as the 16-bit
; unsigned representation 0xFFFF, corrupting the top 4 bits).
; CHECK: .global .align 2 .u16 g_i12_neg = 4095;
@g_i12_neg = addrspace(1) constant i12 -1, align 2

; ----------------------------------------------------------------------------
; Sign-bit case in the .u32 bucket: i24 -1 = 0xFFFFFF must zero-extend to
; 16777215, not emit as -1 (which would corrupt the top 8 bits of the .u32
; storage to 0xFFFFFFFF).
; CHECK: .global .align 4 .u32 g_i24_neg = 16777215;
@g_i24_neg = addrspace(1) constant i24 -1, align 4

; ----------------------------------------------------------------------------
; Positive value at the .u64 bucket: value > 2^32 means it would truncate
; in any smaller bucket, pinning the bucket choice from the value side too.
; i34 (not i33) is used because i33's signed range cannot represent a value
; > 2^32 without setting its sign bit -- a hazard exercised by the next case.
; CHECK: .global .align 8 .u64 g_i34 = 8589934591;
@g_i34 = addrspace(1) constant i34 8589934591, align 8

; ----------------------------------------------------------------------------
; Sign-bit case in the .u64 bucket: i40 -1 = 0xFF_FFFF_FFFF must zero-extend
; to 1099511627775 (low 40 bits = 1, top 24 bits = 0). Pre-fix this would
; emit `.u64 g = -1` (cubin byte pattern 0xFFFF_FFFF_FFFF_FFFF), silently
; corrupting the top 24 bits.
; CHECK: .global .align 8 .u64 g_i40_neg = 1099511627775;
@g_i40_neg = addrspace(1) constant i40 -1, align 8
