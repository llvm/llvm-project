; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; fp128 globals are lowered to byte arrays. An fp128 nested in an aggregate is
; buffered one element at a time, and that per-element path must emit the same
; 16 little-endian bytes as a scalar fp128.

; CHECK-DAG: .visible .global .align 16 .b8 array_nonzero[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 63};
@array_nonzero = global [1 x fp128] [fp128 0xL00000000000000003FFF000000000000]

; CHECK-DAG: .visible .global .align 16 .b8 array_multi[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64};
@array_multi = global [2 x fp128] [fp128 0xL00000000000000003FFF000000000000, fp128 0xL00000000000000004000000000000000]

; Trailing zeros of the struct's tail padding are trimmed by the printer.
; CHECK-DAG: .visible .global .align 16 .b8 struct_nonzero[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 64, 7};
%struct.WithFloat128 = type { fp128, i32 }
@struct_nonzero = global %struct.WithFloat128 { fp128 0xL00000000000000004000800000000000, i32 7 }

; CHECK-DAG: .visible .global .align 16 .b8 scalar_nonzero[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 63};
@scalar_nonzero = global fp128 0xL00000000000000003FFF000000000000

; CHECK-DAG: .visible .global .align 16 .b8 array_zero[16];
@array_zero = global [1 x fp128] zeroinitializer
