; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

; PR12281
; Test generataion of code for vmull instruction when multiplying 128-bit
; vectors that were created by sign-extending smaller vector sizes.
;
; The vmull operation requires 64-bit vectors, so we must extend the original
; vector size to 64 bits for vmull operation.
; Previously failed with an assertion because the <4 x i8> vector was too small
; for vmull.

; Vector x Constant
; v4i8
;
define void @sextload_v4i8_c(ptr %v) nounwind {
;CHECK-LABEL: sextload_v4i8_c:
entry:
  %0 = load <4 x i8>, ptr %v, align 8
  %v0  = sext <4 x i8> %0 to <4 x i32>
;CHECK: vmull
  %v1 = mul <4 x i32>  %v0, <i32 3, i32 3, i32 3, i32 3>
  store <4 x i32> %v1, ptr undef, align 8
  ret void;
}

; v2i8
;
define void @sextload_v2i8_c(ptr %v) nounwind {
;CHECK-LABEL: sextload_v2i8_c:
entry:
  %0   = load <2 x i8>, ptr %v, align 8
  %v0  = sext <2 x i8>  %0 to <2 x i64>
;CHECK: vmull
  %v1  = mul <2 x i64>  %v0, <i64 3, i64 3>
  store <2 x i64> %v1, ptr undef, align 8
  ret void;
}

; v2i16
;
define void @sextload_v2i16_c(ptr %v) nounwind {
;CHECK-LABEL: sextload_v2i16_c:
entry:
  %0   = load <2 x i16>, ptr %v, align 8
  %v0  = sext <2 x i16>  %0 to <2 x i64>
;CHECK: vmull
  %v1  = mul <2 x i64>  %v0, <i64 3, i64 3>
  store <2 x i64> %v1, ptr undef, align 8
  ret void;
}


; Vector x Vector
; v4i8
;
define void @sextload_v4i8_v(ptr %v, ptr %p) nounwind {
;CHECK-LABEL: sextload_v4i8_v:
entry:
  %0 = load <4 x i8>, ptr %v, align 8
  %v0  = sext <4 x i8> %0 to <4 x i32>

  %1  = load <4 x i8>, ptr %p, align 8
  %v2 = sext <4 x i8> %1 to <4 x i32>
;CHECK: vmull
  %v1 = mul <4 x i32>  %v0, %v2
  store <4 x i32> %v1, ptr undef, align 8
  ret void;
}

; v2i8
;
define void @sextload_v2i8_v(ptr %v, ptr %p) nounwind {
;CHECK-LABEL: sextload_v2i8_v:
entry:
  %0 = load <2 x i8>, ptr %v, align 8
  %v0  = sext <2 x i8> %0 to <2 x i64>

  %1  = load <2 x i8>, ptr %p, align 8
  %v2 = sext <2 x i8> %1 to <2 x i64>
;CHECK: vmull
  %v1 = mul <2 x i64>  %v0, %v2
  store <2 x i64> %v1, ptr undef, align 8
  ret void;
}

; v2i16
;
define void @sextload_v2i16_v(ptr %v, ptr %p) nounwind {
;CHECK-LABEL: sextload_v2i16_v:
entry:
  %0 = load <2 x i16>, ptr %v, align 8
  %v0  = sext <2 x i16> %0 to <2 x i64>

  %1  = load <2 x i16>, ptr %p, align 8
  %v2 = sext <2 x i16> %1 to <2 x i64>
;CHECK: vmull
  %v1 = mul <2 x i64>  %v0, %v2
  store <2 x i64> %v1, ptr undef, align 8
  ret void;
}


; Vector(small) x Vector(big)
; v4i8 x v4i16
;
define void @sextload_v4i8_vs(ptr %v, ptr %p) nounwind {
;CHECK-LABEL: sextload_v4i8_vs:
entry:
  %0 = load <4 x i8>, ptr %v, align 8
  %v0  = sext <4 x i8> %0 to <4 x i32>

  %1  = load <4 x i16>, ptr %p, align 8
  %v2 = sext <4 x i16> %1 to <4 x i32>
;CHECK: vmull
  %v1 = mul <4 x i32>  %v0, %v2
  store <4 x i32> %v1, ptr undef, align 8
  ret void;
}

; v2i8
; v2i8 x v2i16
define void @sextload_v2i8_vs(ptr %v, ptr %p) nounwind {
;CHECK-LABEL: sextload_v2i8_vs:
entry:
  %0 = load <2 x i8>, ptr %v, align 8
  %v0  = sext <2 x i8> %0 to <2 x i64>

  %1  = load <2 x i16>, ptr %p, align 8
  %v2 = sext <2 x i16> %1 to <2 x i64>
;CHECK: vmull
  %v1 = mul <2 x i64>  %v0, %v2
  store <2 x i64> %v1, ptr undef, align 8
  ret void;
}

; v2i16
; v2i16 x v2i32
define void @sextload_v2i16_vs(ptr %v, ptr %p) nounwind {
;CHECK-LABEL: sextload_v2i16_vs:
entry:
  %0 = load <2 x i16>, ptr %v, align 8
  %v0  = sext <2 x i16> %0 to <2 x i64>

  %1  = load <2 x i32>, ptr %p, align 8
  %v2 = sext <2 x i32> %1 to <2 x i64>
;CHECK: vmull
  %v1 = mul <2 x i64>  %v0, %v2
  store <2 x i64> %v1, ptr undef, align 8
  ret void;
}
