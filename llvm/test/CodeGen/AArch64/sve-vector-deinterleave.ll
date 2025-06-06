; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+sve2 | FileCheck %s

define {<vscale x 2 x half>, <vscale x 2 x half>} @vector_deinterleave_nxv2f16_nxv4f16(<vscale x 4 x half> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv2f16_nxv4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z1.s, z0.s, z0.s
; CHECK-NEXT:    uzp2 z2.s, z0.s, z0.s
; CHECK-NEXT:    uunpklo z0.d, z1.s
; CHECK-NEXT:    uunpklo z1.d, z2.s
; CHECK-NEXT:    ret
  %retval = call {<vscale x 2 x half>, <vscale x 2 x half>} @llvm.vector.deinterleave2.nxv4f16(<vscale x 4 x half> %vec)
  ret {<vscale x 2 x half>, <vscale x 2 x half>} %retval
}

define {<vscale x 4 x half>, <vscale x 4 x half>} @vector_deinterleave_nxv4f16_nxv8f16(<vscale x 8 x half> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv4f16_nxv8f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z1.h, z0.h, z0.h
; CHECK-NEXT:    uzp2 z2.h, z0.h, z0.h
; CHECK-NEXT:    uunpklo z0.s, z1.h
; CHECK-NEXT:    uunpklo z1.s, z2.h
; CHECK-NEXT:    ret
  %retval = call {<vscale x 4 x half>, <vscale x 4 x half>} @llvm.vector.deinterleave2.nxv8f16(<vscale x 8 x half> %vec)
  ret {<vscale x 4 x half>, <vscale x 4 x half>} %retval
}

define {<vscale x 8 x half>, <vscale x 8 x half>} @vector_deinterleave_nxv8f16_nxv16f16(<vscale x 16 x half> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv8f16_nxv16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z2.h, z0.h, z1.h
; CHECK-NEXT:    uzp2 z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.d, z2.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 8 x half>, <vscale x 8 x half>} @llvm.vector.deinterleave2.nxv16f16(<vscale x 16 x half> %vec)
  ret {<vscale x 8 x half>, <vscale x 8 x half>} %retval
}

define {<vscale x 2 x float>, <vscale x 2 x float>} @vector_deinterleave_nxv2f32_nxv4f32(<vscale x 4 x float> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv2f32_nxv4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z1.s, z0.s, z0.s
; CHECK-NEXT:    uzp2 z2.s, z0.s, z0.s
; CHECK-NEXT:    uunpklo z0.d, z1.s
; CHECK-NEXT:    uunpklo z1.d, z2.s
; CHECK-NEXT:    ret
  %retval = call {<vscale x 2 x float>, <vscale x 2 x float>} @llvm.vector.deinterleave2.nxv4f32(<vscale x 4 x float> %vec)
  ret {<vscale x 2 x float>, <vscale x 2 x float>} %retval
}

define {<vscale x 4 x float>, <vscale x 4 x float>} @vector_deinterleave_nxv4f32_nxv8f32(<vscale x 8 x float> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv4f32_nxv8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z2.s, z0.s, z1.s
; CHECK-NEXT:    uzp2 z1.s, z0.s, z1.s
; CHECK-NEXT:    mov z0.d, z2.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 4 x float>, <vscale x 4 x float>} @llvm.vector.deinterleave2.nxv8f32(<vscale x 8 x float> %vec)
  ret {<vscale x 4 x float>, <vscale x 4 x float>} %retval
}

define {<vscale x 2 x double>, <vscale x 2 x double>} @vector_deinterleave_nxv2f64_nxv4f64(<vscale x 4 x double> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv2f64_nxv4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z2.d, z0.d, z1.d
; CHECK-NEXT:    uzp2 z1.d, z0.d, z1.d
; CHECK-NEXT:    mov z0.d, z2.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 2 x double>, <vscale x 2 x double>} @llvm.vector.deinterleave2.nxv4f64(<vscale x 4 x double> %vec)
  ret {<vscale x 2 x double>, <vscale x 2 x double>} %retval
}

define {<vscale x 2 x bfloat>, <vscale x 2 x bfloat>} @vector_deinterleave_nxv2bf16_nxv4bf16(<vscale x 4 x bfloat> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv2bf16_nxv4bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z1.s, z0.s, z0.s
; CHECK-NEXT:    uzp2 z2.s, z0.s, z0.s
; CHECK-NEXT:    uunpklo z0.d, z1.s
; CHECK-NEXT:    uunpklo z1.d, z2.s
; CHECK-NEXT:    ret
  %retval = call {<vscale x 2 x bfloat>, <vscale x 2 x bfloat>} @llvm.vector.deinterleave2.nxv4bf16(<vscale x 4 x bfloat> %vec)
  ret {<vscale x 2 x bfloat>, <vscale x 2 x bfloat>} %retval
}

define {<vscale x 4 x bfloat>, <vscale x 4 x bfloat>} @vector_deinterleave_nxv4bf16_nxv8bf16(<vscale x 8 x bfloat> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv4bf16_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z1.h, z0.h, z0.h
; CHECK-NEXT:    uzp2 z2.h, z0.h, z0.h
; CHECK-NEXT:    uunpklo z0.s, z1.h
; CHECK-NEXT:    uunpklo z1.s, z2.h
; CHECK-NEXT:    ret
  %retval = call {<vscale x 4 x bfloat>, <vscale x 4 x bfloat>} @llvm.vector.deinterleave2.nxv8bf16(<vscale x 8 x bfloat> %vec)
  ret {<vscale x 4 x bfloat>, <vscale x 4 x bfloat>} %retval
}

define {<vscale x 8 x bfloat>, <vscale x 8 x bfloat>} @vector_deinterleave_nxv8bf16_nxv16bf16(<vscale x 16 x bfloat> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv8bf16_nxv16bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z2.h, z0.h, z1.h
; CHECK-NEXT:    uzp2 z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.d, z2.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 8 x bfloat>, <vscale x 8 x bfloat>} @llvm.vector.deinterleave2.nxv16bf16(<vscale x 16 x bfloat> %vec)
  ret {<vscale x 8 x bfloat>, <vscale x 8 x bfloat>} %retval
}

; Integers

define {<vscale x 16 x i8>, <vscale x 16 x i8>} @vector_deinterleave_nxv16i8_nxv32i8(<vscale x 32 x i8> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv16i8_nxv32i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z2.b, z0.b, z1.b
; CHECK-NEXT:    uzp2 z1.b, z0.b, z1.b
; CHECK-NEXT:    mov z0.d, z2.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 16 x i8>, <vscale x 16 x i8>} @llvm.vector.deinterleave2.nxv32i8(<vscale x 32 x i8> %vec)
  ret {<vscale x 16 x i8>, <vscale x 16 x i8>} %retval
}

define {<vscale x 8 x i16>, <vscale x 8 x i16>} @vector_deinterleave_nxv8i16_nxv16i16(<vscale x 16 x i16> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv8i16_nxv16i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z2.h, z0.h, z1.h
; CHECK-NEXT:    uzp2 z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.d, z2.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 8 x i16>, <vscale x 8 x i16>} @llvm.vector.deinterleave2.nxv16i16(<vscale x 16 x i16> %vec)
  ret {<vscale x 8 x i16>, <vscale x 8 x i16>} %retval
}

define {<vscale x 4 x i32>, <vscale x 4 x i32>} @vector_deinterleave_nxv4i32_nxvv8i32(<vscale x 8 x i32> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv4i32_nxvv8i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z2.s, z0.s, z1.s
; CHECK-NEXT:    uzp2 z1.s, z0.s, z1.s
; CHECK-NEXT:    mov z0.d, z2.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 4 x i32>, <vscale x 4 x i32>} @llvm.vector.deinterleave2.nxv8i32(<vscale x 8 x i32> %vec)
  ret {<vscale x 4 x i32>, <vscale x 4 x i32>} %retval
}

define {<vscale x 2 x i64>, <vscale x 2 x i64>} @vector_deinterleave_nxv2i64_nxv4i64(<vscale x 4 x i64> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv2i64_nxv4i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z2.d, z0.d, z1.d
; CHECK-NEXT:    uzp2 z1.d, z0.d, z1.d
; CHECK-NEXT:    mov z0.d, z2.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 2 x i64>, <vscale x 2 x i64>} @llvm.vector.deinterleave2.nxv4i64(<vscale x 4 x i64> %vec)
  ret {<vscale x 2 x i64>, <vscale x 2 x i64>} %retval
}

define {<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>} @vector_deinterleave_nxv16i8_nxv64i8(<vscale x 64 x i8> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv16i8_nxv64i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z4.b, z2.b, z3.b
; CHECK-NEXT:    uzp1 z5.b, z0.b, z1.b
; CHECK-NEXT:    uzp2 z3.b, z2.b, z3.b
; CHECK-NEXT:    uzp2 z6.b, z0.b, z1.b
; CHECK-NEXT:    uzp1 z0.b, z5.b, z4.b
; CHECK-NEXT:    uzp2 z2.b, z5.b, z4.b
; CHECK-NEXT:    uzp1 z1.b, z6.b, z3.b
; CHECK-NEXT:    uzp2 z3.b, z6.b, z3.b
; CHECK-NEXT:    ret
  %retval = call {<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>} @llvm.vector.deinterleave4.nxv64i8(<vscale x 64 x i8> %vec)
  ret {<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>} %retval
}

define {<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>} @vector_deinterleave_nxv8i16_nxv32i16(<vscale x 32 x i16> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv8i16_nxv32i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z4.h, z2.h, z3.h
; CHECK-NEXT:    uzp1 z5.h, z0.h, z1.h
; CHECK-NEXT:    uzp2 z3.h, z2.h, z3.h
; CHECK-NEXT:    uzp2 z6.h, z0.h, z1.h
; CHECK-NEXT:    uzp1 z0.h, z5.h, z4.h
; CHECK-NEXT:    uzp2 z2.h, z5.h, z4.h
; CHECK-NEXT:    uzp1 z1.h, z6.h, z3.h
; CHECK-NEXT:    uzp2 z3.h, z6.h, z3.h
; CHECK-NEXT:    ret
  %retval = call {<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>} @llvm.vector.deinterleave4.nxv32i16(<vscale x 32 x i16> %vec)
  ret {<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>} %retval
}

define {<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>} @vector_deinterleave_nxv4i32_nxv16i32(<vscale x 16 x i32> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv4i32_nxv16i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z4.s, z2.s, z3.s
; CHECK-NEXT:    uzp1 z5.s, z0.s, z1.s
; CHECK-NEXT:    uzp2 z3.s, z2.s, z3.s
; CHECK-NEXT:    uzp2 z6.s, z0.s, z1.s
; CHECK-NEXT:    uzp1 z0.s, z5.s, z4.s
; CHECK-NEXT:    uzp2 z2.s, z5.s, z4.s
; CHECK-NEXT:    uzp1 z1.s, z6.s, z3.s
; CHECK-NEXT:    uzp2 z3.s, z6.s, z3.s
; CHECK-NEXT:    ret
  %retval = call {<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>} @llvm.vector.deinterleave4.nxv16i32(<vscale x 16 x i32> %vec)
  ret {<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>} %retval
}

define {<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>} @vector_deinterleave_nxv2i64_nxv8i64(<vscale x 8 x i64> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv2i64_nxv8i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z4.d, z2.d, z3.d
; CHECK-NEXT:    uzp1 z5.d, z0.d, z1.d
; CHECK-NEXT:    uzp2 z3.d, z2.d, z3.d
; CHECK-NEXT:    uzp2 z6.d, z0.d, z1.d
; CHECK-NEXT:    uzp1 z0.d, z5.d, z4.d
; CHECK-NEXT:    uzp2 z2.d, z5.d, z4.d
; CHECK-NEXT:    uzp1 z1.d, z6.d, z3.d
; CHECK-NEXT:    uzp2 z3.d, z6.d, z3.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>} @llvm.vector.deinterleave4.nxv8i64(<vscale x 8 x i64> %vec)
  ret {<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>} %retval
}

define {<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>} @vector_deinterleave_nxv2i64_nxv16i64(<vscale x 16 x i64> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv2i64_nxv16i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z24.d, z6.d, z7.d
; CHECK-NEXT:    uzp1 z25.d, z4.d, z5.d
; CHECK-NEXT:    uzp1 z26.d, z2.d, z3.d
; CHECK-NEXT:    uzp1 z27.d, z0.d, z1.d
; CHECK-NEXT:    uzp2 z6.d, z6.d, z7.d
; CHECK-NEXT:    uzp2 z4.d, z4.d, z5.d
; CHECK-NEXT:    uzp2 z2.d, z2.d, z3.d
; CHECK-NEXT:    uzp2 z0.d, z0.d, z1.d
; CHECK-NEXT:    uzp1 z5.d, z25.d, z24.d
; CHECK-NEXT:    uzp2 z24.d, z25.d, z24.d
; CHECK-NEXT:    uzp1 z7.d, z27.d, z26.d
; CHECK-NEXT:    uzp1 z28.d, z4.d, z6.d
; CHECK-NEXT:    uzp2 z25.d, z27.d, z26.d
; CHECK-NEXT:    uzp1 z29.d, z0.d, z2.d
; CHECK-NEXT:    uzp2 z26.d, z4.d, z6.d
; CHECK-NEXT:    uzp2 z27.d, z0.d, z2.d
; CHECK-NEXT:    uzp1 z0.d, z7.d, z5.d
; CHECK-NEXT:    uzp1 z2.d, z25.d, z24.d
; CHECK-NEXT:    uzp2 z4.d, z7.d, z5.d
; CHECK-NEXT:    uzp1 z1.d, z29.d, z28.d
; CHECK-NEXT:    uzp1 z3.d, z27.d, z26.d
; CHECK-NEXT:    uzp2 z5.d, z29.d, z28.d
; CHECK-NEXT:    uzp2 z6.d, z25.d, z24.d
; CHECK-NEXT:    uzp2 z7.d, z27.d, z26.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>} @llvm.vector.deinterleave8.nxv16i64(<vscale x 16 x i64> %vec)
  ret {<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>} %retval
}

; Predicated
define {<vscale x 16 x i1>, <vscale x 16 x i1>} @vector_deinterleave_nxv16i1_nxv32i1(<vscale x 32 x i1> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv16i1_nxv32i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 p2.b, p0.b, p1.b
; CHECK-NEXT:    uzp2 p1.b, p0.b, p1.b
; CHECK-NEXT:    mov p0.b, p2.b
; CHECK-NEXT:    ret
  %retval = call {<vscale x 16 x i1>, <vscale x 16 x i1>} @llvm.vector.deinterleave2.nxv32i1(<vscale x 32 x i1> %vec)
  ret {<vscale x 16 x i1>, <vscale x 16 x i1>} %retval
}

define {<vscale x 8 x i1>, <vscale x 8 x i1>} @vector_deinterleave_nxv8i1_nxv16i1(<vscale x 16 x i1> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv8i1_nxv16i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 p1.b, p0.b, p0.b
; CHECK-NEXT:    uzp2 p2.b, p0.b, p0.b
; CHECK-NEXT:    punpklo p0.h, p1.b
; CHECK-NEXT:    punpklo p1.h, p2.b
; CHECK-NEXT:    ret
  %retval = call {<vscale x 8 x i1>, <vscale x 8 x i1>} @llvm.vector.deinterleave2.nxv16i1(<vscale x 16 x i1> %vec)
  ret {<vscale x 8 x i1>, <vscale x 8 x i1>} %retval
}

define {<vscale x 4 x i1>, <vscale x 4 x i1>} @vector_deinterleave_nxv4i1_nxv8i1(<vscale x 8 x i1> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv4i1_nxv8i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 p1.h, p0.h, p0.h
; CHECK-NEXT:    uzp2 p2.h, p0.h, p0.h
; CHECK-NEXT:    punpklo p0.h, p1.b
; CHECK-NEXT:    punpklo p1.h, p2.b
; CHECK-NEXT:    ret
  %retval = call {<vscale x 4 x i1>, <vscale x 4 x i1>} @llvm.vector.deinterleave2.nxv8i1(<vscale x 8 x i1> %vec)
  ret {<vscale x 4 x i1>, <vscale x 4 x i1>} %retval
}

define {<vscale x 2 x i1>, <vscale x 2 x i1>} @vector_deinterleave_nxv2i1_nxv4i1(<vscale x 4 x i1> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv2i1_nxv4i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 p1.s, p0.s, p0.s
; CHECK-NEXT:    uzp2 p2.s, p0.s, p0.s
; CHECK-NEXT:    punpklo p0.h, p1.b
; CHECK-NEXT:    punpklo p1.h, p2.b
; CHECK-NEXT:    ret
  %retval = call {<vscale x 2 x i1>, <vscale x 2 x i1>} @llvm.vector.deinterleave2.nxv4i1(<vscale x 4 x i1> %vec)
  ret {<vscale x 2 x i1>, <vscale x 2 x i1>} %retval
}


; Split illegal types

define {<vscale x 4 x i64>, <vscale x 4 x i64>} @vector_deinterleave_nxv4i64_nxv8i64(<vscale x 8 x i64> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv4i64_nxv8i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z4.d, z2.d, z3.d
; CHECK-NEXT:    uzp1 z5.d, z0.d, z1.d
; CHECK-NEXT:    uzp2 z6.d, z0.d, z1.d
; CHECK-NEXT:    uzp2 z3.d, z2.d, z3.d
; CHECK-NEXT:    mov z0.d, z5.d
; CHECK-NEXT:    mov z1.d, z4.d
; CHECK-NEXT:    mov z2.d, z6.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 4 x i64>, <vscale x 4 x i64>} @llvm.vector.deinterleave2.nxv8i64(<vscale x 8 x i64> %vec)
  ret {<vscale x 4 x i64>, <vscale x 4 x i64>} %retval
}

define {<vscale x 8 x i64>, <vscale x 8 x i64>} @vector_deinterleave_nxv8i64_nxv16i64(<vscale x 16 x i64> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv8i64_nxv16i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uzp1 z24.d, z2.d, z3.d
; CHECK-NEXT:    uzp1 z25.d, z0.d, z1.d
; CHECK-NEXT:    uzp1 z26.d, z4.d, z5.d
; CHECK-NEXT:    uzp1 z27.d, z6.d, z7.d
; CHECK-NEXT:    uzp2 z28.d, z0.d, z1.d
; CHECK-NEXT:    uzp2 z29.d, z2.d, z3.d
; CHECK-NEXT:    uzp2 z30.d, z4.d, z5.d
; CHECK-NEXT:    uzp2 z7.d, z6.d, z7.d
; CHECK-NEXT:    mov z0.d, z25.d
; CHECK-NEXT:    mov z1.d, z24.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    mov z4.d, z28.d
; CHECK-NEXT:    mov z5.d, z29.d
; CHECK-NEXT:    mov z6.d, z30.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 8 x i64>, <vscale x 8 x i64>} @llvm.vector.deinterleave2.nxv16i64(<vscale x 16 x i64> %vec)
  ret {<vscale x 8 x i64>, <vscale x 8 x i64>} %retval
}


; Promote illegal type size

define {<vscale x 8 x i8>, <vscale x 8 x i8>} @vector_deinterleave_nxv8i8_nxv16i8(<vscale x 16 x i8> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv8i8_nxv16i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uunpkhi z1.h, z0.b
; CHECK-NEXT:    uunpklo z2.h, z0.b
; CHECK-NEXT:    uzp1 z0.h, z2.h, z1.h
; CHECK-NEXT:    uzp2 z1.h, z2.h, z1.h
; CHECK-NEXT:    ret
  %retval = call {<vscale x 8 x i8>, <vscale x 8 x i8>} @llvm.vector.deinterleave2.nxv16i8(<vscale x 16 x i8> %vec)
  ret {<vscale x 8 x i8>, <vscale x 8 x i8>} %retval
}

define {<vscale x 4 x i16>, <vscale x 4 x i16>} @vector_deinterleave_nxv4i16_nxv8i16(<vscale x 8 x i16> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv4i16_nxv8i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uunpkhi z1.s, z0.h
; CHECK-NEXT:    uunpklo z2.s, z0.h
; CHECK-NEXT:    uzp1 z0.s, z2.s, z1.s
; CHECK-NEXT:    uzp2 z1.s, z2.s, z1.s
; CHECK-NEXT:    ret
  %retval = call {<vscale x 4 x i16>, <vscale x 4 x i16>} @llvm.vector.deinterleave2.nxv8i16(<vscale x 8 x i16> %vec)
  ret {<vscale x 4 x i16>, <vscale x 4 x i16>} %retval
}

define {<vscale x 2 x i32>, <vscale x 2 x i32>} @vector_deinterleave_nxv2i32_nxv4i32(<vscale x 4 x i32> %vec) {
; CHECK-LABEL: vector_deinterleave_nxv2i32_nxv4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uunpkhi z1.d, z0.s
; CHECK-NEXT:    uunpklo z2.d, z0.s
; CHECK-NEXT:    uzp1 z0.d, z2.d, z1.d
; CHECK-NEXT:    uzp2 z1.d, z2.d, z1.d
; CHECK-NEXT:    ret
  %retval = call {<vscale x 2 x i32>,<vscale x 2 x i32>} @llvm.vector.deinterleave2.nxv4i32(<vscale x 4 x i32> %vec)
  ret {<vscale x 2 x i32>, <vscale x 2 x i32>} %retval
}

; Floating declarations
declare {<vscale x 2 x half>,<vscale x 2 x half>} @llvm.vector.deinterleave2.nxv4f16(<vscale x 4 x half>)
declare {<vscale x 4 x half>, <vscale x 4 x half>} @llvm.vector.deinterleave2.nxv8f16(<vscale x 8 x half>)
declare {<vscale x 2 x float>, <vscale x 2 x float>} @llvm.vector.deinterleave2.nxv4f32(<vscale x 4 x float>)
declare {<vscale x 8 x half>, <vscale x 8 x half>} @llvm.vector.deinterleave2.nxv16f16(<vscale x 16 x half>)
declare {<vscale x 4 x float>, <vscale x 4 x float>} @llvm.vector.deinterleave2.nxv8f32(<vscale x 8 x float>)
declare {<vscale x 2 x double>, <vscale x 2 x double>} @llvm.vector.deinterleave2.nxv4f64(<vscale x 4 x double>)

; Integer declarations
declare {<vscale x 16 x i8>, <vscale x 16 x i8>} @llvm.vector.deinterleave2.nxv32i8(<vscale x 32 x i8>)
declare {<vscale x 8 x i16>, <vscale x 8 x i16>} @llvm.vector.deinterleave2.nxv16i16(<vscale x 16 x i16>)
declare {<vscale x 4 x i32>, <vscale x 4 x i32>} @llvm.vector.deinterleave2.nxv8i32(<vscale x 8 x i32>)
declare {<vscale x 2 x i64>, <vscale x 2 x i64>} @llvm.vector.deinterleave2.nxv4i64(<vscale x 4 x i64>)

; Predicated declarations
declare {<vscale x 16 x i1>, <vscale x 16 x i1>} @llvm.vector.deinterleave2.nxv32i1(<vscale x 32 x i1>)
declare {<vscale x 8 x i1>, <vscale x 8 x i1>} @llvm.vector.deinterleave2.nxv16i1(<vscale x 16 x i1>)
declare {<vscale x 4 x i1>, <vscale x 4 x i1>} @llvm.vector.deinterleave2.nxv8i1(<vscale x 8 x i1>)
declare {<vscale x 2 x i1>, <vscale x 2 x i1>} @llvm.vector.deinterleave2.nxv4i1(<vscale x 4 x i1>)

; Illegal size type
declare {<vscale x 4 x i64>, <vscale x 4 x i64>} @llvm.vector.deinterleave2.nxv8i64(<vscale x 8 x i64>)
declare {<vscale x 8 x i64>, <vscale x 8 x i64>} @llvm.vector.deinterleave2.nxv16i64(<vscale x 16 x i64>)

declare {<vscale x 8 x i8>, <vscale x 8 x i8>} @llvm.vector.deinterleave2.nxv16i8(<vscale x 16 x i8>)
declare {<vscale x 4 x i16>, <vscale x 4 x i16>} @llvm.vector.deinterleave2.nxv8i16(<vscale x 8 x i16>)
declare {<vscale x 2 x i32>, <vscale x 2 x i32>} @llvm.vector.deinterleave2.nxv4i32(<vscale x 4 x i32>)
