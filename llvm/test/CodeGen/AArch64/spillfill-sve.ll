; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+sve -mattr=+bf16 < %s | FileCheck %s

; This file checks that unpredicated load/store instructions to locals
; use the right instructions and offsets.

; Data fills

define void @fill_nxv16i8() {
; CHECK-LABEL: fill_nxv16i8
; CHECK-DAG: ld1b    { z{{[01]}}.b }, p0/z, [sp]
; CHECK-DAG: ld1b    { z{{[01]}}.b }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 16 x i8>
  %local1 = alloca <vscale x 16 x i8>
  load volatile <vscale x 16 x i8>, <vscale x 16 x i8>* %local0
  load volatile <vscale x 16 x i8>, <vscale x 16 x i8>* %local1
  ret void
}

define void @fill_nxv8i8() {
; CHECK-LABEL: fill_nxv8i8
; CHECK-DAG: ld1b    { z{{[01]}}.h }, p0/z, [sp]
; CHECK-DAG: ld1b    { z{{[01]}}.h }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 8 x i8>
  %local1 = alloca <vscale x 8 x i8>
  load volatile <vscale x 8 x i8>, <vscale x 8 x i8>* %local0
  load volatile <vscale x 8 x i8>, <vscale x 8 x i8>* %local1
  ret void
}

define <vscale x 8 x i16> @fill_signed_nxv8i8() {
; CHECK-LABEL: fill_signed_nxv8i8
; CHECK-DAG: ld1sb    { z{{[01]}}.h }, p0/z, [sp]
; CHECK-DAG: ld1sb    { z{{[01]}}.h }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 8 x i8>
  %local1 = alloca <vscale x 8 x i8>
  %a = load volatile <vscale x 8 x i8>, <vscale x 8 x i8>* %local0
  %a_ext = sext <vscale x 8 x i8> %a to <vscale x 8 x i16>
  %b = load volatile <vscale x 8 x i8>, <vscale x 8 x i8>* %local1
  %b_ext = sext <vscale x 8 x i8> %b to <vscale x 8 x i16>
  %sum = add <vscale x 8 x i16> %a_ext, %b_ext
  ret <vscale x 8 x i16> %sum
}

define void @fill_nxv4i8() {
; CHECK-LABEL: fill_nxv4i8
; CHECK-DAG: ld1b    { z{{[01]}}.s }, p0/z, [sp, #3, mul vl]
; CHECK-DAG: ld1b    { z{{[01]}}.s }, p0/z, [sp, #2, mul vl]
  %local0 = alloca <vscale x 4 x i8>
  %local1 = alloca <vscale x 4 x i8>
  load volatile <vscale x 4 x i8>, <vscale x 4 x i8>* %local0
  load volatile <vscale x 4 x i8>, <vscale x 4 x i8>* %local1
  ret void
}

define <vscale x 4 x i32> @fill_signed_nxv4i8() {
; CHECK-LABEL: fill_signed_nxv4i8
; CHECK-DAG: ld1sb    { z{{[01]}}.s }, p0/z, [sp, #3, mul vl]
; CHECK-DAG: ld1sb    { z{{[01]}}.s }, p0/z, [sp, #2, mul vl]
  %local0 = alloca <vscale x 4 x i8>
  %local1 = alloca <vscale x 4 x i8>
  %a = load volatile <vscale x 4 x i8>, <vscale x 4 x i8>* %local0
  %a_ext = sext <vscale x 4 x i8> %a to <vscale x 4 x i32>
  %b = load volatile <vscale x 4 x i8>, <vscale x 4 x i8>* %local1
  %b_ext = sext <vscale x 4 x i8> %b to <vscale x 4 x i32>
  %sum = add <vscale x 4 x i32> %a_ext, %b_ext
  ret <vscale x 4 x i32> %sum
}

define void @fill_nxv2i8() {
; CHECK-LABEL: fill_nxv2i8
; CHECK-DAG: ld1b    { z{{[01]}}.d }, p0/z, [sp, #7, mul vl]
; CHECK-DAG: ld1b    { z{{[01]}}.d }, p0/z, [sp, #6, mul vl]
  %local0 = alloca <vscale x 2 x i8>
  %local1 = alloca <vscale x 2 x i8>
  load volatile <vscale x 2 x i8>, <vscale x 2 x i8>* %local0
  load volatile <vscale x 2 x i8>, <vscale x 2 x i8>* %local1
  ret void
}

define <vscale x 2 x i64> @fill_signed_nxv2i8() {
; CHECK-LABEL: fill_signed_nxv2i8
; CHECK-DAG: ld1sb    { z{{[01]}}.d }, p0/z, [sp, #7, mul vl]
; CHECK-DAG: ld1sb    { z{{[01]}}.d }, p0/z, [sp, #6, mul vl]
  %local0 = alloca <vscale x 2 x i8>
  %local1 = alloca <vscale x 2 x i8>
  %a = load volatile <vscale x 2 x i8>, <vscale x 2 x i8>* %local0
  %a_ext = sext <vscale x 2 x i8> %a to <vscale x 2 x i64>
  %b = load volatile <vscale x 2 x i8>, <vscale x 2 x i8>* %local1
  %b_ext = sext <vscale x 2 x i8> %b to <vscale x 2 x i64>
  %sum = add <vscale x 2 x i64> %a_ext, %b_ext
  ret <vscale x 2 x i64> %sum
}

define void @fill_nxv8i16() {
; CHECK-LABEL: fill_nxv8i16
; CHECK-DAG: ld1h    { z{{[01]}}.h }, p0/z, [sp]
; CHECK-DAG: ld1h    { z{{[01]}}.h }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 8 x i16>
  %local1 = alloca <vscale x 8 x i16>
  load volatile <vscale x 8 x i16>, <vscale x 8 x i16>* %local0
  load volatile <vscale x 8 x i16>, <vscale x 8 x i16>* %local1
  ret void
}

define void @fill_nxv4i16() {
; CHECK-LABEL: fill_nxv4i16
; CHECK-DAG: ld1h    { z{{[01]}}.s }, p0/z, [sp]
; CHECK-DAG: ld1h    { z{{[01]}}.s }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 4 x i16>
  %local1 = alloca <vscale x 4 x i16>
  load volatile <vscale x 4 x i16>, <vscale x 4 x i16>* %local0
  load volatile <vscale x 4 x i16>, <vscale x 4 x i16>* %local1
  ret void
}

define <vscale x 4 x i32> @fill_signed_nxv4i16() {
; CHECK-LABEL: fill_signed_nxv4i16
; CHECK-DAG: ld1sh    { z{{[01]}}.s }, p0/z, [sp]
; CHECK-DAG: ld1sh    { z{{[01]}}.s }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 4 x i16>
  %local1 = alloca <vscale x 4 x i16>
  %a = load volatile <vscale x 4 x i16>, <vscale x 4 x i16>* %local0
  %a_ext = sext <vscale x 4 x i16> %a to <vscale x 4 x i32>
  %b = load volatile <vscale x 4 x i16>, <vscale x 4 x i16>* %local1
  %b_ext = sext <vscale x 4 x i16> %b to <vscale x 4 x i32>
  %sum = add <vscale x 4 x i32> %a_ext, %b_ext
  ret <vscale x 4 x i32> %sum
}

define void @fill_nxv2i16() {
; CHECK-LABEL: fill_nxv2i16
; CHECK-DAG: ld1h    { z{{[01]}}.d }, p0/z, [sp, #3, mul vl]
; CHECK-DAG: ld1h    { z{{[01]}}.d }, p0/z, [sp, #2, mul vl]
  %local0 = alloca <vscale x 2 x i16>
  %local1 = alloca <vscale x 2 x i16>
  load volatile <vscale x 2 x i16>, <vscale x 2 x i16>* %local0
  load volatile <vscale x 2 x i16>, <vscale x 2 x i16>* %local1
  ret void
}

define <vscale x 2 x i64> @fill_signed_nxv2i16() {
; CHECK-LABEL: fill_signed_nxv2i16
; CHECK-DAG: ld1sh    { z{{[01]}}.d }, p0/z, [sp, #3, mul vl]
; CHECK-DAG: ld1sh    { z{{[01]}}.d }, p0/z, [sp, #2, mul vl]
  %local0 = alloca <vscale x 2 x i16>
  %local1 = alloca <vscale x 2 x i16>
  %a = load volatile <vscale x 2 x i16>, <vscale x 2 x i16>* %local0
  %a_ext = sext <vscale x 2 x i16> %a to <vscale x 2 x i64>
  %b = load volatile <vscale x 2 x i16>, <vscale x 2 x i16>* %local1
  %b_ext = sext <vscale x 2 x i16> %b to <vscale x 2 x i64>
  %sum = add <vscale x 2 x i64> %a_ext, %b_ext
  ret <vscale x 2 x i64> %sum
}

define void @fill_nxv4i32() {
; CHECK-LABEL: fill_nxv4i32
; CHECK-DAG: ld1w    { z{{[01]}}.s }, p0/z, [sp]
; CHECK-DAG: ld1w    { z{{[01]}}.s }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 4 x i32>
  %local1 = alloca <vscale x 4 x i32>
  load volatile <vscale x 4 x i32>, <vscale x 4 x i32>* %local0
  load volatile <vscale x 4 x i32>, <vscale x 4 x i32>* %local1
  ret void
}

define void @fill_nxv2i32() {
; CHECK-LABEL: fill_nxv2i32
; CHECK-DAG: ld1w    { z{{[01]}}.d }, p0/z, [sp]
; CHECK-DAG: ld1w    { z{{[01]}}.d }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 2 x i32>
  %local1 = alloca <vscale x 2 x i32>
  load volatile <vscale x 2 x i32>, <vscale x 2 x i32>* %local0
  load volatile <vscale x 2 x i32>, <vscale x 2 x i32>* %local1
  ret void
}

define <vscale x 2 x i64> @fill_signed_nxv2i32() {
; CHECK-LABEL: fill_signed_nxv2i32
; CHECK-DAG: ld1sw    { z{{[01]}}.d }, p0/z, [sp]
; CHECK-DAG: ld1sw    { z{{[01]}}.d }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 2 x i32>
  %local1 = alloca <vscale x 2 x i32>
  %a = load volatile <vscale x 2 x i32>, <vscale x 2 x i32>* %local0
  %a_ext = sext <vscale x 2 x i32> %a to <vscale x 2 x i64>
  %b = load volatile <vscale x 2 x i32>, <vscale x 2 x i32>* %local1
  %b_ext = sext <vscale x 2 x i32> %b to <vscale x 2 x i64>
  %sum = add <vscale x 2 x i64> %a_ext, %b_ext
  ret <vscale x 2 x i64> %sum
}

define void @fill_nxv2i64() {
; CHECK-LABEL: fill_nxv2i64
; CHECK-DAG: ld1d    { z{{[01]}}.d }, p0/z, [sp]
; CHECK-DAG: ld1d    { z{{[01]}}.d }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 2 x i64>
  %local1 = alloca <vscale x 2 x i64>
  load volatile <vscale x 2 x i64>, <vscale x 2 x i64>* %local0
  load volatile <vscale x 2 x i64>, <vscale x 2 x i64>* %local1
  ret void
}

define void @fill_nxv8bf16() {
; CHECK-LABEL: fill_nxv8bf16
; CHECK-DAG: ld1h    { z{{[01]}}.h }, p0/z, [sp]
; CHECK-DAG: ld1h    { z{{[01]}}.h }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 8 x bfloat>
  %local1 = alloca <vscale x 8 x bfloat>
  load volatile <vscale x 8 x bfloat>, <vscale x 8 x bfloat>* %local0
  load volatile <vscale x 8 x bfloat>, <vscale x 8 x bfloat>* %local1
  ret void
}

define void @fill_nxv8f16() {
; CHECK-LABEL: fill_nxv8f16
; CHECK-DAG: ld1h    { z{{[01]}}.h }, p0/z, [sp]
; CHECK-DAG: ld1h    { z{{[01]}}.h }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 8 x half>
  %local1 = alloca <vscale x 8 x half>
  load volatile <vscale x 8 x half>, <vscale x 8 x half>* %local0
  load volatile <vscale x 8 x half>, <vscale x 8 x half>* %local1
  ret void
}

define void @fill_nxv4f32() {
; CHECK-LABEL: fill_nxv4f32
; CHECK-DAG: ld1w    { z{{[01]}}.s }, p0/z, [sp]
; CHECK-DAG: ld1w    { z{{[01]}}.s }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 4 x float>
  %local1 = alloca <vscale x 4 x float>
  load volatile <vscale x 4 x float>, <vscale x 4 x float>* %local0
  load volatile <vscale x 4 x float>, <vscale x 4 x float>* %local1
  ret void
}

define void @fill_nxv2f64() {
; CHECK-LABEL: fill_nxv2f64
; CHECK-DAG: ld1d    { z{{[01]}}.d }, p0/z, [sp]
; CHECK-DAG: ld1d    { z{{[01]}}.d }, p0/z, [sp, #1, mul vl]
  %local0 = alloca <vscale x 2 x double>
  %local1 = alloca <vscale x 2 x double>
  load volatile <vscale x 2 x double>, <vscale x 2 x double>* %local0
  load volatile <vscale x 2 x double>, <vscale x 2 x double>* %local1
  ret void
}


; Data spills

define void @spill_nxv16i8(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1) {
; CHECK-LABEL: spill_nxv16i8
; CHECK-DAG: st1b    { z{{[01]}}.b }, p0, [sp]
; CHECK-DAG: st1b    { z{{[01]}}.b }, p0, [sp, #1, mul vl]
  %local0 = alloca <vscale x 16 x i8>
  %local1 = alloca <vscale x 16 x i8>
  store volatile <vscale x 16 x i8> %v0, <vscale x 16 x i8>* %local0
  store volatile <vscale x 16 x i8> %v1, <vscale x 16 x i8>* %local1
  ret void
}

define void @spill_nxv8i8(<vscale x 8 x i8> %v0, <vscale x 8 x i8> %v1) {
; CHECK-LABEL: spill_nxv8i8
; CHECK-DAG: st1b    { z{{[01]}}.h }, p0, [sp]
; CHECK-DAG: st1b    { z{{[01]}}.h }, p0, [sp, #1, mul vl]
  %local0 = alloca <vscale x 8 x i8>
  %local1 = alloca <vscale x 8 x i8>
  store volatile <vscale x 8 x i8> %v0, <vscale x 8 x i8>* %local0
  store volatile <vscale x 8 x i8> %v1, <vscale x 8 x i8>* %local1
  ret void
}

define void @spill_nxv4i8(<vscale x 4 x i8> %v0, <vscale x 4 x i8> %v1) {
; CHECK-LABEL: spill_nxv4i8
; CHECK-DAG: st1b    { z{{[01]}}.s }, p0, [sp, #3, mul vl]
; CHECK-DAG: st1b    { z{{[01]}}.s }, p0, [sp, #2, mul vl]
  %local0 = alloca <vscale x 4 x i8>
  %local1 = alloca <vscale x 4 x i8>
  store volatile <vscale x 4 x i8> %v0, <vscale x 4 x i8>* %local0
  store volatile <vscale x 4 x i8> %v1, <vscale x 4 x i8>* %local1
  ret void
}

define void @spill_nxv2i8(<vscale x 2 x i8> %v0, <vscale x 2 x i8> %v1) {
; CHECK-LABEL: spill_nxv2i8
; CHECK-DAG: st1b    { z{{[01]}}.d }, p0, [sp, #7, mul vl]
; CHECK-DAG: st1b    { z{{[01]}}.d }, p0, [sp, #6, mul vl]
  %local0 = alloca <vscale x 2 x i8>
  %local1 = alloca <vscale x 2 x i8>
  store volatile <vscale x 2 x i8> %v0, <vscale x 2 x i8>* %local0
  store volatile <vscale x 2 x i8> %v1, <vscale x 2 x i8>* %local1
  ret void
}

define void @spill_nxv8i16(<vscale x 8 x i16> %v0, <vscale x 8 x i16> %v1) {
; CHECK-LABEL: spill_nxv8i16
; CHECK-DAG: st1h    { z{{[01]}}.h }, p0, [sp]
; CHECK-DAG: st1h    { z{{[01]}}.h }, p0, [sp, #1, mul vl]
  %local0 = alloca <vscale x 8 x i16>
  %local1 = alloca <vscale x 8 x i16>
  store volatile <vscale x 8 x i16> %v0, <vscale x 8 x i16>* %local0
  store volatile <vscale x 8 x i16> %v1, <vscale x 8 x i16>* %local1
  ret void
}

define void @spill_nxv4i16(<vscale x 4 x i16> %v0, <vscale x 4 x i16> %v1) {
; CHECK-LABEL: spill_nxv4i16
; CHECK-DAG: st1h    { z{{[01]}}.s }, p0, [sp]
; CHECK-DAG: st1h    { z{{[01]}}.s }, p0, [sp, #1, mul vl]
  %local0 = alloca <vscale x 4 x i16>
  %local1 = alloca <vscale x 4 x i16>
  store volatile <vscale x 4 x i16> %v0, <vscale x 4 x i16>* %local0
  store volatile <vscale x 4 x i16> %v1, <vscale x 4 x i16>* %local1
  ret void
}

define void @spill_nxv2i16(<vscale x 2 x i16> %v0, <vscale x 2 x i16> %v1) {
; CHECK-LABEL: spill_nxv2i16
; CHECK-DAG: st1h    { z{{[01]}}.d }, p0, [sp, #3, mul vl]
; CHECK-DAG: st1h    { z{{[01]}}.d }, p0, [sp, #2, mul vl]
  %local0 = alloca <vscale x 2 x i16>
  %local1 = alloca <vscale x 2 x i16>
  store volatile <vscale x 2 x i16> %v0, <vscale x 2 x i16>* %local0
  store volatile <vscale x 2 x i16> %v1, <vscale x 2 x i16>* %local1
  ret void
}

define void @spill_nxv4i32(<vscale x 4 x i32> %v0, <vscale x 4 x i32> %v1) {
; CHECK-LABEL: spill_nxv4i32
; CHECK-DAG: st1w    { z{{[01]}}.s }, p0, [sp]
; CHECK-DAG: st1w    { z{{[01]}}.s }, p0, [sp, #1, mul vl]
  %local0 = alloca <vscale x 4 x i32>
  %local1 = alloca <vscale x 4 x i32>
  store volatile <vscale x 4 x i32> %v0, <vscale x 4 x i32>* %local0
  store volatile <vscale x 4 x i32> %v1, <vscale x 4 x i32>* %local1
  ret void
}

define void @spill_nxv2i32(<vscale x 2 x i32> %v0, <vscale x 2 x i32> %v1) {
; CHECK-LABEL: spill_nxv2i32
; CHECK-DAG: st1w    { z{{[01]}}.d }, p0, [sp]
; CHECK-DAG: st1w    { z{{[01]}}.d }, p0, [sp, #1, mul vl]
  %local0 = alloca <vscale x 2 x i32>
  %local1 = alloca <vscale x 2 x i32>
  store volatile <vscale x 2 x i32> %v0, <vscale x 2 x i32>* %local0
  store volatile <vscale x 2 x i32> %v1, <vscale x 2 x i32>* %local1
  ret void
}

define void @spill_nxv2i64(<vscale x 2 x i64> %v0, <vscale x 2 x i64> %v1) {
; CHECK-LABEL: spill_nxv2i64
; CHECK-DAG: st1d    { z{{[01]}}.d }, p0, [sp]
; CHECK-DAG: st1d    { z{{[01]}}.d }, p0, [sp, #1, mul vl]
  %local0 = alloca <vscale x 2 x i64>
  %local1 = alloca <vscale x 2 x i64>
  store volatile <vscale x 2 x i64> %v0, <vscale x 2 x i64>* %local0
  store volatile <vscale x 2 x i64> %v1, <vscale x 2 x i64>* %local1
  ret void
}

define void @spill_nxv8f16(<vscale x 8 x half> %v0, <vscale x 8 x half> %v1) {
; CHECK-LABEL: spill_nxv8f16
; CHECK-DAG: st1h    { z{{[01]}}.h }, p0, [sp]
; CHECK-DAG: st1h    { z{{[01]}}.h }, p0, [sp, #1, mul vl]
  %local0 = alloca <vscale x 8 x half>
  %local1 = alloca <vscale x 8 x half>
  store volatile <vscale x 8 x half> %v0, <vscale x 8 x half>* %local0
  store volatile <vscale x 8 x half> %v1, <vscale x 8 x half>* %local1
  ret void
}

define void @spill_nxv8bf16(<vscale x 8 x bfloat> %v0, <vscale x 8 x bfloat> %v1) {
; CHECK-LABEL: spill_nxv8bf16
; CHECK-DAG: st1h    { z{{[01]}}.h }, p0, [sp]
; CHECK-DAG: st1h    { z{{[01]}}.h }, p0, [sp, #1, mul vl]
  %local0 = alloca <vscale x 8 x bfloat>
  %local1 = alloca <vscale x 8 x bfloat>
  store volatile <vscale x 8 x bfloat> %v0, <vscale x 8 x bfloat>* %local0
  store volatile <vscale x 8 x bfloat> %v1, <vscale x 8 x bfloat>* %local1
  ret void
}

define void @spill_nxv4f32(<vscale x 4 x float> %v0, <vscale x 4 x float> %v1) {
; CHECK-LABEL: spill_nxv4f32
; CHECK-DAG: st1w    { z{{[01]}}.s }, p0, [sp]
; CHECK-DAG: st1w    { z{{[01]}}.s }, p0, [sp, #1, mul vl]
  %local0 = alloca <vscale x 4 x float>
  %local1 = alloca <vscale x 4 x float>
  store volatile <vscale x 4 x float> %v0, <vscale x 4 x float>* %local0
  store volatile <vscale x 4 x float> %v1, <vscale x 4 x float>* %local1
  ret void
}

define void @spill_nxv2f64(<vscale x 2 x double> %v0, <vscale x 2 x double> %v1) {
; CHECK-LABEL: spill_nxv2f64
; CHECK-DAG: st1d    { z{{[01]}}.d }, p0, [sp]
; CHECK-DAG: st1d    { z{{[01]}}.d }, p0, [sp, #1, mul vl]
  %local0 = alloca <vscale x 2 x double>
  %local1 = alloca <vscale x 2 x double>
  store volatile <vscale x 2 x double> %v0, <vscale x 2 x double>* %local0
  store volatile <vscale x 2 x double> %v1, <vscale x 2 x double>* %local1
  ret void
}

; Predicate fills

define void @fill_nxv16i1() {
; CHECK-LABEL: fill_nxv16i1
; CHECK-DAG: ldr    p{{[01]}}, [sp, #7, mul vl]
; CHECK-DAG: ldr    p{{[01]}}, [sp, #6, mul vl]
  %local0 = alloca <vscale x 16 x i1>
  %local1 = alloca <vscale x 16 x i1>
  load volatile <vscale x 16 x i1>, <vscale x 16 x i1>* %local0
  load volatile <vscale x 16 x i1>, <vscale x 16 x i1>* %local1
  ret void
}

; Predicate spills

define void @spill_nxv16i1(<vscale x 16 x i1> %v0, <vscale x 16 x i1> %v1) {
; CHECK-LABEL: spill_nxv16i1
; CHECK-DAG: str    p{{[01]}}, [sp, #7, mul vl]
; CHECK-DAG: str    p{{[01]}}, [sp, #6, mul vl]
  %local0 = alloca <vscale x 16 x i1>
  %local1 = alloca <vscale x 16 x i1>
  store volatile <vscale x 16 x i1> %v0, <vscale x 16 x i1>* %local0
  store volatile <vscale x 16 x i1> %v1, <vscale x 16 x i1>* %local1
  ret void
}
