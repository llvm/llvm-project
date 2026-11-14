; Test that v1024i32 mul nuw with operands that are statically known to fit
; in 16 bits is correctly lowered to LX_mpy_uw_uh.
;
; DetectUseZxth is extended to recognize:
;   - lshr v1024i32, splat(C) where C >= 16  (result <= 65535)
;   - and  v1024i32, splat(0xFFFF)            (result <= 65535)
;   - splat_vector(small_constant)            (constant <= 65535)
;   - splat_vector(zext i16 x)               (scalar fits in 16 bits)
;
; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx < %s | FileCheck %s

; All mul nuw v1024i32 operations must lower to mpy (LX_mpy_uw_uh).
; This IR was referenced from the softmax exp kernel pattern crash.

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048-v16384:16384:16384-v32768:16384:16384"
target triple = "hexagon"

; scale is i16 (fits in 16 bits)
; CHECK-LABEL: test_mul_splat_zext:
; CHECK: .uw = mpy({{.*}}.uh,{{.*}}.uh)
define void @test_mul_splat_zext(ptr %input, ptr %output, i16 zeroext %scale) {
entry:
  ; zext i16 -> i32: recognized as HLXZext32 via ZERO_EXTEND
  %scale32 = zext i16 %scale to i32
  %scale_insert = insertelement <1024 x i32> poison, i32 %scale32, i64 0
  ; splat(zext i16)
  %scale_splat = shufflevector <1024 x i32> %scale_insert, <1024 x i32> poison, <1024 x i32> zeroinitializer
  %data = load <1024 x i16>, ptr %input, align 2048
  ; zext v1024i16 -> v1024i32
  %data_ext = zext <1024 x i16> %data to <1024 x i32>
  ; mul nuw: splat(zext i16) * zext(i16) — both fit in 16 bits
  %mul = mul nuw <1024 x i32> %scale_splat, %data_ext
  tail call void @llvm.hexagon.LX.SS16b.new(ptr %output, <1024 x i32> %mul)
  ret void
}

; CHECK-LABEL: test_mul_and_splat:
; CHECK: .uw = mpy({{.*}}.uh,{{.*}}.uh)
define void @test_mul_and_splat(<1024 x i32> %input, ptr %output) {
entry:
  ; and with 0xFFFF — result fits in 16 bits (recognized by AND case)
  %masked = and <1024 x i32> %input, splat (i32 65535)
  ; mul nuw: (and ... 0xFFFF) * splat(112) — both fit in 16 bits
  %mul = mul nuw <1024 x i32> %masked, splat (i32 112)
  tail call void @llvm.hexagon.LX.SS16b.new(ptr %output, <1024 x i32> %mul)
  ret void
}

; CHECK-LABEL: test_mul_srl_and:
; CHECK: .uw = mpy({{.*}}.uh,{{.*}}.uh)
define void @test_mul_srl_and(<1024 x i32> %input0, <1024 x i32> %input1, ptr %output) {
entry:
  ; and with 0xFFFF — result fits in 16 bits (recognized by AND case)
  %masked = and <1024 x i32> %input0, splat (i32 65535)
  %add = add <1024 x i32> %input1, splat (i32 -114262016)
  ; lshr by constant 16 — result fits in 16 bits
  %shr = lshr <1024 x i32> %add, splat (i32 16)
  ; mul nuw: (lshr ... 16) * (and ... 0xFFFF) — both fit in 16 bits
  %mul = mul nuw <1024 x i32> %shr, %masked
  tail call void @llvm.hexagon.LX.SS16b.new(ptr %output, <1024 x i32> %mul)
  ret void
}

declare void @llvm.hexagon.LX.SS16b.new(ptr, <1024 x i32>)
