; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -wasm-keep-registers -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -mattr=+simd128 | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -wasm-keep-registers -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals | FileCheck %s --check-prefixes CHECK,NO-SIMD128

; Test that vector float-to-int and int-to-float instructions lower correctly

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: convert_s_v4f32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype convert_s_v4f32 (v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.convert_i32x4_s $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
define <4 x float> @convert_s_v4f32(<4 x i32> %x) {
  %a = sitofp <4 x i32> %x to <4 x float>
  ret <4 x float> %a
}

; CHECK-LABEL: convert_u_v4f32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype convert_u_v4f32 (v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.convert_i32x4_u $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
define <4 x float> @convert_u_v4f32(<4 x i32> %x) {
  %a = uitofp <4 x i32> %x to <4 x float>
  ret <4 x float> %a
}

; CHECK-LABEL: convert_s_v2f64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NOT: f64x2.convert_i64x2_s
; SIMD128-NEXT: .functype convert_s_v2f64 (v128) -> (v128){{$}}
define <2 x double> @convert_s_v2f64(<2 x i64> %x) {
  %a = sitofp <2 x i64> %x to <2 x double>
  ret <2 x double> %a
}

; CHECK-LABEL: convert_u_v2f64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NOT: f64x2.convert_i64x2_u
; SIMD128-NEXT: .functype convert_u_v2f64 (v128) -> (v128){{$}}
define <2 x double> @convert_u_v2f64(<2 x i64> %x) {
  %a = uitofp <2 x i64> %x to <2 x double>
  ret <2 x double> %a
}

; CHECK-LABEL: trunc_sat_s_v4i32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype trunc_sat_s_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.trunc_sat_f32x4_s $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
define <4 x i32> @trunc_sat_s_v4i32(<4 x float> %x) {
  %a = fptosi <4 x float> %x to <4 x i32>
  ret <4 x i32> %a
}

; CHECK-LABEL: trunc_sat_u_v4i32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype trunc_sat_u_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.trunc_sat_f32x4_u $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
define <4 x i32> @trunc_sat_u_v4i32(<4 x float> %x) {
  %a = fptoui <4 x float> %x to <4 x i32>
  ret <4 x i32> %a
}

; CHECK-LABEL: trunc_sat_s_v2i64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NOT: i64x2.trunc_sat_f64x2_s
; SIMD128-NEXT: .functype trunc_sat_s_v2i64 (v128) -> (v128){{$}}
define <2 x i64> @trunc_sat_s_v2i64(<2 x double> %x) {
  %a = fptosi <2 x double> %x to <2 x i64>
  ret <2 x i64> %a
}

; CHECK-LABEL: trunc_sat_u_v2i64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NOT: i64x2.trunc_sat_f64x2_u
; SIMD128-NEXT: .functype trunc_sat_u_v2i64 (v128) -> (v128){{$}}
define <2 x i64> @trunc_sat_u_v2i64(<2 x double> %x) {
  %a = fptoui <2 x double> %x to <2 x i64>
  ret <2 x i64> %a
}

; CHECK-LABEL: demote_zero_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype demote_zero_v4f32 (v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.demote_zero_f64x2 $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
define <4 x float> @demote_zero_v4f32(<2 x double> %x) {
  %v = shufflevector <2 x double> %x, <2 x double> zeroinitializer,
         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %a = fptrunc <4 x double> %v to <4 x float>
  ret <4 x float> %a
}

; CHECK-LABEL: demote_zero_v4f32_2:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype demote_zero_v4f32_2 (v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.demote_zero_f64x2 $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
define <4 x float> @demote_zero_v4f32_2(<2 x double> %x) {
  %v = fptrunc <2 x double> %x to <2 x float>
  %a = shufflevector <2 x float> %v, <2 x float> zeroinitializer,
         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %a
}

; CHECK-LABEL: convert_low_s_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype convert_low_s_v2f64 (v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.convert_low_i32x4_s $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
define <2 x double> @convert_low_s_v2f64(<4 x i32> %x) {
  %v = shufflevector <4 x i32> %x, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %a = sitofp <2 x i32> %v to <2 x double>
  ret <2 x double> %a
}

; CHECK-LABEL: convert_low_u_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype convert_low_u_v2f64 (v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.convert_low_i32x4_u $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
define <2 x double> @convert_low_u_v2f64(<4 x i32> %x) {
  %v = shufflevector <4 x i32> %x, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %a = uitofp <2 x i32> %v to <2 x double>
  ret <2 x double> %a
}


; CHECK-LABEL: convert_low_s_v2f64_2:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype convert_low_s_v2f64_2 (v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.convert_low_i32x4_s $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
define <2 x double> @convert_low_s_v2f64_2(<4 x i32> %x) {
  %v = sitofp <4 x i32> %x to <4 x double>
  %a = shufflevector <4 x double> %v, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %a
}

; CHECK-LABEL: convert_low_u_v2f64_2:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype convert_low_u_v2f64_2 (v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.convert_low_i32x4_u $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
define <2 x double> @convert_low_u_v2f64_2(<4 x i32> %x) {
  %v = uitofp <4 x i32> %x to <4 x double>
  %a = shufflevector <4 x double> %v, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %a
}

; CHECK-LABEL: promote_low_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype promote_low_v2f64 (v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.promote_low_f32x4 $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
define <2 x double> @promote_low_v2f64(<4 x float> %x) {
  %v = shufflevector <4 x float> %x, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  %a = fpext <2 x float> %v to <2 x double>
  ret <2 x double> %a
}

; CHECK-LABEL: promote_low_v2f64_2:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype promote_low_v2f64_2 (v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.promote_low_f32x4 $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
define <2 x double> @promote_low_v2f64_2(<4 x float> %x) {
  %v = fpext <4 x float> %x to <4 x double>
  %a = shufflevector <4 x double> %v, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %a
}
