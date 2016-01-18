; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=avx < %s | FileCheck %s -check-prefix=CHECK --check-prefix=AVX
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=avx2 < %s | FileCheck %s -check-prefix=CHECK --check-prefix=AVX2

; Check constant loads of every 128-bit and 256-bit vector type 
; for size optimization using splat ops available with AVX and AVX2.

; There is no AVX broadcast from double to 128-bit vector because movddup has been around since SSE3 (grrr).
define <2 x double> @splat_v2f64(<2 x double> %x) #0 {
  %add = fadd <2 x double> %x, <double 1.0, double 1.0>
  ret <2 x double> %add
; CHECK-LABEL: splat_v2f64
; CHECK: vmovddup
; CHECK: vaddpd 
; CHECK-NEXT: retq
}

define <4 x double> @splat_v4f64(<4 x double> %x) #1 {
  %add = fadd <4 x double> %x, <double 1.0, double 1.0, double 1.0, double 1.0>
  ret <4 x double> %add
; CHECK-LABEL: splat_v4f64
; CHECK: vbroadcastsd 
; CHECK-NEXT: vaddpd
; CHECK-NEXT: retq
}

define <4 x float> @splat_v4f32(<4 x float> %x) #0 {
  %add = fadd <4 x float> %x, <float 1.0, float 1.0, float 1.0, float 1.0>
  ret <4 x float> %add
; CHECK-LABEL: splat_v4f32
; CHECK: vbroadcastss 
; CHECK-NEXT: vaddps
; CHECK-NEXT: retq
}

define <8 x float> @splat_v8f32(<8 x float> %x) #1 {
  %add = fadd <8 x float> %x, <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>
  ret <8 x float> %add
; CHECK-LABEL: splat_v8f32
; CHECK: vbroadcastss 
; CHECK-NEXT: vaddps
; CHECK-NEXT: retq
}

; AVX can't do integer splats, so fake it: use vmovddup to splat 64-bit value.
; We also generate vmovddup for AVX2 because it's one byte smaller than vpbroadcastq.
define <2 x i64> @splat_v2i64(<2 x i64> %x) #1 {
  %add = add <2 x i64> %x, <i64 1, i64 1>
  ret <2 x i64> %add
; CHECK-LABEL: splat_v2i64
; CHECK: vmovddup 
; CHECK: vpaddq
; CHECK-NEXT: retq
}

; AVX can't do 256-bit integer ops, so we split this into two 128-bit vectors,
; and then we fake it: use vmovddup to splat 64-bit value.
define <4 x i64> @splat_v4i64(<4 x i64> %x) #0 {
  %add = add <4 x i64> %x, <i64 1, i64 1, i64 1, i64 1>
  ret <4 x i64> %add
; CHECK-LABEL: splat_v4i64
; AVX: vmovddup
; AVX: vpaddq 
; AVX: vpaddq 
; AVX2: vpbroadcastq 
; AVX2: vpaddq 
; CHECK: retq
}

; AVX can't do integer splats, so fake it: use vbroadcastss to splat 32-bit value.
define <4 x i32> @splat_v4i32(<4 x i32> %x) #1 {
  %add = add <4 x i32> %x, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %add
; CHECK-LABEL: splat_v4i32
; AVX: vbroadcastss
; AVX2: vpbroadcastd 
; CHECK-NEXT: vpaddd 
; CHECK-NEXT: retq
}

; AVX can't do integer splats, so fake it: use vbroadcastss to splat 32-bit value.
define <8 x i32> @splat_v8i32(<8 x i32> %x) #0 {
  %add = add <8 x i32> %x, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %add
; CHECK-LABEL: splat_v8i32
; AVX: vbroadcastss
; AVX: vpaddd 
; AVX: vpaddd 
; AVX2: vpbroadcastd 
; AVX2: vpaddd 
; CHECK: retq
}

; AVX can't do integer splats, and there's no broadcast fakery for 16-bit. Could use pshuflw, etc?
define <8 x i16> @splat_v8i16(<8 x i16> %x) #1 {
  %add = add <8 x i16> %x, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %add
; CHECK-LABEL: splat_v8i16
; AVX-NOT: broadcast
; AVX2: vpbroadcastw 
; CHECK: vpaddw 
; CHECK-NEXT: retq
}

; AVX can't do integer splats, and there's no broadcast fakery for 16-bit. Could use pshuflw, etc?
define <16 x i16> @splat_v16i16(<16 x i16> %x) #0 {
  %add = add <16 x i16> %x, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <16 x i16> %add
; CHECK-LABEL: splat_v16i16
; AVX-NOT: broadcast
; AVX: vpaddw 
; AVX: vpaddw 
; AVX2: vpbroadcastw 
; AVX2: vpaddw 
; CHECK: retq
}

; AVX can't do integer splats, and there's no broadcast fakery for 8-bit. Could use pshufb, etc?
define <16 x i8> @splat_v16i8(<16 x i8> %x) #1 {
  %add = add <16 x i8> %x, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %add
; CHECK-LABEL: splat_v16i8
; AVX-NOT: broadcast
; AVX2: vpbroadcastb 
; CHECK: vpaddb 
; CHECK-NEXT: retq
}

; AVX can't do integer splats, and there's no broadcast fakery for 8-bit. Could use pshufb, etc?
define <32 x i8> @splat_v32i8(<32 x i8> %x) #0 {
  %add = add <32 x i8> %x, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <32 x i8> %add
; CHECK-LABEL: splat_v32i8
; AVX-NOT: broadcast
; AVX: vpaddb 
; AVX: vpaddb 
; AVX2: vpbroadcastb 
; AVX2: vpaddb 
; CHECK: retq
}

; PR23259: Verify that ISel doesn't crash with a 'fatal error in backend'
; due to a missing AVX pattern to select a v2i64 X86ISD::BROADCAST of a
; loadi64 with multiple uses.

@A = common global <3 x i64> zeroinitializer, align 32

define <8 x i64> @pr23259() #1 {
entry:
  %0 = load <4 x i64>, <4 x i64>* bitcast (<3 x i64>* @A to <4 x i64>*), align 32
  %1 = shufflevector <4 x i64> %0, <4 x i64> undef, <3 x i32> <i32 undef, i32 undef, i32 2>
  %shuffle = shufflevector <3 x i64> <i64 1, i64 undef, i64 undef>, <3 x i64> %1, <8 x i32> <i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

attributes #0 = { optsize }
attributes #1 = { minsize }
