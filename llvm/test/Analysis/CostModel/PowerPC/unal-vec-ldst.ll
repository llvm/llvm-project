; RUN: opt < %s  -passes="print<cost-model>" 2>&1 -disable-output | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define <16 x i8> @test_l_v16i8(ptr %p) #0 {
entry:
  %r = load <16 x i8>, ptr %p, align 1
  ret <16 x i8> %r

; CHECK-LABEL: test_l_v16i8
; CHECK: cost of 2 for instruction:   %r = load <16 x i8>, ptr %p, align 1
}

define <32 x i8> @test_l_v32i8(ptr %p) #0 {
entry:
  %r = load <32 x i8>, ptr %p, align 1
  ret <32 x i8> %r

; CHECK-LABEL: test_l_v32i8
; CHECK: cost of 4 for instruction:   %r = load <32 x i8>, ptr %p, align 1
}

define <8 x i16> @test_l_v8i16(ptr %p) #0 {
entry:
  %r = load <8 x i16>, ptr %p, align 2
  ret <8 x i16> %r

; CHECK-LABEL: test_l_v8i16
; CHECK: cost of 2 for instruction:   %r = load <8 x i16>, ptr %p, align 2
}

define <16 x i16> @test_l_v16i16(ptr %p) #0 {
entry:
  %r = load <16 x i16>, ptr %p, align 2
  ret <16 x i16> %r

; CHECK-LABEL: test_l_v16i16
; CHECK: cost of 4 for instruction:   %r = load <16 x i16>, ptr %p, align 2
}

define <4 x i32> @test_l_v4i32(ptr %p) #0 {
entry:
  %r = load <4 x i32>, ptr %p, align 4
  ret <4 x i32> %r

; CHECK-LABEL: test_l_v4i32
; CHECK: cost of 2 for instruction:   %r = load <4 x i32>, ptr %p, align 4
}

define <8 x i32> @test_l_v8i32(ptr %p) #0 {
entry:
  %r = load <8 x i32>, ptr %p, align 4
  ret <8 x i32> %r

; CHECK-LABEL: test_l_v8i32
; CHECK: cost of 4 for instruction:   %r = load <8 x i32>, ptr %p, align 4
}

define <2 x i64> @test_l_v2i64(ptr %p) #0 {
entry:
  %r = load <2 x i64>, ptr %p, align 8
  ret <2 x i64> %r

; CHECK-LABEL: test_l_v2i64
; CHECK: cost of 1 for instruction:   %r = load <2 x i64>, ptr %p, align 8
}

define <4 x i64> @test_l_v4i64(ptr %p) #0 {
entry:
  %r = load <4 x i64>, ptr %p, align 8
  ret <4 x i64> %r

; CHECK-LABEL: test_l_v4i64
; CHECK: cost of 2 for instruction:   %r = load <4 x i64>, ptr %p, align 8
}

define <4 x float> @test_l_v4float(ptr %p) #0 {
entry:
  %r = load <4 x float>, ptr %p, align 4
  ret <4 x float> %r

; CHECK-LABEL: test_l_v4float
; CHECK: cost of 2 for instruction:   %r = load <4 x float>, ptr %p, align 4
}

define <8 x float> @test_l_v8float(ptr %p) #0 {
entry:
  %r = load <8 x float>, ptr %p, align 4
  ret <8 x float> %r

; CHECK-LABEL: test_l_v8float
; CHECK: cost of 4 for instruction:   %r = load <8 x float>, ptr %p, align 4
}

define <2 x double> @test_l_v2double(ptr %p) #0 {
entry:
  %r = load <2 x double>, ptr %p, align 8
  ret <2 x double> %r

; CHECK-LABEL: test_l_v2double
; CHECK: cost of 1 for instruction:   %r = load <2 x double>, ptr %p, align 8
}

define <4 x double> @test_l_v4double(ptr %p) #0 {
entry:
  %r = load <4 x double>, ptr %p, align 8
  ret <4 x double> %r

; CHECK-LABEL: test_l_v4double
; CHECK: cost of 2 for instruction:   %r = load <4 x double>, ptr %p, align 8
}

define <16 x i8> @test_l_p8v16i8(ptr %p) #2 {
entry:
  %r = load <16 x i8>, ptr %p, align 1
  ret <16 x i8> %r

; CHECK-LABEL: test_l_p8v16i8
; CHECK: cost of 1 for instruction:   %r = load <16 x i8>, ptr %p, align 1
}

define <32 x i8> @test_l_p8v32i8(ptr %p) #2 {
entry:
  %r = load <32 x i8>, ptr %p, align 1
  ret <32 x i8> %r

; CHECK-LABEL: test_l_p8v32i8
; CHECK: cost of 2 for instruction:   %r = load <32 x i8>, ptr %p, align 1
}

define <8 x i16> @test_l_p8v8i16(ptr %p) #2 {
entry:
  %r = load <8 x i16>, ptr %p, align 2
  ret <8 x i16> %r

; CHECK-LABEL: test_l_p8v8i16
; CHECK: cost of 1 for instruction:   %r = load <8 x i16>, ptr %p, align 2
}

define <16 x i16> @test_l_p8v16i16(ptr %p) #2 {
entry:
  %r = load <16 x i16>, ptr %p, align 2
  ret <16 x i16> %r

; CHECK-LABEL: test_l_p8v16i16
; CHECK: cost of 2 for instruction:   %r = load <16 x i16>, ptr %p, align 2
}

define <4 x i32> @test_l_p8v4i32(ptr %p) #2 {
entry:
  %r = load <4 x i32>, ptr %p, align 4
  ret <4 x i32> %r

; CHECK-LABEL: test_l_p8v4i32
; CHECK: cost of 1 for instruction:   %r = load <4 x i32>, ptr %p, align 4
}

define <8 x i32> @test_l_p8v8i32(ptr %p) #2 {
entry:
  %r = load <8 x i32>, ptr %p, align 4
  ret <8 x i32> %r

; CHECK-LABEL: test_l_p8v8i32
; CHECK: cost of 2 for instruction:   %r = load <8 x i32>, ptr %p, align 4
}

define <2 x i64> @test_l_p8v2i64(ptr %p) #2 {
entry:
  %r = load <2 x i64>, ptr %p, align 8
  ret <2 x i64> %r

; CHECK-LABEL: test_l_p8v2i64
; CHECK: cost of 1 for instruction:   %r = load <2 x i64>, ptr %p, align 8
}

define <4 x i64> @test_l_p8v4i64(ptr %p) #2 {
entry:
  %r = load <4 x i64>, ptr %p, align 8
  ret <4 x i64> %r

; CHECK-LABEL: test_l_p8v4i64
; CHECK: cost of 2 for instruction:   %r = load <4 x i64>, ptr %p, align 8
}

define <4 x float> @test_l_p8v4float(ptr %p) #2 {
entry:
  %r = load <4 x float>, ptr %p, align 4
  ret <4 x float> %r

; CHECK-LABEL: test_l_p8v4float
; CHECK: cost of 1 for instruction:   %r = load <4 x float>, ptr %p, align 4
}

define <8 x float> @test_l_p8v8float(ptr %p) #2 {
entry:
  %r = load <8 x float>, ptr %p, align 4
  ret <8 x float> %r

; CHECK-LABEL: test_l_p8v8float
; CHECK: cost of 2 for instruction:   %r = load <8 x float>, ptr %p, align 4
}

define <2 x double> @test_l_p8v2double(ptr %p) #2 {
entry:
  %r = load <2 x double>, ptr %p, align 8
  ret <2 x double> %r

; CHECK-LABEL: test_l_p8v2double
; CHECK: cost of 1 for instruction:   %r = load <2 x double>, ptr %p, align 8
}

define <4 x double> @test_l_p8v4double(ptr %p) #2 {
entry:
  %r = load <4 x double>, ptr %p, align 8
  ret <4 x double> %r

; CHECK-LABEL: test_l_p8v4double
; CHECK: cost of 2 for instruction:   %r = load <4 x double>, ptr %p, align 8
}

define void @test_s_v16i8(ptr %p, <16 x i8> %v) #0 {
entry:
  store <16 x i8> %v, ptr %p, align 1
  ret void

; CHECK-LABEL: test_s_v16i8
; CHECK: cost of 1 for instruction:   store <16 x i8> %v, ptr %p, align 1
}

define void @test_s_v32i8(ptr %p, <32 x i8> %v) #0 {
entry:
  store <32 x i8> %v, ptr %p, align 1
  ret void

; CHECK-LABEL: test_s_v32i8
; CHECK: cost of 2 for instruction:   store <32 x i8> %v, ptr %p, align 1
}

define void @test_s_v8i16(ptr %p, <8 x i16> %v) #0 {
entry:
  store <8 x i16> %v, ptr %p, align 2
  ret void

; CHECK-LABEL: test_s_v8i16
; CHECK: cost of 1 for instruction:   store <8 x i16> %v, ptr %p, align 2
}

define void @test_s_v16i16(ptr %p, <16 x i16> %v) #0 {
entry:
  store <16 x i16> %v, ptr %p, align 2
  ret void

; CHECK-LABEL: test_s_v16i16
; CHECK: cost of 2 for instruction:   store <16 x i16> %v, ptr %p, align 2
}

define void @test_s_v4i32(ptr %p, <4 x i32> %v) #0 {
entry:
  store <4 x i32> %v, ptr %p, align 4
  ret void

; CHECK-LABEL: test_s_v4i32
; CHECK: cost of 1 for instruction:   store <4 x i32> %v, ptr %p, align 4
}

define void @test_s_v8i32(ptr %p, <8 x i32> %v) #0 {
entry:
  store <8 x i32> %v, ptr %p, align 4
  ret void

; CHECK-LABEL: test_s_v8i32
; CHECK: cost of 2 for instruction:   store <8 x i32> %v, ptr %p, align 4
}

define void @test_s_v2i64(ptr %p, <2 x i64> %v) #0 {
entry:
  store <2 x i64> %v, ptr %p, align 8
  ret void

; CHECK-LABEL: test_s_v2i64
; CHECK: cost of 1 for instruction:   store <2 x i64> %v, ptr %p, align 8
}

define void @test_s_v4i64(ptr %p, <4 x i64> %v) #0 {
entry:
  store <4 x i64> %v, ptr %p, align 8
  ret void

; CHECK-LABEL: test_s_v4i64
; CHECK: cost of 2 for instruction:   store <4 x i64> %v, ptr %p, align 8
}

define void @test_s_v4float(ptr %p, <4 x float> %v) #0 {
entry:
  store <4 x float> %v, ptr %p, align 4
  ret void

; CHECK-LABEL: test_s_v4float
; CHECK: cost of 1 for instruction:   store <4 x float> %v, ptr %p, align 4
}

define void @test_s_v8float(ptr %p, <8 x float> %v) #0 {
entry:
  store <8 x float> %v, ptr %p, align 4
  ret void

; CHECK-LABEL: test_s_v8float
; CHECK: cost of 2 for instruction:   store <8 x float> %v, ptr %p, align 4
}

define void @test_s_v2double(ptr %p, <2 x double> %v) #0 {
entry:
  store <2 x double> %v, ptr %p, align 8
  ret void

; CHECK-LABEL: test_s_v2double
; CHECK: cost of 1 for instruction:   store <2 x double> %v, ptr %p, align 8
}

define void @test_s_v4double(ptr %p, <4 x double> %v) #0 {
entry:
  store <4 x double> %v, ptr %p, align 8
  ret void

; CHECK-LABEL: test_s_v4double
; CHECK: cost of 2 for instruction:   store <4 x double> %v, ptr %p, align 8
}

attributes #0 = { nounwind "target-cpu"="pwr7" }
attributes #2 = { nounwind "target-cpu"="pwr8" }

