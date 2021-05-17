; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck %s

; Usually MIPS hosts uses a legacy (non IEEE 754-2008) encoding for NaNs.
; Tests like `nan_f32` failed in attempt to compare hard-coded IEEE 754-2008
; NaN value and a legacy NaN value provided by a system.
; XFAIL: mips-, mipsel-, mips64-, mips64el-

; Test that basic immediates assemble as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: zero_i32:
; CHECK-NEXT: .functype zero_i32 () -> (i32){{$}}
; CHECK-NEXT: i32.const $push[[NUM:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @zero_i32() {
  ret i32 0
}

; CHECK-LABEL: one_i32:
; CHECK-NEXT: .functype one_i32 () -> (i32){{$}}
; CHECK-NEXT: i32.const $push[[NUM:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @one_i32() {
  ret i32 1
}

; CHECK-LABEL: max_i32:
; CHECK-NEXT: .functype max_i32 () -> (i32){{$}}
; CHECK-NEXT: i32.const $push[[NUM:[0-9]+]]=, 2147483647{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @max_i32() {
  ret i32 2147483647
}

; CHECK-LABEL: min_i32:
; CHECK-NEXT: .functype min_i32 () -> (i32){{$}}
; CHECK-NEXT: i32.const $push[[NUM:[0-9]+]]=, -2147483648{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @min_i32() {
  ret i32 -2147483648
}

; CHECK-LABEL: zero_i64:
; CHECK-NEXT: .functype zero_i64 () -> (i64){{$}}
; CHECK-NEXT: i64.const $push[[NUM:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @zero_i64() {
  ret i64 0
}

; CHECK-LABEL: one_i64:
; CHECK-NEXT: .functype one_i64 () -> (i64){{$}}
; CHECK-NEXT: i64.const $push[[NUM:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @one_i64() {
  ret i64 1
}

; CHECK-LABEL: max_i64:
; CHECK-NEXT: .functype max_i64 () -> (i64){{$}}
; CHECK-NEXT: i64.const $push[[NUM:[0-9]+]]=, 9223372036854775807{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @max_i64() {
  ret i64 9223372036854775807
}

; CHECK-LABEL: min_i64:
; CHECK-NEXT: .functype min_i64 () -> (i64){{$}}
; CHECK-NEXT: i64.const $push[[NUM:[0-9]+]]=, -9223372036854775808{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @min_i64() {
  ret i64 -9223372036854775808
}

; CHECK-LABEL: negzero_f32:
; CHECK-NEXT: .functype negzero_f32 () -> (f32){{$}}
; CHECK-NEXT: f32.const $push[[NUM:[0-9]+]]=, -0x0p0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @negzero_f32() {
  ret float -0.0
}

; CHECK-LABEL: zero_f32:
; CHECK-NEXT: .functype zero_f32 () -> (f32){{$}}
; CHECK-NEXT: f32.const $push[[NUM:[0-9]+]]=, 0x0p0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @zero_f32() {
  ret float 0.0
}

; CHECK-LABEL: one_f32:
; CHECK-NEXT: .functype one_f32 () -> (f32){{$}}
; CHECK-NEXT: f32.const $push[[NUM:[0-9]+]]=, 0x1p0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @one_f32() {
  ret float 1.0
}

; CHECK-LABEL: two_f32:
; CHECK-NEXT: .functype two_f32 () -> (f32){{$}}
; CHECK-NEXT: f32.const $push[[NUM:[0-9]+]]=, 0x1p1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @two_f32() {
  ret float 2.0
}

; CHECK-LABEL: nan_f32:
; CHECK-NEXT: .functype nan_f32 () -> (f32){{$}}
; CHECK-NEXT: f32.const $push[[NUM:[0-9]+]]=, nan{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @nan_f32() {
  ret float 0x7FF8000000000000
}

; CHECK-LABEL: negnan_f32:
; CHECK-NEXT: .functype negnan_f32 () -> (f32){{$}}
; CHECK-NEXT: f32.const $push[[NUM:[0-9]+]]=, -nan{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @negnan_f32() {
  ret float 0xFFF8000000000000
}

; CHECK-LABEL: inf_f32:
; CHECK-NEXT: .functype inf_f32 () -> (f32){{$}}
; CHECK-NEXT: f32.const $push[[NUM:[0-9]+]]=, infinity{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @inf_f32() {
  ret float 0x7FF0000000000000
}

; CHECK-LABEL: neginf_f32:
; CHECK-NEXT: .functype neginf_f32 () -> (f32){{$}}
; CHECK-NEXT: f32.const $push[[NUM:[0-9]+]]=, -infinity{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @neginf_f32() {
  ret float 0xFFF0000000000000
}

; CHECK-LABEL: custom_nan_f32:
; CHECK-NEXT: .functype custom_nan_f32 () -> (f32){{$}}
; CHECK-NEXT: f32.const $push[[NUM:[0-9]+]]=, -nan:0x6bcdef{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @custom_nan_f32() {
  ret float 0xFFFD79BDE0000000
}

; CHECK-LABEL: custom_nans_f32:
; CHECK-NEXT: .functype custom_nans_f32 () -> (f32){{$}}
; CHECK-NEXT: f32.const $push[[NUM:[0-9]+]]=, -nan:0x2bcdef{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @custom_nans_f32() {
  ret float 0xFFF579BDE0000000
}

; CHECK-LABEL: negzero_f64:
; CHECK-NEXT: .functype negzero_f64 () -> (f64){{$}}
; CHECK-NEXT: f64.const $push[[NUM:[0-9]+]]=, -0x0p0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @negzero_f64() {
  ret double -0.0
}

; CHECK-LABEL: zero_f64:
; CHECK-NEXT: .functype zero_f64 () -> (f64){{$}}
; CHECK-NEXT: f64.const $push[[NUM:[0-9]+]]=, 0x0p0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @zero_f64() {
  ret double 0.0
}

; CHECK-LABEL: one_f64:
; CHECK-NEXT: .functype one_f64 () -> (f64){{$}}
; CHECK-NEXT: f64.const $push[[NUM:[0-9]+]]=, 0x1p0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @one_f64() {
  ret double 1.0
}

; CHECK-LABEL: two_f64:
; CHECK-NEXT: .functype two_f64 () -> (f64){{$}}
; CHECK-NEXT: f64.const $push[[NUM:[0-9]+]]=, 0x1p1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @two_f64() {
  ret double 2.0
}

; CHECK-LABEL: nan_f64:
; CHECK-NEXT: .functype nan_f64 () -> (f64){{$}}
; CHECK-NEXT: f64.const $push[[NUM:[0-9]+]]=, nan{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @nan_f64() {
  ret double 0x7FF8000000000000
}

; CHECK-LABEL: negnan_f64:
; CHECK-NEXT: .functype negnan_f64 () -> (f64){{$}}
; CHECK-NEXT: f64.const $push[[NUM:[0-9]+]]=, -nan{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @negnan_f64() {
  ret double 0xFFF8000000000000
}

; CHECK-LABEL: inf_f64:
; CHECK-NEXT: .functype inf_f64 () -> (f64){{$}}
; CHECK-NEXT: f64.const $push[[NUM:[0-9]+]]=, infinity{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @inf_f64() {
  ret double 0x7FF0000000000000
}

; CHECK-LABEL: neginf_f64:
; CHECK-NEXT: .functype neginf_f64 () -> (f64){{$}}
; CHECK-NEXT: f64.const $push[[NUM:[0-9]+]]=, -infinity{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @neginf_f64() {
  ret double 0xFFF0000000000000
}

;; Custom NaN playloads are currently not always preserved because of the use of
;; native doubles in the MC layer. TODO: fix this problem or decide we don't
;; care about preserving NaN payloads.

; XXX-CHECK-LABEL: custom_nan_f64:
; XXX-CHECK-NEXT: .functype custom_nan_f64 () -> (f64){{$}}
; XXX-CHECK-NEXT: f64.const $push[[NUM:[0-9]+]]=, -nan:0xabcdef0123456{{$}}
; XXX-CHECK-NEXT: return $pop[[NUM]]{{$}}
; define double @custom_nan_f64() {
;   ret double 0xFFFABCDEF0123456
; }

; XXX-CHECK-LABEL: custom_nans_f64:
; XXX-CHECK-NEXT: .functype custom_nans_f64 () -> (f64){{$}}
; XXX-CHECK-NEXT: f64.const $push[[NUM:[0-9]+]]=, -nan:0x2bcdef0123456{{$}}
; XXX-CHECK-NEXT: return $pop[[NUM]]{{$}}
; define double @custom_nans_f64() {
;   ret double 0xFFF2BCDEF0123456
; }
