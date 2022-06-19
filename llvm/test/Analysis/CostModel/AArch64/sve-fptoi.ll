; RUN: opt -passes='print<cost-model>' 2>&1 -disable-output -mtriple aarch64-linux-gnu -mattr=+sve -o - -S < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @sve-fptoi() {
; CHECK-LABEL: 'sve-fptoi'
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %nv1f16_to_si8 = fptosi <vscale x 1 x half> undef to <vscale x 1 x i8>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %nv1f16_to_ui8 = fptoui <vscale x 1 x half> undef to <vscale x 1 x i8>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %nv1f16_to_si32 = fptosi <vscale x 1 x half> undef to <vscale x 1 x i32>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %nv1f16_to_ui32 = fptoui <vscale x 1 x half> undef to <vscale x 1 x i32>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %nv1f16_to_si64 = fptosi <vscale x 1 x half> undef to <vscale x 1 x i64>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %nv1f16_to_ui64 = fptoui <vscale x 1 x half> undef to <vscale x 1 x i64>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %nv1f32_to_si8 = fptosi <vscale x 1 x float> undef to <vscale x 1 x i8>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %nv1f32_to_ui8 = fptoui <vscale x 1 x float> undef to <vscale x 1 x i8>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %nv1f32_to_si16 = fptosi <vscale x 1 x float> undef to <vscale x 1 x i16>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %nv1f32_to_ui16 = fptoui <vscale x 1 x float> undef to <vscale x 1 x i16>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %nv1f32_to_si64 = fptosi <vscale x 1 x float> undef to <vscale x 1 x i64>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %nv1f32_to_ui64 = fptoui <vscale x 1 x float> undef to <vscale x 1 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv1f64_to_si8 = fptosi <vscale x 1 x double> undef to <vscale x 1 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv1f64_to_ui8 = fptoui <vscale x 1 x double> undef to <vscale x 1 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv1f64_to_si16 = fptosi <vscale x 1 x double> undef to <vscale x 1 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv1f64_to_ui16 = fptoui <vscale x 1 x double> undef to <vscale x 1 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv1f64_to_si32 = fptosi <vscale x 1 x double> undef to <vscale x 1 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv1f64_to_ui32 = fptoui <vscale x 1 x double> undef to <vscale x 1 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f16_to_si8 = fptosi <vscale x 2 x half> undef to <vscale x 2 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f16_to_ui8 = fptoui <vscale x 2 x half> undef to <vscale x 2 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f16_to_si32 = fptosi <vscale x 2 x half> undef to <vscale x 2 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f16_to_ui32 = fptoui <vscale x 2 x half> undef to <vscale x 2 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f16_to_si64 = fptosi <vscale x 2 x half> undef to <vscale x 2 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f16_to_ui64 = fptoui <vscale x 2 x half> undef to <vscale x 2 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f32_to_si8 = fptosi <vscale x 2 x float> undef to <vscale x 2 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f32_to_ui8 = fptoui <vscale x 2 x float> undef to <vscale x 2 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f32_to_si16 = fptosi <vscale x 2 x float> undef to <vscale x 2 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f32_to_ui16 = fptoui <vscale x 2 x float> undef to <vscale x 2 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f32_to_si64 = fptosi <vscale x 2 x float> undef to <vscale x 2 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f32_to_ui64 = fptoui <vscale x 2 x float> undef to <vscale x 2 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f64_to_si8 = fptosi <vscale x 2 x double> undef to <vscale x 2 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f64_to_ui8 = fptoui <vscale x 2 x double> undef to <vscale x 2 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f64_to_si16 = fptosi <vscale x 2 x double> undef to <vscale x 2 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f64_to_ui16 = fptoui <vscale x 2 x double> undef to <vscale x 2 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f64_to_si32 = fptosi <vscale x 2 x double> undef to <vscale x 2 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f64_to_ui32 = fptoui <vscale x 2 x double> undef to <vscale x 2 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv4f16_to_si8 = fptosi <vscale x 4 x half> undef to <vscale x 4 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv4f16_to_ui8 = fptoui <vscale x 4 x half> undef to <vscale x 4 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv4f16_to_si32 = fptosi <vscale x 4 x half> undef to <vscale x 4 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv4f16_to_ui32 = fptoui <vscale x 4 x half> undef to <vscale x 4 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %nv4f16_to_si64 = fptosi <vscale x 4 x half> undef to <vscale x 4 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %nv4f16_to_ui64 = fptoui <vscale x 4 x half> undef to <vscale x 4 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv4f32_to_si8 = fptosi <vscale x 4 x float> undef to <vscale x 4 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv4f32_to_ui8 = fptoui <vscale x 4 x float> undef to <vscale x 4 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv4f32_to_si16 = fptosi <vscale x 4 x float> undef to <vscale x 4 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv4f32_to_ui16 = fptoui <vscale x 4 x float> undef to <vscale x 4 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %nv4f32_to_si64 = fptosi <vscale x 4 x float> undef to <vscale x 4 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %nv4f32_to_ui64 = fptoui <vscale x 4 x float> undef to <vscale x 4 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nv4f64_to_si8 = fptosi <vscale x 4 x double> undef to <vscale x 4 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nv4f64_to_ui8 = fptoui <vscale x 4 x double> undef to <vscale x 4 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nv4f64_to_si16 = fptosi <vscale x 4 x double> undef to <vscale x 4 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nv4f64_to_ui16 = fptoui <vscale x 4 x double> undef to <vscale x 4 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nv4f64_to_si32 = fptosi <vscale x 4 x double> undef to <vscale x 4 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nv4f64_to_ui32 = fptoui <vscale x 4 x double> undef to <vscale x 4 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv8f16_to_si8 = fptosi <vscale x 8 x half> undef to <vscale x 8 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv8f16_to_ui8 = fptoui <vscale x 8 x half> undef to <vscale x 8 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %nv8f16_to_si32 = fptosi <vscale x 8 x half> undef to <vscale x 8 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %nv8f16_to_ui32 = fptoui <vscale x 8 x half> undef to <vscale x 8 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 10 for instruction:   %nv8f16_to_si64 = fptosi <vscale x 8 x half> undef to <vscale x 8 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 10 for instruction:   %nv8f16_to_ui64 = fptoui <vscale x 8 x half> undef to <vscale x 8 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nv8f32_to_si8 = fptosi <vscale x 8 x float> undef to <vscale x 8 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nv8f32_to_ui8 = fptoui <vscale x 8 x float> undef to <vscale x 8 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nv8f32_to_si16 = fptosi <vscale x 8 x float> undef to <vscale x 8 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nv8f32_to_ui16 = fptoui <vscale x 8 x float> undef to <vscale x 8 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 8 for instruction:   %nv8f32_to_si64 = fptosi <vscale x 8 x float> undef to <vscale x 8 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 8 for instruction:   %nv8f32_to_ui64 = fptoui <vscale x 8 x float> undef to <vscale x 8 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 7 for instruction:   %nv8f64_to_si8 = fptosi <vscale x 8 x double> undef to <vscale x 8 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 7 for instruction:   %nv8f64_to_ui8 = fptoui <vscale x 8 x double> undef to <vscale x 8 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 7 for instruction:   %nv8f64_to_si16 = fptosi <vscale x 8 x double> undef to <vscale x 8 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 7 for instruction:   %nv8f64_to_ui16 = fptoui <vscale x 8 x double> undef to <vscale x 8 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %nv8f64_to_si32 = fptosi <vscale x 8 x double> undef to <vscale x 8 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %nv8f64_to_ui32 = fptoui <vscale x 8 x double> undef to <vscale x 8 x i32>
  %nv1f16_to_si8  = fptosi <vscale x 1 x half> undef to <vscale x 1 x i8>
  %nv1f16_to_ui8  = fptoui <vscale x 1 x half> undef to <vscale x 1 x i8>
  %nv1f16_to_si32 = fptosi <vscale x 1 x half> undef to <vscale x 1 x i32>
  %nv1f16_to_ui32 = fptoui <vscale x 1 x half> undef to <vscale x 1 x i32>
  %nv1f16_to_si64 = fptosi <vscale x 1 x half> undef to <vscale x 1 x i64>
  %nv1f16_to_ui64 = fptoui <vscale x 1 x half> undef to <vscale x 1 x i64>

  %nv1f32_to_si8  = fptosi <vscale x 1 x float> undef to <vscale x 1 x i8>
  %nv1f32_to_ui8  = fptoui <vscale x 1 x float> undef to <vscale x 1 x i8>
  %nv1f32_to_si16 = fptosi <vscale x 1 x float> undef to <vscale x 1 x i16>
  %nv1f32_to_ui16 = fptoui <vscale x 1 x float> undef to <vscale x 1 x i16>
  %nv1f32_to_si64 = fptosi <vscale x 1 x float> undef to <vscale x 1 x i64>
  %nv1f32_to_ui64 = fptoui <vscale x 1 x float> undef to <vscale x 1 x i64>

  %nv1f64_to_si8  = fptosi <vscale x 1 x double> undef to <vscale x 1 x i8>
  %nv1f64_to_ui8  = fptoui <vscale x 1 x double> undef to <vscale x 1 x i8>
  %nv1f64_to_si16 = fptosi <vscale x 1 x double> undef to <vscale x 1 x i16>
  %nv1f64_to_ui16 = fptoui <vscale x 1 x double> undef to <vscale x 1 x i16>
  %nv1f64_to_si32 = fptosi <vscale x 1 x double> undef to <vscale x 1 x i32>
  %nv1f64_to_ui32 = fptoui <vscale x 1 x double> undef to <vscale x 1 x i32>

  %nv2f16_to_si8  = fptosi <vscale x 2 x half> undef to <vscale x 2 x i8>
  %nv2f16_to_ui8  = fptoui <vscale x 2 x half> undef to <vscale x 2 x i8>
  %nv2f16_to_si32 = fptosi <vscale x 2 x half> undef to <vscale x 2 x i32>
  %nv2f16_to_ui32 = fptoui <vscale x 2 x half> undef to <vscale x 2 x i32>
  %nv2f16_to_si64 = fptosi <vscale x 2 x half> undef to <vscale x 2 x i64>
  %nv2f16_to_ui64 = fptoui <vscale x 2 x half> undef to <vscale x 2 x i64>

  %nv2f32_to_si8  = fptosi <vscale x 2 x float> undef to <vscale x 2 x i8>
  %nv2f32_to_ui8  = fptoui <vscale x 2 x float> undef to <vscale x 2 x i8>
  %nv2f32_to_si16 = fptosi <vscale x 2 x float> undef to <vscale x 2 x i16>
  %nv2f32_to_ui16 = fptoui <vscale x 2 x float> undef to <vscale x 2 x i16>
  %nv2f32_to_si64 = fptosi <vscale x 2 x float> undef to <vscale x 2 x i64>
  %nv2f32_to_ui64 = fptoui <vscale x 2 x float> undef to <vscale x 2 x i64>

  %nv2f64_to_si8  = fptosi <vscale x 2 x double> undef to <vscale x 2 x i8>
  %nv2f64_to_ui8  = fptoui <vscale x 2 x double> undef to <vscale x 2 x i8>
  %nv2f64_to_si16 = fptosi <vscale x 2 x double> undef to <vscale x 2 x i16>
  %nv2f64_to_ui16 = fptoui <vscale x 2 x double> undef to <vscale x 2 x i16>
  %nv2f64_to_si32 = fptosi <vscale x 2 x double> undef to <vscale x 2 x i32>
  %nv2f64_to_ui32 = fptoui <vscale x 2 x double> undef to <vscale x 2 x i32>

  %nv4f16_to_si8  = fptosi <vscale x 4 x half> undef to <vscale x 4 x i8>
  %nv4f16_to_ui8  = fptoui <vscale x 4 x half> undef to <vscale x 4 x i8>
  %nv4f16_to_si32 = fptosi <vscale x 4 x half> undef to <vscale x 4 x i32>
  %nv4f16_to_ui32 = fptoui <vscale x 4 x half> undef to <vscale x 4 x i32>
  %nv4f16_to_si64 = fptosi <vscale x 4 x half> undef to <vscale x 4 x i64>
  %nv4f16_to_ui64 = fptoui <vscale x 4 x half> undef to <vscale x 4 x i64>

  %nv4f32_to_si8  = fptosi <vscale x 4 x float> undef to <vscale x 4 x i8>
  %nv4f32_to_ui8  = fptoui <vscale x 4 x float> undef to <vscale x 4 x i8>
  %nv4f32_to_si16 = fptosi <vscale x 4 x float> undef to <vscale x 4 x i16>
  %nv4f32_to_ui16 = fptoui <vscale x 4 x float> undef to <vscale x 4 x i16>
  %nv4f32_to_si64 = fptosi <vscale x 4 x float> undef to <vscale x 4 x i64>
  %nv4f32_to_ui64 = fptoui <vscale x 4 x float> undef to <vscale x 4 x i64>

  %nv4f64_to_si8  = fptosi <vscale x 4 x double> undef to <vscale x 4 x i8>
  %nv4f64_to_ui8  = fptoui <vscale x 4 x double> undef to <vscale x 4 x i8>
  %nv4f64_to_si16 = fptosi <vscale x 4 x double> undef to <vscale x 4 x i16>
  %nv4f64_to_ui16 = fptoui <vscale x 4 x double> undef to <vscale x 4 x i16>
  %nv4f64_to_si32 = fptosi <vscale x 4 x double> undef to <vscale x 4 x i32>
  %nv4f64_to_ui32 = fptoui <vscale x 4 x double> undef to <vscale x 4 x i32>

  %nv8f16_to_si8  = fptosi <vscale x 8 x half> undef to <vscale x 8 x i8>
  %nv8f16_to_ui8  = fptoui <vscale x 8 x half> undef to <vscale x 8 x i8>
  %nv8f16_to_si32 = fptosi <vscale x 8 x half> undef to <vscale x 8 x i32>
  %nv8f16_to_ui32 = fptoui <vscale x 8 x half> undef to <vscale x 8 x i32>
  %nv8f16_to_si64 = fptosi <vscale x 8 x half> undef to <vscale x 8 x i64>
  %nv8f16_to_ui64 = fptoui <vscale x 8 x half> undef to <vscale x 8 x i64>

  %nv8f32_to_si8  = fptosi <vscale x 8 x float> undef to <vscale x 8 x i8>
  %nv8f32_to_ui8  = fptoui <vscale x 8 x float> undef to <vscale x 8 x i8>
  %nv8f32_to_si16 = fptosi <vscale x 8 x float> undef to <vscale x 8 x i16>
  %nv8f32_to_ui16 = fptoui <vscale x 8 x float> undef to <vscale x 8 x i16>
  %nv8f32_to_si64 = fptosi <vscale x 8 x float> undef to <vscale x 8 x i64>
  %nv8f32_to_ui64 = fptoui <vscale x 8 x float> undef to <vscale x 8 x i64>

  %nv8f64_to_si8  = fptosi <vscale x 8 x double> undef to <vscale x 8 x i8>
  %nv8f64_to_ui8  = fptoui <vscale x 8 x double> undef to <vscale x 8 x i8>
  %nv8f64_to_si16 = fptosi <vscale x 8 x double> undef to <vscale x 8 x i16>
  %nv8f64_to_ui16 = fptoui <vscale x 8 x double> undef to <vscale x 8 x i16>
  %nv8f64_to_si32 = fptosi <vscale x 8 x double> undef to <vscale x 8 x i32>
  %nv8f64_to_ui32 = fptoui <vscale x 8 x double> undef to <vscale x 8 x i32>

  ret void
}
