; RUN: opt -cost-model -analyze -mtriple aarch64-linux-gnu -mattr=+sve -o - -S < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @sve-fptoi() {
  ;CHECK-LABEL: 'sve-fptoi'
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f32_to_si8 = fptosi <vscale x 2 x float> undef to <vscale x 2 x i8>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f32_to_ui8 = fptoui <vscale x 2 x float> undef to <vscale x 2 x i8>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f32_to_si32 = fptosi <vscale x 2 x float> undef to <vscale x 2 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f32_to_ui32 = fptoui <vscale x 2 x float> undef to <vscale x 2 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv2f32_to_si64 = fptosi <vscale x 2 x float> undef to <vscale x 2 x i64>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv2f32_to_ui64 = fptoui <vscale x 2 x float> undef to <vscale x 2 x i64>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv2f64_to_si8 = fptosi <vscale x 2 x double> undef to <vscale x 2 x i8>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv2f64_to_ui8 = fptoui <vscale x 2 x double> undef to <vscale x 2 x i8>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv2f64_to_si32 = fptosi <vscale x 2 x double> undef to <vscale x 2 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv2f64_to_ui32 = fptoui <vscale x 2 x double> undef to <vscale x 2 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f64_to_si64 = fptosi <vscale x 2 x double> undef to <vscale x 2 x i64>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv2f64_to_ui64 = fptoui <vscale x 2 x double> undef to <vscale x 2 x i64>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv4f32_to_si8 = fptosi <vscale x 4 x float> undef to <vscale x 4 x i8>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv4f32_to_ui8 = fptoui <vscale x 4 x float> undef to <vscale x 4 x i8>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv4f32_to_si32 = fptosi <vscale x 4 x float> undef to <vscale x 4 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv4f32_to_ui32 = fptoui <vscale x 4 x float> undef to <vscale x 4 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 5 for instruction:   %nv4f32_to_si64 = fptosi <vscale x 4 x float> undef to <vscale x 4 x i64>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 5 for instruction:   %nv4f32_to_ui64 = fptoui <vscale x 4 x float> undef to <vscale x 4 x i64>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv4f64_to_si8 = fptosi <vscale x 4 x double> undef to <vscale x 4 x i8>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv4f64_to_ui8 = fptoui <vscale x 4 x double> undef to <vscale x 4 x i8>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv4f64_to_si32 = fptosi <vscale x 4 x double> undef to <vscale x 4 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv4f64_to_ui32 = fptoui <vscale x 4 x double> undef to <vscale x 4 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv4f64_to_si64 = fptosi <vscale x 4 x double> undef to <vscale x 4 x i64>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv4f64_to_ui64 = fptoui <vscale x 4 x double> undef to <vscale x 4 x i64>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv8f32_to_si8 = fptosi <vscale x 8 x float> undef to <vscale x 8 x i8>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nv8f32_to_ui8 = fptoui <vscale x 8 x float> undef to <vscale x 8 x i8>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv8f32_to_si32 = fptosi <vscale x 8 x float> undef to <vscale x 8 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv8f32_to_ui32 = fptoui <vscale x 8 x float> undef to <vscale x 8 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv8f32_to_si64 = fptosi <vscale x 8 x float> undef to <vscale x 8 x i64>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv8f32_to_ui64 = fptoui <vscale x 8 x float> undef to <vscale x 8 x i64>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv8f64_to_si8 = fptosi <vscale x 8 x double> undef to <vscale x 8 x i8>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv8f64_to_ui8 = fptoui <vscale x 8 x double> undef to <vscale x 8 x i8>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv8f64_to_si32 = fptosi <vscale x 8 x double> undef to <vscale x 8 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nv8f64_to_ui32 = fptoui <vscale x 8 x double> undef to <vscale x 8 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %nv8f64_to_si64 = fptosi <vscale x 8 x double> undef to <vscale x 8 x i64>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %nv8f64_to_ui64 = fptoui <vscale x 8 x double> undef to <vscale x 8 x i64>

  %nv2f32_to_si8  = fptosi <vscale x 2 x float> undef to <vscale x 2 x i8>
  %nv2f32_to_ui8  = fptoui <vscale x 2 x float> undef to <vscale x 2 x i8>
  %nv2f32_to_si32 = fptosi <vscale x 2 x float> undef to <vscale x 2 x i32>
  %nv2f32_to_ui32 = fptoui <vscale x 2 x float> undef to <vscale x 2 x i32>
  %nv2f32_to_si64 = fptosi <vscale x 2 x float> undef to <vscale x 2 x i64>
  %nv2f32_to_ui64 = fptoui <vscale x 2 x float> undef to <vscale x 2 x i64>

  %nv2f64_to_si8  = fptosi <vscale x 2 x double> undef to <vscale x 2 x i8>
  %nv2f64_to_ui8  = fptoui <vscale x 2 x double> undef to <vscale x 2 x i8>
  %nv2f64_to_si32 = fptosi <vscale x 2 x double> undef to <vscale x 2 x i32>
  %nv2f64_to_ui32 = fptoui <vscale x 2 x double> undef to <vscale x 2 x i32>
  %nv2f64_to_si64 = fptosi <vscale x 2 x double> undef to <vscale x 2 x i64>
  %nv2f64_to_ui64 = fptoui <vscale x 2 x double> undef to <vscale x 2 x i64>

  %nv4f32_to_si8  = fptosi <vscale x 4 x float> undef to <vscale x 4 x i8>
  %nv4f32_to_ui8  = fptoui <vscale x 4 x float> undef to <vscale x 4 x i8>
  %nv4f32_to_si32 = fptosi <vscale x 4 x float> undef to <vscale x 4 x i32>
  %nv4f32_to_ui32 = fptoui <vscale x 4 x float> undef to <vscale x 4 x i32>
  %nv4f32_to_si64 = fptosi <vscale x 4 x float> undef to <vscale x 4 x i64>
  %nv4f32_to_ui64 = fptoui <vscale x 4 x float> undef to <vscale x 4 x i64>

  %nv4f64_to_si8  = fptosi <vscale x 4 x double> undef to <vscale x 4 x i8>
  %nv4f64_to_ui8  = fptoui <vscale x 4 x double> undef to <vscale x 4 x i8>
  %nv4f64_to_si32 = fptosi <vscale x 4 x double> undef to <vscale x 4 x i32>
  %nv4f64_to_ui32 = fptoui <vscale x 4 x double> undef to <vscale x 4 x i32>
  %nv4f64_to_si64 = fptosi <vscale x 4 x double> undef to <vscale x 4 x i64>
  %nv4f64_to_ui64 = fptoui <vscale x 4 x double> undef to <vscale x 4 x i64>

  %nv8f32_to_si8  = fptosi <vscale x 8 x float> undef to <vscale x 8 x i8>
  %nv8f32_to_ui8  = fptoui <vscale x 8 x float> undef to <vscale x 8 x i8>
  %nv8f32_to_si32 = fptosi <vscale x 8 x float> undef to <vscale x 8 x i32>
  %nv8f32_to_ui32 = fptoui <vscale x 8 x float> undef to <vscale x 8 x i32>
  %nv8f32_to_si64 = fptosi <vscale x 8 x float> undef to <vscale x 8 x i64>
  %nv8f32_to_ui64 = fptoui <vscale x 8 x float> undef to <vscale x 8 x i64>

  %nv8f64_to_si8  = fptosi <vscale x 8 x double> undef to <vscale x 8 x i8>
  %nv8f64_to_ui8  = fptoui <vscale x 8 x double> undef to <vscale x 8 x i8>
  %nv8f64_to_si32 = fptosi <vscale x 8 x double> undef to <vscale x 8 x i32>
  %nv8f64_to_ui32 = fptoui <vscale x 8 x double> undef to <vscale x 8 x i32>
  %nv8f64_to_si64 = fptosi <vscale x 8 x double> undef to <vscale x 8 x i64>
  %nv8f64_to_ui64 = fptoui <vscale x 8 x double> undef to <vscale x 8 x i64>

  ret void
}
