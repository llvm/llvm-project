; RUN: opt -cost-model -analyze -mtriple aarch64-linux-gnu -mattr=+sve -S -o - < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @sve_fpext() {
  ;CHECK-LABEL: 'sve_fpext'
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nxv2_f16_to_f32 = fpext <vscale x 2 x half> undef to <vscale x 2 x float>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nxv4_f16_to_f32 = fpext <vscale x 4 x half> undef to <vscale x 4 x float>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nxv8_f16_to_f32 = fpext <vscale x 8 x half> undef to <vscale x 8 x float>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nxv2_f16_to_f64 = fpext <vscale x 2 x half> undef to <vscale x 2 x double>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nxv4_f16_to_f64 = fpext <vscale x 4 x half> undef to <vscale x 4 x double>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %nxv8_f16_to_f64 = fpext <vscale x 8 x half> undef to <vscale x 8 x double>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nxv2_f32_to_f64 = fpext <vscale x 2 x float> undef to <vscale x 2 x double>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %nxv4_f32_to_f64 = fpext <vscale x 4 x float> undef to <vscale x 4 x double>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %nxv8_f32_to_f64 = fpext <vscale x 8 x float> undef to <vscale x 8 x double>
  %nxv2_f16_to_f32 = fpext <vscale x 2 x half> undef to <vscale x 2 x float>
  %nxv4_f16_to_f32 = fpext <vscale x 4 x half> undef to <vscale x 4 x float>
  %nxv8_f16_to_f32 = fpext <vscale x 8 x half> undef to <vscale x 8 x float>

  %nxv2_f16_to_f64 = fpext <vscale x 2 x half> undef to <vscale x 2 x double>
  %nxv4_f16_to_f64 = fpext <vscale x 4 x half> undef to <vscale x 4 x double>
  %nxv8_f16_to_f64 = fpext <vscale x 8 x half> undef to <vscale x 8 x double>

  %nxv2_f32_to_f64 = fpext <vscale x 2 x float> undef to <vscale x 2 x double>
  %nxv4_f32_to_f64 = fpext <vscale x 4 x float> undef to <vscale x 4 x double>
  %nxv8_f32_to_f64 = fpext <vscale x 8 x float> undef to <vscale x 8 x double>

  ret void
}
