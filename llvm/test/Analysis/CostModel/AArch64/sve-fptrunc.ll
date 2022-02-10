; RUN: opt -passes='print<cost-model>' 2>&1 -disable-output -mtriple aarch64-linux-gnu -mattr=+sve -S -o - < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @sve_fptruncs() {
  ;CHECK-LABEL: 'sve_fptruncs'
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nxv2_f16_from_f32 = fptrunc <vscale x 2 x float> undef to <vscale x 2 x half>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nxv4_f16_from_f32 = fptrunc <vscale x 4 x float> undef to <vscale x 4 x half>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nxv8_f16_from_f32 = fptrunc <vscale x 8 x float> undef to <vscale x 8 x half>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nxv2_f16_from_f64 = fptrunc <vscale x 2 x double> undef to <vscale x 2 x half>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nxv4_f16_from_f64 = fptrunc <vscale x 4 x double> undef to <vscale x 4 x half>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 7 for instruction:   %nxv8_f16_from_f64 = fptrunc <vscale x 8 x double> undef to <vscale x 8 x half>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %nxv2_f32_from_f64 = fptrunc <vscale x 2 x double> undef to <vscale x 2 x float>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %nxv4_f32_from_f64 = fptrunc <vscale x 4 x double> undef to <vscale x 4 x float>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %nxv8_f32_from_f64 = fptrunc <vscale x 8 x double> undef to <vscale x 8 x float>
  %nxv2_f16_from_f32 = fptrunc <vscale x 2 x float> undef to <vscale x 2 x half>
  %nxv4_f16_from_f32 = fptrunc <vscale x 4 x float> undef to <vscale x 4 x half>
  %nxv8_f16_from_f32 = fptrunc <vscale x 8 x float> undef to <vscale x 8 x half>

  %nxv2_f16_from_f64 = fptrunc <vscale x 2 x double> undef to <vscale x 2 x half>
  %nxv4_f16_from_f64 = fptrunc <vscale x 4 x double> undef to <vscale x 4 x half>
  %nxv8_f16_from_f64 = fptrunc <vscale x 8 x double> undef to <vscale x 8 x half>

  %nxv2_f32_from_f64 = fptrunc <vscale x 2 x double> undef to <vscale x 2 x float>
  %nxv4_f32_from_f64 = fptrunc <vscale x 4 x double> undef to <vscale x 4 x float>
  %nxv8_f32_from_f64 = fptrunc <vscale x 8 x double> undef to <vscale x 8 x float>

  ret void
}
