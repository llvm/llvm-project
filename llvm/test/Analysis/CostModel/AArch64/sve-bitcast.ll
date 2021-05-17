; RUN: opt -mtriple=aarch64-linux-gnu -mattr=+sve -cost-model -analyze < %s | FileCheck %s

; CHECK: Found an estimated cost of 0 for instruction:   %b = bitcast <vscale x 2 x double> %a to <vscale x 2 x i64>

define <vscale x 2 x i64> @foo(<vscale x 2 x double> %a, i32 %x) {
  %b = bitcast <vscale x 2 x double> %a to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %b
}
