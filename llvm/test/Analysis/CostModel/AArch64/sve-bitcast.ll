; RUN: opt -mtriple=aarch64-linux-gnu -mattr=+sve -cost-model -analyze < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read clang/test/CodeGen/aarch64-sve-intrinsics/README for instructions on how to resolve it.
; WARN-NOT: warning

; CHECK: Found an estimated cost of 0 for instruction:   %b = bitcast <vscale x 2 x double> %a to <vscale x 2 x i64>

define <vscale x 2 x i64> @foo(<vscale x 2 x double> %a, i32 %x) {
  %b = bitcast <vscale x 2 x double> %a to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %b
}
