; RUN: opt -mtriple=aarch64-linux-gnu -mattr=+sve -cost-model -analyze < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; CHECK: Found an estimated cost of 0 for instruction:   %0 = trunc <vscale x 2 x i64> %v to <vscale x 2 x i32>

define void @trunc_nxv2i64_to_nxv2i32(<vscale x 2 x i32>* %ptr, <vscale x 2 x i64> %v) {
entry:
  %0 = trunc <vscale x 2 x i64> %v to <vscale x 2 x i32>
  store <vscale x 2 x i32> %0, <vscale x 2 x i32>* %ptr
  ret void
}
