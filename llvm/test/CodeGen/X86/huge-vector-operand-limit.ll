; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512bw -O0 | FileCheck %s

; Vectors and aggregates with more than 65535 elements (#214666, #7622). These
; shapes compile to a lot of code, so only completion is checked; the lane
; values are checked in huge-vector-operand-limit-lanes.ll.

define void @exact_65536(ptr %p) {
; CHECK-LABEL: exact_65536:
; CHECK: retq
  %v = load <65536 x i8>, ptr %p, align 16
  %s = shl <65536 x i8> %v, splat (i8 5)
  store <65536 x i8> %s, ptr %p, align 16
  ret void
}

define void @exact_131072(ptr %p) {
; CHECK-LABEL: exact_131072:
; CHECK: retq
  %v = load <131072 x i8>, ptr %p, align 16
  %s = shl <131072 x i8> %v, splat (i8 5)
  store <131072 x i8> %s, ptr %p, align 16
  ret void
}

define i8 @scalarized_arg_65537(<65537 x i8> %v) {
; CHECK-LABEL: scalarized_arg_65537:
; CHECK: retq
  %e = extractelement <65537 x i8> %v, i32 65536
  ret i8 %e
}

define void @merge_values_65536(ptr %p, ptr %q) {
; CHECK-LABEL: merge_values_65536:
; CHECK: retq
  %v = load [65536 x i8], ptr %p
  store [65536 x i8] %v, ptr %q
  ret void
}
