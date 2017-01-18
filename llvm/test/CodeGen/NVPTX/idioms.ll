; Check that various LLVM idioms get lowered to NVPTX as expected.

; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s

; CHECK-LABEL: abs_i16(
define i16 @abs_i16(i16 %a) {
; CHECK: abs.s16
  %neg = sub i16 0, %a
  %abs.cond = icmp sge i16 %a, 0
  %abs = select i1 %abs.cond, i16 %a, i16 %neg
  ret i16 %abs
}

; CHECK-LABEL: abs_i32(
define i32 @abs_i32(i32 %a) {
; CHECK: abs.s32
  %neg = sub i32 0, %a
  %abs.cond = icmp sge i32 %a, 0
  %abs = select i1 %abs.cond, i32 %a, i32 %neg
  ret i32 %abs
}

; CHECK-LABEL: abs_i64(
define i64 @abs_i64(i64 %a) {
; CHECK: abs.s64
  %neg = sub i64 0, %a
  %abs.cond = icmp sge i64 %a, 0
  %abs = select i1 %abs.cond, i64 %a, i64 %neg
  ret i64 %abs
}
