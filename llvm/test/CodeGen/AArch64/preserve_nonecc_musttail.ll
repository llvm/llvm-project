; RUN: not llc -mtriple=aarch64-unknown-unknown %s -o - 2>&1 | FileCheck %s

; Incompatible calling convention causes following error message.

; CHECK: cannot guarantee tail call due to mismatched calling conv

declare preserve_nonecc void @callee(ptr)
define void @caller(ptr %a) {
  musttail call preserve_nonecc void @callee(ptr %a)
  ret void
}
