; RUN: opt < %s -mtriple=x86_64 -internalize -S | FileCheck %s

$c2 = comdat any

;; Without an exported symbol, the module ID is empty, and we don't rename
;; comdat for ELF. This does not matter in practice because all the symbols
;; will be optimized out.

; CHECK: define internal void @c2_a() comdat($c2) {
define void @c2_a() comdat($c2) {
  ret void
}

; CHECK: define internal void @c2_b() comdat($c2) {
define void @c2_b() comdat($c2) {
  ret void
}
