; RUN: opt < %s -passes=loweratomic -S | FileCheck %s

define void @barrier() {
; CHECK-LABEL: @barrier(
  fence seq_cst
; CHECK-NEXT: ret
  ret void
}
