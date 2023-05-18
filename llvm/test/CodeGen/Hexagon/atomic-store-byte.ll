; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test that the address for a store conditional for a byte is aligned
; correctly to use the memw_locked instruction.

; CHECK: [[REG:(r[0-9]+)]] = and(r{{[0-9]+}},#-4)
; CHECK: = memw_locked([[REG]])
; CHECK: memw_locked([[REG]],p{{[0-4]}}) =

@foo.a00 = internal global i8 0, align 1

; Function Attrs: nofree norecurse nounwind
define dso_local void @foo() local_unnamed_addr #0 {
entry:
  %0 = cmpxchg volatile ptr @foo.a00, i8 0, i8 1 seq_cst seq_cst
  ret void
}

