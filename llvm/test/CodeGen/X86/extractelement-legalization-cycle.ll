; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

; When the extractelement is converted to a load the store can be re-used.
; This will, however, introduce a cycle into the selection DAG (the load
; of the extractelement index is dependent on the store, and so after the
; conversion it becomes dependent on the new load, which is dependent on
; the index).  Make sure we skip the store, and conservatively instead
; use a store to the stack.

define float @foo(ptr %i, ptr %v) {
; CHECK-LABEL: foo:
; CHECK:    movaps %xmm0, -[[OFFSET:[0-9]+]](%rsp)
; CHECK:    movss -[[OFFSET]](%rsp,{{.*}}), %xmm0 {{.*}}
; CHECK-NEXT:    retq
  %1 = load <4 x float>, ptr %v, align 16
  %mul = fmul <4 x float> %1, %1
  store <4 x float> %mul, ptr %v, align 16
  %2 = load i32, ptr %i, align 4
  %vecext = extractelement <4 x float> %mul, i32 %2
  ret float %vecext
}
