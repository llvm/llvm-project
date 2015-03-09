; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=generic < %s | FileCheck %s

; Verify that the DAGCombiner doesn't wrongly remove the 'and' from the dag.

define i8 @foo(<4 x i8>* %V) {
; CHECK-LABEL: foo:
; CHECK: pand
; CHECK: ret
entry:
  %Vp = bitcast <4 x i8>* %V to <3 x i8>*
  %V3i8 = load <3 x i8>* %Vp, align 4
  %0 = and <3 x i8> %V3i8, <i8 undef, i8 undef, i8 95>
  %1 = extractelement <3 x i8> %0, i64 2
  ret i8 %1
}
