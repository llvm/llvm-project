; RUN: opt < %s -inline -instcombine -S | FileCheck %s

; This test case exposed a bug in instcombine where the early
; DCE of a call wasn't recognized as changing the IR.
; So when runOnFunction propagated the "made changes" upwards
; to the CallGraphSCCPass it signalled that no changes had been
; made, so CallGraphSCCPass assumed that the old CallGraph,
; as known by that pass manager, still was up-to-date.
;
; This was detected as an assert when trying to remove the
; no longer used function 'bar' (due to incorrect reference
; count in the CallGraph).

define void @foo() {
; CHECK-LABEL: @foo(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret void
;
entry:
  %call = call i32 @bar()
  ret void
}

define internal i32 @bar() {
; CHECK-NOT: bar
entry:
  ret i32 42
}

