; The only use of "typeid1" is in a dead function. Export nothing.

; RUN: opt -S -passes=lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/use-typeid1-dead.yaml -lowertypetests-write-summary=%t %s | FileCheck %s
; RUN: FileCheck --check-prefix=SUMMARY %s < %t

@foo = constant i32 42, !type !0

!0 = !{i32 0, !"typeid1"}

; CHECK-NOT: @__typeid_typeid1_global_addr =

; SUMMARY:      TypeIdMap:
; SUMMARY-NEXT: WithGlobalValueDeadStripping: true
; SUMMARY-NEXT: ...
