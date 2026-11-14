; RUN: opt -passes=wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-write-summary=%t -disable-output %s
; RUN: FileCheck %s < %t

; CHECK: ---
; CHECK-NEXT: GlobalValueMap:
; CHECK-NEXT: TypeIdMap:
; CHECK-NEXT: WithGlobalValueDeadStripping: false
; CHECK-NEXT: ...
