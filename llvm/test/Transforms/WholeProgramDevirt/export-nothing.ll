; RUN: opt -wholeprogramdevirt -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-write-summary=%t -o /dev/null %s
; RUN: FileCheck %s < %t

; CHECK: ---
; CHECK-NEXT: GlobalValueMap:
; CHECK-NEXT: TypeIdMap:
; CHECK-NEXT: WithGlobalValueDeadStripping: false
; CHECK-NEXT: ...
