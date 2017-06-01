; RUN: opt -lowertypetests -lowertypetests-summary-action=export -lowertypetests-write-summary=%t -o /dev/null %s
; RUN: FileCheck %s < %t

; CHECK: ---
; CHECK-NEXT: GlobalValueMap:
; CHECK-NEXT: TypeIdMap:
; CHECK-NEXT: ...
