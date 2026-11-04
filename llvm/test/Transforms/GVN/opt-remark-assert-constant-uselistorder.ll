; RUN: opt -passes='gvn' -pass-remarks-output=%t.yaml %s
; RUN: FileCheck %s < %t.yaml

; Check that there's no assert from trying to the uses of the constant
; null.

; CHECK: --- !Missed
; CHECK-NEXT: Pass:            gvn
; CHECK-NEXT: Name:            LoadClobbered
; CHECK-NEXT: Function:        c
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'load of type '
; CHECK-NEXT:   - Type:            i64
; CHECK-NEXT:   - String:          ' not eliminated'
; CHECK-NEXT:   - String:          ' because it is clobbered by '
; CHECK-NEXT:   - ClobberedBy:     store
; CHECK-NEXT: ...
define void @c(ptr addrspace(21) %a) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %load = load i64, ptr addrspace(21) null, align 1
  store i64 %load, ptr addrspace(21) %a, align 1
  br label %for.cond
}
