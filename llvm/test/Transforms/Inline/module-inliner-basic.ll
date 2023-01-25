; RUN: opt -passes=module-inline                                    -S < %s | FileCheck %s
; RUN: opt -passes=module-inline -inline-priority-mode=size         -S < %s | FileCheck %s
; RUN: opt -passes=module-inline -inline-priority-mode=cost         -S < %s | FileCheck %s
; RUN: opt -passes=module-inline -inline-priority-mode=cost-benefit -S < %s | FileCheck %s
; RUN: opt -passes=module-inline -inline-priority-mode=ml           -S < %s | FileCheck %s

define i32 @callee(i32 %a) {
entry:
  %add = add nsw i32 %a, 1
  ret i32 %add
}

define i32 @caller(i32 %a) {
entry:
  %call = call i32 @callee(i32 %a)
  ret i32 %call
}

; CHECK-LABEL: @caller
; CHECK-NOT:     call
; CHECK:         ret
