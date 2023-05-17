; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

; Store instructions are allowed users for byval arguments only.
define internal void @callee(ptr %arg) nounwind {
; CHECK-LABEL: define {{[^@]+}}@callee
; CHECK-SAME: (ptr [[ARG:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TEMP:%.*]] = load i32, ptr [[ARG]], align 4
; CHECK-NEXT:    [[SUM:%.*]] = add i32 [[TEMP]], 1
; CHECK-NEXT:    store i32 [[SUM]], ptr [[ARG]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %temp = load i32, ptr %arg, align 4
  %sum = add i32 %temp, 1
  store i32 %sum, ptr %arg, align 4
  ret void
}

define i32 @caller(ptr %arg) nounwind {
; CHECK-LABEL: define {{[^@]+}}@caller
; CHECK-SAME: (ptr [[ARG:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @callee(ptr [[ARG]]) #[[ATTR0]]
; CHECK-NEXT:    ret i32 0
;
entry:
  call void @callee(ptr %arg) nounwind
  ret i32 0
}
