; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

; Store instructions are allowed users for byval arguments only.
define internal void @callee(i32* %arg) nounwind {
; CHECK-LABEL: define {{[^@]+}}@callee
; CHECK-SAME: (i32* [[ARG:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TEMP:%.*]] = load i32, i32* [[ARG]], align 4
; CHECK-NEXT:    [[SUM:%.*]] = add i32 [[TEMP]], 1
; CHECK-NEXT:    store i32 [[SUM]], i32* [[ARG]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %temp = load i32, i32* %arg, align 4
  %sum = add i32 %temp, 1
  store i32 %sum, i32* %arg, align 4
  ret void
}

define i32 @caller(i32* %arg) nounwind {
; CHECK-LABEL: define {{[^@]+}}@caller
; CHECK-SAME: (i32* [[ARG:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @callee(i32* [[ARG]]) #[[ATTR0]]
; CHECK-NEXT:    ret i32 0
;
entry:
  call void @callee(i32* %arg) nounwind
  ret i32 0
}
