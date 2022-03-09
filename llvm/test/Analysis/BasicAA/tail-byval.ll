; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

declare void @takebyval(i32* byval(i32) %p)

define i32 @tailbyval() {
entry:
  %p = alloca i32
  store i32 42, i32* %p
  tail call void @takebyval(i32* byval(i32) %p)
  %rv = load i32, i32* %p
  ret i32 %rv
}
; FIXME: This should be Just Ref.
; CHECK-LABEL: Function: tailbyval: 1 pointers, 1 call sites
; CHECK-NEXT:   Both ModRef:  Ptr: i32* %p       <->  tail call void @takebyval(i32* byval(i32) %p)
