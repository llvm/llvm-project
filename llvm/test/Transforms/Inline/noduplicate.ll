; RUN: opt < %s -passes=inline -pass-remarks-missed=inline -inline-cost-full -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes=inline -pass-remarks-missed=inline -S 2>&1 | FileCheck %s

declare void @foo()
declare void @bar()

define void @callee() {
entry:
  call void @foo() noduplicate
  ; Just to inflate the cost
  call void @bar() "call-inline-cost"="1000"
  ret void
}

define void @caller() {
; CHECK: 'callee' not inlined into 'caller' because it should never be inlined (cost=never): noduplicate
; CHECK: define void @caller()
; CHECK-NEXT: call void @callee()
; CHECK-NEXT: ret void
  call void @callee()
  ret void
}
