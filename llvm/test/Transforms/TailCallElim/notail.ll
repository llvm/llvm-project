; RUN: opt < %s -passes=tailcallelim -verify-dom-info -S | FileCheck %s

; CHECK: tail call void @callee0()
; CHECK: notail call void @callee1()

define void @foo1(i32 %a) {
entry:
  %tobool = icmp eq i32 %a, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:
  call void @callee0()
  br label %if.end

if.else:
  notail call void @callee1()
  br label %if.end

if.end:
  ret void
}

declare void @callee0()
declare void @callee1()
