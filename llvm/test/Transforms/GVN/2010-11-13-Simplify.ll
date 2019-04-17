; RUN: opt < %s -basicaa -gvn -S | FileCheck %s

declare i32 @foo(i32) readnone

define i1 @bar() {
; CHECK-LABEL: @bar(
  %a = call i32 @foo (i32 0) readnone
  %b = call i32 @foo (i32 0) readnone
  %c = and i32 %a, %b
  %x = call i32 @foo (i32 %a) readnone
  %y = call i32 @foo (i32 %c) readnone
  %z = icmp eq i32 %x, %y
  ret i1 %z
; CHECK: ret i1 true
} 
