; distilled from 255.vortex
; RUN: opt < %s -passes=globaldce -S | FileCheck %s

; CHECK-NOT: testfunc

declare ptr @getfunc()

define internal i1 @testfunc() {
  %F = call ptr @getfunc()                ; <ptr> [#uses=1]
  %c = icmp eq ptr %F, @testfunc          ; <i1> [#uses=1]
  ret i1 %c
}

