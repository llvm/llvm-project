; RUN: opt -instcombine -S < %s | FileCheck %s

declare void @foo(i32)

define void @g() {
; CHECK-LABEL: @g(
 entry:
; CHECK: call void @foo(i32 0) [ "deopt"() ]
  call void bitcast (void (i32)* @foo to void ()*) ()  [ "deopt"() ]
  ret void
}
