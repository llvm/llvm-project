; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

declare i32 @pers(...)
declare void @f()

define void @f1() personality ptr @pers {
  invoke void @f() to label %normal unwind label %lp

normal:
  ret void

lp:
; CHECK:Only two-valued landingpads are supported
  landingpad {ptr} cleanup
  ret void
}
