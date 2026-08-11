; UNSUPPORTED: system-windows
; RUN: not --crash llc %s 2>&1 | FileCheck %s
; RUN: not --crash llc --global-isel %s 2>&1 | FileCheck %s

declare i32 @pers(...)
declare void @f()

define void @main() personality ptr @pers {
  invoke void @f() to label %normal unwind label %lp

normal:
  ret void

lp:
  landingpad {ptr} cleanup
  ret void
}
; CHECK:LLVM ERROR: Only two-valued landingpads are supported
