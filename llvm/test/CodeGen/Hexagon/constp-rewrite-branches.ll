; RUN: llc -O2 -march=hexagon < %s | FileCheck %s

define i32 @foo(i32 %x) {
  %p = icmp eq i32 %x, 0
  br i1 %p, label %zero, label %nonzero
nonzero:
  %v1 = add i32 %x, 1
  %c = icmp eq i32 %x, %v1
; This branch will be rewritten by HCP.  A bug would cause both branches to
; go away, leaving no path to "ret -1".
  br i1 %c, label %zero, label %other
zero:
  ret i32 0
other:
; CHECK: -1
  ret i32 -1
}
