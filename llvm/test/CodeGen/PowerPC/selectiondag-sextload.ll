; RUN: llc --mtriple=powerpc64le-linux-gnu < %s | FileCheck %s

; It tests in function DAGCombiner::visitSIGN_EXTEND_INREG
; signext will not be combined with extload, and causes extra zext.

declare void @g(i32 signext)

define void @foo(i8* %p) {
entry:
  br label %while.body

while.body:
  %0 = load i8, i8* %p, align 1
  %conv = zext i8 %0 to i32
  %cmp = icmp sgt i8 %0, 0
  br i1 %cmp, label %if.then, label %while.body
; CHECK:     lbz
; CHECK:     extsb.
; CHECK-NOT: rlwinm
; CHECK:     ble

if.then:
  tail call void @g(i32 signext %conv)
  br label %while.body
}

