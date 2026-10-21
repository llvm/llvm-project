; RUN: opt < %s -passes=simplifycfg -S | FileCheck %s

define ptr @test(i16 %0, ptr %p) {
entry:
  %cmp.i1 = icmp eq i16 %0, 0
  %cmp.i2 = icmp eq i16 %0, 0
  %or1 = or i1 %cmp.i1, %cmp.i2
  br i1 %or1, label %bbl1, label %bbl2, !prof !0

bbl1:
  %load1 = load ptr, ptr %p, align 8
  ret ptr %load1

bbl2:
  ret ptr null
}

!0 = !{!"unknown", !"simple-loop-unswitch"}
