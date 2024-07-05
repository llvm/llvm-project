; RUN: opt -S -passes=instcombine < %s | FileCheck %s

define i1 @test1() {
entry:
  %cmp = icmp ne i16 bitcast (<16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false> to i16), 0
  ret i1 %cmp
}
; CHECK-LABEL: define i1 @test1(
; CHECK:  ret i1 true
