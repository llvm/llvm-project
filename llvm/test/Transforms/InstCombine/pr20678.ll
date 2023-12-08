; RUN: opt -S -passes=instcombine < %s | FileCheck %s

define i1 @test1() {
entry:
  ret i1 icmp ne (i16 bitcast (<16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false> to i16), i16 0)
}
; CHECK-LABEL: define i1 @test1(
; CHECK:  ret i1 true
