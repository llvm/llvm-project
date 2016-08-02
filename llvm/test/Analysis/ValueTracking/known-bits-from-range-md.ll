; RUN: opt -S -instsimplify < %s | FileCheck %s

define i1 @test0(i8* %ptr) {
; CHECK-LABEL: @test0(
 entry:
  %val = load i8, i8* %ptr, !range !{i8 -50, i8 0}
  %and = and i8 %val, 128
  %is.eq = icmp eq i8 %and, 128
  ret i1 %is.eq
; CHECK: ret i1 true
}

define i1 @test1(i8* %ptr) {
; CHECK-LABEL: @test1(
 entry:
  %val = load i8, i8* %ptr, !range !{i8 64, i8 128}
  %and = and i8 %val, 64
  %is.eq = icmp eq i8 %and, 64
  ret i1 %is.eq
; CHECK: ret i1 true
}

define i1 @test2(i8* %ptr) {
; CHECK-LABEL: @test2(
 entry:
; CHECK: load
; CHECK: and
; CHECK: icmp eq
; CHECK: ret
  %val = load i8, i8* %ptr, !range !{i8 64, i8 129}
  %and = and i8 %val, 64
  %is.eq = icmp eq i8 %and, 64
  ret i1 %is.eq
}
