; RUN: llc -mtriple=i686-- -asm-verbose=false < %s | FileCheck %s

; Check that merging switch cases that differ in one bit works.
; CHECK-LABEL: test1
; CHECK: orl $2
; CHECK-NEXT: cmpl $6
define void @test1(i32 %variable) nounwind {
entry:
  switch i32 %variable, label %if.end [
    i32 4, label %if.then
    i32 6, label %if.then
  ]

if.then:
  %call = tail call i32 (...) @bar() nounwind
  ret void

if.end:
  ret void
}

; CHECK-LABEL: test2
; CHECK: testl $2147483647
define void @test2(i32 %variable) nounwind {
entry:
  switch i32 %variable, label %if.end [
    i32 0, label %if.then
    i32 -2147483648, label %if.then
  ]

if.then:
  %call = tail call i32 (...) @bar() nounwind
  ret void

if.end:
  ret void
}

declare i32 @bar(...) nounwind
