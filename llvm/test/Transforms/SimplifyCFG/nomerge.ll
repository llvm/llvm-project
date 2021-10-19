; RUN: opt < %s -O1 -S | FileCheck %s

; The attribute nomerge prevents the 3 bar() calls from being sunk/hoisted into
; one inside a function. Check that there are still 3 tail calls.

; Test case for preventing sinking
; CHECK-LABEL: define void @sink
; CHECK: if.then:
; CHECK-NEXT: tail call void @bar()
; CHECK: if.then2:
; CHECK-NEXT: tail call void @bar()
; CHECK: if.end3:
; CHECK-NEXT: tail call void @bar()
define void @sink(i32 %i) {
entry:
  switch i32 %i, label %if.end3 [
    i32 5, label %if.then
    i32 7, label %if.then2
  ]

if.then:
  tail call void @bar() #0
  br label %if.end3

if.then2:
  tail call void @bar() #0
  br label %if.end3

if.end3:
  tail call void @bar() #0
  ret void
}

; Test case for preventing hoisting
; CHECK-LABEL: define void @hoist
; CHECK: if.then:
; CHECK-NEXT: tail call void @bar()
; CHECK: if.then2:
; CHECK-NEXT: tail call void @bar()
; CHECK: if.end:
; CHECK-NEXT: tail call void @bar()
define void @hoist(i32 %i) {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %cmp = icmp eq i32 %0, 5
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @bar() #1
  unreachable

if.else:
  %1 = load i32, i32* %i.addr, align 4
  %cmp1 = icmp eq i32 %i, 7
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  tail call void @bar() #1
  unreachable

if.end:
  tail call void @bar() #1
  unreachable
}

declare void @bar()

attributes #0 = { nomerge }
attributes #1 = { noreturn nomerge }
