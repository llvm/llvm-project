; RUN: opt < %s -postdomtree -analyze | FileCheck %s
; RUN: opt < %s -passes='print<postdomtree>' 2>&1 | FileCheck %s

@a = external global i32, align 4

define void @fn1() {
entry:
  store i32 5, i32* @a, align 4
  %call = call i32 (...) @foo()
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %loop

loop:                                             ; preds = %loop, %if.then
  %0 = load i32, i32* @a, align 4
  call void @bar(i32 %0)
  br label %loop

if.end:                                           ; preds = %entry
  store i32 6, i32* @a, align 4
  ret void
}

declare i32 @foo(...)
declare void @bar(i32)


; CHECK:      Inorder PostDominator Tree:
; CHECK-NEXT:  [1]  <<exit node>>
; CHECK:         [2] %loop
; CHECK-NEXT:      [3] %if.then
; CHECK: Roots: %if.end %loop
