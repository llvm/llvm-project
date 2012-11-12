; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 | FileCheck %s

; CHECK: _f
; CHECK-NOT: ands
; CHECK: cmp
; CHECK: blxle _g

define i32 @f(i32 %a, i32 %b) nounwind ssp {
entry:
  %and = and i32 %b, %a
  %cmp = icmp slt i32 %and, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void (...)* @g(i32 %a, i32 %b) nounwind
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 %and
}

declare void @g(...)
