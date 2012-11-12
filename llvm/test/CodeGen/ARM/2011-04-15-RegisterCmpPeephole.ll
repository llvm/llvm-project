; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 | FileCheck %s

; CHECK: _f
; CHECK: adds
; CHECK-NOT: cmp
; CHECK: blxeq _h

define i32 @f(i32 %a, i32 %b) nounwind ssp {
entry:
  %add = add nsw i32 %b, %a
  %cmp = icmp eq i32 %add, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void (...)* @h(i32 %a, i32 %b) nounwind
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 %add
}

; CHECK: _g
; CHECK: orrs
; CHECK-NOT: cmp
; CHECK: blxeq _h

define i32 @g(i32 %a, i32 %b) nounwind ssp {
entry:
  %add = or i32 %b, %a
  %cmp = icmp eq i32 %add, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void (...)* @h(i32 %a, i32 %b) nounwind
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 %add
}

declare void @h(...)
