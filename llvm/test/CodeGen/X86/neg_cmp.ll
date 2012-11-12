; RUN: llc < %s -march=x86-64 | FileCheck %s

; rdar://11245199
; PR12545
define void @f(i32 %x, i32 %y) nounwind uwtable ssp {
entry:
; CHECK: f:
; CHECK-NOT: neg
; CHECK: add
  %sub = sub i32 0, %y
  %cmp = icmp eq i32 %x, %sub
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @g() nounwind
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare void @g()
