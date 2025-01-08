; RUN: llc < %s -mtriple=mipsel -verify-machineinstrs
; Make sure machine verifier understands the last instruction of a basic block
; is not the terminator instruction after delay slot filler pass is run.

@g = external global i32

define void @foo() nounwind {
entry:
  %0 = load i32, ptr @g, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %add = add nsw i32 %0, 10
  store i32 %add, ptr @g, align 4
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}

