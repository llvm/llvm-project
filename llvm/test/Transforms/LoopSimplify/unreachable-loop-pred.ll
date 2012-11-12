; RUN: opt -S -loop-simplify -disable-output -verify-loop-info -verify-dom-info < %s
; PR5235

; When loopsimplify inserts a preheader for this loop, it should add the new
; block to the enclosing loop and not get confused by the unreachable
; bogus loop entry.

define void @is_extract_cab() nounwind {
entry:
  br label %header

header:                                       ; preds = %if.end206, %cond.end66, %if.end23
  br label %while.body115

while.body115:                                    ; preds = %9, %if.end192, %if.end101
  br i1 undef, label %header, label %while.body115

foo:
  br label %while.body115
}
