; RUN: llc < %s
; PR4769
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"

define i16 @foo() nounwind readnone {
entry:
  %result = alloca i16, align 1                   ; <i16*> [#uses=2]
  store volatile i16 0, i16* %result
  %tmp = load volatile i16, i16* %result               ; <i16> [#uses=1]
  ret i16 %tmp
}

define i16 @main() nounwind {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %call = call i16 @bar() nounwind                ; <i16> [#uses=1]
  %tobool = icmp eq i16 %call, 0                  ; <i1> [#uses=1]
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  %result.i = alloca i16, align 1                 ; <i16*> [#uses=2]
  store volatile i16 0, i16* %result.i
  %tmp.i = load volatile i16, i16* %result.i           ; <i16> [#uses=0]
  ret i16 0
}

declare i16 @bar()
