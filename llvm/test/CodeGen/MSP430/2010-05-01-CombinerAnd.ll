; RUN: llc < %s
; PR7001

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"
target triple = "msp430-elf"

define i16 @main() nounwind {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  br i1 undef, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %while.cond
  br label %land.end

land.end:                                         ; preds = %land.rhs, %while.cond
  %0 = phi i1 [ false, %while.cond ], [ undef, %land.rhs ] ; <i1> [#uses=1]
  br i1 %0, label %while.body, label %while.end

while.body:                                       ; preds = %land.end
  %tmp4 = load i16* undef                         ; <i16> [#uses=0]
  br label %while.cond

while.end:                                        ; preds = %land.end
  ret i16 undef
}
