; RUN: opt < %s -adce | llvm-dis
; RUN: opt < %s -adce -verify-dom-info | llvm-dis

define void @foo() {
entry:
  br label %switch
switch:                    ; preds = %entry
  switch i32 undef, label %default [
    i32 2, label %two
    i32 5, label %five
    i32 4, label %four
  ]
four:                      ; preds = %switch
  br label %exit
five:                      ; preds = %switch
  br label %exit
two:                       ; preds = %switch
  br label %exit
default:                   ; preds = %switch
  br label %exit
exit:                      ; preds = %default, %two, %five, %four
  ret void
}

