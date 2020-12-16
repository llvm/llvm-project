; RUN: opt < %s -instcombine -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | grep " = phi " | count 1

; instcombine should sort the PHI operands so that simplifycfg can see the
; duplicate and remove it.

define i32 @foo(i1 %t) {
entry:
  call void @bar()
  br i1 %t, label %true, label %false
true:
  call void @bar()
  br label %false
false:
  %a = phi i32 [ 2, %true ], [ 5, %entry ]
  %b = phi i32 [ 5, %entry ], [ 2, %true ]
  call void @bar()
  %c = add i32 %a, %b
  ret i32 %c
}

declare void @bar()
