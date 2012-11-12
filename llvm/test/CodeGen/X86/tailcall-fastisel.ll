; RUN: llc < %s -march=x86-64 -tailcallopt -fast-isel | not grep TAILCALL

; Fast-isel shouldn't attempt to cope with tail calls.

%0 = type { i64, i32, i8* }

define fastcc i8* @"visit_array_aux<`Reference>"(%0 %arg, i32 %arg1) nounwind {
fail:                                             ; preds = %entry
  %tmp20 = tail call fastcc i8* @"visit_array_aux<`Reference>"(%0 %arg, i32 undef) ; <i8*> [#uses=1]
  ret i8* %tmp20
}

define i32 @foo() nounwind {
entry:
 %0 = tail call i32 (...)* @bar() nounwind       ; <i32> [#uses=1]
 ret i32 %0
}

declare i32 @bar(...) nounwind
