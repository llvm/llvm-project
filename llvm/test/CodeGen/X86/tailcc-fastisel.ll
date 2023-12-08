; RUN: llc < %s -mtriple=x86_64-apple-darwin -fast-isel -fast-isel-abort=1 | FileCheck %s

%0 = type { i64, i32, ptr }

define tailcc ptr @"visit_array_aux<`Reference>"(%0 %arg, i32 %arg1) nounwind {
fail:                                             ; preds = %entry
  %tmp20 = tail call tailcc ptr @"visit_array_aux<`Reference>"(%0 %arg, i32 undef) ; <ptr> [#uses=1]
; CHECK: jmp "_visit_array_aux<`Reference>" ## TAILCALL
  ret ptr %tmp20
}

define i32 @foo() nounwind {
entry:
 %0 = tail call i32 (...) @bar() nounwind       ; <i32> [#uses=1]
 ret i32 %0
}

declare i32 @bar(...) nounwind
