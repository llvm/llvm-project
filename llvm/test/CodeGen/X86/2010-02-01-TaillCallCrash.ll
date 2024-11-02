; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu
; PR6196

%"char[]" = type [1 x i8]

@.str = external constant %"char[]", align 1      ; <ptr> [#uses=1]

define i32 @regex_subst() nounwind {
entry:
  %0 = tail call i32 @.str(i32 0) nounwind ; <i32> [#uses=1]
  ret i32 %0
}
