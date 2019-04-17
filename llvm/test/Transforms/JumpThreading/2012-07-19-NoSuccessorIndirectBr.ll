; RUN: opt < %s -jump-threading
; PR 13405
; Just check that it doesn't crash / assert

define i32 @f() nounwind {
entry:
  indirectbr i8* undef, []
}
