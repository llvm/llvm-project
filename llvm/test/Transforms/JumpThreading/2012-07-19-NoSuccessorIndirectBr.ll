; RUN: opt -S -passes=jump-threading < %s
; PR 13405
; Just check that it doesn't crash / assert

define i32 @f() nounwind {
entry:
  indirectbr ptr undef, []
}
