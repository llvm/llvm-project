; RUN: verify-uselistorder %s

@g = global i8 0
declare i32 @__gxx_personality_v0(...)

define void @f1() prefix ptr @g prologue ptr @g personality ptr @__gxx_personality_v0 {
  ret void
}

define void @f2() prefix ptr @g prologue ptr @g personality ptr @__gxx_personality_v0 {
  ret void
}
