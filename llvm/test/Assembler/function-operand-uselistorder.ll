; RUN: verify-uselistorder %s

@g = global i8 0

define void @f1() prefix ptr @g prologue ptr @g personality ptr @g {
  ret void
}

define void @f2() prefix ptr @g prologue ptr @g personality ptr @g {
  ret void
}
