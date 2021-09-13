; RUN: verify-uselistorder %s

@g = global i8 0

define void @f1() prefix i8* @g prologue i8* @g personality i8* @g {
  ret void
}

define void @f2() prefix i8* @g prologue i8* @g personality i8* @g {
  ret void
}
