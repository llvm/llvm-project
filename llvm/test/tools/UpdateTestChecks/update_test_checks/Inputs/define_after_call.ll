; RUN: opt < %s -S | FileCheck %s

define i32 @a() {
  %val = call i32 @b(i32 0)
  ret i32 %val
}

define i32 @b(i32 %arg) {
  ret i32 %arg
}
