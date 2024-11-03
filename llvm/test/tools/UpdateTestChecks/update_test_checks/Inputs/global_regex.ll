; RUN: opt < %s -S | FileCheck %s

@Bar = global i32 0
@OnlyFoo = global i32 1
@Baz = global i32 2

define i32 @t() {
  %v1 = load i32, ptr @Bar
  %v2 = load i32, ptr @OnlyFoo
  %v3 = load i32, ptr @Baz
  %a1 = add i32 %v1, %v2
  %a2 = add i32 %a1, %v3
  ret i32 %a2
}

