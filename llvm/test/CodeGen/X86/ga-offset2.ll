; RUN: llc < %s -mtriple=i686-apple-darwin -relocation-model=dynamic-no-pic | FileCheck %s

@var = external hidden global i32
@p = external hidden global i32*

define void @f() {
; CHECK:  movl    $_var+40, _p
  store i32* getelementptr (i32, i32* @var, i64 10), i32** @p
  ret void
}
