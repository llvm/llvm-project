; RUN: llc < %s -mtriple=i686-apple-darwin -relocation-model=dynamic-no-pic | FileCheck %s

@var = external hidden global i32
@p = external hidden global ptr

define void @f() {
; CHECK:  movl    $_var+40, _p
  store ptr getelementptr (i32, ptr @var, i64 10), ptr @p
  ret void
}
