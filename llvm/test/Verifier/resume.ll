; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

define void @test1() {
entry:
  resume { i8*, i32 } undef
; CHECK: ResumeInst needs to be in a function with a personality.
}
