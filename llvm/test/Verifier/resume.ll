; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

define void @test1() {
entry:
  resume { ptr, i32 } undef
; CHECK: ResumeInst needs to be in a function with a personality.
}
