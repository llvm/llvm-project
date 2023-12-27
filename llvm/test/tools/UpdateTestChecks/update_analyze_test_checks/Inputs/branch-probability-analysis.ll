; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

define void @test1() {
entry:
  br label %exit

exit:
  ret void
}
