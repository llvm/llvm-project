; RUN(some-feature): opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s --check-prefix=GATED

define void @test1() {
entry:
  br label %exit

exit:
  ret void
}
