; RUN: sed 's/foo/entry/' %s | opt -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s --check-prefix=CHECK

define void @test1() {
foo:
  br label %exit

exit:
  ret void
}
