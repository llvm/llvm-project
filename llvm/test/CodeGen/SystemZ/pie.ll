; RUN: llc -mtriple=s390x-linux-gnu -relocation-model=pic < %s | FileCheck %s

@foo = dso_local global i32 42

define dso_local ptr @get_foo() {
  ret ptr @foo
}

; CHECK: larl    %r2, foo{{$}}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"PIE Level", i32 2}
