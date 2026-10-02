; RUN: llc -mtriple=s390x-linux-gnu -stop-after=systemz-isel --simplify-mir < %s | FileCheck %s

@G = external global i64
define i64 @fun0() {
; CHECK:    %{{.*}}:addr64bit = LGRL target-flags(systemz-got) @G
  %Res = load i64, ptr @G
  ret i64 %Res
}

@x = thread_local(initialexec) global i32 0
define ptr@fun1() {
; CHECK:    %{{.*}}:addr64bit = LARL target-flags(systemz-indntpoff) @x
  ret ptr@x
}
