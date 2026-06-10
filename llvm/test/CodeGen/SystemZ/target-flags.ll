; RUN: llc -mtriple=s390x-linux-gnu -stop-after=systemz-isel --simplify-mir < %s | FileCheck %s

@G = external global i64

define i64 @fun() {
entry:
; CHECK:    %{{.*}}:addr64bit = LGRL target-flags(systemz-got) @G
  %Res = load i64, ptr @G
  ret i64 %Res
}
