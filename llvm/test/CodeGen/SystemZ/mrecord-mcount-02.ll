; RUN: not --crash llc %s -mtriple=s390x-linux-gnu -o - 2>&1 | FileCheck %s
;
; CHECK: LLVM ERROR: mrecord-mcount only supported with fentry-call

define void @test1() #0 {
entry:
  ret void
}

attributes #0 = { "instrument-function-entry-inlined"="mcount" "mrecord-mcount" }
