; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: mrecord-mcount only supported with fentry-call

define void @test1() #0 {
entry:
  ret void
}

attributes #0 = { "instrument-function-entry-inlined"="mcount" "mrecord-mcount" }
