; RUN: llc < %s -march=nvptx64 -verify-machineinstrs -o /dev/null
; Regression test: don't crash when analyzing branches like that one time

define void @crash() {
entry:
  br label %loop

loop:
  br label %loop
}
