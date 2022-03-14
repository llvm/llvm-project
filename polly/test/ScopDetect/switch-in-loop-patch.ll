; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s

; CHECK-NOT: Valid

; Verify that we do not detect loops where the loop latch is a switch statement.
; Such loops are not yet supported by Polly.

define void @f() {
b:
  br label %d

d:
  switch i8 0, label %e [
    i8 71, label %d
    i8 56, label %d
  ]

e:
 ret void
}

