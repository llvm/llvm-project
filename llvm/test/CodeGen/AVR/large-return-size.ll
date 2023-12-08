; RUN: llc < %s -march=avr | FileCheck %s
define void @call_more_than_64_bits() {
; CHECK-LABEL: call_more_than_64_bits
entry-block:
  %foo = call { i64, i1 } @more_than_64_bits()
  ret void
}

declare { i64, i1 } @more_than_64_bits()
