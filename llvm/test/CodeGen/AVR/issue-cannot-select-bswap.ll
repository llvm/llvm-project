; RUN: llc < %s -march=avr | FileCheck %s

declare i16 @llvm.bswap.i16(i16)

define i16 @foo(i16) {
; CHECK-LABEL: foo:
entry-block:
  %1 = tail call i16 @llvm.bswap.i16(i16 %0)
  ret i16 %1
}
