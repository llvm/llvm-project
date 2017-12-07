; This tests that llc generates 'ret' instruction in assembly output.

; RUN: llc < %s -march=nios2 2>&1 | FileCheck %s --check-prefix=ARCH

; ARCH: ret

define i32 @f(i32 %i) {
  ret i32 %i
}
