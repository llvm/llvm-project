; This tests that llc accepts Nios2 processors.

; RUN: not not llc < %s -asm-verbose=false -march=nios2 -mcpu=nios2r1 2>&1 | FileCheck %s --check-prefix=ARCH
; RUN: not not llc < %s -asm-verbose=false -march=nios2 -mcpu=nios2r2 2>&1 | FileCheck %s --check-prefix=ARCH

; ARCH-NOT: is not a recognized processor

define i32 @f(i32 %i) {
  ret i32 %i
}
