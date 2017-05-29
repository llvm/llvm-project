; This tests that llc accepts Nios2 target.

; RUN: not not llc < %s -asm-verbose=false -march=nios2 2>&1 | FileCheck %s --check-prefix=ARCH
; RUN: not not llc < %s -asm-verbose=false -mtriple=nios2 2>&1 | FileCheck %s --check-prefix=TRIPLE

; ARCH-NOT: invalid target
; TRIPLE-NOT: unable to get target

define i32 @f(i32 %i) {
  ret i32 %i
}
