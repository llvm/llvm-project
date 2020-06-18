# RUN: not llvm-mc -triple i386-unknown-unknown < %s 2>&1 | FileCheck %s

add  $1\@, %eax
# CHECK: :[[@LINE-1]]:8: error: unexpected token in argument list

.macro A @
  mov  %eax, %eax
.endm
# CHECK: :[[@LINE-3]]:10: error: expected identifier in '.macro' directive

.rept 2
  addi $8, $8, \@
.endr
# CHECK: error: unknown token in expression
# CHECK: :[[@LINE-4]]:1: note: while in macro instantiation
# CHECK-NEXT: .rept 2

.rep 3
  addi $9, $9, \@
.endr
# CHECK: error: unknown token in expression
# CHECK: :[[@LINE-4]]:1: note: while in macro instantiation
# CHECK-NEXT: .rep 3
