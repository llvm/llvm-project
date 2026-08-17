# RUN: not llvm-mc -triple x86_64 %s -o /dev/null 2>&1 | FileCheck %s

# CHECK: <instantiation>:1:1: error: unknown directive
# CHECK: macro-like-diagnostic.s:5:1: note: while in macro instantiation
.irp reg,%rax
  .invalid_irp_directive_here \reg
.endr

# CHECK: <instantiation>:1:1: error: unknown directive
# CHECK: macro-like-diagnostic.s:11:1: note: while in macro instantiation
.irpc char,a
  .invalid_irpc_directive_here \char
.endr

# CHECK: <instantiation>:1:1: error: unknown directive
# CHECK: macro-like-diagnostic.s:17:1: note: while in macro instantiation
.rept 1
  .invalid_rept_directive_here
.endr
