# RUN: not llvm-mc -triple riscv32 -mattr=+zihintntl < %s 2>&1 | FileCheck --match-full-lines %s
# RUN: not llvm-mc -triple riscv64 -mattr=+zihintntl < %s 2>&1 | FileCheck --match-full-lines %s
#
# ntl.* hints are always available even without Zihintntl
# (riscv-non-isa/riscv-elf-psabi-doc#474).
#
# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck --match-full-lines %s
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck --match-full-lines %s

ntl.p1 1 # CHECK: <stdin>:[[@LINE]]:8: error: unexpected extra operand for instruction
ntl.pall 2 # CHECK: <stdin>:[[@LINE]]:10: error: unexpected extra operand for instruction
ntl.s1 3 # CHECK: <stdin>:[[@LINE]]:8: error: unexpected extra operand for instruction
ntl.all 4 # CHECK: <stdin>:[[@LINE]]:9: error: unexpected extra operand for instruction

ntl.p1 t0, t1 # CHECK: <stdin>:[[@LINE]]:8: error: unexpected extra operand for instruction
ntl.pall t0, t1 # CHECK: <stdin>:[[@LINE]]:10: error: unexpected extra operand for instruction
ntl.s1 t0, t1 # CHECK: <stdin>:[[@LINE]]:8: error: unexpected extra operand for instruction
ntl.all t0, t1 # CHECK: <stdin>:[[@LINE]]:9: error: unexpected extra operand for instruction
