// RUN: not llvm-mc -triple=x86_64-apple-macos -show-encoding < %s 2>&1 | FileCheck %s

movl %eax
// CHECK: :[[@LINE-1]]:1: error: too few operands for instruction
jmp L_undefined_temporary_symbol
// CHECK: :[[@LINE-1]]:5: error: assembler local symbol 'L_undefined_temporary_symbol' not defined
