// RUN: env DEBUG_PRODUCER="my producer" llvm-mc -g -triple i386-apple-darwin10 %s -filetype=obj -o %t
// RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

.globl _bar
_bar:
	ret

// CHECK:    DW_AT_producer [DW_FORM_string]	("my producer")
