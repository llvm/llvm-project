# RUN: not llvm-mc -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

.section .foo,"a",@progbits

# CHECK: {{.*}}.s:[[# @LINE+1]]:1: error: changed section type for .foo, expected: 0x1
.section .foo,"a",@init_array

# CHECK: {{.*}}.s:[[# @LINE+1]]:1: error: changed section type for .foo, expected: 0x1
.pushsection .foo,"a",@nobits

.pushsection .foo,"a",@progbits

## GNU as emits SHT_PROGBITS .eh_frame for .cfi_* directives. Don't error.
.section .eh_frame,"a",@progbits
.section .eh_frame,"a",@unwind
.pushsection .eh_frame,"a",@progbits

# CHECK: {{.*}}.s:[[# @LINE+1]]:1: error: changed section type for .eh_frame, expected: 0x70000001
.section .eh_frame,"a",@nobits
