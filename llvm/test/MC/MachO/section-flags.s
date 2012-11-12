// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s
//
// CHECK: # Section 0
// CHECK: 'section_name', '__text
// CHECK: 'flags', 0x80000000
// CHECK: # Section 1
// CHECK: 'section_name', '__data
// CHECK: 'flags', 0x400
        
        .text

        .data
f0:
        movl $0, %eax
