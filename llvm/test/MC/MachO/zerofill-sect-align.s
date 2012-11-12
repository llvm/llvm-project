// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s
//
// Check that the section itself is aligned.

        .byte 0
        
.zerofill __DATA,__bss,_a,1,0
.zerofill __DATA,__bss,_b,4,4

// CHECK: # Symbol 0
// CHECK: ('n_value', 16)
// CHECK: ('_string', '_a')
// CHECK: # Symbol 1
// CHECK: ('n_value', 32)
// CHECK: ('_string', '_b')
