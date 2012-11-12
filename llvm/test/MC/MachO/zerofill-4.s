// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s

.zerofill __DATA,__bss,_fill0,1,0
.zerofill __DATA,__bss,_a,4,2
.zerofill __DATA,__bss,_fill1,1,0
.zerofill __DATA,__bss,_b,4,3
.zerofill __DATA,__bss,_fill2,1,0
.zerofill __DATA,__bss,_c,4,4
.zerofill __DATA,__bss,_fill3,1,0
.zerofill __DATA,__bss,_d,4,5

// CHECK: # Symbol 0
// CHECK: ('n_value', 0)
// CHECK: ('_string', '_fill0')
// CHECK: # Symbol 1
// CHECK: ('n_value', 4)
// CHECK: ('_string', '_a')
// CHECK: # Symbol 2
// CHECK: ('n_value', 8)
// CHECK: ('_string', '_fill1')
// CHECK: # Symbol 3
// CHECK: ('n_value', 16)
// CHECK: ('_string', '_b')
// CHECK: # Symbol 4
// CHECK: ('n_value', 20)
// CHECK: ('_string', '_fill2')
// CHECK: # Symbol 5
// CHECK: ('n_value', 32)
// CHECK: ('_string', '_c')
// CHECK: # Symbol 6
// CHECK: ('n_value', 36)
// CHECK: ('_string', '_fill3')
// CHECK: # Symbol 7
// CHECK: ('n_value', 64)
// CHECK: ('_string', '_d')
