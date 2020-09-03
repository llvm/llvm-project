// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu %s -o - | llvm-objdump -d - | FileCheck  %s

// Test for proper instruction relaxation behavior for the push $imm
// instruction forms. This is the 32-bit version of the push $imm tests from
// relax-arith.s and relax-arith2.s.

// CHECK:      Disassembly of section push8:
// CHECK-EMPTY:
// CHECK-NEXT: <push8>:
// CHECK-NEXT:   0: 66 6a 80                      pushw $-128
// CHECK-NEXT:   3: 66 6a 7f                      pushw $127
// CHECK-NEXT:   6: 6a 80                         pushl $-128
// CHECK-NEXT:   8: 6a 7f                         pushl $127
        .section push8,"x"
        pushw $-128
        pushw $127
        push  $-128
        push  $127

// CHECK:      Disassembly of section push32:
// CHECK-EMPTY:
// CHECK-NEXT: <push32>:
// CHECK-NEXT:   0: 66 68 00 00                   pushw $0
// CHECK-NEXT:   4: 68 00 00 00 00                pushl $0
        .section push32,"x"
        pushw $foo
        push  $foo
