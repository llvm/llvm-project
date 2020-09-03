// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-objdump -d - | FileCheck  %s

// Test that we correctly relax these instructions into versions that use
// 16 or 32 bit immediate values.

bar:
// CHECK:      Disassembly of section imul:
// CHECK-EMPTY:
// CHECK-NEXT: <imul>:
// CHECK-NEXT:   0: 66 69 1d 00 00 00 00 00 00        imulw $0, (%rip), %bx
// CHECK-NEXT:   9: 69 1d 00 00 00 00 00 00 00 00     imull $0, (%rip), %ebx
// CHECK-NEXT:  13: 48 69 1d 00 00 00 00 00 00 00 00  imulq $0, (%rip), %rbx
        .section imul,"x"
        imul $foo, bar(%rip),  %bx
        imul $foo, bar(%rip),  %ebx
        imul $foo, bar(%rip),  %rbx


// CHECK:      <and>:
// CHECK-NEXT:   0: 66 81 25 00 00 00 00 00 00        andw $0, (%rip)
// CHECK-NEXT:   9: 81 25 00 00 00 00 00 00 00 00     andl $0, (%rip)
// CHECK-NEXT:  13: 48 81 25 00 00 00 00 00 00 00 00  andq $0, (%rip)
        .section and,"x"
        andw $foo, bar(%rip)
        andl $foo, bar(%rip)
        andq $foo, bar(%rip)

// CHECK:      <or>:
// CHECK-NEXT:   0: 66 81 0d 00 00 00 00 00 00        orw $0, (%rip)
// CHECK-NEXT:   9: 81 0d 00 00 00 00 00 00 00 00     orl $0, (%rip)
// CHECK-NEXT:  13: 48 81 0d 00 00 00 00 00 00 00 00  orq $0, (%rip)
        .section or,"x"
        orw $foo, bar(%rip)
        orl $foo, bar(%rip)
        orq $foo, bar(%rip)

// CHECK:      <xor>:
// CHECK-NEXT:   0: 66 81 35 00 00 00 00 00 00        xorw $0, (%rip)
// CHECK-NEXT:   9: 81 35 00 00 00 00 00 00 00 00     xorl $0, (%rip)
// CHECK-NEXT:  13: 48 81 35 00 00 00 00 00 00 00 00  xorq $0, (%rip)
        .section xor,"x"
        xorw $foo, bar(%rip)
        xorl $foo, bar(%rip)
        xorq $foo, bar(%rip)

// CHECK:      <add>:
// CHECK-NEXT:   0: 66 81 05 00 00 00 00 00 00        addw $0, (%rip)
// CHECK-NEXT:   9: 81 05 00 00 00 00 00 00 00 00     addl $0, (%rip)
// CHECK-NEXT:  13: 48 81 05 00 00 00 00 00 00 00 00  addq $0, (%rip)
        .section add,"x"
        addw $foo, bar(%rip)
        addl $foo, bar(%rip)
        addq $foo, bar(%rip)

// CHECK:      <sub>:
// CHECK-NEXT:   0: 66 81 2d 00 00 00 00 00 00        subw $0, (%rip)
// CHECK-NEXT:   9: 81 2d 00 00 00 00 00 00 00 00     subl $0, (%rip)
// CHECK-NEXT:  13: 48 81 2d 00 00 00 00 00 00 00 00  subq $0, (%rip)
        .section sub,"x"
        subw $foo, bar(%rip)
        subl $foo, bar(%rip)
        subq $foo, bar(%rip)

// CHECK:      <cmp>:
// CHECK-NEXT:   0: 66 81 3d 00 00 00 00 00 00        cmpw $0, (%rip)
// CHECK-NEXT:   9: 81 3d 00 00 00 00 00 00 00 00     cmpl $0, (%rip)
// CHECK-NEXT:  13: 48 81 3d 00 00 00 00 00 00 00 00  cmpq $0, (%rip)
        .section cmp,"x"
        cmpw $foo, bar(%rip)
        cmpl $foo, bar(%rip)
        cmpq $foo, bar(%rip)
