// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-objdump -d - | FileCheck  %s

// Test that we avoid relaxing these instructions and instead generate versions
// that use 8-bit immediate values.

bar:
// CHECK:      Disassembly of section imul:
// CHECK-EMPTY:
// CHECK-NEXT: imul:
// CHECK-NEXT:   0: 66 6b db 80                   imulw $-128, %bx, %bx
// CHECK-NEXT:   4: 66 6b 1c 25 00 00 00 00 7f    imulw $127, 0, %bx
// CHECK-NEXT:   d: 6b db 00                      imull $0, %ebx, %ebx
// CHECK-NEXT:  10: 6b 1c 25 00 00 00 00 01       imull $1, 0, %ebx
// CHECK-NEXT:  18: 48 6b db ff                   imulq $-1, %rbx, %rbx
// CHECK-NEXT:  1c: 48 6b 1c 25 00 00 00 00 2a    imulq $42, 0, %rbx
        .section imul,"x"
        imul $-128, %bx,  %bx
        imul $127, bar,  %bx
        imul $0, %ebx, %ebx
        imul $1, bar,  %ebx
        imul $-1, %rbx, %rbx
        imul $42, bar,  %rbx


// CHECK:      Disassembly of section and:
// CHECK-EMPTY:
// CHECK-NEXT: and:
// CHECK-NEXT:   0: 66 83 e3 7f                   andw $127, %bx
// CHECK-NEXT:   4: 66 83 24 25 00 00 00 00 00    andw $0, 0
// CHECK-NEXT:   d: 83 e3 01                      andl $1, %ebx
// CHECK-NEXT:  10: 83 24 25 00 00 00 00 ff       andl $-1, 0
// CHECK-NEXT:  18: 48 83 e3 2a                   andq $42, %rbx
// CHECK-NEXT:  1c: 48 83 24 25 00 00 00 00 80    andq $-128, 0
        .section and,"x"
        and  $127, %bx
        andw $0, bar
        and  $1, %ebx
        andl $-1, bar
        and  $42, %rbx
        andq $-128, bar

// CHECK:      Disassembly of section or:
// CHECK-EMPTY:
// CHECK-NEXT: or:
// CHECK-NEXT:   0: 66 83 cb 00                   orw $0, %bx
// CHECK-NEXT:   4: 66 83 0c 25 00 00 00 00 01    orw $1, 0
// CHECK-NEXT:   d: 83 cb ff                      orl $-1, %ebx
// CHECK-NEXT:  10: 83 0c 25 00 00 00 00 2a       orl $42, 0
// CHECK-NEXT:  18: 48 83 cb 80                   orq $-128, %rbx
// CHECK-NEXT:  1c: 48 83 0c 25 00 00 00 00 7f    orq $127, 0
        .section or,"x"
        or  $0, %bx
        orw $1, bar
        or  $-1, %ebx
        orl $42, bar
        or  $-128, %rbx
        orq $127, bar

// CHECK:      Disassembly of section xor:
// CHECK-EMPTY:
// CHECK-NEXT: xor:
// CHECK-NEXT:   0: 66 83 f3 01                   xorw $1, %bx
// CHECK-NEXT:   4: 66 83 34 25 00 00 00 00 ff    xorw $-1, 0
// CHECK-NEXT:   d: 83 f3 2a                      xorl $42, %ebx
// CHECK-NEXT:  10: 83 34 25 00 00 00 00 80       xorl $-128, 0
// CHECK-NEXT:  18: 48 83 f3 7f                   xorq $127, %rbx
// CHECK-NEXT:  1c: 48 83 34 25 00 00 00 00 00    xorq $0, 0
        .section xor,"x"
        xor  $1, %bx
        xorw $-1, bar
        xor  $42, %ebx
        xorl $-128, bar
        xor  $127, %rbx
        xorq $0, bar

// CHECK:      Disassembly of section add:
// CHECK-EMPTY:
// CHECK-NEXT: add:
// CHECK-NEXT:   0: 66 83 c3 ff                   addw $-1, %bx
// CHECK-NEXT:   4: 66 83 04 25 00 00 00 00 2a    addw $42, 0
// CHECK-NEXT:   d: 83 c3 80                      addl $-128, %ebx
// CHECK-NEXT:  10: 83 04 25 00 00 00 00 7f       addl $127, 0
// CHECK-NEXT:  18: 48 83 c3 00                   addq $0, %rbx
// CHECK-NEXT:  1c: 48 83 04 25 00 00 00 00 01    addq $1, 0
        .section add,"x"
        add  $-1, %bx
        addw $42, bar
        add  $-128, %ebx
        addl $127, bar
        add  $0, %rbx
        addq $1, bar

// CHECK:      Disassembly of section sub:
// CHECK-EMPTY:
// CHECK-NEXT: sub:
// CHECK-NEXT:   0: 66 83 eb 2a                   subw $42, %bx
// CHECK-NEXT:   4: 66 83 2c 25 00 00 00 00 80    subw $-128, 0
// CHECK-NEXT:   d: 83 eb 7f                      subl $127, %ebx
// CHECK-NEXT:  10: 83 2c 25 00 00 00 00 00       subl $0, 0
// CHECK-NEXT:  18: 48 83 eb 01                   subq $1, %rbx
// CHECK-NEXT:  1c: 48 83 2c 25 00 00 00 00 ff    subq $-1, 0
        .section sub,"x"
        sub  $42, %bx
        subw $-128, bar
        sub  $127, %ebx
        subl $0, bar
        sub  $1, %rbx
        subq $-1, bar

// CHECK:      Disassembly of section cmp:
// CHECK-EMPTY:
// CHECK-NEXT: cmp:
// CHECK-NEXT:   0: 66 83 fb 80                   cmpw $-128, %bx
// CHECK-NEXT:   4: 66 83 3c 25 00 00 00 00 7f    cmpw $127, 0
// CHECK-NEXT:   d: 83 fb 00                      cmpl $0, %ebx
// CHECK-NEXT:  10: 83 3c 25 00 00 00 00 01       cmpl $1, 0
// CHECK-NEXT:  18: 48 83 fb ff                   cmpq $-1, %rbx
// CHECK-NEXT:  1c: 48 83 3c 25 00 00 00 00 2a    cmpq $42, 0
        .section cmp,"x"
        cmp  $-128, %bx
        cmpw $127, bar
        cmp  $0, %ebx
        cmpl $1, bar
        cmp  $-1, %rbx
        cmpq $42, bar

// CHECK:      Disassembly of section push:
// CHECK-EMPTY:
// CHECK-NEXT: push:
// CHECK-NEXT:   0: 66 6a 80                      pushw $-128
// CHECK-NEXT:   3: 66 6a 7f                      pushw $127
// CHECK-NEXT:   6: 6a 80                         pushq $-128
// CHECK-NEXT:   8: 6a 7f                         pushq $127
        .section push,"x"
        pushw $-128
        pushw $127
        push  $-128
        push  $127
