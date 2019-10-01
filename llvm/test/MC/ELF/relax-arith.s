// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-objdump -d - | FileCheck  %s

// Test that we correctly relax these instructions into versions that use
// 16 or 32 bit immediate values.

bar:
// CHECK:      Disassembly of section imul:
// CHECK-EMPTY:
// CHECK-NEXT: imul:
// CHECK-NEXT:   0: 66 69 db 00 00                       imulw $0, %bx, %bx
// CHECK-NEXT:   5: 66 69 1c 25 00 00 00 00 00 00        imulw $0, 0, %bx
// CHECK-NEXT:   f: 69 db 00 00 00 00                    imull $0, %ebx, %ebx
// CHECK-NEXT:  15: 69 1c 25 00 00 00 00 00 00 00 00     imull $0, 0, %ebx
// CHECK-NEXT:  20: 48 69 db 00 00 00 00                 imulq $0, %rbx, %rbx
// CHECK-NEXT:  27: 48 69 1c 25 00 00 00 00 00 00 00 00  imulq $0, 0, %rbx
        .section imul,"x"
        imul $foo, %bx,  %bx
        imul $foo, bar,  %bx
        imul $foo, %ebx, %ebx
        imul $foo, bar,  %ebx
        imul $foo, %rbx, %rbx
        imul $foo, bar,  %rbx

// CHECK:      Disassembly of section and:
// CHECK-EMPTY:
// CHECK-NEXT: and:
// CHECK-NEXT:   0: 66 81 e3 00 00                       andw $0, %bx
// CHECK-NEXT:   5: 66 81 24 25 00 00 00 00 00 00        andw $0, 0
// CHECK-NEXT:   f: 81 e3 00 00 00 00                    andl $0, %ebx
// CHECK-NEXT:  15: 81 24 25 00 00 00 00 00 00 00 00     andl $0, 0
// CHECK-NEXT:  20: 48 81 e3 00 00 00 00                 andq $0, %rbx
// CHECK-NEXT:  27: 48 81 24 25 00 00 00 00 00 00 00 00  andq $0, 0
        .section and,"x"
        and  $foo, %bx
        andw $foo, bar
        and  $foo, %ebx
        andl $foo, bar
        and  $foo, %rbx
        andq $foo, bar

// CHECK:      Disassembly of section or:
// CHECK-EMPTY:
// CHECK-NEXT: or:
// CHECK-NEXT:   0: 66 81 cb 00 00                       orw $0, %bx
// CHECK-NEXT:   5: 66 81 0c 25 00 00 00 00 00 00        orw $0, 0
// CHECK-NEXT:   f: 81 cb 00 00 00 00                    orl $0, %ebx
// CHECK-NEXT:  15: 81 0c 25 00 00 00 00 00 00 00 00     orl $0, 0
// CHECK-NEXT:  20: 48 81 cb 00 00 00 00                 orq $0, %rbx
// CHECK-NEXT:  27: 48 81 0c 25 00 00 00 00 00 00 00 00  orq $0, 0
        .section or,"x"
        or  $foo, %bx
        orw $foo, bar
        or  $foo, %ebx
        orl $foo, bar
        or  $foo, %rbx
        orq $foo, bar

// CHECK:      Disassembly of section xor:
// CHECK-EMPTY:
// CHECK-NEXT: xor:
// CHECK-NEXT:   0: 66 81 f3 00 00                       xorw $0, %bx
// CHECK-NEXT:   5: 66 81 34 25 00 00 00 00 00 00        xorw $0, 0
// CHECK-NEXT:   f: 81 f3 00 00 00 00                    xorl $0, %ebx
// CHECK-NEXT:  15: 81 34 25 00 00 00 00 00 00 00 00     xorl $0, 0
// CHECK-NEXT:  20: 48 81 f3 00 00 00 00                 xorq $0, %rbx
// CHECK-NEXT:  27: 48 81 34 25 00 00 00 00 00 00 00 00  xorq $0, 0
        .section xor,"x"
        xor  $foo, %bx
        xorw $foo, bar
        xor  $foo, %ebx
        xorl $foo, bar
        xor  $foo, %rbx
        xorq $foo, bar

// CHECK:      Disassembly of section add:
// CHECK-EMPTY:
// CHECK-NEXT: add:
// CHECK-NEXT:   0: 66 81 c3 00 00                       addw $0, %bx
// CHECK-NEXT:   5: 66 81 04 25 00 00 00 00 00 00        addw $0, 0
// CHECK-NEXT:   f: 81 c3 00 00 00 00                    addl $0, %ebx
// CHECK-NEXT:  15: 81 04 25 00 00 00 00 00 00 00 00     addl $0, 0
// CHECK-NEXT:  20: 48 81 c3 00 00 00 00                 addq $0, %rbx
// CHECK-NEXT:  27: 48 81 04 25 00 00 00 00 00 00 00 00  addq $0, 0
        .section add,"x"
        add  $foo, %bx
        addw $foo, bar
        add  $foo, %ebx
        addl $foo, bar
        add  $foo, %rbx
        addq $foo, bar

// CHECK:      Disassembly of section sub:
// CHECK-EMPTY:
// CHECK-NEXT: sub:
// CHECK-NEXT:   0: 66 81 eb 00 00                       subw $0, %bx
// CHECK-NEXT:   5: 66 81 2c 25 00 00 00 00 00 00        subw $0, 0
// CHECK-NEXT:   f: 81 eb 00 00 00 00                    subl $0, %ebx
// CHECK-NEXT:  15: 81 2c 25 00 00 00 00 00 00 00 00     subl $0, 0
// CHECK-NEXT:  20: 48 81 eb 00 00 00 00                 subq $0, %rbx
// CHECK-NEXT:  27: 48 81 2c 25 00 00 00 00 00 00 00 00  subq $0, 0
        .section sub,"x"
        sub  $foo, %bx
        subw $foo, bar
        sub  $foo, %ebx
        subl $foo, bar
        sub  $foo, %rbx
        subq $foo, bar

// CHECK:      Disassembly of section cmp:
// CHECK-EMPTY:
// CHECK-NEXT: cmp:
// CHECK-NEXT:   0: 66 81 fb 00 00                       cmpw $0, %bx
// CHECK-NEXT:   5: 66 81 3c 25 00 00 00 00 00 00        cmpw $0, 0
// CHECK-NEXT:   f: 81 fb 00 00 00 00                    cmpl $0, %ebx
// CHECK-NEXT:  15: 81 3c 25 00 00 00 00 00 00 00 00     cmpl $0, 0
// CHECK-NEXT:  20: 48 81 fb 00 00 00 00                 cmpq $0, %rbx
// CHECK-NEXT:  27: 48 81 3c 25 00 00 00 00 00 00 00 00  cmpq $0, 0
        .section cmp,"x"
        cmp  $foo, %bx
        cmpw $foo, bar
        cmp  $foo, %ebx
        cmpl $foo, bar
        cmp  $foo, %rbx
        cmpq $foo, bar

// CHECK:      Disassembly of section push:
// CHECK-EMPTY:
// CHECK-NEXT: push:
// CHECK-NEXT:   0: 66 68 00 00                          pushw $0
// CHECK-NEXT:   4: 68 00 00 00 00                       pushq $0
        .section push,"x"
        pushw $foo
        push  $foo

// CHECK:      Disassembly of section adc:
// CHECK-EMPTY:
// CHECK-NEXT: adc:
// CHECK-NEXT:   0: 66 81 d3 00 00                       adcw $0, %bx
// CHECK-NEXT:   5: 66 81 14 25 00 00 00 00 00 00        adcw $0, 0
// CHECK-NEXT:   f: 81 d3 00 00 00 00                    adcl $0, %ebx
// CHECK-NEXT:  15: 81 14 25 00 00 00 00 00 00 00 00     adcl $0, 0
// CHECK-NEXT:  20: 48 81 d3 00 00 00 00                 adcq $0, %rbx
// CHECK-NEXT:  27: 48 81 14 25 00 00 00 00 00 00 00 00  adcq $0, 0
        .section adc,"x"
        adc  $foo, %bx
        adcw $foo, bar
        adc  $foo, %ebx
        adcl $foo, bar
        adc  $foo, %rbx
        adcq $foo, bar

// CHECK:      Disassembly of section sbb:
// CHECK-EMPTY:
// CHECK-NEXT: sbb:
// CHECK-NEXT:   0: 66 81 db 00 00                       sbbw $0, %bx
// CHECK-NEXT:   5: 66 81 1c 25 00 00 00 00 00 00        sbbw $0, 0
// CHECK-NEXT:   f: 81 db 00 00 00 00                    sbbl $0, %ebx
// CHECK-NEXT:  15: 81 1c 25 00 00 00 00 00 00 00 00     sbbl $0, 0
// CHECK-NEXT:  20: 48 81 db 00 00 00 00                 sbbq $0, %rbx
// CHECK-NEXT:  27: 48 81 1c 25 00 00 00 00 00 00 00 00  sbbq $0, 0
        .section sbb,"x"
        sbb  $foo, %bx
        sbbw $foo, bar
        sbb  $foo, %ebx
        sbbl $foo, bar
        sbb  $foo, %rbx
        sbbq $foo, bar
