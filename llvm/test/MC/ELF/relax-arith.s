// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that we correctly relax these instructions into versions that use
// 16 or 32 bit immediate values.

bar:
// CHECK: 'imul'
// CHECK: ('_section_data', '6669db00 0066691c 25000000 00000069 db000000 00691c25 00000000 00000000 4869db00 00000048 691c2500 00000000 000000')
        .section imul
        imul $foo, %bx,  %bx
        imul $foo, bar,  %bx
        imul $foo, %ebx, %ebx
        imul $foo, bar,  %ebx
        imul $foo, %rbx, %rbx
        imul $foo, bar,  %rbx

// CHECK: and'
// CHECK:('_section_data', '6681e300 00668124 25000000 00000081 e3000000 00812425 00000000 00000000 4881e300 00000048 81242500 00000000 000000')
        .section and
        and  $foo, %bx
        andw $foo, bar
        and  $foo, %ebx
        andl $foo, bar
        and  $foo, %rbx
        andq $foo, bar

// CHECK: 'or'
// CHECK: ('_section_data', '6681cb00 0066810c 25000000 00000081 cb000000 00810c25 00000000 00000000 4881cb00 00000048 810c2500 00000000 000000')
        .section or
        or  $foo, %bx
        orw $foo, bar
        or  $foo, %ebx
        orl $foo, bar
        or  $foo, %rbx
        orq $foo, bar

// CHECK: 'xor'
// CHECK: ('_section_data', '6681f300 00668134 25000000 00000081 f3000000 00813425 00000000 00000000 4881f300 00000048 81342500 00000000 000000')
        .section xor
        xor  $foo, %bx
        xorw $foo, bar
        xor  $foo, %ebx
        xorl $foo, bar
        xor  $foo, %rbx
        xorq $foo, bar

// CHECK: 'add'
// CHECK: ('_section_data', '6681c300 00668104 25000000 00000081 c3000000 00810425 00000000 00000000 4881c300 00000048 81042500 00000000 000000')
        .section add
        add  $foo, %bx
        addw $foo, bar
        add  $foo, %ebx
        addl $foo, bar
        add  $foo, %rbx
        addq $foo, bar

// CHECK: 'sub'
// CHECK: ('_section_data', '6681eb00 0066812c 25000000 00000081 eb000000 00812c25 00000000 00000000 4881eb00 00000048 812c2500 00000000 000000')
        .section sub
        sub  $foo, %bx
        subw $foo, bar
        sub  $foo, %ebx
        subl $foo, bar
        sub  $foo, %rbx
        subq $foo, bar

// CHECK: 'cmp'
// CHECK: ('_section_data', '6681fb00 0066813c 25000000 00000081 fb000000 00813c25 00000000 00000000 4881fb00 00000048 813c2500 00000000 000000')
        .section cmp
        cmp  $foo, %bx
        cmpw $foo, bar
        cmp  $foo, %ebx
        cmpl $foo, bar
        cmp  $foo, %rbx
        cmpq $foo, bar
