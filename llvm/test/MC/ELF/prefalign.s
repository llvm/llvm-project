# RUN: llvm-mc -triple x86_64 %s -o - | FileCheck --check-prefix=ASM %s
# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o %t
# RUN: llvm-readelf -SW %t | FileCheck --check-prefix=OBJ %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DIS %s

## MinAlign >= PrefAlign: the three-way rule is bounded by MinAlign regardless
## of body size, so sh_addralign stays at MinAlign.
# ASM: .section .text.f1
# ASM: .p2align 2
# ASM: .prefalign 2, .Lf1_end, 0
# OBJ: .text.f1          PROGBITS        0000000000000000 {{[0-9a-f]+}} 000003 00  AX  0   0  4
.section .text.f1,"ax",@progbits
.p2align 2
.prefalign 2, .Lf1_end, 0
.rept 3
clc
.endr
.Lf1_end:

## Multiple .prefalign on the same end symbol: effective PrefAlign is the maximum.
# ASM: .section .text.f2
# ASM: .prefalign 8, .Lf2_end, 0
# ASM: .prefalign 16, .Lf2_end, 0
# ASM: .prefalign 8, .Lf2_end, 0
# OBJ: .text.f2          PROGBITS        0000000000000000 {{[0-9a-f]+}} 000009 00  AX  0   0 16
.section .text.f2,"ax",@progbits
.p2align 2
.prefalign 8, .Lf2_end, 0
.prefalign 16, .Lf2_end, 0
.prefalign 8, .Lf2_end, 0
.rept 9
clc
.endr
.Lf2_end:

## Multiple functions in a section, each with its own .prefalign.
## nop fill; f3b's 5-byte padding is a NOP.
## f3b: ComputedAlign=8,  padding=5
## f3c: ComputedAlign=16, padding=0
# ASM: .prefalign 16, .Lf3a_end, nop
# ASM: .prefalign 16, .Lf3b_end, nop
# ASM: .prefalign 16, .Lf3c_end, 204
# OBJ: .text.f3          PROGBITS        0000000000000000 {{[0-9a-f]+}} 000020 00  AX  0   0 16
# DIS: Disassembly of section .text.f3:
# DIS:       0: clc
# DIS-NEXT:  1: clc
# DIS-NEXT:  2: clc
# DIS-NEXT:  3: nopl
# DIS-NEXT:  8: stc
# DIS:       f: stc
# DIS-NEXT: 10: clc
# DIS:      1f: clc
# DIS-EMPTY:
.section .text.f3,"ax",@progbits
.p2align 2
.prefalign 16, .Lf3a_end, nop
.rept 3
clc
.endr
.Lf3a_end:
.prefalign 16, .Lf3b_end, nop
.rept 8
stc
.endr
.Lf3b_end:
.prefalign 16, .Lf3c_end, 0xcc
.rept 16
clc
.endr
.Lf3c_end:
## No-op prefalign
.prefalign 16, .Lf3d_end, 0xcc
.Lf3d_end:
.prefalign 16, .Lf3a_end, 0xcc

## Two functions in one section where the second function's padding depends on
## the first function's size.
# OBJ: .text.f4          PROGBITS        0000000000000000 {{[0-9a-f]+}} 00001e 00  AX  0   0 16
# DIS: Disassembly of section .text.f4:
# DIS:       0: pushq
# DIS:       7: retq
# DIS-NEXT:  8: nopl
# DIS-NEXT: 10: movl
# DIS:      1d: retq
# DIS-EMPTY:
.section .text.f4,"ax",@progbits
.p2align 2
.prefalign 16, .Lf4a_end, nop
pushq %rbp
movq %rsp, %rbp
xorl %eax, %eax
popq %rbp
retq
.Lf4a_end:
.prefalign 16, .Lf4b_end, nop
movl $0, 0
xorl %eax, %eax
retq
.Lf4b_end:

## sh_addralign stays at 32, not downgraded by .prefalign.
# OBJ: .text.f5          PROGBITS        0000000000000000 {{[0-9a-f]+}} 000003 00  AX  0   0 32
.section .text.f5,"ax",@progbits
.p2align 5
.prefalign 16, .Lf5_end, 0
.rept 3
clc
.endr
.Lf5_end:

## .prefalign in a BSS section with zero fill.
# ASM: .bss
# ASM: .prefalign 16, .Lbss_end, 0
# OBJ: .bss              NOBITS          0000000000000000 {{[0-9a-f]+}} 000004 00  WA  0   0  4
.bss
.p2align 2
.prefalign 16, .Lbss_end, 0
.space 4
.Lbss_end:
