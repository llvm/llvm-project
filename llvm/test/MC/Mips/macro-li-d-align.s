# RUN: llvm-mc %s -triple=mips -mcpu=mips32 -filetype=obj -o %t.o32
# RUN: llvm-objdump -dr --no-show-raw-insn %t.o32 | FileCheck %s --check-prefix=O32
# RUN: llvm-objdump -s -j .rodata %t.o32 | FileCheck %s --check-prefix=DATA
# RUN: llvm-mc %s -triple=mips64 -mcpu=mips64 -target-abi=n32 -filetype=obj -o %t.n32
# RUN: llvm-objdump -dr --no-show-raw-insn %t.n32 | FileCheck %s --check-prefix=N32
# RUN: llvm-objdump -s -j .rodata %t.n32 | FileCheck %s --check-prefix=DATA

# li.s leaves .rodata at an offset that is not a multiple of 8. The label for
# the li.d constant must be placed after the alignment padding, not before it.

        .text
        li.s    $f0, 1.1
        li.d    $f2, 2.2
        li.s    $f4, 3.3
        li.d    $4, 4.4

# O32:      lwc1 $f0, 0x0($1)
# O32-NEXT:   R_MIPS_LO16 .rodata
# O32:      ldc1 $f2, 0x8($1)
# O32-NEXT:   R_MIPS_LO16 .rodata
# O32:      lwc1 $f4, 0x10($1)
# O32-NEXT:   R_MIPS_LO16 .rodata
# O32:      addiu $1, $1, 0x18
# O32-NEXT:   R_MIPS_LO16 .rodata
# O32-NEXT: lw $4, 0x0($1)
# O32-NEXT: lw $5, 0x4($1)

# N32:      lwc1 $f0, 0x0($1)
# N32-NEXT:   R_MIPS_LO16 .rodata{{$}}
# N32:      ldc1 $f2, 0x0($1)
# N32-NEXT:   R_MIPS_LO16 .rodata+0x8
# N32:      lwc1 $f4, 0x0($1)
# N32-NEXT:   R_MIPS_LO16 .rodata+0x10
# N32:      addiu $1, $1, 0x0
# N32-NEXT:   R_MIPS_LO16 .rodata+0x18
# N32-NEXT: ld $4, 0x0($1)

# DATA:      Contents of section .rodata:
# DATA-NEXT:  0000 3f8ccccd 00000000 40019999 9999999a
# DATA-NEXT:  0010 40533333 00000000 40119999 9999999a
