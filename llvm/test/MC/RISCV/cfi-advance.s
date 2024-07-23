# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=-relax %s -o %t.o
# RUN: llvm-readelf -sr %t.o | FileCheck %s --check-prefix=NORELAX
# RUN: llvm-dwarfdump --debug-frame %t.o 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-DWARFDUMP %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+relax %s -o %t.relax.o
# RUN: llvm-readelf -sr %t.relax.o | FileCheck %s --check-prefix=RELAX

# NORELAX:      Relocation section '.rela.text1' at offset {{.*}} contains 1 entries:
# NORELAX-NEXT:  Offset     Info    Type                Sym. Value  Symbol's Name + Addend
# NORELAX-NEXT: 00000000  00000313 R_RISCV_CALL_PLT       00000004   .L0 + 0
# NORELAX-EMPTY:
# NORELAX-NEXT: Relocation section '.rela.eh_frame' at offset {{.*}} contains 1 entries:
# NORELAX:       Offset     Info    Type                Sym. Value  Symbol's Name + Addend
# NORELAX-NEXT: 0000001c  00000139 R_RISCV_32_PCREL       00000000   .L0 + 0
# NORELAX-EMPTY:
# NORELAX:      Symbol table '.symtab' contains 13 entries:
# NORELAX-NEXT:    Num:    Value  Size Type    Bind   Vis       Ndx Name
# NORELAX-NEXT:      0: 00000000     0 NOTYPE  LOCAL  DEFAULT   UND
# NORELAX-NEXT:      1: 00000000     0 NOTYPE  LOCAL  DEFAULT     2 .L0 {{$}}
# NORELAX:           3: 00000004     0 NOTYPE  LOCAL  DEFAULT     2 .L0{{$}}
# NORELAX-NOT: .L0

# RELAX:        Relocation section '.rela.eh_frame' at offset {{.*}} contains 5 entries:
# RELAX-NEXT:    Offset     Info    Type                Sym. Value  Symbol's Name + Addend
# RELAX-NEXT:   0000001c  00000139 R_RISCV_32_PCREL       00000000   .L0 + 0
# RELAX-NEXT:   00000020  00000c23 R_RISCV_ADD32          0001017a   .L0 + 0
# RELAX-NEXT:   00000020  00000127 R_RISCV_SUB32          00000000   .L0 + 0
# RELAX-NEXT:   00000035  00000b35 R_RISCV_SET6           00010176   .L0 + 0
# RELAX-NEXT:   00000035  00000934 R_RISCV_SUB6           0001016e   .L0 + 0
# RELAX-EMPTY:
# RELAX:        Symbol table '.symtab' contains 16 entries:
# RELAX-NEXT:      Num:    Value  Size Type    Bind   Vis       Ndx Name
# RELAX-NEXT:        0: 00000000     0 NOTYPE  LOCAL  DEFAULT   UND
# RELAX-NEXT:        1: 00000000     0 NOTYPE  LOCAL  DEFAULT     2 .L0 {{$}}
# RELAX:             3: 00000004     0 NOTYPE  LOCAL  DEFAULT     2 .L0{{$}}
# RELAX:             9: 0001016e     0 NOTYPE  LOCAL  DEFAULT     2 .L0 {{$}}
# RELAX:            11: 00010176     0 NOTYPE  LOCAL  DEFAULT     2 .L0 {{$}}
# RELAX:            12: 0001017a     0 NOTYPE  LOCAL  DEFAULT     2 .L0 {{$}}

# CHECK-DWARFDUMP: DW_CFA_advance_loc1: 104
# CHECK-DWARFDUMP-NEXT: DW_CFA_def_cfa_offset: +8
# CHECK-DWARFDUMP-NEXT: DW_CFA_advance_loc2: 259
# CHECK-DWARFDUMP-NEXT: DW_CFA_def_cfa_offset: +8
# CHECK-DWARFDUMP-NEXT: DW_CFA_advance_loc4: 65539
# CHECK-DWARFDUMP-NEXT: DW_CFA_def_cfa_offset: +8
# CHECK-DWARFDUMP-NEXT: DW_CFA_advance_loc: 10
# CHECK-DWARFDUMP-NEXT: DW_CFA_def_cfa_offset: +8
        .text
        .globl  test                            # -- Begin function test
        .p2align        1
        .type   test,@function
test:
        .cfi_startproc
        nop
## This looks similar to fake label names ".L0 ". Even if this is ".L0 ",
## the assembler will not conflate it with fake labels.
.L0:
        .zero 100, 0x90
        .cfi_def_cfa_offset 8
        nop
        .zero 255, 0x90
        .cfi_def_cfa_offset 8
        nop
        .zero 65535, 0x90
        .cfi_def_cfa_offset 8
        nop
        .p2align 3
        .cfi_def_cfa_offset 8
        nop
        .cfi_endproc

.section .text1,"ax"
call .L0
