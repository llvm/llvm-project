## Test relocations without specifiers. See also relocation-specifier.s for relocations with specifiers.
# RUN: llvm-mc %s -triple=sparcv9 | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc %s -triple=sparcv9 -filetype=obj -o %t
# RUN: llvm-objdump -Dr %t | FileCheck %s --check-prefix=OBJDUMP

# ASM:      call local
# ASM:      call local1
# ASM-NEXT: call undef
# OBJDUMP:      call 0x14
# OBJDUMP-NEXT: call 0x0
# OBJDUMP-NEXT:   R_SPARC_WDISP30 .text1+0x4
# OBJDUMP-NEXT: call 0x0
# OBJDUMP-NEXT:   R_SPARC_WDISP30 undef{{$}}
call local
call local1
call undef

# ASM:      or %g1, sym, %g3
# ASM-NEXT: or %g1, sym+4, %g3
# OBJDUMP:      or %g1, 0x0, %g3
# OBJDUMP-NEXT: 0000000c:  R_SPARC_13   sym{{$}}
# OBJDUMP-NEXT: or %g1, 0x0, %g3
# OBJDUMP-NEXT: 00000010:  R_SPARC_13   sym+0x4
or %g1, sym, %g3
or %g1, (sym+4), %g3

local:

# ASM:      brz %g1, undef
# ASM:      brlz %g1, .Ltmp{{.}}-8
# OBJDUMP:      brz %g1, 0x0
# OBJDUMP-NEXT:   R_SPARC_WDISP16 undef
# OBJDUMP-NEXT: brlz %g1, 0xfffe
# OBJDUMP-NEXT: bg %icc, 0x0
# OBJDUMP-NEXT:   R_SPARC_WDISP19 undef
# OBJDUMP-NEXT: bg %icc, 0x7fffe
# OBJDUMP-NEXT: cbn 0x0
# OBJDUMP-NEXT:   R_SPARC_WDISP22 undef
# OBJDUMP-NEXT: cbn 0x3ffffe
brz %g1, undef
brlz %g1, .-8
bg %icc, undef
bg %icc, .-8
cbn undef
cbn .-8

.section .text1,"ax"
nop
local1:

# OBJDUMP-LABEL: .data:
# OBJDUMP:      0: R_SPARC_32 .text1+0x8
# OBJDUMP:      4: R_SPARC_DISP32 .text1+0x8
# OBJDUMP:      8: R_SPARC_64 .text1+0x8
# OBJDUMP:     10: R_SPARC_DISP64 .text1+0x8
.data
.word local1+4
.word local1+4-.
.xword local1+4
.xword local1+4-.

# OBJDUMP:     18: R_SPARC_8 .text1+0x8
# OBJDUMP:     19: R_SPARC_DISP8 .text1+0x8
# OBJDUMP:     1a: R_SPARC_16 .text1+0x8
# OBJDUMP:     1c: R_SPARC_DISP16 .text1+0x8
.byte local1+4
.byte local1+4-.
.half local1+4
.half local1+4-.

# This test needs to placed last in the file
# ASM: .half	a-.Ltmp{{.}}{{$}}
.half a - .
.byte a - .
a:
