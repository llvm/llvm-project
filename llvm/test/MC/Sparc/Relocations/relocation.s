## Test relocations without specifiers. See also relocation-specifier.s for relocations with specifiers.
# RUN: llvm-mc %s -triple=sparcv9 | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc %s -triple=sparcv9 -filetype=obj -o %t
# RUN: llvm-objdump -dr %t | FileCheck %s --check-prefix=OBJDUMP

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

.section .text1,"ax"
nop
local1:

.data
# This test needs to placed last in the file
# ASM: .half	a-.Ltmp0
.half a - .
.byte a - .
a:
