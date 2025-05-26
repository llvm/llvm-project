# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-objdump --no-print-imm-hex -d -r %t | FileCheck %s --check-prefix=OBJ

# RUN: not llvm-mc -filetype=obj -triple=x86_64 --defsym ERR=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# ASM:      movabsq $puts@PLTOFF, %rax
# OBJ:      movabsq $0, %rax
# OBJ-NEXT:   0000000000000002: R_X86_64_PLTOFF64 puts{{$}}
# OBJ:      movabsq $0, %rax
# OBJ-NEXT:   000000000000000c: R_X86_64_PLTOFF64 .text{{$}}

movabsq $puts@PLTOFF, %rax
movabsq $.text@PLTOFF, %rax

.ifdef ERR
# ERR: {{.*}}.s:[[#@LINE+1]]:1: error: 64 bit reloc applied to a field with a different size
movl $puts@PLTOFF, %eax
.endif
