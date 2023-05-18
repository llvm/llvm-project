# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o %t
# RUN: llvm-objdump --no-print-imm-hex -dr %t | FileCheck %s
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYM

# RUN: not llvm-mc -filetype=obj -triple x86_64 --defsym ERR=1 %s 2>&1 | FileCheck %s --check-prefix=ERR

## If a fixup symbol is equated to an undefined symbol, convert the fixup
## to be against the target symbol, even if there is a variant (@PLT).
# CHECK:      callq {{.*}}
# CHECK-NEXT:   R_X86_64_PLT32  __GI_memcpy-0x4
# CHECK:      movabsq $0, %rax
# CHECK-NEXT:   R_X86_64_64  __GI_memcpy+0x2
memcpy = __GI_memcpy
call memcpy@PLT
movabsq $memcpy+2, %rax

# CHECK:      movq (%rip), %rax
# CHECK-NEXT:   R_X86_64_REX_GOTPCRELX  abs-0x4
movq abs@GOTPCREL(%rip), %rax
abs = 42

# CHECK:      movabsq $0, %rbx
# CHECK-NEXT:   R_X86_64_64  data_alias
.globl data_alias
.set data_alias, data
movabsq $data_alias, %rbx

## A local alias to a defined symbol still references a section symbol.
# CHECK:      movabsq $0, %rbx
# CHECK-NEXT:   R_X86_64_64  .data+0x1
.set data_alias_l, data
movabsq $data_alias_l, %rbx

.data
.byte 0
.globl data
data:

.ifdef ERR
.text
## Note, GNU as emits a relocation for this erroneous fixup.
# ERR: {{.*}}.s:[[#@LINE+2]]:1: error: expected relocatable expression
memcpy_plus_1 = __GI_memcpy+1
call memcpy_plus_1@PLT
.endif

## Redirected symbols do not have a symbol table entry.
# SYM:      NOTYPE  LOCAL  DEFAULT UND
# SYM-NEXT: NOTYPE  LOCAL  DEFAULT ABS abs
# SYM-NEXT: NOTYPE  LOCAL  DEFAULT   4 data_alias_l
# SYM-NEXT: SECTION LOCAL  DEFAULT   4 .data
# SYM-NEXT: NOTYPE  GLOBAL DEFAULT UND __GI_memcpy
# SYM-NEXT: NOTYPE  GLOBAL DEFAULT   4 data_alias
# SYM-NEXT: NOTYPE  GLOBAL DEFAULT   4 data
# SYM-NOT:  {{.}}
