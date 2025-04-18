# RUN: llvm-mc -triple=riscv64 -filetype=obj %s | llvm-readobj -r - | FileCheck %s --check-prefixes=CHECK,RV64
# RUN: llvm-mc -triple=riscv32 -filetype=obj %s | llvm-readobj -r - | FileCheck %s --check-prefixes=CHECK,RV32

# RUN: not llvm-mc -triple=riscv64 -filetype=obj %s --defsym ERR=1 -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

.globl g
g:
l:

# CHECK:      Section ({{.*}}) .rela.data {
# CHECK-NEXT:   0x0 R_RISCV_PLT32 l 0x0
# CHECK-NEXT:   0x4 R_RISCV_PLT32 extern 0x4
# CHECK-NEXT:   0x8 R_RISCV_PLT32 g 0x8
# CHECK-NEXT: }
.data
.word %pltpcrel(l)
.word %pltpcrel(extern + 4), %pltpcrel(g + 8)

# CHECK:      Section ({{.*}}) .rela.data1 {
# CHECK-NEXT:   0x0 R_RISCV_GOT32_PCREL data1 0x0
# CHECK-NEXT:   0x4 R_RISCV_GOT32_PCREL extern 0x4
# RV32-NEXT:    0x8 R_RISCV_GOT32_PCREL extern 0xFFFFFFFB
# RV64-NEXT:    0x8 R_RISCV_GOT32_PCREL extern 0xFFFFFFFFFFFFFFFB
# CHECK-NEXT: }
.section .data1,"aw"
data1:
.word %gotpcrel(data1)
.word %gotpcrel(extern+4), %gotpcrel(extern-5)

.ifdef ERR
# ERR: [[#@LINE+1]]:7: error: %pltpcrel can only be used in a .word directive
.quad %pltpcrel(g)

# ERR: [[#@LINE+1]]:7: error: expected relocatable expression
.word %pltpcrel(g-.)

# ERR: [[#@LINE+1]]:7: error: expected relocatable expression
.word %pltpcrel(extern - und)

# ERR: [[#@LINE+1]]:7: error: %gotpcrel can only be used in a .word directive
.quad %gotpcrel(g)

# ERR: [[#@LINE+1]]:7: error: expected relocatable expression
.word %gotpcrel(extern - .)

# ERR: [[#@LINE+1]]:7: error: expected relocatable expression
.word %gotpcrel(extern - und)
.endif
